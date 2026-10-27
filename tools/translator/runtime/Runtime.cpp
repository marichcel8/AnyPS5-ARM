#include "GuestState.hpp"
#include "../../../core/libs/prx/libkernel/System/include/UserContext.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unordered_map>

// Emitted by the translator: the guest image (its lowest segment at offset 0), its size, the table
// of translated functions sorted by guest offset, the guest entry point and the PT_TLS layout.
struct TranslatedFunction {
    std::uint64_t offset;
    void (*code)(GuestState*);
};

// The System V va_list the guest's variadic calls are read through (SceTypes.hpp VaList).
struct GuestVaList {
    std::uint32_t gpOffset;
    std::uint32_t fpOffset;
    void* overflowArea;
    void* registerSaveArea;
};

// What aps5_enter saves from a host call: the AAPCS argument registers. On return x[0], x[1], q[0]
// and q[1] hold the results.
struct EnterFrame {
    std::uint64_t x[8];
    std::uint8_t q[8][16];
};

extern "C" {
extern std::uint8_t aps5_guest_image[];
extern const std::uint64_t aps5_guest_image_size;
extern const TranslatedFunction aps5_guest_functions[];
extern const std::uint64_t aps5_guest_function_count;
extern const std::uint64_t aps5_guest_entry;
extern const std::uint64_t aps5_guest_tls[4]; // image offset, file size, block size, alignment
extern const char* const aps5_profile_path;

void aps5_host_bridge(GuestState* state, std::uint64_t target);
void aps5_enter();
int aps5_invoke_translated(const void*, const void*, const void*, const void*);
[[noreturn]] void aps5_native_restore(const std::uint64_t* registers);
[[noreturn]] void aps5_start_on_stack(void* top, void (*start)(std::uint64_t, std::uint64_t, std::uint64_t), std::uint64_t, std::uint64_t, std::uint64_t);
extern std::uint64_t aps5_capture_targets[4];
double aps5_x87_load80(const std::uint8_t* source);
void aps5_x87_store80(std::uint8_t* target, double value);
}

namespace {

constexpr std::size_t GuestStackBytes = 8u << 20;
constexpr std::size_t TcbBytes = 0x100;
constexpr std::uint64_t StackGuard = 0x5a17c0de2b6e4d91ull;

thread_local GuestState* threadState = nullptr;
// The state of the guest call into the host that is running on this thread, for variadic exports.
thread_local GuestState* bridgedState = nullptr;

// Fibers and contexts move a native stack to another host thread, so code that may have switched
// stacks looks the thread-local state up again instead of using an address computed before.
[[gnu::noinline]] GuestState*& Current() {
    asm volatile("" ::: "memory");
    return threadState;
}

[[gnu::noinline]] GuestState*& Bridged() {
    asm volatile("" ::: "memory");
    return bridgedState;
}

// Thrown when the guest unwinder resumes an older guest frame; aps5_call_impl returns to the
// translated code, which returns frame by frame until GuestState.resumeRip is its landing pad.
struct GuestResume {};

// The x86-64 DWARF register numbers 0-15 as GuestState registers.
constexpr unsigned DwarfRegisters[16] = {Rax, Rdx, Rcx, Rbx, Rsi, Rdi, Rbp, Rsp, R8, R9, R10, R11, R12, R13, R14, R15};

// The functions aps5_call hands to aps5_prepare_capture, in aps5_capture_targets order.
enum CaptureTarget { CaptureSetjmp, CaptureSigsetjmp, CaptureGetcontext, CaptureSwapcontext };

// The libkernel signal mask call over the guest's sigset_t (how: 1 block, 3 set).
using GuestSigprocmask = int (*)(int how, const void* set, void* old);
GuestSigprocmask guestSigprocmask = nullptr;
constexpr int GuestSigBlock = 1;
constexpr int GuestSigSetmask = 3;

// A setjmp that may still be jumped to. The guest's 96-byte jmp_buf holds the x86-64 state in the
// layout libc used under Rosetta (return address, rbx, rsp, rbp, r12-r15, mxcsr, x87 control word)
// plus a marker; the native registers of the translated caller of setjmp live here, keyed by the
// buffer's address.
constexpr std::uint64_t JumpMagic = 0x6a6d70c0de5a17b1ull;
constexpr std::size_t JumpMagicWord = 9;
constexpr std::size_t JumpPurgeSize = 64;

// sigsetjmp's buffer has the signal mask at byte 72 and whether it was saved at byte 88.
constexpr std::size_t SignalMaskOffset = 72;
constexpr std::size_t SignalSavedOffset = 88;

struct JumpRecord {
    std::uint64_t native[24];
    std::uint64_t guestRsp;
    GuestState* bridged;
};

thread_local std::unordered_map<std::uint64_t, JumpRecord> jumpRecords;
std::uint64_t longjmpTarget = 0;
std::uint64_t siglongjmpTarget = 0;

// A getcontext or swapcontext that may be resumed: the native registers of its translated caller and
// the GuestState it ran on, keyed by the guest ucontext_t. A context made by makecontext starts a
// coroutine on a native stack of its own, kept per guest stack.
struct ContextRecord {
    std::uint64_t native[24];
    GuestState* state;
    GuestState* bridged;
    std::uint64_t rip;
};

struct Coroutine {
    std::uint8_t* nativeStack;
    std::size_t nativeBytes;
    GuestState* state;
};

std::mutex contextLock;
std::unordered_map<std::uint64_t, ContextRecord> contextRecords;
std::unordered_map<std::uint64_t, Coroutine> coroutines;
thread_local const UserUcontext* swapTarget = nullptr;

// libc's exports with long double in their signature (libc/src/GuestX87.cpp), by address.
struct X87Export {
    const void* function;
    const char* signature;
};

std::unordered_map<std::uint64_t, const char*> x87Exports;
std::uint64_t x87Low = ~std::uint64_t(0);
std::uint64_t x87High = 0;

// A guest stack per thread for sceKernelRaiseException handlers, and the GuestState they interrupted.
constexpr std::size_t SignalStackBytes = 256u << 10;
thread_local std::uint8_t* signalStack = nullptr;
thread_local GuestState interrupted;

struct sigaction previousBus {}, previousSegv {};
// APS5_NATIVE_TRACE: report each redirected host call into guest code.
bool tracing = false;

std::uint64_t ImageBase() {
    return reinterpret_cast<std::uint64_t>(aps5_guest_image);
}

const TranslatedFunction* Find(std::uint64_t address) {
    const auto base = ImageBase();
    if (address < base || address - base >= aps5_guest_image_size) return nullptr;
    const auto offset = address - base;
    const auto* begin = aps5_guest_functions;
    const auto* end = aps5_guest_functions + aps5_guest_function_count;
    const auto* found = std::lower_bound(begin, end, offset, [](const TranslatedFunction& function, std::uint64_t value) { return function.offset < value; });
    return found != end && found->offset == offset ? found : nullptr;
}

[[noreturn]] void Fail(const char* what, std::uint64_t address) {
    std::fprintf(stderr, "[translator] %s 0x%" PRIx64 " (guest offset 0x%" PRIx64 ")\n", what, address, address - ImageBase());
    std::abort();
}

// FreeBSD TLS variant II, as the PS5 lays it out: the executable's block ends at the thread pointer,
// and the thread control block at the thread pointer starts with a pointer to itself; the stack
// guard is at fs:0x28.
std::uint64_t NewThreadPointer() {
    const auto offset = aps5_guest_tls[0], fileSize = aps5_guest_tls[1], blockSize = aps5_guest_tls[2];
    const auto alignment = std::max<std::uint64_t>(aps5_guest_tls[3], 16);
    const auto staticSize = (blockSize + alignment - 1) / alignment * alignment;
    auto* block = static_cast<std::uint8_t*>(std::aligned_alloc(alignment, staticSize + TcbBytes));
    if (block == nullptr) Fail("cannot allocate guest TLS of", staticSize);
    std::memset(block, 0, staticSize + TcbBytes);
    if (fileSize != 0) std::memcpy(block, aps5_guest_image + offset, fileSize);
    const auto threadPointer = reinterpret_cast<std::uint64_t>(block + staticSize);
    std::memcpy(block + staticSize, &threadPointer, 8);
    std::memcpy(block + staticSize + 0x28, &StackGuard, 8);
    return threadPointer;
}

GuestState* NewState() {
    auto* state = static_cast<GuestState*>(std::aligned_alloc(alignof(GuestState), sizeof(GuestState)));
    std::memset(state, 0, sizeof(*state));
    state->mxcsr = 0x1f80;
    state->x87Control = 0x037f;
    void* stack = mmap(nullptr, GuestStackBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (stack == MAP_FAILED) Fail("cannot map a guest stack for", 0);
    state->gpr[Rsp] = reinterpret_cast<std::uint64_t>(stack) + GuestStackBytes - 64;
    state->fsBase = NewThreadPointer();
    return state;
}

GuestState* ThreadState() {
    auto*& state = Current();
    if (state == nullptr) state = NewState();
    return state;
}

const void* ResolveCallback(const void* function) {
    const auto* found = Find(reinterpret_cast<std::uint64_t>(function));
    return found != nullptr ? reinterpret_cast<const void*>(found->code) : nullptr;
}

// Host code calling a guest function pointer executes the guest image, which is not executable.
// The fault is redirected to aps5_enter with the translated function in x16, so every host callback
// site (thread entries, atexit, module init, library callbacks) runs translated code unchanged.
void Redirect(int signal, siginfo_t* info, void* context) {
    auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
    const auto pc = static_cast<std::uint64_t>(machine->__ss.__pc);
    if (const auto* function = Find(pc)) {
        machine->__ss.__x[16] = reinterpret_cast<std::uint64_t>(function->code);
        machine->__ss.__pc = reinterpret_cast<std::uint64_t>(&aps5_enter);
        if (tracing) std::fprintf(stderr, "[translator] host call into guest offset 0x%" PRIx64 "\n", pc - ImageBase());
        return;
    }
    const auto& previous = signal == SIGBUS ? previousBus : previousSegv;
    if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
        previous.sa_sigaction(signal, info, context);
        return;
    }
    if (previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN) {
        previous.sa_handler(signal);
        return;
    }
    if (pc >= ImageBase() && pc - ImageBase() < aps5_guest_image_size) Fail("host call into untranslated guest code at", pc);
    // The process dies on the re-raised fault; say where it was first.
    Dl_info where {};
    const bool named = dladdr(reinterpret_cast<void*>(pc), &where) != 0;
    std::fprintf(stderr, "[translator] signal %d at pc 0x%" PRIx64 " (%s+0x%" PRIx64 "), address %p, lr 0x%" PRIx64 "\n", signal, pc,
                 named && where.dli_sname ? where.dli_sname : "?", named && where.dli_saddr ? pc - reinterpret_cast<std::uint64_t>(where.dli_saddr) : 0,
                 info->si_addr, static_cast<std::uint64_t>(machine->__ss.__lr));
    // Re-raise with the default action.
    struct sigaction fallback {};
    fallback.sa_handler = SIG_DFL;
    sigaction(signal, &fallback, nullptr);
}

struct ProfileRow {
    std::uint64_t offset;
    std::uint64_t count;
};

struct Profile {
    std::mutex lock;
    std::map<std::uint64_t, std::uint64_t> earlier;
};

Profile* profile = nullptr;

void WriteProfile() {
    unsigned long bytes = 0;
    const auto* rows = reinterpret_cast<const ProfileRow*>(getsectiondata(&_mh_execute_header, "__DATA", "__aps5_prof", &bytes));
    const std::lock_guard<std::mutex> guard(profile->lock);
    auto counts = profile->earlier;
    for (std::size_t i = 0; rows != nullptr && i < bytes / sizeof(ProfileRow); ++i) {
        if (rows[i].count != 0) counts[rows[i].offset] += rows[i].count;
    }
    const std::string partial = std::string(aps5_profile_path) + ".part";
    FILE* file = std::fopen(partial.c_str(), "w");
    if (file == nullptr) return;
    for (const auto& [offset, count] : counts) std::fprintf(file, "%" PRIx64 " %" PRIu64 "\n", offset, count);
    if (std::fclose(file) == 0) std::rename(partial.c_str(), aps5_profile_path);
}

void StartProfile() {
    if (aps5_profile_path == nullptr) return;
    profile = new Profile;
    if (FILE* file = std::fopen(aps5_profile_path, "r")) {
        std::uint64_t offset = 0, count = 0;
        while (std::fscanf(file, "%" SCNx64 " %" SCNu64, &offset, &count) == 2) profile->earlier[offset] += count;
        std::fclose(file);
    }
    std::atexit(WriteProfile);
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            WriteProfile();
        }
    }).detach();
}

// Fills the guest jmp_buf for the setjmp or sigsetjmp call whose return address is at rsp, and leaves
// the state as the first return of setjmp: rax 0, return address popped. Records older than rsp
// belong to frames that have returned, which can no longer be jumped to.
JumpRecord& PrepareJump(GuestState* state, std::uint64_t buffer, bool signal) {
    const auto rsp = state->gpr[Rsp];
    if (jumpRecords.size() >= JumpPurgeSize) {
        for (auto it = jumpRecords.begin(); it != jumpRecords.end();) it = it->second.guestRsp < rsp ? jumpRecords.erase(it) : std::next(it);
    }
    auto& record = jumpRecords[buffer];
    record.guestRsp = rsp + 8;
    record.bridged = Bridged();
    std::uint64_t words[10] = {};
    std::memcpy(&words[0], reinterpret_cast<const void*>(rsp), 8);
    words[1] = state->gpr[Rbx];
    words[2] = rsp + 8;
    words[3] = state->gpr[Rbp];
    for (unsigned i = 0; i < 4; ++i) words[4 + i] = state->gpr[R12 + i];
    std::memcpy(reinterpret_cast<std::uint8_t*>(&words[8]), &state->mxcsr, 4);
    std::memcpy(reinterpret_cast<std::uint8_t*>(&words[8]) + 4, &state->x87Control, 2);
    words[JumpMagicWord] = JumpMagic;
    std::memcpy(reinterpret_cast<void*>(buffer), words, signal ? SignalMaskOffset : sizeof(words));
    if (signal) {
        const auto save = static_cast<std::uint32_t>(state->gpr[Rsi]);
        std::memcpy(reinterpret_cast<void*>(buffer + SignalSavedOffset), &save, 4);
        if (save != 0 && guestSigprocmask != nullptr) guestSigprocmask(GuestSigBlock, nullptr, reinterpret_cast<void*>(buffer + SignalMaskOffset));
    }
    state->gpr[Rax] = 0;
    state->gpr[Rsp] = rsp + 8;
    return record;
}

[[noreturn]] void GuestLongjmp(GuestState* state, bool signal) {
    const auto buffer = state->gpr[Rdi];
    const auto value = static_cast<std::uint32_t>(state->gpr[Rsi]);
    std::uint64_t words[10];
    std::memcpy(words, reinterpret_cast<const void*>(buffer), sizeof(words));
    const auto found = jumpRecords.find(buffer);
    if (found == jumpRecords.end() || (!signal && words[JumpMagicWord] != JumpMagic) || found->second.guestRsp != words[2]) Fail("longjmp to a jmp_buf without a live setjmp at", buffer);
    if (signal) {
        std::uint32_t saved = 0;
        std::memcpy(&saved, reinterpret_cast<const void*>(buffer + SignalSavedOffset), 4);
        if (saved != 0 && guestSigprocmask != nullptr) guestSigprocmask(GuestSigSetmask, reinterpret_cast<const void*>(buffer + SignalMaskOffset), nullptr);
    }
    state->gpr[Rbx] = words[1];
    state->gpr[Rsp] = words[2];
    state->gpr[Rbp] = words[3];
    for (unsigned i = 0; i < 4; ++i) state->gpr[R12 + i] = words[4 + i];
    std::memcpy(&state->mxcsr, reinterpret_cast<const std::uint8_t*>(&words[8]), 4);
    std::memcpy(&state->x87Control, reinterpret_cast<const std::uint8_t*>(&words[8]) + 4, 2);
    state->gpr[Rax] = value != 0 ? value : 1;
    Bridged() = found->second.bridged;
    aps5_native_restore(found->second.native);
}

constexpr std::size_t FxsaveControl = 0, FxsaveStatus = 2, FxsaveMxcsr = 24, FxsaveX87 = 32, FxsaveXmm = 160;
constexpr std::uint64_t ContextFpFormatXmm = 0x10002, ContextOwnedFpu = 0x20001;

std::uint64_t PackFlags(const GuestState* state) {
    return 0x202 | std::uint64_t(state->cf) | std::uint64_t(state->pf) << 2 | std::uint64_t(state->af) << 4 | std::uint64_t(state->zf) << 6 | std::uint64_t(state->sf) << 7 |
           std::uint64_t(state->df) << 10 | std::uint64_t(state->of) << 11;
}

void UnpackFlags(GuestState* state, std::uint64_t flags) {
    state->cf = flags & 1;
    state->pf = (flags >> 2) & 1;
    state->af = (flags >> 4) & 1;
    state->zf = (flags >> 6) & 1;
    state->sf = (flags >> 7) & 1;
    state->df = (flags >> 10) & 1;
    state->of = (flags >> 11) & 1;
}

// The guest's ucontext_t for the getcontext or swapcontext call whose return address is at rsp, as
// libkernel fills it on x86-64: the registers after the call returned with rax 0.
void SaveContext(const GuestState* state, UserUcontext* context) {
    UserUcontext captured {};
    if (guestSigprocmask != nullptr) guestSigprocmask(GuestSigBlock, nullptr, captured.sigmask);
    auto& m = captured.mcontext;
    m.rdi = state->gpr[Rdi];
    m.rsi = state->gpr[Rsi];
    m.rdx = state->gpr[Rdx];
    m.rcx = state->gpr[Rcx];
    m.r8 = state->gpr[R8];
    m.r9 = state->gpr[R9];
    m.rbx = state->gpr[Rbx];
    m.rbp = state->gpr[Rbp];
    m.r10 = state->gpr[R10];
    m.r11 = state->gpr[R11];
    m.r12 = state->gpr[R12];
    m.r13 = state->gpr[R13];
    m.r14 = state->gpr[R14];
    m.r15 = state->gpr[R15];
    std::memcpy(&m.rip, reinterpret_cast<const void*>(state->gpr[Rsp]), 8);
    m.rsp = state->gpr[Rsp] + 8;
    m.rflags = PackFlags(state) & ~std::uint64_t(1);
    m.cs = 0x43;
    m.ss = 0x3b;
    m.len = sizeof(UserMcontext);
    m.fpformat = ContextFpFormatXmm;
    m.ownedfp = ContextOwnedFpu;
    auto* fx = reinterpret_cast<std::uint8_t*>(m.fpstate);
    const std::uint16_t status = static_cast<std::uint16_t>((state->x87Status & ~0x3800u) | (state->x87Top & 7u) << 11);
    std::memcpy(fx + FxsaveControl, &state->x87Control, 2);
    std::memcpy(fx + FxsaveStatus, &status, 2);
    std::memcpy(fx + FxsaveMxcsr, &state->mxcsr, 4);
    for (unsigned i = 0; i < 8; ++i) aps5_x87_store80(fx + FxsaveX87 + i * 16, state->x87[(state->x87Top + i) & 7]);
    for (unsigned i = 0; i < 16; ++i) std::memcpy(fx + FxsaveXmm + i * 16, state->ymm[i], 16);
    std::memcpy(context, &captured, offsetof(UserUcontext, link));
}

void LoadContext(GuestState* state, const UserMcontext& m) {
    state->gpr[Rax] = m.rax;
    state->gpr[Rbx] = m.rbx;
    state->gpr[Rcx] = m.rcx;
    state->gpr[Rdx] = m.rdx;
    state->gpr[Rsi] = m.rsi;
    state->gpr[Rdi] = m.rdi;
    state->gpr[Rbp] = m.rbp;
    state->gpr[Rsp] = m.rsp;
    state->gpr[R8] = m.r8;
    state->gpr[R9] = m.r9;
    state->gpr[R10] = m.r10;
    state->gpr[R11] = m.r11;
    state->gpr[R12] = m.r12;
    state->gpr[R13] = m.r13;
    state->gpr[R14] = m.r14;
    state->gpr[R15] = m.r15;
    UnpackFlags(state, m.rflags);
    const auto* fx = reinterpret_cast<const std::uint8_t*>(m.fpstate);
    std::uint16_t status = 0;
    std::memcpy(&state->x87Control, fx + FxsaveControl, 2);
    std::memcpy(&status, fx + FxsaveStatus, 2);
    state->x87Top = static_cast<std::uint8_t>((status >> 11) & 7);
    state->x87Status = static_cast<std::uint16_t>(status & ~0x3800u);
    std::memcpy(&state->mxcsr, fx + FxsaveMxcsr, 4);
    for (unsigned i = 0; i < 8; ++i) state->x87[(state->x87Top + i) & 7] = aps5_x87_load80(fx + FxsaveX87 + i * 16);
    for (unsigned i = 0; i < 16; ++i) {
        std::memcpy(state->ymm[i], fx + FxsaveXmm + i * 16, 16);
        std::memset(state->ymm[i] + 16, 0, 16);
    }
}

GuestState* NewFiberState(std::uint64_t rsp) {
    auto* state = static_cast<GuestState*>(std::aligned_alloc(alignof(GuestState), sizeof(GuestState)));
    std::memset(state, 0, sizeof(*state));
    const auto* current = Current();
    state->mxcsr = current != nullptr ? current->mxcsr : 0x1f80;
    state->x87Control = current != nullptr ? current->x87Control : 0x037f;
    state->fsBase = current != nullptr ? current->fsBase : NewThreadPointer();
    state->gpr[Rsp] = rsp;
    return state;
}

[[noreturn]] void ResumeContext(const UserUcontext* context) {
    if (guestSigprocmask != nullptr) guestSigprocmask(GuestSigSetmask, context->sigmask, nullptr);
    const auto& m = context->mcontext;
    std::unique_lock<std::mutex> guard(contextLock);
    if (const auto found = contextRecords.find(reinterpret_cast<std::uint64_t>(context)); found != contextRecords.end() && found->second.rip == m.rip) {
        const auto record = found->second;
        guard.unlock();
        LoadContext(record.state, m);
        if (auto* current = Current(); current != nullptr && current != record.state) record.state->fsBase = current->fsBase;
        Current() = record.state;
        Bridged() = record.bridged;
        aps5_native_restore(record.native);
    }
    if (m.rip >= ImageBase() && m.rip - ImageBase() < aps5_guest_image_size) Fail("setcontext to a context without a live getcontext at", m.rip);
    const auto stack = reinterpret_cast<std::uint64_t>(context->stackPointer);
    auto& coroutine = coroutines[stack];
    const std::size_t bytes = (context->stackSize * 16 + (1u << 20) + 0x3fff) & ~std::size_t(0x3fff);
    if (coroutine.nativeStack != nullptr && coroutine.nativeBytes < bytes) {
        munmap(coroutine.nativeStack, coroutine.nativeBytes);
        coroutine.nativeStack = nullptr;
    }
    if (coroutine.nativeStack == nullptr) {
        void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (mapping == MAP_FAILED) Fail("cannot map a native stack for the context at", reinterpret_cast<std::uint64_t>(context));
        coroutine.nativeStack = static_cast<std::uint8_t*>(mapping);
        coroutine.nativeBytes = bytes;
    }
    if (coroutine.state == nullptr) coroutine.state = NewFiberState(0);
    auto* state = coroutine.state;
    guard.unlock();
    if (auto* current = Current()) state->fsBase = current->fsBase;
    LoadContext(state, m);
    Current() = state;
    Bridged() = nullptr;
    aps5_start_on_stack(coroutine.nativeStack + coroutine.nativeBytes, reinterpret_cast<void (*)(std::uint64_t, std::uint64_t, std::uint64_t)>(m.rip), m.rdi, m.rsi, m.rdx);
}

[[noreturn]] void SwapContinue(std::uint64_t*) {
    ResumeContext(swapTarget);
}

// A guest call of a libc export whose signature has long double: the guest passes long double
// arguments in memory after the return address and takes a long double result in st(0); the host
// takes and returns doubles in vector registers. A long double* argument points at an 80-bit value,
// which the host function reads and writes as a double.
void CallX87(GuestState* state, std::uint64_t target, const char* signature) {
    static constexpr unsigned integers[] = {Rdi, Rsi, Rdx, Rcx, R8, R9};
    alignas(16) GuestState call {};
    std::uint64_t stack[12] = {};
    call.gpr[Rsp] = reinterpret_cast<std::uint64_t>(stack);
    double pointed[8] = {};
    std::uint64_t written[8] = {};
    unsigned gp = 0, fp = 0, hostGp = 0, hostFp = 0, memory = 0, pointers = 0;
    auto hostInteger = [&](std::uint64_t value) {
        if (hostGp < 6) call.gpr[integers[hostGp]] = value;
        else if (hostGp < 8) stack[1 + hostGp - 6] = value;
        else Fail("x87 export with more than 8 integer arguments at", target);
        ++hostGp;
    };
    auto hostDouble = [&](double value) {
        if (hostFp >= 8) Fail("x87 export with more than 8 floating-point arguments at", target);
        std::memcpy(call.ymm[hostFp++], &value, 8);
    };
    for (const char* letter = signature + 1; *letter != 0; ++letter) {
        switch (*letter) {
        case 'i':
            hostInteger(state->gpr[integers[gp++]]);
            break;
        case 'I':
            hostInteger(state->gpr[integers[gp++]]);
            hostInteger(state->gpr[integers[gp++]]);
            break;
        case 'd': case 'f':
            if (hostFp >= 8) Fail("x87 export with more than 8 floating-point arguments at", target);
            std::memcpy(call.ymm[hostFp++], state->ymm[fp++], 16);
            break;
        case 'L':
            hostDouble(aps5_x87_load80(reinterpret_cast<const std::uint8_t*>(state->gpr[Rsp] + 8 + 16 * memory++)));
            break;
        case 'P': case 'p': {
            const auto guest = state->gpr[integers[gp++]];
            if (guest == 0) {
                hostInteger(0);
                break;
            }
            pointed[pointers] = aps5_x87_load80(reinterpret_cast<const std::uint8_t*>(guest));
            if (*letter == 'P') written[pointers] = guest;
            hostInteger(reinterpret_cast<std::uint64_t>(&pointed[pointers++]));
            break;
        }
        default:
            Fail("x87 export with an unknown signature at", target);
        }
    }
    aps5_host_bridge(&call, target);
    for (unsigned i = 0; i < pointers; ++i) {
        if (written[i] != 0) aps5_x87_store80(reinterpret_cast<std::uint8_t*>(written[i]), pointed[i]);
    }
    auto push = [&](const std::uint8_t* lane) {
        double value = 0;
        std::memcpy(&value, lane, 8);
        state->x87Top = static_cast<std::uint8_t>((state->x87Top - 1) & 7);
        state->x87[state->x87Top] = value;
    };
    switch (signature[0]) {
    case 'L':
        push(call.ymm[0]);
        break;
    case 'C':
        push(call.ymm[1]);
        push(call.ymm[0]);
        break;
    case 'd': case 'f':
        std::memcpy(state->ymm[0], call.ymm[0], 16);
        break;
    case 'I':
        state->gpr[Rdx] = call.gpr[Rdx];
        state->gpr[Rax] = call.gpr[Rax];
        break;
    case 'i':
        state->gpr[Rax] = call.gpr[Rax];
        break;
    default:
        break;
    }
}

void InstallRedirect() {
    tracing = std::getenv("APS5_NATIVE_TRACE") != nullptr;
    if (auto* targets = reinterpret_cast<void (*)(void**, void**)>(dlsym(RTLD_DEFAULT, "Aps5JumpTargets_nid_no_patch"))) {
        void* jumpSet = nullptr;
        void* jumpLong = nullptr;
        targets(&jumpSet, &jumpLong);
        aps5_capture_targets[CaptureSetjmp] = reinterpret_cast<std::uint64_t>(jumpSet);
        longjmpTarget = reinterpret_cast<std::uint64_t>(jumpLong);
    }
    using ContextTargets = void (*)(void**, void**, void**, void**, void**);
    if (auto* targets = reinterpret_cast<ContextTargets>(dlsym(RTLD_DEFAULT, "Aps5ContextTargets_nid_no_patch"))) {
        void* found[5] = {};
        targets(&found[0], &found[1], &found[2], &found[3], &found[4]);
        if (found[0] != nullptr) aps5_capture_targets[CaptureSigsetjmp] = reinterpret_cast<std::uint64_t>(found[0]);
        siglongjmpTarget = reinterpret_cast<std::uint64_t>(found[1]);
        if (found[2] != nullptr) aps5_capture_targets[CaptureGetcontext] = reinterpret_cast<std::uint64_t>(found[2]);
        if (found[3] != nullptr) aps5_capture_targets[CaptureSwapcontext] = reinterpret_cast<std::uint64_t>(found[3]);
        guestSigprocmask = reinterpret_cast<GuestSigprocmask>(found[4]);
    }
    if (auto* exports = reinterpret_cast<void (*)(const void**, std::size_t*)>(dlsym(RTLD_DEFAULT, "Aps5X87Exports_nid_no_patch"))) {
        const void* table = nullptr;
        std::size_t count = 0;
        exports(&table, &count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto& entry = static_cast<const X87Export*>(table)[i];
            x87Exports.emplace(reinterpret_cast<std::uint64_t>(entry.function), entry.signature);
            x87Low = std::min(x87Low, reinterpret_cast<std::uint64_t>(entry.function));
            x87High = std::max(x87High, reinterpret_cast<std::uint64_t>(entry.function));
        }
    }
    if (auto* hooks = reinterpret_cast<void (*)(const void* (*)(const void*), int (*)(const void*, const void*, const void*, const void*))>(dlsym(RTLD_DEFAULT, "Aps5CallbackHooks_nid_no_patch")))
        hooks(ResolveCallback, aps5_invoke_translated);
    struct sigaction action {};
    action.sa_sigaction = Redirect;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    sigaction(SIGBUS, &action, &previousBus);
    sigaction(SIGSEGV, &action, &previousSegv);
}

}

extern "C" {

std::uint64_t aps5_capture_targets[4] = {~0ull, ~0ull, ~0ull, ~0ull};

struct Capture {
    std::uint64_t* native;
    void (*then)(std::uint64_t*);
};

// Called by aps5_call (HostBridge.S) for a guest call to a function in aps5_capture_targets: does the
// guest side of the call and returns the area for the native registers of the translated caller,
// and for swapcontext the continuation that switches to the other context.
Capture aps5_prepare_capture(GuestState* state, std::uint64_t target) {
    if (target == aps5_capture_targets[CaptureSetjmp] || target == aps5_capture_targets[CaptureSigsetjmp])
        return {PrepareJump(state, state->gpr[Rdi], target == aps5_capture_targets[CaptureSigsetjmp]).native, nullptr};
    static thread_local std::uint64_t discarded[24];
    auto* context = reinterpret_cast<UserUcontext*>(state->gpr[Rdi]);
    const bool swap = target == aps5_capture_targets[CaptureSwapcontext];
    const auto* next = swap ? reinterpret_cast<const UserUcontext*>(state->gpr[Rsi]) : nullptr;
    if (context == nullptr || (swap && next == nullptr)) {
        state->gpr[Rax] = ~std::uint64_t(0);
        state->gpr[Rsp] += 8;
        return {discarded, nullptr};
    }
    SaveContext(state, context);
    const std::lock_guard<std::mutex> guard(contextLock);
    auto& record = contextRecords[reinterpret_cast<std::uint64_t>(context)];
    record.state = state;
    record.bridged = Bridged();
    record.rip = context->mcontext.rip;
    state->gpr[Rax] = 0;
    state->gpr[Rsp] += 8;
    if (!swap) return {record.native, nullptr};
    swapTarget = next;
    return {record.native, SwapContinue};
}

// libkernel's setcontext on a guest ucontext_t.
[[noreturn]] void aps5_context_set(const UserUcontext* context) {
    ResumeContext(context);
}

// For the libc unwinder: the x86-64 registers (DWARF numbers 0-16) of the guest call into the host
// that is running on this thread, as its caller sees them once the call has returned.
int aps5_guest_context(std::uint64_t* registers) {
    const auto* state = Bridged();
    if (state == nullptr) return -1;
    for (unsigned i = 0; i < 16; ++i) registers[i] = state->gpr[DwarfRegisters[i]];
    std::memcpy(&registers[16], reinterpret_cast<const void*>(state->gpr[Rsp]), 8);
    registers[7] = state->gpr[Rsp] + 8;
    return 0;
}

// Installs the registers the libc unwinder computed for a landing pad and resumes the guest there.
[[noreturn]] void aps5_guest_resume(const std::uint64_t* registers) {
    auto* state = Bridged();
    if (state == nullptr) Fail("guest unwinder resumed without a guest call running at", registers[16]);
    for (unsigned i = 0; i < 16; ++i) state->gpr[DwarfRegisters[i]] = registers[i];
    state->resumeRip = registers[16];
    state->resumeRsp = registers[7];
    throw GuestResume {};
}

// For libSceFiber: a GuestState for a fiber whose guest stack ends at rsp, with this thread's TLS. A
// fiber without a context (rsp 0) runs below the thread's guest stack.
GuestState* aps5_fiber_state(std::uint64_t rsp, std::uint32_t mxcsr, std::uint16_t x87Control) {
    if (rsp == 0) rsp = (ThreadState()->gpr[Rsp] - 256) & ~std::uint64_t(15);
    auto* state = NewFiberState(rsp);
    if (mxcsr != 0) {
        state->mxcsr = mxcsr;
        state->x87Control = x87Control;
    }
    return state;
}

// Makes next the guest state of this host thread and returns the one before; a fiber that moved to
// another host thread gets this thread's TLS.
GuestState* aps5_fiber_switch(GuestState* next) {
    auto* previous = Current();
    if (previous != nullptr && previous != next) next->fsBase = previous->fsBase;
    Current() = next;
    Bridged() = next;
    return previous;
}

void aps5_fiber_release(GuestState* state) {
    std::free(state);
}

// For libkernel's sceKernelRaiseException delivery on this thread: the guest registers (16 in DWARF
// order, rip, rflags, mxcsr, x87 control) and a guest stack for the handler. They are exact while the
// thread is in a guest call into the host; interrupted translated code may hold newer values, and
// its GuestState comes back unchanged.
int aps5_signal_enter(std::uint64_t* registers) {
    auto* state = Current();
    if (state == nullptr) return -1;
    if (signalStack == nullptr) {
        void* mapping = mmap(nullptr, SignalStackBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) return -1;
        signalStack = static_cast<std::uint8_t*>(mapping);
    }
    const bool exact = Bridged() == state;
    for (unsigned i = 0; i < 16; ++i) registers[i] = state->gpr[DwarfRegisters[i]];
    registers[16] = 0;
    if (exact) {
        std::memcpy(&registers[16], reinterpret_cast<const void*>(state->gpr[Rsp]), 8);
        registers[7] = state->gpr[Rsp] + 8;
    }
    registers[17] = PackFlags(state);
    registers[18] = state->mxcsr;
    registers[19] = state->x87Control;
    std::memcpy(&interrupted, state, sizeof(GuestState));
    state->gpr[Rsp] = reinterpret_cast<std::uint64_t>(signalStack + SignalStackBytes - 64);
    return 0;
}

void aps5_signal_leave(const std::uint64_t* registers) {
    auto* state = Current();
    std::memcpy(state, &interrupted, sizeof(GuestState));
    if (Bridged() != state) return;
    for (unsigned i = 0; i < 16; ++i) {
        if (DwarfRegisters[i] != Rsp) state->gpr[DwarfRegisters[i]] = registers[i];
    }
    UnpackFlags(state, registers[17]);
    state->mxcsr = static_cast<std::uint32_t>(registers[18]);
    state->x87Control = static_cast<std::uint16_t>(registers[19]);
}

// A guest call or jump whose target the translator could not resolve statically. The caller has
// pushed the guest return address; a translated callee pops it with its ret, a host callee does not.
void aps5_call_impl(GuestState* state, std::uint64_t target) {
    if (const auto* function = Find(target)) {
        function->code(state);
        return;
    }
    if (target >= ImageBase() && target - ImageBase() < aps5_guest_image_size) {
        Dl_info self {};
        dladdr(reinterpret_cast<void*>(&aps5_call_impl), &self);
        std::fprintf(stderr, "[translator] from translated code at %p (executable offset 0x%llx)\n", __builtin_return_address(0),
                     static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(__builtin_return_address(0)) - reinterpret_cast<std::uint64_t>(self.dli_fbase)));
        Fail("call into untranslated guest code at", target);
    }
    if (target == longjmpTarget || target == siglongjmpTarget) GuestLongjmp(state, target == siglongjmpTarget);
    auto* outer = Bridged();
    Bridged() = state;
    try {
        const auto x87 = target - x87Low <= x87High - x87Low ? x87Exports.find(target) : x87Exports.end();
        if (x87 != x87Exports.end()) CallX87(state, target, x87->second);
        else aps5_host_bridge(state, target);
    } catch (const GuestResume&) {
        Bridged() = outer;
        return;
    }
    Bridged() = outer;
    state->gpr[Rsp] += 8;
}

// Called by aps5_enter: runs a translated function for a host caller on this thread's guest state.
// A host function that guest code called keeps the guest's callee-saved registers and rsp, because
// the nested guest function preserves them like any System V callee.
void aps5_enter_guest(EnterFrame* frame, void (*code)(GuestState*)) {
    auto* state = ThreadState();
    static constexpr unsigned arguments[] = {Rdi, Rsi, Rdx, Rcx, R8, R9};
    for (unsigned i = 0; i < 6; ++i) state->gpr[arguments[i]] = frame->x[i];
    for (unsigned i = 0; i < 8; ++i) {
        std::memcpy(state->ymm[i], frame->q[i], 16);
        std::memset(state->ymm[i] + 16, 0, 16);
    }
    const auto saved = state->gpr[Rsp];
    // Below an interrupted guest frame's red zone; arguments 7 and 8 on the stack, then a return address.
    auto rsp = ((saved - 128) & ~std::uint64_t(15)) - 16;
    std::memcpy(reinterpret_cast<void*>(rsp), &frame->x[6], 16);
    rsp -= 8;
    std::memset(reinterpret_cast<void*>(rsp), 0, 8);
    state->gpr[Rsp] = rsp;
    code(state);
    // A landing pad in an older guest frame, behind the host code that called this guest function.
    if (state->resumeRip != 0) throw GuestResume {};
    state->gpr[Rsp] = saved;
    frame->x[0] = state->gpr[Rax];
    frame->x[1] = state->gpr[Rdx];
    std::memcpy(frame->q[0], state->ymm[0], 16);
    std::memcpy(frame->q[1], state->ymm[1], 16);
}

// For the host libraries' variadic exports: a System V va_list over the arguments of the guest call
// running on this thread, after fixedGp integer and fixedFp vector named arguments. saveArea is 176
// bytes the caller keeps alive while it reads the list.
int aps5_guest_va_start(GuestVaList* list, void* saveArea, unsigned fixedGp, unsigned fixedFp) {
    const auto* state = Bridged();
    if (state == nullptr) return -1;
    static constexpr unsigned arguments[] = {Rdi, Rsi, Rdx, Rcx, R8, R9};
    auto* area = static_cast<std::uint8_t*>(saveArea);
    for (unsigned i = 0; i < 6; ++i) std::memcpy(area + i * 8, &state->gpr[arguments[i]], 8);
    for (unsigned i = 0; i < 8; ++i) std::memcpy(area + 48 + i * 16, state->ymm[i], 16);
    list->gpOffset = fixedGp * 8;
    list->fpOffset = 48 + fixedFp * 16;
    list->overflowArea = reinterpret_cast<void*>(state->gpr[Rsp] + 8);
    list->registerSaveArea = area;
    return 0;
}

[[noreturn]] void aps5_trap(GuestState* state, std::uint64_t address) {
    static_cast<void>(state);
    Fail("guest trap (ud2/int3) at", address);
}

[[noreturn]] void aps5_unsupported(GuestState* state, std::uint64_t address) {
    static_cast<void>(state);
    Fail("untranslated instruction at", address);
}

}

int main(int argc, char** argv, char** envp) {
    InstallRedirect();
    StartProfile();
    // The FreeBSD start block the guest entry receives: argc, argv[], NULL, envp[], NULL.
    int envc = 0;
    while (envp != nullptr && envp[envc] != nullptr) ++envc;
    auto* block = static_cast<std::uint64_t*>(std::calloc(static_cast<std::size_t>(argc + envc) + 3, sizeof(std::uint64_t)));
    block[0] = static_cast<std::uint64_t>(argc);
    for (int i = 0; i < argc; ++i) block[1 + i] = reinterpret_cast<std::uint64_t>(argv[i]);
    for (int i = 0; i < envc; ++i) block[2 + argc + i] = reinterpret_cast<std::uint64_t>(envp[i]);
    const auto entryAddress = ImageBase() + aps5_guest_entry;
    if (Find(entryAddress) == nullptr) Fail("no translated guest entry at", entryAddress);
    auto* entry = reinterpret_cast<int (*)(void*, void*)>(entryAddress);
    // With libkernel, the guest runs on its own thread while the main thread serves AppKit.
    using StartGuest = void (*)(void (*)(void*, void*), void*);
    if (auto* start = reinterpret_cast<StartGuest>(dlsym(RTLD_DEFAULT, "Aps5StartGuest_nid_no_patch"))) {
        start(reinterpret_cast<void (*)(void*, void*)>(entryAddress), block);
    }
    return entry(block, nullptr);
}

extern "C" {

void aps5_cpuid(GuestState* state) {
    const auto leaf = static_cast<std::uint32_t>(state->gpr[Rax]);
    const auto subleaf = static_cast<std::uint32_t>(state->gpr[Rcx]);
    std::uint32_t a = 0, b = 0, c = 0, d = 0;
    static const char brand[49] = "AnyPS5 x86-64 (AMD Zen 2 compatible)";
    switch (leaf) {
    case 0:
        a = 0xd;
        b = 0x68747541;
        d = 0x69746e65;
        c = 0x444d4163;
        break;
    case 1:
        a = 0x00870f10;
        b = 0x00080800;
        c = (1u << 0) | (1u << 9) | (1u << 12) | (1u << 13) | (1u << 19) | (1u << 20) | (1u << 23) | (1u << 26) | (1u << 27) | (1u << 28) | (1u << 29);
        d = (1u << 0) | (1u << 4) | (1u << 8) | (1u << 15) | (1u << 19) | (1u << 23) | (1u << 24) | (1u << 25) | (1u << 26);
        break;
    case 7:
        if (subleaf == 0) b = (1u << 3) | (1u << 5) | (1u << 8);
        break;
    case 0xd:
        if (subleaf == 0) {
            a = 7;
            b = 0x340;
            c = 0x340;
        }
        break;
    case 0x80000000:
        a = 0x80000004;
        b = 0x68747541;
        d = 0x69746e65;
        c = 0x444d4163;
        break;
    case 0x80000001:
        c = (1u << 0) | (1u << 5) | (1u << 6);
        d = (1u << 11) | (1u << 20) | (1u << 29);
        break;
    case 0x80000002: case 0x80000003: case 0x80000004: {
        std::uint32_t words[4];
        std::memcpy(words, brand + (leaf - 0x80000002) * 16, 16);
        a = words[0];
        b = words[1];
        c = words[2];
        d = words[3];
        break;
    }
    default:
        break;
    }
    state->gpr[Rax] = a;
    state->gpr[Rbx] = b;
    state->gpr[Rcx] = c;
    state->gpr[Rdx] = d;
}

double aps5_x87_load80(const std::uint8_t* source) {
    std::uint64_t mantissa = 0;
    std::uint16_t top = 0;
    std::memcpy(&mantissa, source, 8);
    std::memcpy(&top, source + 8, 2);
    const int exponent = top & 0x7fff;
    const double sign = (top & 0x8000) != 0 ? -1.0 : 1.0;
    if (exponent == 0x7fff) {
        if ((mantissa << 1) == 0) return sign * HUGE_VAL;
        return std::copysign(std::nan(""), sign);
    }
    if (mantissa == 0) return sign * 0.0;
    return sign * std::ldexp(static_cast<double>(mantissa), (exponent == 0 ? 1 : exponent) - 16383 - 63);
}

void aps5_x87_store80(std::uint8_t* target, double value) {
    std::uint64_t mantissa = 0;
    std::uint16_t top = std::signbit(value) ? 0x8000 : 0;
    const double magnitude = std::fabs(value);
    if (std::isnan(value)) {
        mantissa = 0xc000000000000000ull;
        top |= 0x7fff;
    } else if (std::isinf(value)) {
        mantissa = 0x8000000000000000ull;
        top |= 0x7fff;
    } else if (magnitude != 0.0) {
        int exponent = 0;
        const double fraction = std::frexp(magnitude, &exponent);
        mantissa = static_cast<std::uint64_t>(std::ldexp(fraction, 64));
        top |= static_cast<std::uint16_t>(exponent - 1 + 16383);
    }
    std::memcpy(target, &mantissa, 8);
    std::memcpy(target + 8, &top, 2);
}

double aps5_x87_loadbcd(const std::uint8_t* source) {
    double value = 0.0;
    for (int i = 8; i >= 0; --i) value = value * 100.0 + (source[i] >> 4) * 10.0 + (source[i] & 0x0f);
    return (source[9] & 0x80) != 0 ? -value : value;
}

void aps5_x87_storebcd(std::uint8_t* target, double value) {
    std::memset(target, 0, 10);
    const double magnitude = std::fabs(value);
    if (!(magnitude < 1e18)) {
        target[7] = 0xc0;
        target[8] = 0xff;
        target[9] = 0xff;
        return;
    }
    auto digits = static_cast<std::uint64_t>(magnitude);
    for (int i = 0; i < 9; ++i) {
        target[i] = static_cast<std::uint8_t>(digits % 10 | (digits / 10 % 10) << 4);
        digits /= 100;
    }
    if (std::signbit(value)) target[9] = 0x80;
}

}
