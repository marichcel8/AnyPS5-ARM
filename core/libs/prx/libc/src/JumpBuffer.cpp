#include "prx/libc/include/JumpBuffer.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/general/AsmMacros.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace JumpBuffer {
namespace {

constexpr std::size_t MaskOffset = 72;
constexpr std::size_t MaskSavedOffset = 88;
constexpr int SigBlock = 1;
constexpr int SigSetmask = 3;

std::atomic<Sigprocmask> guestSigprocmask{nullptr};

}

extern "C" {

void GuestSigprocmaskRegister_nid_no_patch(Sigprocmask sigprocmask) {
    guestSigprocmask.store(sigprocmask);
}

void APS5_VABI JumpBufferSaveMask_nid_no_patch(unsigned char* buffer) {
    const auto sigprocmask = guestSigprocmask.load();
    const std::int32_t saved = sigprocmask != nullptr && sigprocmask(SigBlock, nullptr, buffer + MaskOffset) == 0;
    std::memcpy(buffer + MaskSavedOffset, &saved, sizeof(saved));
}

void APS5_VABI JumpBufferRestoreMask_nid_no_patch(const unsigned char* buffer) {
    std::int32_t saved = 0;
    std::memcpy(&saved, buffer + MaskSavedOffset, sizeof(saved));
    const auto sigprocmask = guestSigprocmask.load();
    if (saved != 0 && sigprocmask != nullptr) sigprocmask(SigSetmask, buffer + MaskOffset, nullptr);
}

}

}

// setjmp/longjmp must capture the guest's own frame, so they are written directly in assembly with
// the guest (System V) calling convention. The saved state fits the guest's 96-byte jmp_buf:
// return address, rbx, rsp, rbp, r12-r15, MXCSR, the x87 control word and, as FreeBSD's do, the
// signal mask at 72 with the flag sigsetjmp keeps at 88, once libkernel has registered sigprocmask.

#if defined(__x86_64__)
#if defined(_WIN32) || defined(__APPLE__)
#define APS5_JUMP_ASM_CALL(name) "    call " APS5_ASM_SYMBOL(name) "\n"
#else
#define APS5_JUMP_ASM_CALL(name) "    call " name "@PLT\n"
#endif

asm(".text\n"
    APS5_ASM_FUNCTION("setjmp_nid_postfix")
    "    push %rdi\n"
    APS5_JUMP_ASM_CALL("JumpBufferSaveMask_nid_no_patch")
    "    pop %rdi\n"
    "    mov (%rsp), %rax\n"
    "    mov %rax, 0(%rdi)\n"
    "    mov %rbx, 8(%rdi)\n"
    "    lea 8(%rsp), %rax\n"
    "    mov %rax, 16(%rdi)\n"
    "    mov %rbp, 24(%rdi)\n"
    "    mov %r12, 32(%rdi)\n"
    "    mov %r13, 40(%rdi)\n"
    "    mov %r14, 48(%rdi)\n"
    "    mov %r15, 56(%rdi)\n"
    "    stmxcsr 64(%rdi)\n"
    "    fnstcw 68(%rdi)\n"
    "    xor %eax, %eax\n"
    "    ret\n"
    APS5_ASM_FUNCTION("longjmp_nid_postfix")
    "    mov %rdi, %rbx\n"
    "    mov %esi, %r12d\n"
    "    and $-16, %rsp\n"
    APS5_JUMP_ASM_CALL("JumpBufferRestoreMask_nid_no_patch")
#ifdef _WIN32
    "    mov 16(%rbx), %rdi\n"
    "    call GuestStackSwitch_nid_no_patch\n"
#endif
    "    mov %rbx, %rdi\n"
    "    mov %r12d, %esi\n"
    "    mov %esi, %eax\n"
    "    test %eax, %eax\n"
    "    jnz 1f\n"
    "    inc %eax\n"
    "1:\n"
    "    mov 8(%rdi), %rbx\n"
    "    mov 16(%rdi), %rsp\n"
    "    mov 24(%rdi), %rbp\n"
    "    mov 32(%rdi), %r12\n"
    "    mov 40(%rdi), %r13\n"
    "    mov 48(%rdi), %r14\n"
    "    mov 56(%rdi), %r15\n"
    "    ldmxcsr 64(%rdi)\n"
    "    fldcw 68(%rdi)\n"
    "    jmp *0(%rdi)\n");
#elif defined(__aarch64__)
namespace {

constexpr std::size_t SavedWords = 25;

std::mutex savedMutex;
std::unordered_map<const void*, std::array<std::uint64_t, SavedWords>> savedStates;

}

extern "C" {

void Aps5SetjmpStore_nid_no_patch(unsigned char* buffer, const std::uint64_t* block, std::uint64_t callerStack) {
    JumpBuffer::JumpBufferSaveMask_nid_no_patch(buffer);
    std::array<std::uint64_t, SavedWords> state{};
    std::memcpy(state.data(), block, (SavedWords - 1) * sizeof(std::uint64_t));
    state[SavedWords - 1] = callerStack;
    const std::lock_guard lock(savedMutex);
    savedStates[buffer] = state;
}

const std::uint64_t* Aps5LongjmpLoad_nid_no_patch(const unsigned char* buffer) {
    const std::lock_guard lock(savedMutex);
    const auto saved = savedStates.find(buffer);
    if (saved == savedStates.end()) NotImplemented_nid_no_patch("longjmp to a buffer that setjmp did not fill");
    static thread_local std::array<std::uint64_t, SavedWords> state;
    state = saved->second;
    JumpBuffer::JumpBufferRestoreMask_nid_no_patch(buffer);
    return state.data();
}

}

asm(".text\n"
    ".p2align 2\n"
    APS5_ASM_FUNCTION("setjmp_nid_postfix")
    "    stp x29, x30, [sp, #-192]!\n"
    "    mov x29, sp\n"
    "    stp x19, x20, [sp, #16]\n"
    "    stp x21, x22, [sp, #32]\n"
    "    stp x23, x24, [sp, #48]\n"
    "    stp x25, x26, [sp, #64]\n"
    "    stp x27, x28, [sp, #80]\n"
    "    stp d8, d9, [sp, #96]\n"
    "    stp d10, d11, [sp, #112]\n"
    "    stp d12, d13, [sp, #128]\n"
    "    stp d14, d15, [sp, #144]\n"
    "    mov x1, sp\n"
    "    add x2, sp, #192\n"
    "    bl " APS5_ASM_SYMBOL("Aps5SetjmpStore_nid_no_patch") "\n"
    "    mov w0, #0\n"
    "    ldp x29, x30, [sp], #192\n"
    "    ret\n"
    ".p2align 2\n"
    APS5_ASM_FUNCTION("longjmp_nid_postfix")
    "    mov w20, w1\n"
    "    bl " APS5_ASM_SYMBOL("Aps5LongjmpLoad_nid_no_patch") "\n"
    "    mov x9, x0\n"
    "    cmp w20, #0\n"
    "    csinc w10, w20, wzr, ne\n"
    "    ldp x29, x30, [x9]\n"
    "    ldp x19, x20, [x9, #16]\n"
    "    ldp x21, x22, [x9, #32]\n"
    "    ldp x23, x24, [x9, #48]\n"
    "    ldp x25, x26, [x9, #64]\n"
    "    ldp x27, x28, [x9, #80]\n"
    "    ldp d8, d9, [x9, #96]\n"
    "    ldp d10, d11, [x9, #112]\n"
    "    ldp d12, d13, [x9, #128]\n"
    "    ldp d14, d15, [x9, #144]\n"
    "    ldr x11, [x9, #192]\n"
    "    mov sp, x11\n"
    "    mov w0, w10\n"
    "    ret\n");

extern "C" {

int APS5_VABI setjmp_nid_postfix(std::uint64_t* buffer);
[[noreturn]] void APS5_VABI longjmp_nid_postfix(std::uint64_t* buffer, int value);

void Aps5JumpTargets_nid_no_patch(void** setjmpTarget, void** longjmpTarget) {
    *setjmpTarget = reinterpret_cast<void*>(&setjmp_nid_postfix);
    *longjmpTarget = reinterpret_cast<void*>(&longjmp_nid_postfix);
}

}
#endif
