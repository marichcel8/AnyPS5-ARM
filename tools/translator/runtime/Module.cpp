#include "Image.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mach-o/getsect.h>
#include <mach-o/loader.h>
#include <pthread.h>

// Linked into every translated guest module (a .guest.prx dylib) with the object aps5-translator
// writes for it. The module's initializer registers its image with the executable's runtime, sets
// up its TLS like the relinker's macOS mode (MacOsImage.cpp), runs DT_INIT with the pending
// sceKernelLoadStartModule arguments and DT_INIT_ARRAY, and leaves DT_FINI_ARRAY and DT_FINI to
// __cxa_atexit against this image, so they run at exit or when the module is unloaded.

namespace {

struct GuestTlsDescriptor {
    std::uint64_t version;
    std::uint64_t templateAddress;
    std::uint64_t templateSize;
    std::uint64_t blockSize;
    std::uint64_t alignment;
    std::uint64_t key;
};

struct TlsImport {
    std::uint64_t offset;
    const std::uint64_t* index; // {module, offset}
    std::uint64_t field;
    std::int64_t addend;
};

struct PendingModuleArgs {
    std::size_t args;
    const void* argp;
};

enum ModuleTable { Init, Fini, InitArray, InitArrayCount, FiniArray, FiniArrayCount };

constexpr std::uint64_t Absent = ~std::uint64_t{0};

}

// Built with hidden visibility: the module's own tables and aps5_module_thread_pointer stay inside it.
extern "C" {
extern std::uint8_t aps5_guest_image[];
extern const std::uint64_t aps5_guest_image_size;
extern const TranslatedFunction aps5_guest_functions[];
extern const std::uint64_t aps5_guest_function_count;
extern const char* const aps5_profile_path;
extern const std::uint64_t aps5_guest_module[6];
extern const TlsImport aps5_guest_tls_imports[];
extern const std::uint64_t aps5_guest_tls_import_count;
extern GuestTlsDescriptor aps5_guest_tls_descriptor;
extern const mach_header_64 __dso_handle;

void Aps5GuestTlsRegister_nid_no_patch(GuestTlsDescriptor* descriptor);
void* Aps5GuestTlsAllocateThreadPointer_nid_no_patch(const GuestTlsDescriptor* descriptor);
const PendingModuleArgs* __aps5_get_pending_module_args_nid_no_patch();
void __aps5_set_module_init_result_nid_no_patch(int result);
int __cxa_atexit(void (*function)(void*), void* argument, void* handle);
}

namespace {

std::uint64_t Base() {
    return reinterpret_cast<std::uint64_t>(aps5_guest_image);
}

[[noreturn]] void Fail(const char* what, std::uint64_t address) {
    Dl_info self {};
    dladdr(&__dso_handle, &self);
    std::fprintf(stderr, "[translator] %s: %s 0x%" PRIx64 "\n", self.dli_fname != nullptr ? self.dli_fname : "guest module", what, address);
    std::abort();
}

// Runs the guest function at an address, which may be in another image, with up to three arguments.
int Call(std::uint64_t address, const void* first = nullptr, const void* second = nullptr) {
    const void* code = aps5_translated(address);
    if (code == nullptr) Fail("no translated initializer or finalizer at", address);
    return aps5_invoke_translated(first, second, nullptr, code);
}

void CallSlots(std::uint64_t array, std::uint64_t count, bool reverse) {
    if (array == Absent) return;
    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t function = 0;
        std::memcpy(&function, aps5_guest_image + array + 8 * (reverse ? count - 1 - i : i), 8);
        if (function != 0 && function != Absent) Call(function);
    }
}

void Finalize(void*) {
    CallSlots(aps5_guest_module[FiniArray], aps5_guest_module[FiniArrayCount], true);
    if (aps5_guest_module[Fini] != Absent) Call(Base() + aps5_guest_module[Fini]);
}

void Initialize() {
    unsigned long bytes = 0;
    auto* rows = reinterpret_cast<ProfileRow*>(getsectiondata(&__dso_handle, "__DATA", "__aps5_prof", &bytes));
    Dl_info self {};
    dladdr(&__dso_handle, &self);
    const TranslatedImage image{Base(), aps5_guest_image_size, aps5_guest_functions, aps5_guest_function_count, aps5_profile_path, rows, bytes / sizeof(ProfileRow),
                                self.dli_fname != nullptr ? self.dli_fname : "guest module"};
    aps5_register_image(&image);
    if (aps5_guest_tls_descriptor.version == 1) Aps5GuestTlsRegister_nid_no_patch(&aps5_guest_tls_descriptor);
    for (std::uint64_t i = 0; i < aps5_guest_tls_import_count; ++i) {
        const auto& slot = aps5_guest_tls_imports[i];
        const auto value = slot.index[slot.field] + static_cast<std::uint64_t>(slot.addend);
        std::memcpy(aps5_guest_image + slot.offset, &value, 8);
    }
    if (aps5_guest_module[Init] != Absent) {
        const auto* pending = __aps5_get_pending_module_args_nid_no_patch();
        const int result = Call(Base() + aps5_guest_module[Init], reinterpret_cast<const void*>(pending->args), pending->argp);
        __aps5_set_module_init_result_nid_no_patch(result);
    }
    CallSlots(aps5_guest_module[InitArray], aps5_guest_module[InitArrayCount], false);
    if (aps5_guest_module[Fini] != Absent || aps5_guest_module[FiniArray] != Absent) __cxa_atexit(Finalize, nullptr, const_cast<mach_header_64*>(&__dso_handle));
}

struct Registration {
    Registration() { Initialize(); }
    ~Registration() { aps5_unregister_image(Base()); }
};

const Registration registration;

}

extern "C" {

// The thread pointer of this module's guest code: its own TLS block, allocated on a thread's first
// access, as libkernel's __tls_get_addr finds it.
std::uint64_t aps5_module_thread_pointer() {
    void* threadPointer = pthread_getspecific(static_cast<pthread_key_t>(aps5_guest_tls_descriptor.key));
    if (threadPointer == nullptr) threadPointer = Aps5GuestTlsAllocateThreadPointer_nid_no_patch(&aps5_guest_tls_descriptor);
    return reinterpret_cast<std::uint64_t>(threadPointer);
}

}
