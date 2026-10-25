#ifdef __APPLE__
#include "prx/libc/include/general/VabiMacros.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

struct GuestTlsDescriptor {
    std::uint64_t version;
    std::uint64_t templateAddress;
    std::uint64_t templateSize;
    std::uint64_t blockSize;
    std::uint64_t alignment;
    std::uint64_t key;
};

struct TlsIndex {
    const GuestTlsDescriptor* module;
    std::uint64_t offset;
};

extern "C" {
void Aps5GuestTlsRegister_nid_no_patch(GuestTlsDescriptor* descriptor);
void* APS5_VABI __tls_get_addr_nid_postfix(void* index);
}

static void Require(bool value) { if (!value) std::abort(); }

int main() {
    static const char initial[8] = {'g', 'u', 'e', 's', 't', 't', 'l', 's'};
    GuestTlsDescriptor descriptor{1, reinterpret_cast<std::uint64_t>(initial), sizeof(initial), 32, 16, 0};
    Aps5GuestTlsRegister_nid_no_patch(&descriptor);
    TlsIndex first{&descriptor, 0};
    TlsIndex fifth{&descriptor, 4};
    TlsIndex zeroed{&descriptor, 16};
    auto* block = static_cast<char*>(__tls_get_addr_nid_postfix(&first));
    Require(std::memcmp(block, initial, sizeof(initial)) == 0);
    Require(static_cast<char*>(__tls_get_addr_nid_postfix(&fifth)) == block + 4);
    Require(*static_cast<std::uint64_t*>(__tls_get_addr_nid_postfix(&zeroed)) == 0);
    block[0] = 'G';
    Require(static_cast<char*>(__tls_get_addr_nid_postfix(&first))[0] == 'G');
    char* other = nullptr;
    std::thread([&] { other = static_cast<char*>(__tls_get_addr_nid_postfix(&first)); Require(other[0] == 'g'); }).join();
    Require(other != block);
}
#else
#include "prx/libkernel/include/Api.hpp"
#include "SceTypes.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#if defined(_WIN32)
#include <windows.h>
#else
#include <link.h>
#endif

#if defined(_WIN32)
extern "C" ULONG _tls_index;

namespace {

using Cell = char;
constexpr Cell Initial = 0;

KernelTlsIndex CellIndex() { return {reinterpret_cast<std::uintptr_t>(&_tls_index), 0}; }

Cell* ThreadCell() {
    auto* blocks = *reinterpret_cast<char***>(reinterpret_cast<char*>(NtCurrentTeb()) + 0x58);
    return blocks[_tls_index];
}

}
#else
namespace {

using Cell = std::uint32_t;
constexpr Cell Initial = 0x5ca1ab1e;
thread_local Cell Value = Initial;

struct Module {
    std::size_t id = 0;
    void* data = nullptr;
};

Module MainModule() {
    Module module;
    dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void* out) {
        auto& found = *static_cast<Module*>(out);
        found = {info->dlpi_tls_modid, info->dlpi_tls_data};
        return 1;
    }, &module);
    return module;
}

KernelTlsIndex CellIndex() {
    const auto module = MainModule();
    return {module.id, static_cast<std::uint64_t>(reinterpret_cast<char*>(&Value) - static_cast<char*>(module.data))};
}

Cell* ThreadCell() { return &Value; }

}
#endif

namespace {

void Require(bool value) { if (!value) std::abort(); }

}

int main() {
    const auto index = CellIndex();
    auto* own = static_cast<Cell*>(__tls_get_addr_nid_postfix(&index));
    Require(own == ThreadCell() && *own == Initial);
    *own = 7;
    Cell* other = nullptr;
    Cell seen = 1;
    std::thread([&] {
        other = static_cast<Cell*>(__tls_get_addr_nid_postfix(&index));
        Require(other == ThreadCell());
        seen = *other;
        *other = 9;
    }).join();
    Require(other != own && seen == Initial && *own == 7);
    *own = Initial;
    bool threw = false;
    try { __tls_get_addr_nid_postfix(nullptr); } catch (const std::invalid_argument&) { threw = true; }
    Require(threw);
}
#endif
