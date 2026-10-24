#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_GUESTREGISTRY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_GUESTREGISTRY_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/GuestWindows.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace AgcDriver::Graphics::Metal {

std::vector<GuestRange> ReadableRanges(const GuestAllocations::Lease& lease);

class GuestRegistrySync {
public:
    explicit GuestRegistrySync(GuestWindows& windows) : windows(windows) {}

    bool Refresh();
    void Unmapped(std::uintptr_t address, std::size_t bytes);
    std::uint64_t Generation() const { return generation.load(std::memory_order_acquire); }

private:
    GuestWindows& windows;
    std::mutex mutex;
    std::atomic<std::uint64_t> generation{~0ull};
};

}

#endif
