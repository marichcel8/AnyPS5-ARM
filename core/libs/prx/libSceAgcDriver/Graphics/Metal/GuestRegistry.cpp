#include "prx/libSceAgcDriver/Graphics/Metal/GuestRegistry.hpp"

namespace AgcDriver::Graphics::Metal {

std::vector<GuestRange> ReadableRanges(const GuestAllocations::Lease& lease) {
    std::vector<GuestRange> ranges;
    ranges.reserve(lease.size());
    for (const auto& range : lease) {
        if (range == nullptr || !range->readable || range->bytes == 0) continue;
        ranges.push_back({range->address, range->bytes, range->writable});
    }
    return ranges;
}

bool GuestRegistrySync::Refresh() {
    const auto live = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (live == generation.load(std::memory_order_acquire)) return false;
    if (GuestAllocations::GuestAllocationsHeldByThisThread_nid_postfix()) return false;
    std::lock_guard lock(mutex);
    const auto before = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (before == generation.load(std::memory_order_relaxed)) return false;
    auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
    const auto ranges = ReadableRanges(lease);
    lease.clear();
    windows.Replace(ranges);
    generation.store(before, std::memory_order_release);
    return true;
}

void GuestRegistrySync::Unmapped(std::uintptr_t address, std::size_t bytes) {
    windows.Unmap(address, bytes);
}

}
