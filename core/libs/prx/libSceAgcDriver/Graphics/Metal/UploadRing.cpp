#include "prx/libSceAgcDriver/Graphics/Metal/UploadRing.hpp"
#include <algorithm>
#include <utility>

namespace AgcDriver::Graphics::Metal {

UploadRing::~UploadRing() {
    if (source == nullptr) return;
    for (const auto& chunk : chunks) source->Release(chunk);
}

UploadRing::UploadRing(UploadRing&& other) noexcept
    : source(std::exchange(other.source, nullptr)), chunkBytes(other.chunkBytes), chunks(std::move(other.chunks)), activeIndex(other.activeIndex), cursor(other.cursor), used(other.used) {
    active = chunks.empty() || other.active == nullptr ? nullptr : &chunks[activeIndex];
    other.active = nullptr;
    other.chunks.clear();
}

UploadRing& UploadRing::operator=(UploadRing&& other) noexcept {
    if (this == &other) return *this;
    if (source != nullptr)
        for (const auto& chunk : chunks) source->Release(chunk);
    source = std::exchange(other.source, nullptr);
    chunkBytes = other.chunkBytes;
    chunks = std::move(other.chunks);
    activeIndex = other.activeIndex;
    cursor = other.cursor;
    used = other.used;
    active = chunks.empty() || other.active == nullptr ? nullptr : &chunks[activeIndex];
    other.active = nullptr;
    other.chunks.clear();
    return *this;
}

void UploadRing::Reset() {
    activeIndex = 0;
    cursor = 0;
    used = 0;
    active = chunks.empty() ? nullptr : &chunks.front();
}

std::uint64_t UploadRing::CapacityBytes() const {
    std::uint64_t total = 0;
    for (const auto& chunk : chunks) total += chunk.bytes;
    return total;
}

UploadAllocation UploadRing::Grow(std::uint64_t bytes, std::uint64_t alignment) {
    while (active != nullptr && activeIndex + 1u < chunks.size()) {
        ++activeIndex;
        active = &chunks[activeIndex];
        cursor = 0;
        if (bytes <= active->bytes) {
            cursor = bytes;
            used += bytes;
            return {active->cpu, active->gpu};
        }
    }
    const auto size = std::max(chunkBytes, (bytes + alignment - 1u) & ~(alignment - 1u));
    auto chunk = source->Acquire(size);
    if (chunk.cpu == nullptr) return {};
    chunks.push_back(chunk);
    activeIndex = chunks.size() - 1u;
    active = &chunks.back();
    cursor = bytes;
    used += bytes;
    return {active->cpu, active->gpu};
}

}
