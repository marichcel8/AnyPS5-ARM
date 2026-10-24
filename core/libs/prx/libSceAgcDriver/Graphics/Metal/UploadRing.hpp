#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_UPLOADRING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_UPLOADRING_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace AgcDriver::Graphics::Metal {

struct UploadChunk {
    void* handle = nullptr;
    std::byte* cpu = nullptr;
    std::uint64_t gpu = 0;
    std::uint64_t bytes = 0;
};

struct UploadAllocation {
    void* cpu = nullptr;
    std::uint64_t gpu = 0;
};

class UploadSource {
public:
    virtual ~UploadSource() = default;
    virtual UploadChunk Acquire(std::uint64_t bytes) = 0;
    virtual void Release(const UploadChunk& chunk) = 0;
};

class UploadRing {
public:
    UploadRing(UploadSource& source, std::uint64_t chunkBytes) : source(&source), chunkBytes(chunkBytes) {}
    ~UploadRing();
    UploadRing(const UploadRing&) = delete;
    UploadRing& operator=(const UploadRing&) = delete;
    UploadRing(UploadRing&& other) noexcept;
    UploadRing& operator=(UploadRing&& other) noexcept;

    UploadAllocation Allocate(std::uint64_t bytes, std::uint64_t alignment) {
        const auto start = (cursor + alignment - 1u) & ~(alignment - 1u);
        if (active != nullptr && start + bytes <= active->bytes) {
            cursor = start + bytes;
            used += bytes;
            return {active->cpu + start, active->gpu + start};
        }
        return Grow(bytes, alignment);
    }

    void Reset();
    [[nodiscard]] std::uint64_t UsedBytes() const { return used; }
    [[nodiscard]] std::uint64_t CapacityBytes() const;
    [[nodiscard]] std::size_t Chunks() const { return chunks.size(); }

private:
    UploadAllocation Grow(std::uint64_t bytes, std::uint64_t alignment);

    UploadSource* source;
    std::uint64_t chunkBytes;
    std::vector<UploadChunk> chunks;
    std::size_t activeIndex = 0;
    UploadChunk* active = nullptr;
    std::uint64_t cursor = 0;
    std::uint64_t used = 0;
};

template <std::size_t Slots>
class SlotAddresses {
public:
    bool Set(std::uint32_t slot, std::uint64_t address) {
        if (addresses[slot] == address) return false;
        addresses[slot] = address;
        return true;
    }

    [[nodiscard]] std::uint64_t Get(std::uint32_t slot) const { return addresses[slot]; }
    void Clear() { addresses.fill(0); }

private:
    std::array<std::uint64_t, Slots> addresses{};
};

}

#endif
