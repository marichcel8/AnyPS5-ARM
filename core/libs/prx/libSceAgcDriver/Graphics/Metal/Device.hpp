#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_DEVICE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_DEVICE_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/GuestWindows.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace AgcDriver::Graphics::Metal {

enum class QueueKind : std::uint8_t { Graphics, Compute };
inline constexpr std::size_t QueueCount = 2;

struct DeviceLimits {
    std::string name;
    std::uint64_t hostPageBytes = 0;
    std::uint64_t maxBufferBytes = 0;
    std::uint64_t maxWindowBytes = 0;
    std::uint64_t recommendedWorkingSetBytes = 0;
    std::uint64_t driverBudgetBytes = 0;
    bool unifiedMemory = false;
    bool metal4 = false;
};

struct NativeBuffer {
    void* handle = nullptr;
    void* contents = nullptr;
    std::uint64_t gpuAddress = 0;
    std::uint64_t bytes = 0;

    explicit operator bool() const { return handle != nullptr; }
};

struct MemoryReport {
    std::uint64_t physFootprintBytes = 0;
    std::uint64_t deviceAllocatedBytes = 0;
    std::uint64_t residentAllocatedBytes = 0;
    std::uint64_t driverBytes = 0;
    std::uint64_t windowBytes = 0;
    std::uint64_t copiedBytes = 0;
    std::uint64_t windows = 0;
    std::uint64_t pendingReleases = 0;
    std::uint64_t residencyCommits = 0;
};

class Device {
public:
    static std::unique_ptr<Device> Create(std::string* failure);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    const DeviceLimits& Limits() const;
    void* NativeDevice() const;
    void* NativeQueue(QueueKind queue) const;
    void* NativeResidencySet() const;

    GuestWindows& Guest();

    NativeBuffer NewSharedBuffer(std::uint64_t bytes, const char* label);
    NativeBuffer NewPrivateBuffer(std::uint64_t bytes, const char* label);
    void ReleaseBuffer(const NativeBuffer& buffer);

    void AddResidentAllocation(void* allocation);
    void RetireAllocation(void* allocation, std::function<void()> destroy);
    void Defer(std::function<void()> destroy);

    void FlushResidency();
    std::uint64_t Submit(QueueKind queue, void* const* commandBuffers, std::size_t count);
    std::uint64_t SubmittedSerial(QueueKind queue) const;
    std::uint64_t CompletedSerial(QueueKind queue) const;
    bool WaitSerial(QueueKind queue, std::uint64_t serial, std::uint32_t timeoutMs);
    bool WaitIdle(std::uint32_t timeoutMs);
    void Collect();

    MemoryReport Memory() const;

    struct State;

private:
    explicit Device(std::unique_ptr<State> state);

    std::unique_ptr<State> state;
};

std::uint64_t PhysFootprintBytes();

}

#endif
