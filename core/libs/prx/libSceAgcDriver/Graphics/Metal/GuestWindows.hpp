#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_GUESTWINDOWS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_GUESTWINDOWS_HPP

#include "BdaAbi.hpp"
#include "prx/libc/include/general/AtomicSharedPtr.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace AgcDriver::Graphics::Metal {

inline constexpr std::uint64_t GuestPageBytes = 4096;

struct HostSpan {
    std::uint64_t base = 0;
    std::uint64_t bytes = 0;

    std::uint64_t End() const { return base + bytes; }
    bool Contains(std::uint64_t address, std::uint64_t length) const { return address >= base && length <= bytes && address - base <= bytes - length; }
};

HostSpan HostPagesCovering(std::uint64_t address, std::uint64_t bytes, std::uint64_t hostPageBytes);
std::uint64_t WindowStep(std::uint64_t maxWindowBytes, std::uint64_t hostPageBytes);
std::vector<HostSpan> PlanWindows(HostSpan span, std::uint64_t maxWindowBytes, std::uint64_t hostPageBytes);

struct WrappedMemory {
    void* handle = nullptr;
    std::uint64_t gpuAddress = 0;
    bool copied = false;
};

class WindowBackend {
public:
    virtual ~WindowBackend() = default;
    virtual WrappedMemory Wrap(HostSpan span, bool writable) = 0;
    virtual void Retire(const WrappedMemory& memory, HostSpan span) = 0;
};

struct GuestRange {
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    bool writable = false;
};

struct Window {
    HostSpan span;
    std::uint64_t ownerBegin = 0;
    std::uint64_t ownerEnd = 0;
    std::uint64_t gpuAddress = 0;
    void* handle = nullptr;
    bool writable = false;
    bool copied = false;
};

struct WindowSnapshot {
    std::uint64_t serial = 0;
    std::vector<Window> windows;
    std::vector<ShaderRecompiler::BdaAbi::Range> table;
};

struct Resolved {
    const Window* window = nullptr;
    std::uint64_t deviceAddress = 0;

    explicit operator bool() const { return window != nullptr; }
};

Resolved Resolve(const WindowSnapshot& snapshot, std::uint64_t address, std::uint64_t bytes);
std::vector<ShaderRecompiler::BdaAbi::Range> BuildBdaTable(std::span<const Window> windows);
std::size_t BdaTableBytes(std::size_t entries);
void WriteBdaTable(std::span<std::byte> destination, std::span<const ShaderRecompiler::BdaAbi::Range> entries);

struct WindowCounters {
    std::uint64_t windows = 0;
    std::uint64_t wrappedBytes = 0;
    std::uint64_t copiedBytes = 0;
    std::uint64_t wrapFailures = 0;
    std::uint64_t publishes = 0;
};

class GuestWindows {
public:
    GuestWindows(WindowBackend& backend, std::uint64_t hostPageBytes, std::uint64_t maxWindowBytes);
    ~GuestWindows();
    GuestWindows(const GuestWindows&) = delete;
    GuestWindows& operator=(const GuestWindows&) = delete;

    std::shared_ptr<const WindowSnapshot> Snapshot() const { return current.load(std::memory_order_acquire); }
    std::uint64_t Serial() const { return serial.load(std::memory_order_acquire); }

    std::size_t Apply(std::span<const GuestRange> map, std::span<const GuestRange> unmap);
    bool Map(const GuestRange& range);
    void Unmap(std::uint64_t address, std::uint64_t bytes);
    void Replace(std::span<const GuestRange> ranges);
    void Clear();

    std::uint64_t HostPageBytes() const { return hostPage; }
    std::uint64_t MaxWindowBytes() const { return maxWindow; }
    WindowCounters Counters() const;

private:
    bool wrapOwner(const GuestRange& range, std::vector<Window>& out);
    void retire(const Window& window);
    void publish(std::vector<Window> windows);

    WindowBackend& backend;
    const std::uint64_t hostPage;
    const std::uint64_t maxWindow;
    mutable std::mutex mutex;
    AtomicSharedPtr<const WindowSnapshot> current;
    std::atomic<std::uint64_t> serial{0};
    WindowCounters counters;
};

}

#endif
