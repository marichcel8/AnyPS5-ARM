#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr std::uint64_t MiB = 1ull << 20u;
constexpr std::size_t InlineCommits = 8;
constexpr MTLResourceOptions SharedOptions = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
constexpr MTLResourceOptions PrivateOptions = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;

std::uint64_t envMegabytes(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == 0) return fallback;
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || parsed == 0) return fallback;
    return parsed * MiB;
}

std::size_t queueIndex(QueueKind queue) {
    return static_cast<std::size_t>(queue);
}

NSString* labelString(const char* label) {
    return label != nullptr ? [NSString stringWithUTF8String:label] : nil;
}

struct PendingRelease {
    id<MTLAllocation> allocation;
    std::function<void()> destroy;
    std::array<std::uint64_t, QueueCount> serials;
    std::uint64_t driverBytes;
};

}

struct Device::State final : WindowBackend {
    id<MTLDevice> device;
    std::array<id<MTL4CommandQueue>, QueueCount> queues;
    std::array<id<MTLSharedEvent>, QueueCount> timelines;
    id<MTLResidencySet> residency;
    std::array<std::mutex, QueueCount> submitMutex;
    std::array<std::atomic<std::uint64_t>, QueueCount> submitted{};
    std::mutex residencyMutex;
    std::atomic<bool> residencyDirty{false};
    std::atomic<std::uint64_t> residencyCommits{0};
    std::atomic<std::uint64_t> gpuBusyNs{0};
    std::atomic<std::uint64_t> gpuMaxNs{0};
    std::atomic<std::uint64_t> gpuBuffers{0};
    mutable std::mutex releaseMutex;
    std::deque<PendingRelease> releases;
    std::atomic<std::uint64_t> driverBytes{0};
    DeviceLimits limits;
    std::unique_ptr<GuestWindows> guest;

    WrappedMemory Wrap(HostSpan span, bool writable) override {
        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(span.base) length:span.bytes options:SharedOptions deallocator:nil];
        bool copied = false;
        if (buffer == nil) {
            if (writable || !reserveDriver(span.bytes)) return {};
            buffer = [device newBufferWithBytes:reinterpret_cast<const void*>(span.base) length:span.bytes options:SharedOptions];
            if (buffer == nil) {
                driverBytes.fetch_sub(span.bytes, std::memory_order_relaxed);
                return {};
            }
            copied = true;
        }
        addResident(buffer);
        const auto gpuAddress = buffer.gpuAddress;
        return {(__bridge_retained void*)buffer, gpuAddress, copied};
    }

    void Retire(const WrappedMemory& memory, HostSpan span) override {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)memory.handle;
        deferRelease(buffer, memory.copied ? span.bytes : 0);
    }

    bool reserveDriver(std::uint64_t bytes) {
        const auto before = driverBytes.fetch_add(bytes, std::memory_order_relaxed);
        if (before + bytes <= limits.driverBudgetBytes) return true;
        driverBytes.fetch_sub(bytes, std::memory_order_relaxed);
        return false;
    }

    void addResident(id<MTLAllocation> buffer) {
        std::lock_guard lock(residencyMutex);
        [residency addAllocation:buffer];
        residencyDirty.store(true, std::memory_order_release);
    }

    std::array<std::uint64_t, QueueCount> submittedNow() const {
        std::array<std::uint64_t, QueueCount> serials{};
        for (std::size_t index = 0; index < QueueCount; ++index) serials[index] = submitted[index].load(std::memory_order_acquire);
        return serials;
    }

    void deferRelease(id<MTLAllocation> allocation, std::uint64_t bytes, std::function<void()> destroy = {}) {
        std::lock_guard lock(releaseMutex);
        releases.push_back({allocation, std::move(destroy), submittedNow(), bytes});
    }

    NativeBuffer newBuffer(std::uint64_t bytes, MTLResourceOptions options, const char* label) {
        if (bytes == 0 || !reserveDriver(bytes)) return {};
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:options];
        if (buffer == nil) {
            driverBytes.fetch_sub(bytes, std::memory_order_relaxed);
            return {};
        }
        if (label != nullptr) buffer.label = labelString(label);
        addResident(buffer);
        void* contents = (options & MTLResourceStorageModePrivate) != 0 ? nullptr : buffer.contents;
        const auto gpuAddress = buffer.gpuAddress;
        return {(__bridge_retained void*)buffer, contents, gpuAddress, bytes};
    }
};

std::unique_ptr<Device> Device::Create(std::string* failure) {
    @autoreleasepool {
        const auto fail = [&](const std::string& what) -> std::unique_ptr<Device> {
            if (failure != nullptr) *failure = what;
            return nullptr;
        };
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (const char* requested = std::getenv("ANYPS5_GPU"); requested != nullptr && requested[0] != '\0') {
            const auto lower = [](std::string text) {
                std::transform(text.begin(), text.end(), text.begin(), [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
                return text;
            };
            const auto wanted = lower(requested);
            std::string names;
            id<MTLDevice> chosen = nil;
            for (id<MTLDevice> candidate in MTLCopyAllDevices()) {
                const std::string name = candidate.name != nil ? candidate.name.UTF8String : "";
                names += (names.empty() ? "" : ", ") + name;
                if (chosen == nil && lower(name).find(wanted) != std::string::npos) chosen = candidate;
            }
            if (chosen == nil) return fail(std::string("no usable device whose name contains ANYPS5_GPU=\"") + requested + "\"; devices: " + names);
            device = chosen;
        }
        if (device == nil) return fail("no Metal device");
        if (![device supportsFamily:MTLGPUFamilyMetal4]) return fail("the GPU does not support Metal 4");
        auto state = std::make_unique<State>();
        state->device = device;
        auto& limits = state->limits;
        limits.name = device.name != nil ? device.name.UTF8String : "";
        limits.hostPageBytes = static_cast<std::uint64_t>(getpagesize());
        limits.maxBufferBytes = device.maxBufferLength;
        limits.recommendedWorkingSetBytes = device.recommendedMaxWorkingSetSize;
        limits.unifiedMemory = device.hasUnifiedMemory;
        limits.metal4 = true;
        const auto window = std::min<std::uint64_t>(limits.maxBufferBytes, envMegabytes("APS5_METAL_WINDOW_MIB", limits.maxBufferBytes));
        limits.maxWindowBytes = window & ~(limits.hostPageBytes - 1);
        limits.driverBudgetBytes = envMegabytes("APS5_METAL_DRIVER_MIB", std::min<std::uint64_t>(1024 * MiB, limits.recommendedWorkingSetBytes / 4));
        if (!limits.unifiedMemory) return fail("the GPU has no unified memory");
        if (limits.maxWindowBytes < 2 * limits.hostPageBytes) return fail("the buffer length limit is below two host pages");
        NSError* error = nil;
        MTLResidencySetDescriptor* residency = [MTLResidencySetDescriptor new];
        residency.label = @"AnyPS5 guest and driver memory";
        residency.initialCapacity = 4096;
        state->residency = [device newResidencySetWithDescriptor:residency error:&error];
        if (state->residency == nil) return fail("newResidencySetWithDescriptor");
        static constexpr const char* QueueLabels[QueueCount] = {"AnyPS5 graphics", "AnyPS5 compute"};
        for (std::size_t index = 0; index < QueueCount; ++index) {
            MTL4CommandQueueDescriptor* descriptor = [MTL4CommandQueueDescriptor new];
            descriptor.label = labelString(QueueLabels[index]);
            state->queues[index] = [device newMTL4CommandQueueWithDescriptor:descriptor error:&error];
            if (state->queues[index] == nil) return fail("newMTL4CommandQueueWithDescriptor");
            [state->queues[index] addResidencySet:state->residency];
            state->timelines[index] = [device newSharedEvent];
            if (state->timelines[index] == nil) return fail("newSharedEvent");
            state->timelines[index].signaledValue = 0;
        }
        state->guest = std::make_unique<GuestWindows>(*state, limits.hostPageBytes, limits.maxWindowBytes);
        return std::unique_ptr<Device>(new Device(std::move(state)));
    }
}

Device::Device(std::unique_ptr<State> state) : state(std::move(state)) {}

Device::~Device() {
    if (state == nullptr) return;
    state->guest->Clear();
    WaitIdle(5000);
    Collect();
    state->guest.reset();
    std::deque<PendingRelease> leftovers;
    {
        std::lock_guard lock(state->releaseMutex);
        leftovers.swap(state->releases);
    }
    for (auto& release : leftovers) {
        if (release.destroy) release.destroy();
    }
}

const DeviceLimits& Device::Limits() const {
    return state->limits;
}

void* Device::NativeDevice() const {
    return (__bridge void*)state->device;
}

void* Device::NativeQueue(QueueKind queue) const {
    return (__bridge void*)state->queues[queueIndex(queue)];
}

void* Device::NativeResidencySet() const {
    return (__bridge void*)state->residency;
}

GuestWindows& Device::Guest() {
    return *state->guest;
}

NativeBuffer Device::NewSharedBuffer(std::uint64_t bytes, const char* label) {
    return state->newBuffer(bytes, SharedOptions, label);
}

NativeBuffer Device::NewPrivateBuffer(std::uint64_t bytes, const char* label) {
    return state->newBuffer(bytes, PrivateOptions, label);
}

void Device::ReleaseBuffer(const NativeBuffer& buffer) {
    if (buffer.handle == nullptr) return;
    id<MTLBuffer> native = (__bridge_transfer id<MTLBuffer>)buffer.handle;
    state->deferRelease(native, buffer.bytes);
}

void Device::AddResidentAllocation(void* allocation) {
    if (allocation == nullptr) return;
    state->addResident((__bridge id<MTLAllocation>)allocation);
}

void Device::RetireAllocation(void* allocation, std::function<void()> destroy) {
    id<MTLAllocation> native = (__bridge id<MTLAllocation>)allocation;
    state->deferRelease(native, 0, std::move(destroy));
}

void Device::Defer(std::function<void()> destroy) {
    state->deferRelease(nil, 0, std::move(destroy));
}

void Device::FlushResidency() {
    auto& s = *state;
    if (!s.residencyDirty.load(std::memory_order_acquire)) return;
    std::lock_guard lock(s.residencyMutex);
    if (!s.residencyDirty.exchange(false, std::memory_order_acq_rel)) return;
    [s.residency commit];
    [s.residency requestResidency];
    s.residencyCommits.fetch_add(1, std::memory_order_relaxed);
}

namespace {

bool GpuTimeEnabled() {
    static const bool enabled = std::getenv("APS5_METAL_GPU_TIME") != nullptr;
    return enabled;
}

MTL4CommitOptions* GpuTimeOptions(std::atomic<std::uint64_t>* busyNs, std::atomic<std::uint64_t>* maxNs, std::atomic<std::uint64_t>* buffers) {
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        const auto busy = static_cast<std::uint64_t>(std::max(0.0, feedback.GPUEndTime - feedback.GPUStartTime) * 1.0e9);
        busyNs->fetch_add(busy, std::memory_order_relaxed);
        buffers->fetch_add(1, std::memory_order_relaxed);
        for (auto seen = maxNs->load(std::memory_order_relaxed); busy > seen && !maxNs->compare_exchange_weak(seen, busy, std::memory_order_relaxed);) {}
    }];
    return options;
}

}

std::uint64_t Device::Submit(QueueKind queue, void* const* commandBuffers, std::size_t count) {
    auto& s = *state;
    const auto index = queueIndex(queue);
    FlushResidency();
    std::uint64_t serial = 0;
    {
        std::lock_guard lock(s.submitMutex[index]);
        MTL4CommitOptions* options = GpuTimeEnabled() ? GpuTimeOptions(&s.gpuBusyNs, &s.gpuMaxNs, &s.gpuBuffers) : nil;
        for (std::size_t first = 0; first < count; first += InlineCommits) {
            const auto chunk = std::min(InlineCommits, count - first);
            __unsafe_unretained id<MTL4CommandBuffer> list[InlineCommits];
            for (std::size_t item = 0; item < chunk; ++item) list[item] = (__bridge id<MTL4CommandBuffer>)commandBuffers[first + item];
            if (options != nil) [s.queues[index] commit:list count:chunk options:options];
            else [s.queues[index] commit:list count:chunk];
        }
        serial = s.submitted[index].load(std::memory_order_relaxed) + 1;
        [s.queues[index] signalEvent:s.timelines[index] value:serial];
        s.submitted[index].store(serial, std::memory_order_release);
    }
    Collect();
    return serial;
}

std::uint64_t Device::SubmitPresent(QueueKind queue, void* drawable, void* const* commandBuffers, std::size_t count) {
    auto& s = *state;
    const auto index = queueIndex(queue);
    FlushResidency();
    id<MTLDrawable> target = (__bridge id<MTLDrawable>)drawable;
    std::uint64_t serial = 0;
    {
        std::lock_guard lock(s.submitMutex[index]);
        [s.queues[index] waitForDrawable:target];
        for (std::size_t first = 0; first < count; first += InlineCommits) {
            const auto chunk = std::min(InlineCommits, count - first);
            __unsafe_unretained id<MTL4CommandBuffer> list[InlineCommits];
            for (std::size_t item = 0; item < chunk; ++item) list[item] = (__bridge id<MTL4CommandBuffer>)commandBuffers[first + item];
            [s.queues[index] commit:list count:chunk];
        }
        [s.queues[index] signalDrawable:target];
        serial = s.submitted[index].load(std::memory_order_relaxed) + 1;
        [s.queues[index] signalEvent:s.timelines[index] value:serial];
        s.submitted[index].store(serial, std::memory_order_release);
    }
    [target present];
    Collect();
    return serial;
}

GpuTimeReport Device::TakeGpuTime() {
    auto& s = *state;
    GpuTimeReport report;
    report.enabled = GpuTimeEnabled();
    report.busyMs = static_cast<double>(s.gpuBusyNs.exchange(0, std::memory_order_relaxed)) / 1.0e6;
    report.maxMs = static_cast<double>(s.gpuMaxNs.exchange(0, std::memory_order_relaxed)) / 1.0e6;
    report.commandBuffers = s.gpuBuffers.exchange(0, std::memory_order_relaxed);
    return report;
}

std::uint64_t Device::SubmittedSerial(QueueKind queue) const {
    return state->submitted[queueIndex(queue)].load(std::memory_order_acquire);
}

std::uint64_t Device::CompletedSerial(QueueKind queue) const {
    return state->timelines[queueIndex(queue)].signaledValue;
}

bool Device::WaitSerial(QueueKind queue, std::uint64_t serial, std::uint32_t timeoutMs) {
    id<MTLSharedEvent> timeline = state->timelines[queueIndex(queue)];
    if (timeline.signaledValue >= serial) return true;
    return [timeline waitUntilSignaledValue:serial timeoutMS:timeoutMs];
}

bool Device::WaitIdle(std::uint32_t timeoutMs) {
    bool idle = true;
    for (std::size_t index = 0; index < QueueCount; ++index) idle = WaitSerial(static_cast<QueueKind>(index), SubmittedSerial(static_cast<QueueKind>(index)), timeoutMs) && idle;
    Collect();
    return idle;
}

void* Device::NativeTimeline(QueueKind queue) const {
    return (__bridge void*)state->timelines[queueIndex(queue)];
}

void Device::WaitOnQueue(QueueKind waiter, QueueKind signaler, std::uint64_t serial) {
    if (waiter == signaler || serial == 0) return;
    auto& s = *state;
    id<MTLSharedEvent> timeline = s.timelines[queueIndex(signaler)];
    if (timeline.signaledValue >= serial) return;
    const auto index = queueIndex(waiter);
    std::lock_guard lock(s.submitMutex[index]);
    [s.queues[index] waitForEvent:timeline value:serial];
}

void Device::WaitOnEvent(QueueKind waiter, void* sharedEvent, std::uint64_t value) {
    if (sharedEvent == nullptr) return;
    auto& s = *state;
    id<MTLEvent> event = (__bridge id<MTLEvent>)sharedEvent;
    const auto index = queueIndex(waiter);
    std::lock_guard lock(s.submitMutex[index]);
    [s.queues[index] waitForEvent:event value:value];
}

void Device::Collect() {
    auto& s = *state;
    std::array<std::uint64_t, QueueCount> completed{};
    for (std::size_t index = 0; index < QueueCount; ++index) completed[index] = s.timelines[index].signaledValue;
    std::vector<PendingRelease> ready;
    {
        std::lock_guard lock(s.releaseMutex);
        while (!s.releases.empty()) {
            const auto& front = s.releases.front();
            bool done = true;
            for (std::size_t index = 0; index < QueueCount; ++index) done = done && completed[index] >= front.serials[index];
            if (!done) break;
            ready.push_back(std::move(s.releases.front()));
            s.releases.pop_front();
        }
    }
    if (ready.empty()) return;
    {
        std::lock_guard lock(s.residencyMutex);
        for (const auto& release : ready) {
            if (release.allocation != nil) [s.residency removeAllocation:release.allocation];
        }
        s.residencyDirty.store(true, std::memory_order_release);
    }
    std::uint64_t bytes = 0;
    for (const auto& release : ready) bytes += release.driverBytes;
    if (bytes != 0) s.driverBytes.fetch_sub(bytes, std::memory_order_relaxed);
    FlushResidency();
    for (auto& release : ready) {
        if (release.destroy) release.destroy();
    }
}

MemoryReport Device::Memory() const {
    auto& s = *state;
    MemoryReport report;
    report.physFootprintBytes = PhysFootprintBytes();
    report.deviceAllocatedBytes = s.device.currentAllocatedSize;
    {
        std::lock_guard lock(s.residencyMutex);
        report.residentAllocatedBytes = s.residency.allocatedSize;
    }
    report.driverBytes = s.driverBytes.load(std::memory_order_relaxed);
    const auto counters = s.guest->Counters();
    report.windowBytes = counters.wrappedBytes;
    report.copiedBytes = counters.copiedBytes;
    report.windows = counters.windows;
    {
        std::lock_guard lock(s.releaseMutex);
        report.pendingReleases = s.releases.size();
    }
    report.residencyCommits = s.residencyCommits.load(std::memory_order_relaxed);
    return report;
}

std::uint64_t PhysFootprintBytes() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return 0;
    return info.phys_footprint;
}

}
