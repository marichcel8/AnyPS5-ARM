#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace AgcDriver::Graphics::Metal;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t MiB = 1ull << 20u;

const char* const KernelSource = R"(
#include <metal_stdlib>
using namespace metal;
struct Args {
    device uint* data;
    uint count;
    uint stride;
};
kernel void transform(constant Args& args [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index < args.count) args.data[index] = args.data[index] * 2u + 1u;
}
kernel void touch(constant Args& args [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index < args.count) args.data[index * args.stride] = index + 1u;
}
)";

struct HostArgs {
    std::uint64_t data;
    std::uint32_t count;
    std::uint32_t stride;
};

void Check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

double Microseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

double Mebibytes(std::int64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(MiB);
}

void* MapAnonymous(std::uint64_t bytes) {
    void* memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    Check(memory != MAP_FAILED, "mmap");
    return memory;
}

struct Kernels {
    id<MTLDevice> device;
    id<MTLComputePipelineState> transform;
    id<MTLComputePipelineState> touch;
    id<MTL4ArgumentTable> table;
    id<MTL4CommandAllocator> allocator;
    NativeBuffer args;
};

Kernels MakeKernels(Device& device) {
    Kernels kernels;
    kernels.device = (__bridge id<MTLDevice>)device.NativeDevice();
    NSError* error = nil;
    id<MTLLibrary> library = [kernels.device newLibraryWithSource:[NSString stringWithUTF8String:KernelSource] options:nil error:&error];
    if (library == nil) std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
    Check(library != nil, "compile the probe kernels");
    kernels.transform = [kernels.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"transform"] error:&error];
    kernels.touch = [kernels.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"touch"] error:&error];
    Check(kernels.transform != nil && kernels.touch != nil, "build the probe pipelines");
    MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = 1;
    kernels.table = [kernels.device newArgumentTableWithDescriptor:descriptor error:&error];
    Check(kernels.table != nil, "create the argument table");
    kernels.allocator = [kernels.device newCommandAllocator];
    Check(kernels.allocator != nil, "create the command allocator");
    kernels.args = device.NewSharedBuffer(sizeof(HostArgs), "probe arguments");
    Check(static_cast<bool>(kernels.args), "allocate the argument buffer");
    return kernels;
}

double Run(Device& device, Kernels& kernels, id<MTLComputePipelineState> pipeline, std::uint64_t pointer, std::uint32_t count, std::uint32_t stride) {
    *static_cast<HostArgs*>(kernels.args.contents) = {pointer, count, stride};
    const auto begin = Clock::now();
    id<MTL4CommandBuffer> commands = [kernels.device newCommandBuffer];
    [commands beginCommandBufferWithAllocator:kernels.allocator];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [kernels.table setAddress:kernels.args.gpuAddress atIndex:0];
    [encoder setArgumentTable:kernels.table];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [commands endCommandBuffer];
    void* list[] = {(__bridge void*)commands};
    const auto serial = device.Submit(QueueKind::Compute, list, 1);
    Check(device.WaitSerial(QueueKind::Compute, serial, 10000), "the GPU finished in time");
    [kernels.allocator reset];
    return Microseconds(begin);
}

void PrintReport(const char* when, const MemoryReport& report) {
    std::printf("memory.%s footprint_mib=%.1f device_allocated_mib=%.1f resident_mib=%.1f driver_mib=%.2f window_mib=%.1f copied_mib=%.1f windows=%llu pending=%llu residency_commits=%llu\n", when, Mebibytes(static_cast<std::int64_t>(report.physFootprintBytes)), Mebibytes(static_cast<std::int64_t>(report.deviceAllocatedBytes)), Mebibytes(static_cast<std::int64_t>(report.residentAllocatedBytes)), Mebibytes(static_cast<std::int64_t>(report.driverBytes)), Mebibytes(static_cast<std::int64_t>(report.windowBytes)), Mebibytes(static_cast<std::int64_t>(report.copiedBytes)), static_cast<unsigned long long>(report.windows), static_cast<unsigned long long>(report.pendingReleases), static_cast<unsigned long long>(report.residencyCommits));
}

void ZeroCopy(Device& device, Kernels& kernels) {
    const auto page = device.Limits().hostPageBytes;
    const std::uint64_t regionBytes = 8 * MiB;
    auto* region = static_cast<std::uint8_t*>(MapAnonymous(regionBytes));
    const auto guest = reinterpret_cast<std::uint64_t>(region) + page + GuestPageBytes;
    const std::uint64_t bytes = MiB + 2 * GuestPageBytes;
    auto* words = reinterpret_cast<std::uint32_t*>(guest);
    const auto count = static_cast<std::uint32_t>(bytes / sizeof(std::uint32_t));
    for (std::uint32_t index = 0; index < count; ++index) words[index] = index;
    const std::uint32_t guard = 0xa5a5a5a5u;
    words[-1] = guard;
    words[count] = guard;
    const auto mapBegin = Clock::now();
    Check(device.Guest().Map({guest, bytes, true}), "wrap a 4 KiB aligned guest range without copying");
    const auto mapUs = Microseconds(mapBegin);
    const auto snapshot = device.Guest().Snapshot();
    const auto resolved = Resolve(*snapshot, guest, bytes);
    Check(static_cast<bool>(resolved), "the guest range resolves");
    Check(!resolved.window->copied, "the window is not a copy");
    Check(resolved.window->span.base % page == 0 && resolved.window->span.bytes % page == 0, "the window is host page aligned");
    Check(snapshot->table.size() == 1 && snapshot->table[0].begin == guest && snapshot->table[0].deviceAddress == resolved.deviceAddress, "the BDA entry points at the guest bytes");
    const auto firstUs = Run(device, kernels, kernels.transform, resolved.deviceAddress, count, 1);
    for (std::uint32_t index = 0; index < count; ++index) Check(words[index] == index * 2u + 1u, "the GPU wrote the guest bytes in place");
    Check(words[-1] == guard && words[count] == guard, "the GPU kept to the guest range");
    for (std::uint32_t index = 0; index < count; ++index) words[index] = 7;
    const auto secondUs = Run(device, kernels, kernels.transform, resolved.deviceAddress, count, 1);
    for (std::uint32_t index = 0; index < count; ++index) Check(words[index] == 15u, "the GPU saw the CPU's later writes without an upload");
    const auto neighbour = guest + bytes;
    Check(device.Guest().Map({neighbour, 2 * GuestPageBytes, false}), "a neighbour sharing the host page maps");
    Check(device.Guest().Snapshot()->windows.size() == 2, "the neighbour has its own window");
    device.Guest().Unmap(guest, bytes);
    Check(device.WaitIdle(10000), "the device went idle");
    const auto after = device.Memory();
    Check(after.windows == 1 && after.pendingReleases == 0, "the unmapped window was released after the GPU finished");
    device.Guest().Unmap(neighbour, 2 * GuestPageBytes);
    device.WaitIdle(10000);
    munmap(region, regionBytes);
    std::printf("zero_copy ok map_us=%.1f first_dispatch_us=%.1f second_dispatch_us=%.1f words=%u\n", mapUs, firstUs, secondUs, count);
}

void Footprint(Device& device, Kernels& kernels) {
    const std::uint64_t bytes = 1024 * MiB;
    const auto page = device.Limits().hostPageBytes;
    auto* region = MapAnonymous(bytes);
    const auto address = reinterpret_cast<std::uint64_t>(region);
    const auto before = static_cast<std::int64_t>(PhysFootprintBytes());
    const auto mapBegin = Clock::now();
    const bool mapped = device.Guest().Map({address, bytes, true});
    const auto mapUs = Microseconds(mapBegin);
    const auto wrapped = static_cast<std::int64_t>(PhysFootprintBytes());
    device.FlushResidency();
    const auto resident = static_cast<std::int64_t>(PhysFootprintBytes());
    std::int64_t touched = resident;
    std::uint32_t pages = 0;
    if (mapped) {
        const auto resolved = Resolve(*device.Guest().Snapshot(), address, bytes);
        Check(static_cast<bool>(resolved), "the large range resolves");
        pages = 64;
        Run(device, kernels, kernels.touch, resolved.deviceAddress, pages, static_cast<std::uint32_t>(page / sizeof(std::uint32_t)));
        const auto* words = static_cast<const std::uint32_t*>(region);
        for (std::uint32_t index = 0; index < pages; ++index) Check(words[index * (page / sizeof(std::uint32_t))] == index + 1, "the GPU wrote untouched pages");
        touched = static_cast<std::int64_t>(PhysFootprintBytes());
    }
    std::printf("footprint mapped=%d map_us=%.1f wrap_delta_mib=%.1f residency_delta_mib=%.1f touch_delta_mib=%.1f touched_pages=%u reserved_mib=%llu\n", mapped ? 1 : 0, mapUs, Mebibytes(wrapped - before), Mebibytes(resident - before), Mebibytes(touched - before), pages, static_cast<unsigned long long>(bytes / MiB));
    PrintReport("large_window", device.Memory());
    device.Guest().Unmap(address, bytes);
    device.WaitIdle(10000);
    munmap(region, bytes);
}

void Lookups(Device& device) {
    constexpr std::uint64_t ranges = 2000;
    constexpr std::uint64_t stride = 64 * 1024;
    auto* region = MapAnonymous(ranges * stride);
    const auto base = reinterpret_cast<std::uint64_t>(region);
    std::vector<GuestRange> list;
    list.reserve(ranges);
    for (std::uint64_t index = 0; index < ranges; ++index) list.push_back({base + index * stride + GuestPageBytes, 3 * GuestPageBytes, (index & 1) != 0});
    const auto replaceBegin = Clock::now();
    device.Guest().Replace(list);
    const auto replaceUs = Microseconds(replaceBegin);
    const auto snapshot = device.Guest().Snapshot();
    Check(snapshot->windows.size() == ranges, "every range has a window");
    std::mt19937_64 random(1);
    std::vector<std::uint64_t> probes(1u << 20u);
    for (auto& probe : probes) probe = base + (random() % ranges) * stride + GuestPageBytes + (random() % (3 * GuestPageBytes - 16));
    std::uint64_t sum = 0;
    const auto resolveBegin = Clock::now();
    for (const auto probe : probes) sum += Resolve(*snapshot, probe, 16).deviceAddress;
    const auto resolveNs = Microseconds(resolveBegin) * 1000.0 / static_cast<double>(probes.size());
    Check(sum != 0, "the lookups resolved");
    const auto serialBegin = Clock::now();
    std::uint64_t serials = 0;
    for (std::size_t index = 0; index < probes.size(); ++index) serials += device.Guest().Serial();
    const auto serialNs = Microseconds(serialBegin) * 1000.0 / static_cast<double>(probes.size());
    const auto snapshotBegin = Clock::now();
    for (std::size_t index = 0; index < 100000; ++index) serials += device.Guest().Snapshot()->serial;
    const auto snapshotNs = Microseconds(snapshotBegin) * 1000.0 / 100000.0;
    std::printf("lookups ranges=%llu replace_us=%.1f per_range_us=%.2f resolve_ns=%.1f serial_ns=%.2f snapshot_ns=%.1f check=%llu\n", static_cast<unsigned long long>(ranges), replaceUs, replaceUs / static_cast<double>(ranges), resolveNs, serialNs, snapshotNs, static_cast<unsigned long long>(serials & 1u));
    PrintReport("small_windows", device.Memory());
    device.Guest().Clear();
    device.WaitIdle(10000);
    munmap(region, ranges * stride);
}

}

int main() {
    @autoreleasepool {
        std::string failure;
        const auto createBegin = Clock::now();
        auto device = Device::Create(&failure);
        if (device == nullptr) {
            std::fprintf(stderr, "FAIL: %s\n", failure.c_str());
            return 1;
        }
        const auto& limits = device->Limits();
        std::printf("device name=\"%s\" create_us=%.1f host_page=%llu max_buffer_mib=%llu window_mib=%llu working_set_mib=%llu driver_budget_mib=%llu\n", limits.name.c_str(), Microseconds(createBegin), static_cast<unsigned long long>(limits.hostPageBytes), static_cast<unsigned long long>(limits.maxBufferBytes / MiB), static_cast<unsigned long long>(limits.maxWindowBytes / MiB), static_cast<unsigned long long>(limits.recommendedWorkingSetBytes / MiB), static_cast<unsigned long long>(limits.driverBudgetBytes / MiB));
        PrintReport("start", device->Memory());
        auto kernels = MakeKernels(*device);
        ZeroCopy(*device, kernels);
        Footprint(*device, kernels);
        Lookups(*device);
        device->ReleaseBuffer(kernels.args);
        device->WaitIdle(10000);
        PrintReport("end", device->Memory());
        std::puts("metal guest memory probe: ok");
    }
    return 0;
}
