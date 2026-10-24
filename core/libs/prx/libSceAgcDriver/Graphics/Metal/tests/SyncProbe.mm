#include "prx/libSceAgcDriver/Graphics/Metal/Sync.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <libkern/OSCacheControl.h>
#include <sys/mman.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

using namespace AgcDriver::Graphics::Metal;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t MiB = 1ull << 20u;
constexpr std::uint32_t HeavyRounds = 192;

const char* const KernelSource = R"(
#include <metal_stdlib>
using namespace metal;
struct Args {
    device uint* data;
    uint count;
    uint rounds;
};
kernel void heavy(constant Args& args [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index >= args.count) return;
    uint x = args.data[index];
    for (uint i = 0u; i < args.rounds; ++i) x = x * 1664525u + 1013904223u;
    args.data[index] = x;
}
)";

struct HostArgs {
    std::uint64_t data;
    std::uint32_t count;
    std::uint32_t rounds;
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

std::uint32_t Heavy(std::uint32_t x) {
    for (std::uint32_t i = 0; i < HeavyRounds; ++i) x = x * 1664525u + 1013904223u;
    return x;
}

std::uint32_t Load(const void* address) {
    return std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(const_cast<void*>(address))).load(std::memory_order_relaxed);
}

struct Probe {
    Device& device;
    Sync& sync;
    id<MTLDevice> native;
    id<MTLComputePipelineState> heavy;
    id<MTL4ArgumentTable> table;
    std::uint8_t* region;
    std::uint64_t regionBytes;

    std::uint64_t DeviceAddress(void* pointer, std::uint64_t bytes) const {
        const auto snapshot = device.Guest().Snapshot();
        const auto resolved = Resolve(*snapshot, reinterpret_cast<std::uint64_t>(pointer), bytes);
        Check(static_cast<bool>(resolved), "the range resolves");
        return resolved.deviceAddress;
    }

    void EncodeHeavy(Recording& recording, std::uint64_t address, std::uint32_t count, bool producerBarrier = false) {
        NativeBuffer args = device.NewSharedBuffer(sizeof(HostArgs), "probe args");
        *static_cast<HostArgs*>(args.contents) = {address, count, HeavyRounds};
        id<MTL4CommandBuffer> commands = (__bridge id<MTL4CommandBuffer>)recording.commands;
        id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:heavy];
        [table setAddress:args.gpuAddress atIndex:0];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (producerBarrier) [encoder barrierAfterStages:MTLStageDispatch beforeQueueStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder endEncoding];
        device.ReleaseBuffer(args);
    }
};

std::span<const std::byte> Bytes(const void* data, std::size_t size) {
    return {static_cast<const std::byte*>(data), size};
}

void RoundTrip(Probe& probe) {
    auto* words = reinterpret_cast<std::uint32_t*>(probe.region);
    words[0] = 0;
    words[1] = 0;
    const std::uint64_t value = 0x1122334455667788ull;
    const auto guest = reinterpret_cast<std::uint64_t>(probe.region);
    const auto begin = Clock::now();
    Check(probe.sync.WriteLabel(QueueKind::Graphics, guest, Bytes(&value, 8), 1, true) == LabelOutcome::Queued, "a label with open work is queued");
    Check(probe.sync.HasQueuedLabels(QueueKind::Graphics), "the label is listed as queued");
    Check(probe.sync.LabelPendingIn(guest, 8), "a queued label counts as pending");
    auto recording = probe.sync.Begin(QueueKind::Graphics, "label round trip");
    Check(probe.sync.EncodeLabels(recording) == 1, "one label is encoded");
    Check(!probe.sync.HasQueuedLabels(QueueKind::Graphics), "the queue is drained by the encode");
    const auto serial = probe.sync.Submit(&recording, 1);
    Check(serial != 0 && !recording, "the recording is submitted and cleared");
    Check(probe.sync.WaitLabels(guest, 8, 10000), "the label lands");
    const auto us = Microseconds(begin);
    Check(*reinterpret_cast<std::uint64_t*>(probe.region) == value, "the GPU wrote the 64-bit label whole");
    Check(!probe.sync.LabelPendingIn(guest, 8), "a landed label is settled");
    std::printf("sync.roundtrip_us=%.1f\n", us);
}

void Outcomes(Probe& probe) {
    const auto guest = reinterpret_cast<std::uint64_t>(probe.region);
    const std::uint32_t value = 7;
    probe.device.WaitIdle(10000);
    Check(probe.sync.WriteLabel(QueueKind::Graphics, guest, Bytes(&value, 4), 2, false) == LabelOutcome::Direct, "an idle queue without open work writes on the CPU");
    Check(probe.sync.WriteLabel(QueueKind::Graphics, guest + 2, Bytes(&value, 4), 3, true) == LabelOutcome::Rejected, "an unaligned address is rejected");
    Check(probe.sync.WriteLabel(QueueKind::Graphics, guest, std::span<const std::byte>(), 4, true) == LabelOutcome::Rejected, "an empty label is rejected");
    std::vector<std::byte> big(Sync::MaxLabelBytes + 4);
    Check(probe.sync.WriteLabel(QueueKind::Graphics, guest, big, 5, true) == LabelOutcome::Rejected, "an oversized label is rejected");
    Check(probe.sync.WriteLabel(QueueKind::Graphics, 0x10, Bytes(&value, 4), 6, true) == LabelOutcome::Direct, "an unmapped address falls back to the CPU");
    Check(!probe.sync.HasQueuedLabels(QueueKind::Graphics), "no outcome left a queued label behind");
}

std::uint32_t OrderingRun(Probe& probe, bool producerBarrier, bool flushInit, double* worst) {
    const std::uint32_t count = 4 * 1024 * 1024;
    auto* data = reinterpret_cast<std::uint32_t*>(probe.region + 4096);
    auto* label = reinterpret_cast<std::uint32_t*>(probe.region + 4096 + count * 4 + 64);
    const auto dataDevice = probe.DeviceAddress(data, count * 4ull);
    const auto labelAddress = reinterpret_cast<std::uint64_t>(label);
    std::uint32_t violations = 0;
    for (std::uint32_t round = 1; round <= 12; ++round) {
        for (std::uint32_t index = 0; index < count; ++index) data[index] = index ^ round;
        *label = 0;
        if (flushInit) sys_dcache_flush(data, count * 4ull);
        auto recording = probe.sync.Begin(QueueKind::Compute, "ordering");
        probe.EncodeHeavy(recording, dataDevice, count, producerBarrier);
        const std::uint32_t mark = 0xc0de0000u + round;
        Check(probe.sync.WriteLabel(QueueKind::Compute, labelAddress, Bytes(&mark, 4), round, true) == LabelOutcome::Queued, "the ordering label is queued");
        Check(probe.sync.EncodeLabels(recording) == 1, "the ordering label is encoded");
        const auto begin = Clock::now();
        probe.sync.Submit(&recording, 1);
        while (Load(label) != mark) std::this_thread::yield();
        *worst = std::max(*worst, Microseconds(begin));
        const auto scan = [&](std::uint32_t& pending, std::uint32_t& wrong, std::uint32_t& first) {
            pending = wrong = 0;
            first = count;
            for (std::uint32_t index = 0; index < count; index += 61) {
                const auto value = Load(&data[index]);
                if (value == Heavy(index ^ round)) continue;
                ++(value == (index ^ round) ? pending : wrong);
                first = std::min(first, index);
            }
        };
        std::uint32_t pending = 0, wrong = 0, first = count, later = 0, afterSleep = 0, afterIdle = 0, ignore = 0;
        scan(pending, wrong, first);
        const auto serialNow = probe.device.CompletedSerial(QueueKind::Compute);
        const auto submittedNow = probe.device.SubmittedSerial(QueueKind::Compute);
        scan(later, ignore, ignore);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        scan(afterSleep, ignore, ignore);
        probe.device.WaitIdle(10000);
        scan(afterIdle, ignore, ignore);
        if (pending + wrong != 0) {
            ++violations;
            std::printf("sync.ordering violation producer_barrier=%d flush=%d round=%u unprocessed=%u wrong=%u first_index=%u rescan=%u after_2ms=%u after_idle=%u gpu_done=%d\n", producerBarrier ? 1 : 0, flushInit ? 1 : 0, round, pending, wrong, first, later, afterSleep, afterIdle, serialNow >= submittedNow ? 1 : 0);
        }
        probe.device.WaitIdle(10000);
    }
    return violations;
}

void Ordering(Probe& probe) {
    double worst = 0;
    const auto queueOnly = OrderingRun(probe, false, false, &worst);
    const auto withProducer = OrderingRun(probe, true, false, &worst);
    const auto flushed = OrderingRun(probe, false, true, &worst);
    std::printf("sync.ordering queue_barrier_violations=%u producer_barrier_violations=%u flushed_violations=%u worst_us=%.1f\n", queueOnly, withProducer, flushed, worst);
    Check(flushed == 0, "the label never lands before earlier work");
}

void SameAddress(Probe& probe) {
    auto* words = reinterpret_cast<std::uint32_t*>(probe.region + 8 * MiB);
    const auto guest = reinterpret_cast<std::uint64_t>(words);
    for (int round = 0; round < 200; ++round) {
        std::memset(words, 0, 128 * 1024);
        auto recording = probe.sync.Begin(QueueKind::Graphics, "same address");
        for (std::uint32_t value = 1; value <= 64; ++value) {
            Check(probe.sync.WriteLabel(QueueKind::Graphics, guest, Bytes(&value, 4), value, true) == LabelOutcome::Queued, "queue a repeated label");
            const std::uint64_t pair[2] = {value, ~static_cast<std::uint64_t>(value)};
            Check(probe.sync.WriteLabel(QueueKind::Graphics, guest + 64 * 4 + (value % 8) * 16, Bytes(pair, 16), value, true) == LabelOutcome::Queued, "queue a wide label");
        }
        std::vector<std::uint32_t> large(16384);
        for (std::uint32_t index = 0; index < large.size(); ++index) large[index] = index * 2654435761u;
        Check(probe.sync.WriteLabel(QueueKind::Graphics, guest + 4096, Bytes(large.data(), large.size() * 4), 99, true) == LabelOutcome::Queued, "queue a 64 KiB label");
        const std::uint32_t odd[3] = {0xa1, 0xa2, 0xa3};
        Check(probe.sync.WriteLabel(QueueKind::Graphics, guest + 70000 + 4, Bytes(odd, 12), 100, true) == LabelOutcome::Queued, "queue an odd sized label");
        Check(probe.sync.EncodeLabels(recording) == 64 * 2 + 2, "every label is encoded");
        probe.sync.Submit(&recording, 1);
        Check(probe.sync.WaitLabels(guest, 128 * 1024, 10000), "the labels land");
        Check(words[0] == 64, "later labels at one address win");
        for (std::uint32_t slot = 0; slot < 8; ++slot) {
            const auto* pair = reinterpret_cast<const std::uint64_t*>(words + 64 + slot * 4);
            Check(pair[0] == 56 + slot + (slot == 0 ? 8 : 0) && pair[1] == ~pair[0], "wide labels land in order and whole");
        }
        for (std::uint32_t index = 0; index < large.size(); ++index) Check(words[1024 + index] == large[index], "the large label is intact");
        Check(words[70000 / 4 + 1] == 0xa1 && words[70000 / 4 + 2] == 0xa2 && words[70000 / 4 + 3] == 0xa3 && words[70000 / 4 + 4] == 0, "the odd label stays inside its bytes");
    }
    std::puts("sync.same_address ok rounds=200");
}

void CrossQueue(Probe& probe) {
    const std::uint32_t count = 4 * 1024 * 1024;
    auto* data = reinterpret_cast<std::uint32_t*>(probe.region + 16 * MiB);
    auto* label = reinterpret_cast<std::uint32_t*>(probe.region + 16 * MiB + count * 4 + 64);
    const auto dataDevice = probe.DeviceAddress(data, count * 4ull);
    for (std::uint32_t round = 1; round <= 8; ++round) {
        for (std::uint32_t index = 0; index < count; ++index) data[index] = index + round;
        *label = 0;
        auto compute = probe.sync.Begin(QueueKind::Compute, "producer");
        probe.EncodeHeavy(compute, dataDevice, count);
        const auto produced = probe.sync.Submit(&compute, 1);
        probe.sync.WaitOnQueue(QueueKind::Graphics, QueueKind::Compute, produced);
        const std::uint32_t mark = 0xbeef0000u + round;
        Check(probe.sync.WriteLabel(QueueKind::Graphics, reinterpret_cast<std::uint64_t>(label), Bytes(&mark, 4), round, true) == LabelOutcome::Queued, "queue the consumer label");
        auto graphics = probe.sync.Begin(QueueKind::Graphics, "consumer");
        probe.sync.EncodeLabels(graphics);
        probe.sync.Submit(&graphics, 1);
        while (Load(label) != mark) std::this_thread::yield();
        Check(data[count - 1] == Heavy(count - 1 + round) && data[12345] == Heavy(12345 + round), "the graphics label waited for the compute queue");
    }
    probe.device.WaitIdle(10000);
    std::puts("sync.cross_queue ok rounds=8");
}

void GateTest(Probe& probe) {
    auto* label = reinterpret_cast<std::uint32_t*>(probe.region + 30 * MiB);
    *label = 0;
    auto gate = probe.sync.NewGate("probe gate");
    probe.sync.WaitOnGate(QueueKind::Graphics, *gate, 1);
    const std::uint32_t mark = 0x6a7e;
    Check(probe.sync.WriteLabel(QueueKind::Graphics, reinterpret_cast<std::uint64_t>(label), Bytes(&mark, 4), 1, true) == LabelOutcome::Queued, "queue the gated label");
    auto recording = probe.sync.Begin(QueueKind::Graphics, "gated");
    probe.sync.EncodeLabels(recording);
    const auto serial = probe.sync.Submit(&recording, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    Check(probe.device.CompletedSerial(QueueKind::Graphics) < serial && *label == 0, "the queue waits for the CPU gate");
    const auto begin = Clock::now();
    gate->Signal(1);
    Check(probe.device.WaitSerial(QueueKind::Graphics, serial, 10000), "the queue continues after the gate opens");
    Check(*label == mark, "the gated label lands after the gate");
    probe.sync.WaitOnGate(QueueKind::Graphics, *gate, 1);
    std::printf("sync.gate ok release_us=%.1f\n", Microseconds(begin));
}

void Completions(Probe& probe) {
    const std::uint32_t count = 4 * 1024 * 1024;
    auto* data = reinterpret_cast<std::uint32_t*>(probe.region + 32 * MiB);
    const auto dataDevice = probe.DeviceAddress(data, count * 4ull);
    for (std::uint32_t index = 0; index < count; ++index) data[index] = index;
    std::atomic<int> order{0};
    std::atomic<int> first{0}, second{0};
    std::atomic<bool> offThread{false};
    const auto caller = std::this_thread::get_id();
    auto recording = probe.sync.Begin(QueueKind::Compute, "completions");
    probe.EncodeHeavy(recording, dataDevice, count);
    const auto serial = probe.sync.Submit(&recording, 1);
    probe.sync.OnComplete(QueueKind::Compute, serial, [&] { first = ++order; offThread = std::this_thread::get_id() != caller; });
    probe.sync.OnComplete(QueueKind::Compute, serial, [&] { second = ++order; });
    probe.device.WaitSerial(QueueKind::Compute, serial, 10000);
    for (int spin = 0; spin < 2000 && second.load() == 0; ++spin) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Check(first == 1 && second == 2, "completion actions run once, in order");
    Check(offThread.load(), "a deferred action runs off the submitting thread");
    bool inlineRan = false;
    probe.sync.OnComplete(QueueKind::Compute, serial, [&] { inlineRan = true; });
    Check(inlineRan, "an action on a finished serial runs inline");
    std::puts("sync.completions ok");
}

void Throughput(Probe& probe) {
    const auto guest = reinterpret_cast<std::uint64_t>(probe.region + 40 * MiB);
    const auto before = probe.sync.Counters();
    const auto footprint = PhysFootprintBytes();
    const int cycles = 4000;
    double encodeUs = 0, submitUs = 0, beginUs = 0;
    std::uint64_t serial = 0;
    for (int index = 0; index < cycles; ++index) {
        auto t0 = Clock::now();
        auto recording = probe.sync.Begin(QueueKind::Graphics, "throughput");
        beginUs += Microseconds(t0);
        const std::uint64_t value = index + 1;
        probe.sync.WriteLabel(QueueKind::Graphics, guest, Bytes(&value, 8), index, true);
        t0 = Clock::now();
        probe.sync.EncodeLabels(recording);
        encodeUs += Microseconds(t0);
        t0 = Clock::now();
        serial = probe.sync.Submit(&recording, 1);
        submitUs += Microseconds(t0);
        if (index % 6 == 5) probe.device.WaitSerial(QueueKind::Graphics, serial - 3, 10000);
    }
    probe.device.WaitSerial(QueueKind::Graphics, serial, 10000);
    Check(*reinterpret_cast<std::uint64_t*>(guest) == cycles, "the last label of the run landed");
    const auto after = probe.sync.Counters();
    Check(after.allocatorsCreated - before.allocatorsCreated <= 24, "command allocators are recycled");
    Check(after.stagingChunks - before.stagingChunks <= 24, "staging chunks are recycled");
    std::printf("sync.throughput cycles=%d begin_us=%.2f encode_us=%.2f submit_us=%.2f allocators_created=%llu reused=%llu chunks=%llu footprint_delta_kib=%lld\n", cycles, beginUs / cycles, encodeUs / cycles, submitUs / cycles, static_cast<unsigned long long>(after.allocatorsCreated - before.allocatorsCreated), static_cast<unsigned long long>(after.allocatorsReused - before.allocatorsReused), static_cast<unsigned long long>(after.stagingChunks - before.stagingChunks), (static_cast<long long>(PhysFootprintBytes()) - static_cast<long long>(footprint)) / 1024);
    const int batch = 256;
    double writeNs = 0, batchEncodeUs = 0;
    for (int round = 0; round < 40; ++round) {
        auto recording = probe.sync.Begin(QueueKind::Graphics, "batch");
        auto t0 = Clock::now();
        for (int index = 0; index < batch; ++index) {
            const std::uint64_t value = round * batch + index;
            probe.sync.WriteLabel(QueueKind::Graphics, guest + 64 + (index % 64) * 8, Bytes(&value, 8), index, true);
        }
        writeNs += std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
        t0 = Clock::now();
        probe.sync.EncodeLabels(recording);
        batchEncodeUs += Microseconds(t0);
        probe.sync.Submit(&recording, 1);
    }
    probe.device.WaitIdle(10000);
    std::printf("sync.batch labels_per_batch=%d write_ns_per_label=%.0f encode_us_per_batch=%.1f encode_ns_per_label=%.0f\n", batch, writeNs / (40.0 * batch), batchEncodeUs / 40.0, batchEncodeUs * 1000.0 / (40.0 * batch));
}

}

int main() {
    @autoreleasepool {
        std::string failure;
        auto device = Device::Create(&failure);
        Check(device != nullptr, failure.c_str());
        auto sync = Sync::Create(*device, &failure);
        Check(sync != nullptr, failure.c_str());
        const auto page = device->Limits().hostPageBytes;
        const std::uint64_t regionBytes = 64 * MiB;
        auto* region = static_cast<std::uint8_t*>(mmap(nullptr, regionBytes + 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
        Check(region != MAP_FAILED, "mmap");
        auto* aligned = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uint64_t>(region) + page - 1) & ~(page - 1));
        Check(device->Guest().Map({reinterpret_cast<std::uint64_t>(aligned), regionBytes, true}), "map the probe region");
        Probe probe{*device, *sync, (__bridge id<MTLDevice>)device->NativeDevice(), nil, nil, aligned, regionBytes};
        NSError* error = nil;
        id<MTLLibrary> library = [probe.native newLibraryWithSource:[NSString stringWithUTF8String:KernelSource] options:nil error:&error];
        Check(library != nil, "compile the probe kernel");
        probe.heavy = [probe.native newComputePipelineStateWithFunction:[library newFunctionWithName:@"heavy"] error:&error];
        Check(probe.heavy != nil, "build the probe pipeline");
        MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 1;
        probe.table = [probe.native newArgumentTableWithDescriptor:descriptor error:&error];
        Check(probe.table != nil, "create the argument table");
        RoundTrip(probe);
        Outcomes(probe);
        Ordering(probe);
        SameAddress(probe);
        CrossQueue(probe);
        GateTest(probe);
        Completions(probe);
        Throughput(probe);
        const auto counters = sync->Counters();
        std::printf("sync.counters recordings=%llu submits=%llu labels_encoded=%llu dispatches=%llu direct=%llu rejected=%llu dropped=%llu waits=%llu\n", static_cast<unsigned long long>(counters.recordings), static_cast<unsigned long long>(counters.submits), static_cast<unsigned long long>(counters.labelsEncoded), static_cast<unsigned long long>(counters.labelDispatches), static_cast<unsigned long long>(counters.labelsDirect), static_cast<unsigned long long>(counters.labelsRejected), static_cast<unsigned long long>(counters.labelsDropped), static_cast<unsigned long long>(counters.queueWaits));
        Check(counters.labelsDropped == 0, "no label was dropped");
        sync.reset();
        device->Guest().Clear();
        device.reset();
        munmap(region, regionBytes + 2 * page);
        std::puts("agc_metal_sync_probe ok");
    }
    return 0;
}
