#include "prx/libSceAgcDriver/Graphics/Metal/Sync.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr std::uint64_t ChunkBytes = 128 * 1024;
constexpr std::uint32_t TaskDwords = 64;
constexpr std::size_t GroupLabels = 256;
constexpr std::uint64_t ArgsBytes = 32;
constexpr std::uint64_t TaskBytes = 16;
constexpr std::uint64_t Unsealed = std::numeric_limits<std::uint64_t>::max();

constexpr const char* KernelSource = R"AGCMSL(
#include <metal_stdlib>
using namespace metal;
struct LabelTask {
    ulong dst;
    uint src;
    uint dwords;
};
struct LabelArgs {
    device const LabelTask* tasks;
    device const uint* payload;
    uint count;
};
kernel void aps5_write_labels(constant LabelArgs& args [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index >= args.count) return;
    const LabelTask task = args.tasks[index];
    device const uint* source = args.payload + task.src;
    if ((task.dst & 7ul) == 0ul && (task.dwords & 1u) == 0u) {
        device ulong* target = reinterpret_cast<device ulong*>(task.dst);
        const uint pairs = task.dwords >> 1u;
        for (uint i = 0u; i < pairs; ++i) target[i] = (ulong(source[i * 2u + 1u]) << 32u) | ulong(source[i * 2u]);
    } else {
        device uint* target = reinterpret_cast<device uint*>(task.dst);
        for (uint i = 0u; i < task.dwords; ++i) target[i] = source[i];
    }
}
)AGCMSL";

struct HostLabelTask {
    std::uint64_t dst;
    std::uint32_t src;
    std::uint32_t dwords;
};
static_assert(sizeof(HostLabelTask) == TaskBytes);

struct HostLabelArgs {
    std::uint64_t tasks;
    std::uint64_t payload;
    std::uint32_t count;
    std::uint32_t reserved[3];
};
static_assert(sizeof(HostLabelArgs) == ArgsBytes);

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::size_t queueIndex(QueueKind queue) {
    return static_cast<std::size_t>(queue);
}

struct Slot {
    id<MTL4CommandAllocator> allocator = nil;
    id<MTL4CommandBuffer> commands = nil;
    std::uint64_t serial = 0;
    bool inUse = false;
};

struct Chunk {
    NativeBuffer buffer;
    QueueKind queue = QueueKind::Graphics;
    std::uint64_t serial = 0;
    bool inflight = false;
};

struct PendingLabel {
    std::uint64_t address = 0;
    std::uint32_t bytes = 0;
    std::uint64_t stamp = 0;
    std::uint64_t offset = 0;
};

struct ResolvedLabel {
    PendingLabel label;
    std::uint64_t deviceAddress = 0;
};

struct LabelGroup {
    std::vector<std::size_t> items;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::uint64_t tasks = 0;
    std::uint64_t payloadBytes = 0;
    std::uint64_t offset = 0;
};

struct StoreItem {
    std::uint64_t address = 0;
    std::uint32_t bytes = 0;
    std::uint64_t offset = 0;
};

struct StoreBatch {
    std::vector<StoreItem> items;
    std::vector<std::byte> payload;
};

struct QueueState {
    std::mutex slotMutex;
    std::vector<std::unique_ptr<Slot>> slots;
    std::mutex labelMutex;
    std::vector<PendingLabel> pending;
    std::vector<std::byte> payload;
    std::mutex encodeMutex;
    id<MTL4ArgumentTable> table = nil;
};

struct Core {
    CompletionActions actions;
    std::mutex runMutex;
    std::array<id<MTLSharedEvent>, QueueCount> timelines;
    std::atomic<bool> closed{false};
};

void drain(const std::shared_ptr<Core>& core, QueueKind queue) {
    std::lock_guard run(core->runMutex);
    if (core->closed.load(std::memory_order_acquire)) return;
    const auto completed = core->timelines[queueIndex(queue)].signaledValue;
    for (auto& action : core->actions.Take(queue, completed)) action();
}

}

struct Gate::State {
    id<MTLSharedEvent> event = nil;
};

Gate::Gate(std::unique_ptr<State> state) : state(std::move(state)) {}

Gate::~Gate() = default;

void Gate::Signal(std::uint64_t value) {
    if (state->event.signaledValue < value) state->event.signaledValue = value;
}

std::uint64_t Gate::Value() const {
    return state->event.signaledValue;
}

bool Gate::Wait(std::uint64_t value, std::uint32_t timeoutMs) const {
    if (state->event.signaledValue >= value) return true;
    return [state->event waitUntilSignaledValue:value timeoutMS:timeoutMs];
}

void* Gate::Native() const {
    return (__bridge void*)state->event;
}

struct Sync::State {
    Device& device;
    id<MTLDevice> native = nil;
    id<MTLComputePipelineState> pipeline = nil;
    std::array<QueueState, QueueCount> queues;
    std::mutex chunkMutex;
    std::vector<std::unique_ptr<Chunk>> chunks;
    LabelTable table;
    std::shared_ptr<Core> core = std::make_shared<Core>();
    MTLSharedEventListener* listener = nil;
    std::atomic<std::uint64_t> recordings{0};
    std::atomic<std::uint64_t> submits{0};
    std::atomic<std::uint64_t> allocatorsCreated{0};
    std::atomic<std::uint64_t> allocatorsReused{0};
    std::atomic<std::uint64_t> labelsQueued{0};
    std::atomic<std::uint64_t> labelsDirect{0};
    std::atomic<std::uint64_t> labelsRejected{0};
    std::atomic<std::uint64_t> labelsEncoded{0};
    std::atomic<std::uint64_t> labelsDropped{0};
    std::atomic<std::uint64_t> labelDispatches{0};
    std::atomic<std::uint64_t> labelsStored{0};
    std::atomic<bool> gpuLabels{false};
    std::atomic<std::uint64_t> labelBytes{0};
    std::atomic<std::uint64_t> stagingChunks{0};
    std::atomic<std::uint64_t> stagingBytes{0};
    std::atomic<std::uint64_t> queueWaits{0};
    std::atomic<std::uint64_t> completionsInline{0};
    std::atomic<std::uint64_t> completionsDeferred{0};

    explicit State(Device& device) : device(device) {}

    SerialSet completedSerials() const {
        SerialSet completed{};
        for (std::size_t index = 0; index < QueueCount; ++index) completed[index] = device.CompletedSerial(static_cast<QueueKind>(index));
        return completed;
    }

    void settle() {
        table.Settle(completedSerials());
    }

    std::uint32_t acquireChunk(QueueKind queue, std::uint64_t bytes) {
        std::lock_guard lock(chunkMutex);
        const auto completed = completedSerials();
        for (auto& chunk : chunks) {
            if (chunk->inflight && completed[queueIndex(chunk->queue)] >= chunk->serial) chunk->inflight = false;
        }
        std::size_t pick = chunks.size();
        std::size_t small = chunks.size();
        for (std::size_t index = 0; index < chunks.size(); ++index) {
            if (chunks[index]->inflight) continue;
            if (chunks[index]->buffer.bytes >= bytes) {
                pick = index;
                break;
            }
            if (small == chunks.size()) small = index;
        }
        if (pick == chunks.size()) {
            const auto size = std::max(ChunkBytes, alignUp(bytes, 16384));
            if (small != chunks.size()) {
                device.ReleaseBuffer(chunks[small]->buffer);
                chunks[small]->buffer = device.NewSharedBuffer(size, "AnyPS5 label staging");
                if (!chunks[small]->buffer) throw std::runtime_error("AGC graphics: Metal label staging could not be allocated");
                pick = small;
            } else {
                auto chunk = std::make_unique<Chunk>();
                chunk->buffer = device.NewSharedBuffer(size, "AnyPS5 label staging");
                if (!chunk->buffer) throw std::runtime_error("AGC graphics: Metal label staging could not be allocated");
                chunks.push_back(std::move(chunk));
                pick = chunks.size() - 1;
            }
            stagingChunks.fetch_add(1, std::memory_order_relaxed);
            stagingBytes.fetch_add(size, std::memory_order_relaxed);
        }
        chunks[pick]->inflight = true;
        chunks[pick]->queue = queue;
        chunks[pick]->serial = Unsealed;
        return static_cast<std::uint32_t>(pick);
    }

    void sealChunk(std::uint32_t id, std::uint64_t serial) {
        std::lock_guard lock(chunkMutex);
        chunks[id]->serial = serial;
    }

    void store(const StoreBatch& batch, QueueKind queue, std::uint64_t serial) {
        const auto snapshot = device.Guest().Snapshot();
        for (const auto& item : batch.items) {
            const auto resolved = Resolve(*snapshot, item.address, item.bytes);
            if (!resolved || !resolved.window->writable || resolved.window->copied) {
                labelsDropped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const auto* source = batch.payload.data() + item.offset;
            if (item.bytes == 8 && item.address % 8 == 0) {
                std::uint64_t value;
                std::memcpy(&value, source, 8);
                std::atomic_ref<std::uint64_t>(*reinterpret_cast<std::uint64_t*>(item.address)).store(value, std::memory_order_release);
            } else {
                for (std::uint32_t done = 0; done < item.bytes; done += 4) {
                    std::uint32_t value;
                    std::memcpy(&value, source + done, 4);
                    std::atomic_ref<std::uint32_t>(*reinterpret_cast<std::uint32_t*>(item.address + done)).store(value, std::memory_order_relaxed);
                }
                std::atomic_thread_fence(std::memory_order_release);
            }
            labelsStored.fetch_add(1, std::memory_order_relaxed);
        }
        table.Land(queue, serial);
    }

    bool queuedOverlap(std::uint64_t address, std::uint64_t bytes) {
        for (auto& queue : queues) {
            std::lock_guard lock(queue.labelMutex);
            for (const auto& label : queue.pending) {
                if (label.address < address + bytes && address < label.address + label.bytes) return true;
            }
        }
        return false;
    }
};

std::unique_ptr<Sync> Sync::Create(Device& device, std::string* failure) {
    @autoreleasepool {
        const auto fail = [&](const char* what) -> std::unique_ptr<Sync> {
            if (failure != nullptr) *failure = what;
            return nullptr;
        };
        auto state = std::make_unique<State>(device);
        state->native = (__bridge id<MTLDevice>)device.NativeDevice();
        NSError* error = nil;
        id<MTLLibrary> library = [state->native newLibraryWithSource:[NSString stringWithUTF8String:KernelSource] options:nil error:&error];
        if (library == nil) return fail("label kernel library");
        id<MTLFunction> function = [library newFunctionWithName:@"aps5_write_labels"];
        if (function == nil) return fail("label kernel function");
        state->pipeline = [state->native newComputePipelineStateWithFunction:function error:&error];
        if (state->pipeline == nil) return fail("label kernel pipeline");
        for (std::size_t index = 0; index < QueueCount; ++index) {
            MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
            descriptor.maxBufferBindCount = 1;
            state->queues[index].table = [state->native newArgumentTableWithDescriptor:descriptor error:&error];
            if (state->queues[index].table == nil) return fail("label argument table");
            state->core->timelines[index] = (__bridge id<MTLSharedEvent>)device.NativeTimeline(static_cast<QueueKind>(index));
        }
        const char* gpuLabels = std::getenv("APS5_METAL_GPU_LABELS");
        state->gpuLabels.store(gpuLabels != nullptr && *gpuLabels != '0', std::memory_order_relaxed);
        dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0);
        state->listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_queue_create("anyps5.metal.sync", attributes)];
        return std::unique_ptr<Sync>(new Sync(std::move(state)));
    }
}

Sync::Sync(std::unique_ptr<State> state) : state(std::move(state)) {}

Sync::~Sync() {
    if (state == nullptr) return;
    {
        std::lock_guard run(state->core->runMutex);
        state->core->closed.store(true, std::memory_order_release);
    }
    state->core->actions.Clear();
    state->device.WaitIdle(5000);
    for (auto& chunk : state->chunks) state->device.ReleaseBuffer(chunk->buffer);
}

Recording Sync::Begin(QueueKind queue, const char* label) {
    @autoreleasepool {
        auto& s = *state;
        auto& q = s.queues[queueIndex(queue)];
        const auto completed = s.device.CompletedSerial(queue);
        Recording recording;
        recording.queue = queue;
        std::lock_guard lock(q.slotMutex);
        Slot* slot = nullptr;
        for (std::size_t index = 0; index < q.slots.size(); ++index) {
            auto& candidate = *q.slots[index];
            if (candidate.inUse || candidate.serial > completed) continue;
            slot = &candidate;
            recording.slot = static_cast<std::uint32_t>(index);
            [slot->allocator reset];
            s.allocatorsReused.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        if (slot == nullptr) {
            auto created = std::make_unique<Slot>();
            created->allocator = [s.native newCommandAllocator];
            created->commands = [s.native newCommandBuffer];
            if (created->allocator == nil || created->commands == nil) throw std::runtime_error("AGC graphics: Metal command buffer could not be created");
            q.slots.push_back(std::move(created));
            slot = q.slots.back().get();
            recording.slot = static_cast<std::uint32_t>(q.slots.size() - 1);
            s.allocatorsCreated.fetch_add(1, std::memory_order_relaxed);
        }
        slot->inUse = true;
        slot->commands.label = label != nullptr ? [NSString stringWithUTF8String:label] : nil;
        [slot->commands beginCommandBufferWithAllocator:slot->allocator];
        recording.commands = (__bridge void*)slot->commands;
        s.recordings.fetch_add(1, std::memory_order_relaxed);
        return recording;
    }
}

std::uint64_t Sync::Submit(Recording* recordings, std::size_t count) {
    @autoreleasepool {
        if (count == 0) return 0;
        auto& s = *state;
        const auto queue = recordings[0].queue;
        std::vector<void*> list;
        list.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            if (recordings[index].queue != queue || !recordings[index]) throw std::runtime_error("AGC graphics: Metal submit needs open recordings of one queue");
            [(__bridge id<MTL4CommandBuffer>)recordings[index].commands endCommandBuffer];
            list.push_back(recordings[index].commands);
        }
        s.settle();
        const auto serial = s.device.Submit(queue, list.data(), list.size());
        {
            auto& q = s.queues[queueIndex(queue)];
            std::lock_guard lock(q.slotMutex);
            for (std::size_t index = 0; index < count; ++index) {
                auto& slot = *q.slots[recordings[index].slot];
                slot.serial = serial;
                slot.inUse = false;
            }
        }
        std::shared_ptr<StoreBatch> batch;
        for (std::size_t index = 0; index < count; ++index) {
            for (const auto id : recordings[index].staging) s.sealChunk(id, serial);
            const bool stores = !recordings[index].labelOffsets.empty();
            if (stores && !batch) batch = std::make_shared<StoreBatch>();
            for (std::size_t item = 0; item < recordings[index].labels.size(); ++item) {
                auto label = recordings[index].labels[item];
                label.serial = serial;
                label.deferred = stores;
                s.table.Add(label);
                if (stores) batch->items.push_back({label.address, label.bytes, batch->payload.size()});
                if (stores) batch->payload.insert(batch->payload.end(), recordings[index].labelPayload.begin() + recordings[index].labelOffsets[item], recordings[index].labelPayload.begin() + recordings[index].labelOffsets[item] + label.bytes);
            }
            recordings[index] = {};
        }
        if (batch) {
            State* owner = &s;
            OnComplete(queue, serial, [owner, batch, queue, serial] { owner->store(*batch, queue, serial); });
        }
        s.submits.fetch_add(1, std::memory_order_relaxed);
        return serial;
    }
}

void Sync::Discard(Recording& recording) {
    if (!recording) return;
    auto& s = *state;
    [(__bridge id<MTL4CommandBuffer>)recording.commands endCommandBuffer];
    {
        auto& q = s.queues[queueIndex(recording.queue)];
        std::lock_guard lock(q.slotMutex);
        auto& slot = *q.slots[recording.slot];
        slot.serial = 0;
        slot.inUse = false;
    }
    for (const auto id : recording.staging) s.sealChunk(id, 0);
    recording = {};
}

LabelOutcome Sync::WriteLabel(QueueKind queue, std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, bool openWork) {
    auto& s = *state;
    if (bytes.empty() || bytes.size() > MaxLabelBytes || address % 4 != 0 || bytes.size() % 4 != 0) {
        s.labelsRejected.fetch_add(1, std::memory_order_relaxed);
        return LabelOutcome::Rejected;
    }
    const auto snapshot = s.device.Guest().Snapshot();
    const auto resolved = Resolve(*snapshot, address, bytes.size());
    if (!resolved || resolved.window->copied || !resolved.window->writable) {
        s.labelsDirect.fetch_add(1, std::memory_order_relaxed);
        return LabelOutcome::Direct;
    }
    auto& q = s.queues[queueIndex(queue)];
    std::lock_guard lock(q.labelMutex);
    if (!openWork && q.pending.empty() && s.table.Size() == 0 && s.device.CompletedSerial(queue) >= s.device.SubmittedSerial(queue)) {
        s.labelsDirect.fetch_add(1, std::memory_order_relaxed);
        return LabelOutcome::Direct;
    }
    PendingLabel label;
    label.address = address;
    label.bytes = static_cast<std::uint32_t>(bytes.size());
    label.stamp = stamp;
    label.offset = q.payload.size();
    q.payload.insert(q.payload.end(), bytes.begin(), bytes.end());
    q.pending.push_back(label);
    s.labelsQueued.fetch_add(1, std::memory_order_relaxed);
    return LabelOutcome::Queued;
}

std::size_t Sync::EncodeLabels(Recording& recording) {
    @autoreleasepool {
        auto& s = *state;
        auto& q = s.queues[queueIndex(recording.queue)];
        std::vector<PendingLabel> labels;
        std::vector<std::byte> payload;
        {
            std::lock_guard lock(q.labelMutex);
            labels.swap(q.pending);
            payload.swap(q.payload);
        }
        if (labels.empty()) return 0;
        if (!s.gpuLabels.load(std::memory_order_relaxed)) {
            for (const auto& label : labels) {
                recording.labels.push_back({label.address, label.bytes, label.stamp, recording.queue, 0});
                recording.labelOffsets.push_back(recording.labelPayload.size());
                recording.labelPayload.insert(recording.labelPayload.end(), payload.begin() + label.offset, payload.begin() + label.offset + label.bytes);
                s.labelBytes.fetch_add(label.bytes, std::memory_order_relaxed);
            }
            s.labelsEncoded.fetch_add(labels.size(), std::memory_order_relaxed);
            return labels.size();
        }
        const auto snapshot = s.device.Guest().Snapshot();
        std::vector<ResolvedLabel> items;
        items.reserve(labels.size());
        for (const auto& label : labels) {
            const auto resolved = Resolve(*snapshot, label.address, label.bytes);
            if (!resolved || resolved.window->copied || !resolved.window->writable) {
                s.labelsDropped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            items.push_back({label, resolved.deviceAddress});
        }
        if (items.empty()) return 0;
        std::vector<LabelGroup> groups(1);
        for (std::size_t index = 0; index < items.size(); ++index) {
            const auto& label = items[index].label;
            const auto begin = label.address;
            const auto end = label.address + label.bytes;
            auto* group = &groups.back();
            bool clash = group->items.size() >= GroupLabels;
            for (const auto& range : group->ranges) {
                if (range.first < end && begin < range.second) {
                    clash = true;
                    break;
                }
            }
            if (clash) {
                groups.emplace_back();
                group = &groups.back();
            }
            group->items.push_back(index);
            group->ranges.emplace_back(begin, end);
            group->tasks += (label.bytes / 4 + TaskDwords - 1) / TaskDwords;
            group->payloadBytes += label.bytes;
        }
        std::uint64_t total = 0;
        for (auto& group : groups) {
            group.offset = total;
            total += alignUp(ArgsBytes + group.tasks * TaskBytes + alignUp(group.payloadBytes, 16), 16);
        }
        const auto chunkId = s.acquireChunk(recording.queue, total);
        recording.staging.push_back(chunkId);
        NativeBuffer buffer;
        {
            std::lock_guard lock(s.chunkMutex);
            buffer = s.chunks[chunkId]->buffer;
        }
        auto* base = static_cast<std::uint8_t*>(buffer.contents);
        std::vector<std::uint64_t> argAddresses;
        argAddresses.reserve(groups.size());
        for (const auto& group : groups) {
            auto* args = reinterpret_cast<HostLabelArgs*>(base + group.offset);
            auto* tasks = reinterpret_cast<HostLabelTask*>(base + group.offset + ArgsBytes);
            auto* data = base + group.offset + ArgsBytes + group.tasks * TaskBytes;
            std::uint64_t taskIndex = 0;
            std::uint64_t cursor = 0;
            for (const auto itemIndex : group.items) {
                const auto& item = items[itemIndex];
                std::memcpy(data + cursor, payload.data() + item.label.offset, item.label.bytes);
                const auto dwords = item.label.bytes / 4;
                const auto firstDword = static_cast<std::uint32_t>(cursor / 4);
                for (std::uint32_t done = 0; done < dwords; done += TaskDwords) {
                    tasks[taskIndex++] = {item.deviceAddress + static_cast<std::uint64_t>(done) * 4, firstDword + done, std::min<std::uint32_t>(TaskDwords, dwords - done)};
                }
                cursor += item.label.bytes;
                recording.labels.push_back({item.label.address, item.label.bytes, item.label.stamp, recording.queue, 0});
                s.labelBytes.fetch_add(item.label.bytes, std::memory_order_relaxed);
            }
            args->tasks = buffer.gpuAddress + group.offset + ArgsBytes;
            args->payload = buffer.gpuAddress + group.offset + ArgsBytes + group.tasks * TaskBytes;
            args->count = static_cast<std::uint32_t>(group.tasks);
            args->reserved[0] = args->reserved[1] = args->reserved[2] = 0;
            argAddresses.push_back(buffer.gpuAddress + group.offset);
        }
        id<MTL4CommandBuffer> commands = (__bridge id<MTL4CommandBuffer>)recording.commands;
        id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder setComputePipelineState:s.pipeline];
        const auto width = std::min<NSUInteger>(64, s.pipeline.maxTotalThreadsPerThreadgroup);
        {
            std::lock_guard lock(q.encodeMutex);
            for (std::size_t index = 0; index < groups.size(); ++index) {
                if (index != 0) [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
                [q.table setAddress:argAddresses[index] atIndex:0];
                [encoder setArgumentTable:q.table];
                [encoder dispatchThreads:MTLSizeMake(groups[index].tasks, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            }
        }
        [encoder endEncoding];
        s.labelDispatches.fetch_add(groups.size(), std::memory_order_relaxed);
        s.labelsEncoded.fetch_add(items.size(), std::memory_order_relaxed);
        return items.size();
    }
}

bool Sync::HasQueuedLabels(QueueKind queue) const {
    auto& q = state->queues[queueIndex(queue)];
    std::lock_guard lock(q.labelMutex);
    return !q.pending.empty();
}

bool Sync::LabelPendingIn(std::uint64_t address, std::uint64_t bytes) {
    auto& s = *state;
    s.settle();
    return s.table.Overlaps(address, bytes) || s.queuedOverlap(address, bytes);
}

bool Sync::WaitLabels(std::uint64_t address, std::uint64_t bytes, std::uint32_t timeoutMs) {
    auto& s = *state;
    s.settle();
    if (s.queuedOverlap(address, bytes)) return false;
    const auto serials = s.table.SerialsOver(address, bytes);
    for (std::size_t index = 0; index < QueueCount; ++index) {
        if (serials[index] != 0 && !s.device.WaitSerial(static_cast<QueueKind>(index), serials[index], timeoutMs)) return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;) {
        s.settle();
        if (!s.table.Overlaps(address, bytes)) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
}

std::optional<LabelRecord> Sync::NewestLabel(std::uint64_t address, std::uint64_t bytes, std::uint64_t afterStamp) {
    state->settle();
    return state->table.Newest(address, bytes, afterStamp);
}

void Sync::WaitOnQueue(QueueKind waiter, QueueKind signaler, std::uint64_t serial) {
    if (waiter == signaler || serial == 0 || state->device.CompletedSerial(signaler) >= serial) return;
    state->device.WaitOnQueue(waiter, signaler, serial);
    state->queueWaits.fetch_add(1, std::memory_order_relaxed);
}

std::shared_ptr<Gate> Sync::NewGate(const char* label) {
    @autoreleasepool {
        auto inner = std::make_unique<Gate::State>();
        inner->event = [state->native newSharedEvent];
        if (inner->event == nil) throw std::runtime_error("AGC graphics: Metal shared event could not be created");
        inner->event.signaledValue = 0;
        if (label != nullptr) inner->event.label = [NSString stringWithUTF8String:label];
        return std::shared_ptr<Gate>(new Gate(std::move(inner)));
    }
}

void Sync::WaitOnGate(QueueKind waiter, const Gate& gate, std::uint64_t value) {
    if (value == 0 || gate.Value() >= value) return;
    state->device.WaitOnEvent(waiter, gate.Native(), value);
    state->queueWaits.fetch_add(1, std::memory_order_relaxed);
}

void Sync::OnComplete(QueueKind queue, std::uint64_t serial, std::function<void()> action) {
    auto& s = *state;
    const bool done = s.device.CompletedSerial(queue) >= serial;
    const bool fresh = s.core->actions.Push(queue, serial, std::move(action));
    if (done) {
        s.completionsInline.fetch_add(1, std::memory_order_relaxed);
        drain(s.core, queue);
        return;
    }
    s.completionsDeferred.fetch_add(1, std::memory_order_relaxed);
    if (fresh) {
        auto core = s.core;
        [s.core->timelines[queueIndex(queue)] notifyListener:s.listener atValue:serial block:^(id<MTLSharedEvent>, uint64_t) {
            drain(core, queue);
        }];
    }
    if (s.device.CompletedSerial(queue) >= serial) drain(s.core, queue);
}

void Sync::SetGpuLabels(bool enabled) {
    state->gpuLabels.store(enabled, std::memory_order_relaxed);
}

bool Sync::GpuLabels() const {
    return state->gpuLabels.load(std::memory_order_relaxed);
}

void Sync::Trim(std::size_t keepIdle) {
    auto& s = *state;
    for (std::size_t index = 0; index < QueueCount; ++index) {
        auto& q = s.queues[index];
        const auto completed = s.device.CompletedSerial(static_cast<QueueKind>(index));
        std::lock_guard lock(q.slotMutex);
        while (q.slots.size() > keepIdle && !q.slots.back()->inUse && q.slots.back()->serial <= completed) q.slots.pop_back();
    }
    std::lock_guard lock(s.chunkMutex);
    const auto completed = s.completedSerials();
    while (s.chunks.size() > keepIdle && (!s.chunks.back()->inflight || completed[queueIndex(s.chunks.back()->queue)] >= s.chunks.back()->serial)) {
        s.device.ReleaseBuffer(s.chunks.back()->buffer);
        s.chunks.pop_back();
    }
}

SyncCounters Sync::Counters() const {
    auto& s = *state;
    SyncCounters counters;
    counters.recordings = s.recordings.load(std::memory_order_relaxed);
    counters.submits = s.submits.load(std::memory_order_relaxed);
    counters.allocatorsCreated = s.allocatorsCreated.load(std::memory_order_relaxed);
    counters.allocatorsReused = s.allocatorsReused.load(std::memory_order_relaxed);
    counters.labelsQueued = s.labelsQueued.load(std::memory_order_relaxed);
    counters.labelsDirect = s.labelsDirect.load(std::memory_order_relaxed);
    counters.labelsRejected = s.labelsRejected.load(std::memory_order_relaxed);
    counters.labelsEncoded = s.labelsEncoded.load(std::memory_order_relaxed);
    counters.labelsDropped = s.labelsDropped.load(std::memory_order_relaxed);
    counters.labelDispatches = s.labelDispatches.load(std::memory_order_relaxed);
    counters.labelsStored = s.labelsStored.load(std::memory_order_relaxed);
    counters.labelBytes = s.labelBytes.load(std::memory_order_relaxed);
    counters.stagingChunks = s.stagingChunks.load(std::memory_order_relaxed);
    counters.stagingBytes = s.stagingBytes.load(std::memory_order_relaxed);
    counters.queueWaits = s.queueWaits.load(std::memory_order_relaxed);
    counters.completionsInline = s.completionsInline.load(std::memory_order_relaxed);
    counters.completionsDeferred = s.completionsDeferred.load(std::memory_order_relaxed);
    return counters;
}

}
