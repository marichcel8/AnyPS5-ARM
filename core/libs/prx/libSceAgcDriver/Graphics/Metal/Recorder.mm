#include "prx/libSceAgcDriver/Graphics/Metal/Recorder.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <objc/runtime.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace AgcDriver::Graphics::Metal {

static_assert(static_cast<int>(PrimitiveType::TriangleStrip) == MTLPrimitiveTypeTriangleStrip);
static_assert(static_cast<int>(IndexType::UInt32) == MTLIndexTypeUInt32);
static_assert(static_cast<int>(CullMode::Back) == MTLCullModeBack);
static_assert(static_cast<int>(Winding::CounterClockwise) == MTLWindingCounterClockwise);
static_assert(static_cast<int>(DepthClipMode::Clamp) == MTLDepthClipModeClamp);
static_assert(static_cast<int>(FillMode::Lines) == MTLTriangleFillModeLines);
static_assert(static_cast<int>(LoadAction::Clear) == MTLLoadActionClear);
static_assert(static_cast<int>(StoreAction::Unknown) == MTLStoreActionUnknown);
static_assert(static_cast<int>(StoreAction::StoreAndMultisampleResolve) == MTLStoreActionStoreAndMultisampleResolve);

namespace {

constexpr std::uint32_t DepthStencilCacheSize = 64;

std::uint64_t attachmentIdentity(const void* texture, std::uint32_t level, std::uint32_t slice, std::uint32_t plane, std::uint32_t tag) {
    if (texture == nullptr) return 0;
    std::uint64_t value = reinterpret_cast<std::uintptr_t>(texture);
    value ^= (static_cast<std::uint64_t>(level) * 0x9e3779b97f4a7c15ull) ^ (static_cast<std::uint64_t>(slice) << 20u) ^ (static_cast<std::uint64_t>(plane) << 36u) ^ (static_cast<std::uint64_t>(tag) << 60u);
    return value == 0 ? 1 : value;
}

bool envFlag(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == 0) return fallback;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0;
}

bool deferredStoresAvailable() {
    const auto method = protocol_getMethodDescription(@protocol(MTL4RenderCommandEncoder), @selector(setColorStoreAction:atIndex:), YES, YES);
    if (method.name != nullptr) return true;
    return protocol_getMethodDescription(@protocol(MTL4RenderCommandEncoder), @selector(setColorStoreAction:atIndex:), NO, YES).name != nullptr;
}

struct Slot {
    id<MTL4CommandAllocator> allocator;
    id<MTL4CommandBuffer> commandBuffer;
    UploadRing ring;
    std::uint64_t serial = 0;
    bool used = false;

    Slot(UploadSource& source, std::uint64_t chunkBytes) : ring(source, chunkBytes) {}
};

struct DepthStencilEntry {
    std::uint64_t key = 0;
    void* native = nullptr;
};

}

RecorderOptions RecorderOptions::FromEnvironment() {
    RecorderOptions options;
    if (const char* policy = std::getenv("APS5_METAL_PIPELINE_MISS"); policy != nullptr && std::strcmp(policy, "skip") == 0) options.missPolicy = PipelineMissPolicy::Skip;
    options.deferredStores = envFlag("APS5_METAL_DEFERRED_STORES", true);
    options.passBarriers = envFlag("APS5_METAL_PASS_BARRIERS", true);
    return options;
}

struct Recorder::State final : UploadSource {
    Device& device;
    PipelineCache& pipelines;
    RecorderOptions options;
    id<MTLDevice> native;
    std::vector<std::unique_ptr<Slot>> slots;
    Slot* slot = nullptr;
    id<MTL4RenderCommandEncoder> encoder = nil;
    std::array<id<MTL4ArgumentTable>, 2> tables;
    std::array<SlotAddresses<RecorderBufferSlots>, 2> addresses;
    const RenderPipeline* pipeline = nullptr;
    void* boundPipeline = nullptr;
    std::uint64_t depthStencilKey = std::numeric_limits<std::uint64_t>::max();
    void* depthStencil = nullptr;
    void* boundDepthStencil = nullptr;
    std::array<DepthStencilEntry, DepthStencilCacheSize> depthStencilCache{};
    DynamicState dynamic;
    DynamicState boundDynamic;
    bool dynamicDirty = true;
    bool dynamicValid = false;
    PassTracker tracker;
    std::array<AttachmentIntent, PassAttachmentCount> passIntents{};
    std::uint32_t deferredMask = 0;
    std::uint32_t discardMask = 0;
    bool deferredSupport = false;
    RecorderStats stats;

    State(Device& device, PipelineCache& pipelines, const RecorderOptions& options) : device(device), pipelines(pipelines), options(options), native((__bridge id<MTLDevice>)device.NativeDevice()) {}

    UploadChunk Acquire(std::uint64_t bytes) override {
        const auto buffer = device.NewSharedBuffer(bytes, "agc.metal.upload");
        if (!buffer) return {};
        return {buffer.handle, static_cast<std::byte*>(buffer.contents), buffer.gpuAddress, buffer.bytes};
    }

    void Release(const UploadChunk& chunk) override {
        NativeBuffer buffer;
        buffer.handle = chunk.handle;
        buffer.contents = chunk.cpu;
        buffer.gpuAddress = chunk.gpu;
        buffer.bytes = chunk.bytes;
        device.ReleaseBuffer(buffer);
    }

    Slot* acquireSlot() {
        const auto completed = device.CompletedSerial(options.queue);
        Slot* oldest = nullptr;
        for (auto& candidate : slots) {
            if (!candidate->used || candidate->serial <= completed) return candidate.get();
            if (oldest == nullptr || candidate->serial < oldest->serial) oldest = candidate.get();
        }
        if (slots.size() < options.maxSlots || oldest == nullptr) {
            auto created = std::make_unique<Slot>(*this, options.uploadChunkBytes);
            created->allocator = [native newCommandAllocator];
            created->commandBuffer = [native newCommandBuffer];
            if (created->allocator == nil || created->commandBuffer == nil) return nullptr;
            slots.push_back(std::move(created));
            stats.slots = slots.size();
            return slots.back().get();
        }
        ++stats.slotWaits;
        device.WaitSerial(options.queue, oldest->serial, std::numeric_limits<std::uint32_t>::max());
        return oldest;
    }

    void begin() {
        if (slot != nullptr) return;
        slot = acquireSlot();
        if (slot == nullptr) return;
        [slot->allocator reset];
        slot->ring.Reset();
        [slot->commandBuffer beginCommandBufferWithAllocator:slot->allocator];
        if (options.label != nullptr) slot->commandBuffer.label = [NSString stringWithUTF8String:options.label];
    }

    void endEncoder() {
        if (encoder == nil) return;
        if (deferredMask != 0) {
            id untyped = encoder;
            for (std::uint32_t index = 0; index < PassAttachmentCount; ++index) {
                if ((deferredMask & (1u << index)) == 0) continue;
                const bool discarded = (discardMask & (1u << index)) != 0;
                const auto store = static_cast<MTLStoreAction>(FinalStore(passIntents[index], discarded));
                if (discarded) ++stats.storesDiscarded;
                if (index < MaxColorAttachments) [untyped setColorStoreAction:store atIndex:index];
                else if (index == DepthAttachmentIndex) [untyped setDepthStoreAction:store];
                else [untyped setStencilStoreAction:store];
            }
        }
        [encoder endEncoding];
        encoder = nil;
        deferredMask = 0;
        discardMask = 0;
    }

    void resetBound() {
        boundPipeline = nullptr;
        boundDepthStencil = nullptr;
        dynamicValid = false;
        dynamicDirty = true;
    }

    void applyActions(MTLRenderPassAttachmentDescriptor* attachment, std::uint32_t index, const AttachmentIntent& intent) {
        const auto actions = ChooseActions(intent, deferredSupport && options.deferredStores);
        attachment.loadAction = static_cast<MTLLoadAction>(actions.load);
        attachment.storeAction = static_cast<MTLStoreAction>(actions.store);
        if (actions.deferred) {
            deferredMask |= 1u << index;
            ++stats.storesDeferred;
        }
    }

    bool openEncoder(const RenderPassDesc& pass) {
        MTL4RenderPassDescriptor* descriptor = [MTL4RenderPassDescriptor new];
        for (std::uint32_t index = 0; index < pass.colorCount && index < MaxColorAttachments; ++index) {
            const auto& color = pass.colors[index];
            if (color.texture == nullptr) continue;
            MTLRenderPassColorAttachmentDescriptor* attachment = descriptor.colorAttachments[index];
            attachment.texture = (__bridge id<MTLTexture>)color.texture;
            attachment.level = color.level;
            attachment.slice = color.slice;
            attachment.depthPlane = color.depthPlane;
            if (color.resolveTexture != nullptr) attachment.resolveTexture = (__bridge id<MTLTexture>)color.resolveTexture;
            attachment.clearColor = MTLClearColorMake(color.clearColor[0], color.clearColor[1], color.clearColor[2], color.clearColor[3]);
            applyActions(attachment, index, color.intent);
        }
        if (pass.depth.texture != nullptr) {
            MTLRenderPassDepthAttachmentDescriptor* depth = descriptor.depthAttachment;
            depth.texture = (__bridge id<MTLTexture>)pass.depth.texture;
            depth.level = pass.depth.level;
            depth.slice = pass.depth.slice;
            depth.clearDepth = pass.depth.clearDepth;
            applyActions(depth, DepthAttachmentIndex, pass.depth.depthIntent);
            if (pass.depth.stencil) {
                MTLRenderPassStencilAttachmentDescriptor* stencil = descriptor.stencilAttachment;
                stencil.texture = (__bridge id<MTLTexture>)pass.depth.texture;
                stencil.level = pass.depth.level;
                stencil.slice = pass.depth.slice;
                stencil.clearStencil = pass.depth.clearStencil;
                applyActions(stencil, StencilAttachmentIndex, pass.depth.stencilIntent);
            }
        }
        descriptor.renderTargetWidth = pass.width;
        descriptor.renderTargetHeight = pass.height;
        if (pass.layers > 1) descriptor.renderTargetArrayLength = pass.layers;
        if (pass.colorCount == 0 && pass.depth.texture == nullptr) descriptor.defaultRasterSampleCount = pass.samples;
        encoder = [slot->commandBuffer renderCommandEncoderWithDescriptor:descriptor];
        if (encoder == nil) return false;
        if (options.passBarriers) {
            [encoder barrierAfterQueueStages:MTLStageVertex | MTLStageFragment | MTLStageDispatch | MTLStageBlit beforeStages:MTLStageVertex | MTLStageFragment visibilityOptions:MTL4VisibilityOptionDevice];
            ++stats.barriers;
        }
        [encoder setArgumentTable:tables[0] atStages:MTLRenderStageVertex];
        [encoder setArgumentTable:tables[1] atStages:MTLRenderStageFragment];
        resetBound();
        return true;
    }

    DrawOutcome prepare() {
        if (encoder == nil) return DrawOutcome::NoPass;
        if (pipeline == nullptr) return DrawOutcome::NoPipeline;
        void* resolved = pipeline->Native();
        if (resolved == nullptr) {
            if (pipeline->Failed()) return DrawOutcome::PipelineFailed;
            if (options.missPolicy == PipelineMissPolicy::Skip) return DrawOutcome::PipelinePending;
            const auto started = std::chrono::steady_clock::now();
            pipelines.Promote(*pipeline, CompilePriority::Urgent);
            pipeline->Wait(options.pipelineWaitMs);
            ++stats.pipelineWaits;
            stats.pipelineWaitNanos += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
            resolved = pipeline->Native();
            if (resolved == nullptr) return pipeline->Failed() ? DrawOutcome::PipelineFailed : DrawOutcome::PipelinePending;
        }
        if (resolved != boundPipeline) {
            [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)resolved];
            boundPipeline = resolved;
            ++stats.pipelineBinds;
        }
        if (depthStencil != boundDepthStencil && depthStencil != nullptr) {
            [encoder setDepthStencilState:(__bridge id<MTLDepthStencilState>)depthStencil];
            boundDepthStencil = depthStencil;
            ++stats.depthStencilBinds;
        }
        if (dynamicDirty) applyDynamic();
        return DrawOutcome::Drawn;
    }

    void applyDynamic() {
        const auto changed = dynamicValid ? DiffDynamic(boundDynamic, dynamic) : DynamicAll;
        dynamicDirty = false;
        if (changed == 0) return;
        ++stats.dynamicSets;
        if ((changed & DynamicViewport) != 0) {
            const auto& v = dynamic.viewport;
            [encoder setViewport:MTLViewport{v.originX, v.originY, v.width, v.height, v.znear, v.zfar}];
        }
        if ((changed & DynamicScissor) != 0) {
            const auto& s = dynamic.scissor;
            [encoder setScissorRect:MTLScissorRect{s.x, s.y, s.width, s.height}];
        }
        if ((changed & DynamicDepthBias) != 0) [encoder setDepthBias:dynamic.depthBias.constant slopeScale:dynamic.depthBias.slope clamp:dynamic.depthBias.clamp];
        if ((changed & DynamicBlendColor) != 0) [encoder setBlendColorRed:dynamic.blendColor[0] green:dynamic.blendColor[1] blue:dynamic.blendColor[2] alpha:dynamic.blendColor[3]];
        if ((changed & DynamicStencilReference) != 0) [encoder setStencilFrontReferenceValue:dynamic.stencilFrontReference backReferenceValue:dynamic.stencilBackReference];
        if ((changed & DynamicCull) != 0) [encoder setCullMode:static_cast<MTLCullMode>(dynamic.cull)];
        if ((changed & DynamicWinding) != 0) [encoder setFrontFacingWinding:static_cast<MTLWinding>(dynamic.winding)];
        if ((changed & DynamicDepthClip) != 0) [encoder setDepthClipMode:static_cast<MTLDepthClipMode>(dynamic.depthClip)];
        if ((changed & DynamicFill) != 0) [encoder setTriangleFillMode:static_cast<MTLTriangleFillMode>(dynamic.fill)];
        boundDynamic = dynamic;
        dynamicValid = true;
    }
};

Recorder::Recorder(std::unique_ptr<State> state) : state(std::move(state)) {}

Recorder::~Recorder() {
    auto& s = *state;
    if (s.slot != nullptr) Submit();
    for (const auto& slot : s.slots)
        if (slot->used) s.device.WaitSerial(s.options.queue, slot->serial, std::numeric_limits<std::uint32_t>::max());
    s.slots.clear();
}

std::unique_ptr<Recorder> Recorder::Create(Device& device, PipelineCache& pipelines, const RecorderOptions& options, std::string* failure) {
    @autoreleasepool {
        auto state = std::make_unique<State>(device, pipelines, options);
        if (state->native == nil) {
            if (failure != nullptr) *failure = "no Metal device";
            return nullptr;
        }
        MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = RecorderBufferSlots;
        for (auto& table : state->tables) {
            NSError* error = nil;
            table = [state->native newArgumentTableWithDescriptor:descriptor error:&error];
            if (table == nil) {
                if (failure != nullptr) *failure = std::string("argument table: ") + (error.localizedDescription.UTF8String ?: "unknown");
                return nullptr;
            }
        }
        state->deferredSupport = deferredStoresAvailable();
        state->stats.deferredStoreSupport = state->deferredSupport;
        return std::unique_ptr<Recorder>(new Recorder(std::move(state)));
    }
}

void Recorder::Begin() {
    state->begin();
}

bool Recorder::BeginPass(const RenderPassDesc& pass) {
    auto& s = *state;
    s.begin();
    if (s.slot == nullptr) return false;
    PassTargets targets;
    std::array<AttachmentIntent, PassAttachmentCount> intents{};
    for (std::uint32_t index = 0; index < pass.colorCount && index < MaxColorAttachments; ++index) {
        const auto& color = pass.colors[index];
        targets.identities[index] = attachmentIdentity(color.texture, color.level, color.slice, color.depthPlane, 0);
        intents[index] = color.intent;
    }
    targets.identities[DepthAttachmentIndex] = attachmentIdentity(pass.depth.texture, pass.depth.level, pass.depth.slice, 0, 1);
    intents[DepthAttachmentIndex] = pass.depth.depthIntent;
    if (pass.depth.stencil) {
        targets.identities[StencilAttachmentIndex] = attachmentIdentity(pass.depth.texture, pass.depth.level, pass.depth.slice, 0, 2);
        intents[StencilAttachmentIndex] = pass.depth.stencilIntent;
    }
    targets.width = pass.width;
    targets.height = pass.height;
    targets.layers = pass.layers;
    targets.samples = pass.samples;
    if (s.encoder == nil) s.tracker.End();
    const auto decision = s.tracker.Next(targets, intents);
    if (decision.step == PassStep::Continue) return true;
    if (decision.step == PassStep::Restart) {
        s.discardMask |= decision.discardPrevious;
        s.endEncoder();
    }
    s.passIntents = intents;
    @autoreleasepool {
        if (s.openEncoder(pass)) return true;
    }
    s.tracker.End();
    return false;
}

void Recorder::EndPass() {
    state->endEncoder();
    state->tracker.End();
}

void Recorder::DiscardAttachment(std::uint32_t attachment) {
    if (state->encoder != nil && attachment < PassAttachmentCount) state->discardMask |= 1u << attachment;
}

void Recorder::SetPipeline(const RenderPipeline* pipeline) {
    state->pipeline = pipeline;
}

void Recorder::SetDepthStencil(const DepthStencilState& depthStencil) {
    auto& s = *state;
    const auto key = depthStencil.Pack();
    if (key == s.depthStencilKey) return;
    s.depthStencilKey = key;
    auto& entry = s.depthStencilCache[(key * 0x9e3779b97f4a7c15ull) >> 58u];
    if (entry.key != key || entry.native == nullptr) {
        entry.key = key;
        entry.native = s.pipelines.DepthStencil(depthStencil);
    }
    s.depthStencil = entry.native;
}

void Recorder::SetDynamic(const DynamicState& dynamic) {
    auto& s = *state;
    s.dynamic = dynamic;
    s.dynamicDirty = true;
}

void Recorder::SetSlot(std::uint8_t stages, std::uint32_t slot, std::uint64_t address) {
    auto& s = *state;
    for (std::uint32_t stage = 0; stage < 2; ++stage) {
        if ((stages & (1u << stage)) == 0) continue;
        if (!s.addresses[stage].Set(slot, address)) {
            ++s.stats.slotSetsSkipped;
            continue;
        }
        [s.tables[stage] setAddress:address atIndex:slot];
        ++s.stats.slotSets;
    }
}

UploadAllocation Recorder::Upload(std::uint32_t bytes) {
    auto& s = *state;
    s.begin();
    if (s.slot == nullptr) return {};
    s.stats.uploadBytes += bytes;
    return s.slot->ring.Allocate(bytes, s.options.uploadAlignment);
}

std::uint64_t Recorder::PushData(std::uint8_t stages, std::uint32_t slot, const void* data, std::uint32_t bytes) {
    const auto allocation = Upload(bytes);
    if (allocation.cpu == nullptr) return 0;
    std::memcpy(allocation.cpu, data, bytes);
    SetSlot(stages, slot, allocation.gpu);
    return allocation.gpu;
}

DrawOutcome Recorder::Draw(const DrawArgs& args) {
    auto& s = *state;
    const auto outcome = s.prepare();
    if (outcome != DrawOutcome::Drawn) {
        ++s.stats.skippedDraws;
        return outcome;
    }
    [s.encoder drawPrimitives:static_cast<MTLPrimitiveType>(args.primitive) vertexStart:args.vertexStart vertexCount:args.vertexCount instanceCount:args.instanceCount baseInstance:args.baseInstance];
    ++s.stats.draws;
    return outcome;
}

DrawOutcome Recorder::DrawIndexed(const IndexedDrawArgs& args) {
    auto& s = *state;
    const auto outcome = s.prepare();
    if (outcome != DrawOutcome::Drawn) {
        ++s.stats.skippedDraws;
        return outcome;
    }
    [s.encoder drawIndexedPrimitives:static_cast<MTLPrimitiveType>(args.primitive) indexCount:args.indexCount indexType:static_cast<MTLIndexType>(args.indexType) indexBuffer:args.indexAddress indexBufferLength:args.indexBytes instanceCount:args.instanceCount baseVertex:args.baseVertex baseInstance:args.baseInstance];
    ++s.stats.draws;
    return outcome;
}

DrawOutcome Recorder::DrawIndirect(const IndirectDrawArgs& args) {
    auto& s = *state;
    const auto outcome = s.prepare();
    if (outcome != DrawOutcome::Drawn) {
        ++s.stats.skippedDraws;
        return outcome;
    }
    if (args.indexed) [s.encoder drawIndexedPrimitives:static_cast<MTLPrimitiveType>(args.primitive) indexType:static_cast<MTLIndexType>(args.indexType) indexBuffer:args.indexAddress indexBufferLength:args.indexBytes indirectBuffer:args.argumentsAddress];
    else [s.encoder drawPrimitives:static_cast<MTLPrimitiveType>(args.primitive) indirectBuffer:args.argumentsAddress];
    ++s.stats.draws;
    return outcome;
}

void* Recorder::NativeCommandBuffer() {
    EndPass();
    state->begin();
    return state->slot != nullptr ? (__bridge void*)state->slot->commandBuffer : nullptr;
}

void* Recorder::NativeRenderEncoder() const {
    return (__bridge void*)state->encoder;
}

std::uint64_t Recorder::Submit() {
    auto& s = *state;
    if (s.slot == nullptr) return s.device.SubmittedSerial(s.options.queue);
    EndPass();
    [s.slot->commandBuffer endCommandBuffer];
    void* list[] = {(__bridge void*)s.slot->commandBuffer};
    const auto serial = s.device.Submit(s.options.queue, list, 1);
    s.slot->serial = serial;
    s.slot->used = true;
    s.slot = nullptr;
    ++s.stats.submits;
    return serial;
}

RecorderStats Recorder::Stats() const {
    auto stats = state->stats;
    stats.passes = state->tracker.Counters();
    return stats;
}

}
