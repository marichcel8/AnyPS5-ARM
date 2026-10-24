#include "prx/libSceAgcDriver/Graphics/Metal/RenderPassPlan.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/RenderState.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/UploadRing.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

namespace M = AgcDriver::Graphics::Metal;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

class HeapSource final : public M::UploadSource {
public:
    M::UploadChunk Acquire(std::uint64_t bytes) override {
        auto* memory = static_cast<std::byte*>(std::malloc(bytes));
        ++acquired;
        return {memory, memory, next += 0x100000ull, bytes};
    }

    void Release(const M::UploadChunk& chunk) override {
        std::free(chunk.handle);
        ++released;
    }

    std::uint64_t next = 0x1000000ull;
    int acquired = 0;
    int released = 0;
};

void TestActions() {
    M::AttachmentIntent intent;
    auto actions = M::ChooseActions(intent, false);
    Require(actions.load == M::LoadAction::Load && actions.store == M::StoreAction::Store && !actions.deferred, "defined observed target loads and stores");
    intent.clear = true;
    actions = M::ChooseActions(intent, true);
    Require(actions.load == M::LoadAction::Clear && actions.store == M::StoreAction::Unknown && actions.deferred, "clear with deferred store");
    intent = {};
    intent.contentsDefined = false;
    Require(M::ChooseActions(intent, false).load == M::LoadAction::DontCare, "undefined contents are not loaded");
    intent = {};
    intent.transient = true;
    actions = M::ChooseActions(intent, true);
    Require(actions.load == M::LoadAction::DontCare && actions.store == M::StoreAction::DontCare && !actions.deferred, "transient target never touches memory");
    intent.resolve = true;
    Require(M::ChooseActions(intent, true).store == M::StoreAction::MultisampleResolve, "transient msaa resolves only");
    intent = {};
    intent.resolve = true;
    Require(M::FinalStore(intent, false) == M::StoreAction::StoreAndMultisampleResolve, "observed msaa stores and resolves");
    Require(M::FinalStore(intent, true) == M::StoreAction::MultisampleResolve, "discarded msaa resolves only");
    intent = {};
    intent.observed = false;
    Require(M::ChooseActions(intent, true).store == M::StoreAction::DontCare, "unobserved target is not stored");
}

M::PassTargets Targets(std::uint64_t color, std::uint64_t depth, std::uint32_t width = 64, std::uint32_t height = 64) {
    M::PassTargets targets;
    targets.identities[0] = color;
    targets.identities[M::DepthAttachmentIndex] = depth;
    targets.width = width;
    targets.height = height;
    return targets;
}

void TestTracker() {
    M::PassTracker tracker;
    std::array<M::AttachmentIntent, M::PassAttachmentCount> intents{};
    auto decision = tracker.Next(Targets(1, 2), intents);
    Require(decision.step == M::PassStep::Begin, "first pass begins");
    decision = tracker.Next(Targets(1, 2), intents);
    Require(decision.step == M::PassStep::Continue, "same targets continue the pass");
    decision = tracker.Next(Targets(3, 2), intents);
    Require(decision.step == M::PassStep::Restart && decision.discardPrevious == 0, "new target restarts without discards");
    intents[M::DepthAttachmentIndex].clear = true;
    decision = tracker.Next(Targets(3, 2), intents);
    Require(decision.step == M::PassStep::Restart, "a clear restarts even on the same targets");
    Require(decision.discardPrevious == (1u << M::DepthAttachmentIndex), "fully cleared depth discards the previous store");
    decision = tracker.Next(Targets(3, 2, 32, 32), intents);
    Require(decision.discardPrevious == 0, "a smaller clear keeps the previous store");
    intents = {};
    tracker.End();
    decision = tracker.Next(Targets(3, 2), intents);
    Require(decision.step == M::PassStep::Begin, "ended pass begins again");
    const auto& counters = tracker.Counters();
    Require(counters.begun == 2 && counters.continued == 1 && counters.restarted == 3 && counters.discarded == 1, "tracker counters");
}

void TestKeys() {
    M::RenderPipelineState state;
    state.colorCount = 1;
    state.colors[0].format = M::PixelFormat::RGBA8Unorm;
    std::set<std::pair<std::uint64_t, std::uint64_t>> keys;
    auto key = [](const M::RenderPipelineState& value) {
        M::CompileKeyBuilder builder(1);
        value.AddTo(builder);
        const auto finished = builder.Finish();
        return std::make_pair(finished.low, finished.high);
    };
    keys.insert(key(state));
    auto variant = state;
    variant.colors[0].writeMask = M::ColorWriteRed;
    keys.insert(key(variant));
    variant = state;
    variant.colors[0].blend = true;
    keys.insert(key(variant));
    variant.colors[0].destinationRgb = M::BlendFactor::OneMinusSourceAlpha;
    keys.insert(key(variant));
    variant = state;
    variant.depthFormat = M::PixelFormat::Depth32Float;
    keys.insert(key(variant));
    variant = state;
    variant.sampleCount = 4;
    keys.insert(key(variant));
    variant = state;
    variant.topology = M::TopologyClass::Line;
    keys.insert(key(variant));
    variant = state;
    variant.alphaToCoverage = true;
    keys.insert(key(variant));
    Require(keys.size() == 8, "every pipeline state field changes the key");
    variant = state;
    variant.colors[1].format = M::PixelFormat::R32Float;
    Require(key(variant) == key(state), "attachments past the count do not change the key");
    auto blendOff = state;
    blendOff.colors[0].sourceRgb = M::BlendFactor::SourceAlpha;
    Require(blendOff.colors[0].Pack() == state.colors[0].Pack(), "blend factors are ignored while blending is off");

    M::DepthStencilState depth;
    std::set<std::uint64_t> packs{depth.Pack()};
    depth.depthCompare = M::CompareFunction::Less;
    packs.insert(depth.Pack());
    depth.depthWrite = true;
    packs.insert(depth.Pack());
    depth.stencil = true;
    packs.insert(depth.Pack());
    depth.back.pass = M::StencilOperation::Replace;
    packs.insert(depth.Pack());
    depth.front.writeMask = 0x0f;
    packs.insert(depth.Pack());
    Require(packs.size() == 6, "depth-stencil packs are distinct");
    Require(packs.count(0) == 0, "a depth-stencil pack is never zero");
}

void TestDynamic() {
    M::DynamicState a;
    M::DynamicState b = a;
    Require(M::DiffDynamic(a, b) == 0, "equal dynamic state");
    b.viewport.width = 10;
    b.cull = M::CullMode::Back;
    b.stencilBackReference = 3;
    Require(M::DiffDynamic(a, b) == (M::DynamicViewport | M::DynamicCull | M::DynamicStencilReference), "dynamic diff bits");
}

void TestUploadRing() {
    HeapSource source;
    {
        M::UploadRing ring(source, 4096);
        auto first = ring.Allocate(100, 16);
        auto second = ring.Allocate(100, 16);
        Require(first.cpu != nullptr && second.gpu == first.gpu + 112, "bump allocation is aligned");
        Require((second.gpu & 15u) == 0, "allocation alignment");
        ring.Allocate(4000, 16);
        Require(source.acquired == 2 && ring.Chunks() == 2, "overflow takes a new chunk");
        auto big = ring.Allocate(10000, 16);
        Require(big.cpu != nullptr && source.acquired == 3, "oversized allocation takes its own chunk");
        ring.Reset();
        auto again = ring.Allocate(64, 16);
        Require(again.gpu == first.gpu && source.acquired == 3, "reset reuses chunks without allocating");
        ring.Allocate(4064, 16);
        ring.Allocate(64, 16);
        Require(source.acquired == 3, "reset walks the existing chunks");
        M::UploadRing moved(std::move(ring));
        Require(moved.Chunks() == 3, "move keeps chunks");
    }
    Require(source.released == 3, "chunks are released once");

    M::SlotAddresses<14> addresses;
    Require(addresses.Set(3, 0x1000), "first set changes");
    Require(!addresses.Set(3, 0x1000), "same address is skipped");
    Require(addresses.Set(3, 0x2000) && addresses.Get(3) == 0x2000, "new address changes");
}

void BenchmarkDrawBookkeeping() {
    HeapSource source;
    M::UploadRing ring(source, 256u << 10u);
    M::SlotAddresses<14> addresses;
    M::DynamicState bound;
    M::DynamicState next;
    constexpr int Draws = 200000;
    std::uint32_t push[16]{};
    std::uint64_t sink = 0;
    const auto started = std::chrono::steady_clock::now();
    for (int draw = 0; draw < Draws; ++draw) {
        if ((draw & 4095) == 0) ring.Reset();
        push[0] = static_cast<std::uint32_t>(draw);
        auto allocation = ring.Allocate(sizeof(push), 16);
        std::memcpy(allocation.cpu, push, sizeof(push));
        sink += addresses.Set(1, allocation.gpu);
        next.scissor.width = static_cast<std::uint32_t>(draw & 7);
        sink += M::DiffDynamic(bound, next);
        bound = next;
    }
    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
    std::printf("draw bookkeeping: %.1f ns/draw (push 64 B, slot diff, dynamic diff), sink %llu\n", static_cast<double>(nanos) / Draws, static_cast<unsigned long long>(sink));
}

}

int main() {
    TestActions();
    TestTracker();
    TestKeys();
    TestDynamic();
    TestUploadRing();
    BenchmarkDrawBookkeeping();
    std::printf("agc_metal_draw_path: ok\n");
    return 0;
}
