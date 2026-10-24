#include "prx/libSceAgcDriver/Graphics/Metal/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Recorder.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using namespace AgcDriver::Graphics::Metal;

int failures = 0;

void Check(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

const char* const ProbeSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; };
struct Push { float4 color; float4 offset; };
vertex VOut aps5_probe_vs(uint vid [[vertex_id]], device const float4* positions [[buffer(0)]], constant Push& push [[buffer(1)]]) {
    VOut out;
    float4 p = positions[vid];
    p.xy += push.offset.xy;
    p.z = push.offset.z;
    p.y = -p.y;
    out.position = p;
    return out;
}
fragment float4 aps5_probe_fs(constant Push& push [[buffer(1)]]) {
    return push.color;
}
)MSL";

struct Push {
    float color[4];
    float offset[4];
};

constexpr std::uint32_t Size = 64;
constexpr std::uint32_t PositionsSlot = 0;
constexpr std::uint32_t PushSlot = 1;

std::uint32_t Pixel(id<MTLTexture> texture, std::uint32_t x, std::uint32_t y) {
    std::uint32_t value = 0;
    [texture getBytes:&value bytesPerRow:Size * 4u fromRegion:MTLRegionMake2D(x, y, 1, 1) mipmapLevel:0];
    return value;
}

constexpr std::uint32_t Rgba(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
    return static_cast<std::uint32_t>(r) | (static_cast<std::uint32_t>(g) << 8u) | (static_cast<std::uint32_t>(b) << 16u) | (static_cast<std::uint32_t>(a) << 24u);
}

constexpr std::uint32_t Black = Rgba(0, 0, 0, 255);
constexpr std::uint32_t Red = Rgba(255, 0, 0, 255);
constexpr std::uint32_t Green = Rgba(0, 255, 0, 255);
constexpr std::uint32_t Blue = Rgba(0, 0, 255, 255);

Push MakePush(float r, float g, float b, float dx, float dy, float z = 0.5f) {
    return Push{{r, g, b, 1.0f}, {dx, dy, z, 0.0f}};
}

RenderPassDesc ColorPass(id<MTLTexture> target, bool clear, id<MTLTexture> depth = nil) {
    RenderPassDesc pass;
    pass.colorCount = 1;
    pass.colors[0].texture = (__bridge void*)target;
    pass.colors[0].clearColor = {0.0, 0.0, 0.0, 1.0};
    pass.colors[0].intent.clear = clear;
    if (depth != nil) {
        pass.depth.texture = (__bridge void*)depth;
        pass.depth.clearDepth = 1.0;
        pass.depth.depthIntent.clear = true;
        pass.depth.depthIntent.transient = true;
    }
    pass.width = Size;
    pass.height = Size;
    return pass;
}

DynamicState FullViewport(CullMode cull = CullMode::None, Winding winding = Winding::Clockwise) {
    DynamicState dynamic;
    dynamic.viewport = {0.0f, 0.0f, static_cast<float>(Size), static_cast<float>(Size), 0.0f, 1.0f};
    dynamic.scissor = {0, 0, Size, Size};
    dynamic.cull = cull;
    dynamic.winding = winding;
    return dynamic;
}

double Microseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

}

int main() {
    @autoreleasepool {
        std::string failure;
        auto device = Device::Create(&failure);
        if (device == nullptr) {
            std::fprintf(stderr, "FAIL: %s\n", failure.c_str());
            return 1;
        }
        const auto cacheDirectory = std::filesystem::temp_directory_path() / ("aps5_draw_probe_" + std::to_string(::getpid()));
        CompilerConfig config;
        config.cacheRoot = cacheDirectory;
        auto compiler = MetalCompiler::Create(*device, config, &failure);
        if (compiler == nullptr) {
            std::fprintf(stderr, "FAIL: compiler: %s\n", failure.c_str());
            return 1;
        }
        auto pipelines = PipelineCache::Create(*device, *compiler, &failure);
        auto options = RecorderOptions::FromEnvironment();
        options.label = "agc.metal.draw-probe";
        auto recorder = pipelines != nullptr ? Recorder::Create(*device, *pipelines, options, &failure) : nullptr;
        if (recorder == nullptr) {
            std::fprintf(stderr, "FAIL: recorder: %s\n", failure.c_str());
            return 1;
        }
        id<MTLDevice> native = (__bridge id<MTLDevice>)device->NativeDevice();
        std::printf("device name=\"%s\" deferredStores=%d\n", device->Limits().name.c_str(), recorder->Stats().deferredStoreSupport ? 1 : 0);

        MTLTextureDescriptor* targetDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:Size height:Size mipmapped:NO];
        targetDescriptor.storageMode = MTLStorageModeShared;
        targetDescriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        id<MTLTexture> target = [native newTextureWithDescriptor:targetDescriptor];
        device->AddResidentAllocation((__bridge void*)target);
        MTLTextureDescriptor* depthDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:Size height:Size mipmapped:NO];
        depthDescriptor.storageMode = MTLStorageModeMemoryless;
        depthDescriptor.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> depth = [native newTextureWithDescriptor:depthDescriptor];
        Check(target != nil && depth != nil, "render target and memoryless depth are created");

        auto geometry = device->NewSharedBuffer(4096, "agc.metal.draw-probe.geometry");
        auto* positions = static_cast<float*>(geometry.contents);
        const float triangle[12] = {-1.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f};
        const float fullscreen[12] = {-1.0f, -1.0f, 0.0f, 1.0f, 3.0f, -1.0f, 0.0f, 1.0f, -1.0f, 3.0f, 0.0f, 1.0f};
        std::memcpy(positions, triangle, sizeof(triangle));
        std::memcpy(positions + 12, fullscreen, sizeof(fullscreen));
        const auto triangleAddress = geometry.gpuAddress;
        const auto fullscreenAddress = geometry.gpuAddress + sizeof(triangle);
        auto* indirect = reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(geometry.contents) + 1024);
        const std::uint32_t drawArgs[4] = {3, 1, 0, 0};
        const std::uint32_t indexedArgs[5] = {3, 1, 0, 0, 0};
        const std::uint16_t indices[4] = {0, 1, 2, 0};
        std::memcpy(indirect, drawArgs, sizeof(drawArgs));
        std::memcpy(indirect + 8, indexedArgs, sizeof(indexedArgs));
        std::memcpy(indirect + 16, indices, sizeof(indices));
        const auto drawArgsAddress = geometry.gpuAddress + 1024;
        const auto indexedArgsAddress = drawArgsAddress + 32;
        const auto indexAddress = drawArgsAddress + 64;

        auto owner = std::make_shared<std::string>(ProbeSource);
        LibrarySource library;
        library.key = CompileKeyBuilder(0x70726f6265ull).AddString(*owner).Finish();
        library.owner = owner;
        library.text = *owner;
        library.name = "aps5_draw_probe";
        RenderPipelineRequest request;
        request.vertexLibrary = library;
        request.vertexFunction = "aps5_probe_vs";
        request.fragmentLibrary = library;
        request.fragmentFunction = "aps5_probe_fs";
        request.state.colorCount = 1;
        request.state.colors[0].format = PixelFormat::RGBA8Unorm;
        request.label = "aps5.probe";
        const auto compileStart = std::chrono::steady_clock::now();
        auto pipeline = pipelines->Render(request);
        auto depthRequest = request;
        depthRequest.state.depthFormat = PixelFormat::Depth32Float;
        auto depthPipeline = pipelines->Render(depthRequest);
        auto blendRequest = request;
        blendRequest.state.colors[0].blend = true;
        blendRequest.state.colors[0].sourceRgb = BlendFactor::One;
        blendRequest.state.colors[0].destinationRgb = BlendFactor::One;
        auto blendPipeline = pipelines->Render(blendRequest);
        Check(pipelines->Render(request) == pipeline, "an equal request returns the same pipeline");
        Check(pipeline->Wait(60000) && depthPipeline->Wait(60000) && blendPipeline->Wait(60000), "render pipelines compile");
        if (pipeline->Failed()) std::fprintf(stderr, "pipeline error: %s\n", pipeline->Error().c_str());
        std::printf("pipelines: 3 compiled in %.1f ms\n", Microseconds(compileStart) / 1000.0);
        if (failures != 0) return 1;

        auto run = [&](auto&& body) {
            body();
            const auto serial = recorder->Submit();
            Check(device->WaitSerial(QueueKind::Graphics, serial, 10000), "submission completes");
        };
        auto drawTriangle = [&](const Push& push, std::uint64_t address = 0) {
            recorder->SetSlot(StageVertex, PositionsSlot, address != 0 ? address : triangleAddress);
            recorder->PushData(StageAll, PushSlot, &push, sizeof(push));
            DrawArgs args;
            args.vertexCount = 3;
            return recorder->Draw(args);
        };

        run([&] {
            Check(recorder->BeginPass(ColorPass(target, true)), "pass begins");
            recorder->SetPipeline(pipeline.get());
            recorder->SetDepthStencil(DepthStencilState{});
            recorder->SetDynamic(FullViewport());
            Check(drawTriangle(MakePush(1, 0, 0, 0, 0)) == DrawOutcome::Drawn, "triangle draws");
        });
        Check(Pixel(target, 4, 4) == Red, "triangle covers the top-left corner");
        Check(Pixel(target, 4, 60) == Black, "bottom-left stays clear (y orientation)");
        Check(Pixel(target, 60, 60) == Black, "bottom-right stays clear");

        run([&] {
            recorder->BeginPass(ColorPass(target, false));
            recorder->SetPipeline(pipeline.get());
            recorder->SetDynamic(FullViewport());
            recorder->SetSlot(StageVertex, PositionsSlot, triangleAddress);
            const auto green = MakePush(0, 1, 0, 1, 1);
            recorder->PushData(StageAll, PushSlot, &green, sizeof(green));
            IndirectDrawArgs args;
            args.argumentsAddress = drawArgsAddress;
            Check(recorder->DrawIndirect(args) == DrawOutcome::Drawn, "indirect draw");
            const auto blue = MakePush(0, 0, 1, 0, 1);
            recorder->PushData(StageAll, PushSlot, &blue, sizeof(blue));
            args.argumentsAddress = indexedArgsAddress;
            args.indexed = true;
            args.indexType = IndexType::UInt16;
            args.indexAddress = indexAddress;
            args.indexBytes = sizeof(indices);
            Check(recorder->DrawIndirect(args) == DrawOutcome::Drawn, "indexed indirect draw");
        });
        Check(Pixel(target, 4, 4) == Red, "load keeps the earlier contents");
        Check(Pixel(target, 36, 36) == Green, "indirect triangle lands bottom-right");
        Check(Pixel(target, 4, 36) == Blue, "indexed indirect triangle lands bottom-left");

        const auto cwVisible = [&](Winding winding) {
            run([&] {
                recorder->BeginPass(ColorPass(target, true));
                recorder->SetPipeline(pipeline.get());
                recorder->SetDynamic(FullViewport(CullMode::Back, winding));
                drawTriangle(MakePush(1, 0, 0, 0, 0));
            });
            return Pixel(target, 4, 4) == Red;
        };
        const bool clockwiseFront = cwVisible(Winding::Clockwise);
        const bool counterFront = cwVisible(Winding::CounterClockwise);
        std::printf("winding: vulkan-clockwise triangle visible with front=Clockwise %d, front=CounterClockwise %d\n", clockwiseFront ? 1 : 0, counterFront ? 1 : 0);
        Check(clockwiseFront && !counterFront, "vulkan front face maps straight to metal winding");

        run([&] {
            recorder->BeginPass(ColorPass(target, true, depth));
            recorder->SetPipeline(depthPipeline.get());
            DepthStencilState less;
            less.depthCompare = CompareFunction::Less;
            less.depthWrite = true;
            recorder->SetDepthStencil(less);
            recorder->SetDynamic(FullViewport());
            drawTriangle(MakePush(1, 0, 0, 0, 0, 0.3f), fullscreenAddress);
            drawTriangle(MakePush(0, 1, 0, 0, 0, 0.7f), fullscreenAddress);
            drawTriangle(MakePush(0, 0, 1, 0, 0, 0.1f), triangleAddress);
        });
        Check(Pixel(target, 60, 60) == Red, "memoryless depth rejects the farther draw");
        Check(Pixel(target, 4, 4) == Blue, "memoryless depth accepts the nearer draw");

        auto before = recorder->Stats();
        run([&] {
            recorder->BeginPass(ColorPass(target, true));
            recorder->SetPipeline(pipeline.get());
            recorder->SetDepthStencil(DepthStencilState{});
            recorder->SetDynamic(FullViewport());
            drawTriangle(MakePush(1, 0, 0, 0, 0));
            recorder->BeginPass(ColorPass(target, false));
            drawTriangle(MakePush(0, 1, 0, 1, 1));
            recorder->BeginPass(ColorPass(target, true));
            drawTriangle(MakePush(0, 0, 1, 0, 1));
        });
        auto after = recorder->Stats();
        Check(after.passes.continued == before.passes.continued + 1, "a pass on the same target continues the encoder");
        Check(after.passes.discarded == before.passes.discarded + 1, "a full clear of the open target discards its store");
        if (after.deferredStoreSupport) Check(after.storesDiscarded == before.storesDiscarded + 1, "the discarded store is applied");
        Check(Pixel(target, 4, 4) == Black && Pixel(target, 36, 36) == Black && Pixel(target, 4, 36) == Blue, "only the draw after the clear survives");

        constexpr int Draws = 20000;
        double steadyUs = 0;
        double churnUs = 0;
        double gpuUs = 0;
        for (int round = 0; round < 3; ++round) {
            const auto steadyStart = std::chrono::steady_clock::now();
            recorder->BeginPass(ColorPass(target, true));
            recorder->SetPipeline(pipeline.get());
            recorder->SetDepthStencil(DepthStencilState{});
            recorder->SetDynamic(FullViewport());
            recorder->SetSlot(StageVertex, PositionsSlot, triangleAddress);
            DrawArgs args;
            args.vertexCount = 3;
            for (int draw = 0; draw < Draws; ++draw) {
                const auto push = MakePush(static_cast<float>(draw & 255) / 255.0f, 0, 0, 0, 0);
                recorder->PushData(StageAll, PushSlot, &push, sizeof(push));
                recorder->Draw(args);
            }
            steadyUs = Microseconds(steadyStart);
            const auto gpuStart = std::chrono::steady_clock::now();
            const auto serial = recorder->Submit();
            device->WaitSerial(QueueKind::Graphics, serial, 10000);
            gpuUs = Microseconds(gpuStart);

            const auto churnStart = std::chrono::steady_clock::now();
            recorder->BeginPass(ColorPass(target, true));
            DynamicState a = FullViewport();
            DynamicState b = FullViewport(CullMode::Back, Winding::Clockwise);
            b.scissor = {0, 0, Size / 2, Size};
            DepthStencilState state;
            for (int draw = 0; draw < Draws; ++draw) {
                recorder->SetPipeline((draw & 1) != 0 ? blendPipeline.get() : pipeline.get());
                recorder->SetDynamic((draw & 1) != 0 ? b : a);
                state.depthCompare = (draw & 2) != 0 ? CompareFunction::Always : CompareFunction::Never;
                recorder->SetDepthStencil(state);
                recorder->SetSlot(StageVertex, PositionsSlot, (draw & 1) != 0 ? fullscreenAddress : triangleAddress);
                const auto push = MakePush(0, static_cast<float>(draw & 255) / 255.0f, 0, 0, 0);
                recorder->PushData(StageAll, PushSlot, &push, sizeof(push));
                recorder->Draw(args);
            }
            churnUs = Microseconds(churnStart);
            device->WaitSerial(QueueKind::Graphics, recorder->Submit(), 10000);
        }
        const auto stats = recorder->Stats();
        std::printf("cpu per draw: steady %.3f us, full state churn %.3f us (%d draws, last of 3 rounds); gpu drain %.1f us\n", steadyUs / Draws, churnUs / Draws, Draws, gpuUs);
        std::printf("stats: draws=%llu skipped=%llu pipelineBinds=%llu depthStencilBinds=%llu dynamicSets=%llu slotSets=%llu skippedSlotSets=%llu uploadKiB=%llu passes(begun=%llu continued=%llu restarted=%llu discarded=%llu) barriers=%llu slots=%llu slotWaits=%llu pipelineWaits=%llu (%.1f ms)\n",
            static_cast<unsigned long long>(stats.draws), static_cast<unsigned long long>(stats.skippedDraws), static_cast<unsigned long long>(stats.pipelineBinds), static_cast<unsigned long long>(stats.depthStencilBinds), static_cast<unsigned long long>(stats.dynamicSets), static_cast<unsigned long long>(stats.slotSets), static_cast<unsigned long long>(stats.slotSetsSkipped), static_cast<unsigned long long>(stats.uploadBytes >> 10u),
            static_cast<unsigned long long>(stats.passes.begun), static_cast<unsigned long long>(stats.passes.continued), static_cast<unsigned long long>(stats.passes.restarted), static_cast<unsigned long long>(stats.passes.discarded), static_cast<unsigned long long>(stats.barriers), static_cast<unsigned long long>(stats.slots), static_cast<unsigned long long>(stats.slotWaits), static_cast<unsigned long long>(stats.pipelineWaits), static_cast<double>(stats.pipelineWaitNanos) / 1.0e6);
        const auto cache = pipelines->Stats();
        std::printf("pipeline cache: requests=%llu hits=%llu pipelines=%llu depthStencil=%llu\n", static_cast<unsigned long long>(cache.renderRequests), static_cast<unsigned long long>(cache.renderHits), static_cast<unsigned long long>(cache.renderPipelines), static_cast<unsigned long long>(cache.depthStencilStates));
        Check(stats.skippedDraws == 0, "no draw was skipped");
        Check(steadyUs / Draws < 2.0, "steady draws stay under 2 us of CPU each");

        recorder.reset();
        device->ReleaseBuffer(geometry);
        compiler->WaitIdle(10000);
        pipelines.reset();
        compiler.reset();
        device->WaitIdle(10000);
        std::filesystem::remove_all(cacheDirectory);
        std::printf("%s\n", failures == 0 ? "agc_metal_draw_probe: ok" : "agc_metal_draw_probe: FAILED");
        return failures == 0 ? 0 : 1;
    }
}
