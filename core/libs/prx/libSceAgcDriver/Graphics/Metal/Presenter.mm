#include "prx/libSceAgcDriver/Graphics/Metal/Presenter.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/PowerMonitor.hpp"
#include "prx/libSceAgcDriver/Execution/include/AspectFit.hpp"
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <thread>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr std::size_t Slots = 3;
constexpr std::size_t ConstantsStride = 256;
constexpr MTLPixelFormat TargetFormat = MTLPixelFormatBGRA8Unorm;
constexpr double MinUpscaleRatio = 1.1;

const char* const ShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint mode;
    float whiteNits;
    float peakNits;
    uint reserved;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut present_vertex(uint vertex_id [[vertex_id]]) {
    float2 p = float2(float((vertex_id << 1) & 2u), float(vertex_id & 2u));
    VertexOut out;
    out.position = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    out.uv = p;
    return out;
}

float3 srgb_encode(float3 x) {
    x = clamp(x, 0.0, 1.0);
    return select(1.055 * pow(x, float3(1.0 / 2.4)) - 0.055, x * 12.92, x <= 0.0031308);
}

float3 pq_to_nits(float3 e) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    float3 p = pow(clamp(e, 0.0, 1.0), float3(1.0 / m2));
    return 10000.0 * pow(max(p - c1, 0.0) / (c2 - c3 * p), float3(1.0 / m1));
}

fragment float4 present_fragment(VertexOut in [[stage_in]], texture2d<float> source [[texture(0)]], constant Params& params [[buffer(0)]]) {
    constexpr sampler linear_clamp(coord::normalized, filter::linear, address::clamp_to_edge);
    float4 color = source.sample(linear_clamp, in.uv);
    if (params.mode == 1u) {
        color.rgb = srgb_encode(color.rgb);
    } else if (params.mode == 2u) {
        float3 linear2020 = pq_to_nits(color.rgb) / params.whiteNits;
        float3 linear709 = float3(
            dot(linear2020, float3(1.6605, -0.5876, -0.0728)),
            dot(linear2020, float3(-0.1246, 1.1329, -0.0083)),
            dot(linear2020, float3(-0.0182, -0.1006, 1.1187)));
        linear709 = max(linear709, 0.0);
        float peak = max(params.peakNits / params.whiteNits, 1.0);
        float3 mapped = linear709 * (1.0 + linear709 / (peak * peak)) / (1.0 + linear709);
        color.rgb = srgb_encode(mapped);
    }
    return float4(color.rgb, 1.0);
}
)";

struct Params {
    std::uint32_t mode;
    float whiteNits;
    float peakNits;
    std::uint32_t reserved;
};

std::int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double millis(std::int64_t ns) {
    return static_cast<double>(ns) / 1.0e6;
}

std::uint32_t modeFor(SourceEncoding encoding) {
    switch (encoding) {
    case SourceEncoding::Unorm: return 0;
    case SourceEncoding::LinearHalf: return 1;
    default: return 2;
    }
}

double displayRefreshHz() {
    CGDisplayModeRef mode = CGDisplayCopyDisplayMode(CGMainDisplayID());
    if (mode == nullptr) return 60.0;
    const double hz = CGDisplayModeGetRefreshRate(mode);
    CGDisplayModeRelease(mode);
    return hz >= 24.0 ? hz : 60.0;
}

bool equalsIgnoreCase(const char* value, std::string_view expected) {
    if (value == nullptr || std::strlen(value) != expected.size()) return false;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (std::tolower(static_cast<unsigned char>(value[index])) != expected[index]) return false;
    }
    return true;
}

struct ScalerKey {
    std::uint32_t sourceWidth = 0;
    std::uint32_t sourceHeight = 0;
    std::uint32_t outputWidth = 0;
    std::uint32_t outputHeight = 0;
    MTLPixelFormat format = MTLPixelFormatInvalid;

    bool operator==(const ScalerKey& other) const {
        return sourceWidth == other.sourceWidth && sourceHeight == other.sourceHeight && outputWidth == other.outputWidth && outputHeight == other.outputHeight && format == other.format;
    }
};

}

PresenterOptions PresenterOptions::FromEnvironment() {
    PresenterOptions options;
    if (const char* fps = std::getenv("APS5_METAL_FPS"); fps != nullptr && *fps != 0 && !equalsIgnoreCase(fps, "auto")) {
        char* end = nullptr;
        const double parsed = std::strtod(fps, &end);
        if (end != fps && parsed >= 1.0) options.targetFps = parsed;
    }
    if (const char* upscale = std::getenv("APS5_METAL_UPSCALE")) {
        if (equalsIgnoreCase(upscale, "off")) options.upscale = UpscaleMode::Off;
        else if (equalsIgnoreCase(upscale, "on")) options.upscale = UpscaleMode::On;
    }
    if (const char* power = std::getenv("APS5_METAL_POWER")) {
        if (equalsIgnoreCase(power, "saver")) options.profile = PowerProfile::Saver;
        else if (equalsIgnoreCase(power, "performance")) options.profile = PowerProfile::Performance;
    }
    if (const char* drawables = std::getenv("APS5_METAL_DRAWABLES"); drawables != nullptr && (drawables[0] == '2' || drawables[0] == '3')) options.drawables = static_cast<std::uint32_t>(drawables[0] - '0');
    return options;
}

struct Presenter::State {
    Device& device;
    PresenterOptions options;
    CAMetalLayer* layer = nil;
    id<MTLDevice> mtl = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTLRenderPipelineState> pipeline = nil;
    id<MTL4ArgumentTable> table = nil;
    id<MTLResidencySet> ownResidency = nil;
    std::array<id<MTL4CommandAllocator>, Slots> allocators{};
    std::array<id<MTL4CommandBuffer>, Slots> commandBuffers{};
    std::array<std::uint64_t, Slots> slotSerials{};
    std::size_t nextSlot = 0;
    NativeBuffer constants;
    id<MTL4FXSpatialScaler> scaler = nil;
    id<MTLTexture> scaled = nil;
    ScalerKey scalerKey;
    bool scalerUnavailable = false;
    bool scalerSupported = false;
    FramePacer pacer;
    PowerPolicy policy;
    PowerMonitor monitor;
    std::shared_ptr<std::atomic<std::int64_t>> displayed = std::make_shared<std::atomic<std::int64_t>>(0);
    std::atomic<bool> occluded{false};
    std::atomic<double> refreshHz{60.0};
    std::atomic<double> userFps{0.0};
    double configuredRefresh = 0.0;
    double configuredTarget = -1.0;
    PresenterStats stats;
    std::uint64_t lastSerial = 0;

    State(Device& device, const PresenterOptions& options) : device(device), options(options), policy(options.profile) {}

    double targetFps(const PowerAdvice& advice) const {
        const double user = userFps.load(std::memory_order_relaxed);
        if (user > 0.0 && advice.maxFps > 0.0) return std::min(user, advice.maxFps);
        return user > 0.0 ? user : advice.maxFps;
    }

    bool buildPipeline(std::string* failure) {
        NSError* error = nil;
        MTL4CompilerDescriptor* compilerDescriptor = [MTL4CompilerDescriptor new];
        compilerDescriptor.label = @"AnyPS5 present compiler";
        compiler = [mtl newCompilerWithDescriptor:compilerDescriptor error:&error];
        if (compiler == nil) {
            if (failure != nullptr) *failure = "newCompilerWithDescriptor";
            return false;
        }
        id<MTLLibrary> library = [mtl newLibraryWithSource:[NSString stringWithUTF8String:ShaderSource] options:nil error:&error];
        if (library == nil) {
            if (failure != nullptr) *failure = std::string("present shader: ") + (error.localizedDescription.UTF8String ?: "");
            return false;
        }
        MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
        vertex.library = library;
        vertex.name = @"present_vertex";
        MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new];
        fragment.library = library;
        fragment.name = @"present_fragment";
        MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.label = @"AnyPS5 present";
        descriptor.vertexFunctionDescriptor = vertex;
        descriptor.fragmentFunctionDescriptor = fragment;
        descriptor.colorAttachments[0].pixelFormat = TargetFormat;
        pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (pipeline == nil) {
            if (failure != nullptr) *failure = std::string("present pipeline: ") + (error.localizedDescription.UTF8String ?: "");
            return false;
        }
        MTL4ArgumentTableDescriptor* tableDescriptor = [MTL4ArgumentTableDescriptor new];
        tableDescriptor.maxBufferBindCount = 1;
        tableDescriptor.maxTextureBindCount = 1;
        tableDescriptor.label = @"AnyPS5 present arguments";
        table = [mtl newArgumentTableWithDescriptor:tableDescriptor error:&error];
        if (table == nil) {
            if (failure != nullptr) *failure = "newArgumentTableWithDescriptor";
            return false;
        }
        return true;
    }

    bool ensureScaler(id<MTLTexture> source, const AspectFitRect& rect) {
        const ScalerKey key{static_cast<std::uint32_t>(source.width), static_cast<std::uint32_t>(source.height), rect.width, rect.height, source.pixelFormat};
        if (scaler != nil && key == scalerKey) return true;
        if (scalerUnavailable && key == scalerKey) return false;
        if (scaled != nil || scaler != nil) {
            device.WaitSerial(QueueKind::Graphics, lastSerial, 1000);
            [ownResidency removeAllocation:scaled];
            scaled = nil;
            scaler = nil;
        }
        scalerKey = key;
        scalerUnavailable = true;
        MTLFXSpatialScalerDescriptor* descriptor = [MTLFXSpatialScalerDescriptor new];
        descriptor.inputWidth = key.sourceWidth;
        descriptor.inputHeight = key.sourceHeight;
        descriptor.outputWidth = key.outputWidth;
        descriptor.outputHeight = key.outputHeight;
        descriptor.colorTextureFormat = key.format;
        descriptor.outputTextureFormat = TargetFormat;
        descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
        id<MTL4FXSpatialScaler> candidate = [descriptor newSpatialScalerWithDevice:mtl compiler:compiler];
        if (candidate == nil) return false;
        if ((candidate.colorTextureUsage & ~source.usage) != 0) return false;
        MTLTextureDescriptor* textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:TargetFormat width:key.outputWidth height:key.outputHeight mipmapped:NO];
        textureDescriptor.storageMode = MTLStorageModePrivate;
        textureDescriptor.usage = candidate.outputTextureUsage | MTLTextureUsageShaderRead;
        textureDescriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        id<MTLTexture> output = [mtl newTextureWithDescriptor:textureDescriptor];
        if (output == nil) return false;
        output.label = @"AnyPS5 MetalFX output";
        [ownResidency addAllocation:output];
        [ownResidency commit];
        [ownResidency requestResidency];
        scaler = candidate;
        scaled = output;
        scalerUnavailable = false;
        ++stats.scalerRebuilds;
        return true;
    }

    bool wantsUpscale(const PresentSource& source, const AspectFitRect& rect, const PowerAdvice& advice, bool allow) const {
        if (!allow || !scalerSupported || options.upscale == UpscaleMode::Off) return false;
        if (source.encoding != SourceEncoding::Unorm) return false;
        if (options.upscale == UpscaleMode::Auto && !advice.allowUpscaler) return false;
        return static_cast<double>(rect.width) >= static_cast<double>(source.width) * MinUpscaleRatio && static_cast<double>(rect.height) >= static_cast<double>(source.height) * MinUpscaleRatio;
    }

    std::size_t acquireSlot() {
        const auto slot = nextSlot;
        nextSlot = (nextSlot + 1) % Slots;
        device.WaitSerial(QueueKind::Graphics, slotSerials[slot], 2000);
        return slot;
    }

    bool record(std::size_t slot, const PresentSource& source, id<MTLTexture> target, const PowerAdvice& advice, bool allowUpscale, bool& upscaled) {
        id<MTLTexture> input = (__bridge id<MTLTexture>)source.texture;
        const AspectFitRect rect = ComputeContainRect_nid_postfix(source.width, source.height, static_cast<std::uint32_t>(target.width), static_cast<std::uint32_t>(target.height));
        if (rect.width == 0 || rect.height == 0) return false;
        id<MTL4CommandAllocator> allocator = allocators[slot];
        id<MTL4CommandBuffer> commands = commandBuffers[slot];
        [allocator reset];
        [commands beginCommandBufferWithAllocator:allocator];
        id<MTLTexture> sampled = input;
        std::uint32_t mode = modeFor(source.encoding);
        upscaled = false;
        if (wantsUpscale(source, rect, advice, allowUpscale) && ensureScaler(input, rect)) {
            id<MTL4ComputeCommandEncoder> barrier = [commands computeCommandEncoder];
            [barrier barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
            [barrier endEncoding];
            scaler.colorTexture = input;
            scaler.outputTexture = scaled;
            scaler.inputContentWidth = source.width;
            scaler.inputContentHeight = source.height;
            [scaler encodeToCommandBuffer:commands];
            sampled = scaled;
            mode = 0;
            upscaled = true;
        }
        Params params{mode, std::max(source.referenceWhiteNits, 1.0f), std::max(source.peakNits, 1.0f), 0};
        auto* block = static_cast<std::uint8_t*>(constants.contents) + slot * ConstantsStride;
        std::memcpy(block, &params, sizeof(params));
        [table setAddress:constants.gpuAddress + slot * ConstantsStride atIndex:0];
        [table setTexture:sampled.gpuResourceID atIndex:0];
        const bool full = rect.width == target.width && rect.height == target.height;
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = full ? MTLLoadActionDontCare : MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        id<MTL4RenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
        [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageFragment visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder setRenderPipelineState:pipeline];
        [encoder setArgumentTable:table atStages:MTLRenderStageFragment];
        MTLViewport viewport{static_cast<double>(rect.x), static_cast<double>(rect.y), static_cast<double>(rect.width), static_cast<double>(rect.height), 0.0, 1.0};
        [encoder setViewport:viewport];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
        [commands endCommandBuffer];
        return true;
    }
};

std::unique_ptr<Presenter> Presenter::Create(Device& device, void* metalLayer, const PresenterOptions& options, std::string* failure) {
    @autoreleasepool {
        const auto fail = [&](const char* what) -> std::unique_ptr<Presenter> {
            if (failure != nullptr) *failure = what;
            return nullptr;
        };
        auto state = std::make_unique<State>(device, options);
        state->mtl = (__bridge id<MTLDevice>)device.NativeDevice();
        if (!state->buildPipeline(failure)) return nullptr;
        NSError* error = nil;
        MTLResidencySetDescriptor* residency = [MTLResidencySetDescriptor new];
        residency.label = @"AnyPS5 present resources";
        state->ownResidency = [state->mtl newResidencySetWithDescriptor:residency error:&error];
        if (state->ownResidency == nil) return fail("newResidencySetWithDescriptor");
        id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)device.NativeQueue(QueueKind::Graphics);
        [queue addResidencySet:state->ownResidency];
        for (std::size_t slot = 0; slot < Slots; ++slot) {
            state->allocators[slot] = [state->mtl newCommandAllocator];
            state->commandBuffers[slot] = [state->mtl newCommandBuffer];
            if (state->allocators[slot] == nil || state->commandBuffers[slot] == nil) return fail("command allocator");
        }
        state->constants = device.NewSharedBuffer(Slots * ConstantsStride, "present constants");
        if (!state->constants) return fail("constants buffer");
        if (metalLayer != nullptr) {
            CAMetalLayer* layer = (__bridge CAMetalLayer*)metalLayer;
            layer.device = state->mtl;
            layer.pixelFormat = TargetFormat;
            layer.framebufferOnly = YES;
            layer.displaySyncEnabled = YES;
            layer.allowsNextDrawableTimeout = YES;
            layer.presentsWithTransaction = NO;
            layer.maximumDrawableCount = options.drawables;
            CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            layer.colorspace = space;
            CGColorSpaceRelease(space);
            if (layer.residencySet != nil) [queue addResidencySet:layer.residencySet];
            state->layer = layer;
        }
        state->scalerSupported = [MTLFXSpatialScalerDescriptor supportsMetal4FX:state->mtl];
        const double refresh = displayRefreshHz();
        state->refreshHz.store(refresh);
        state->userFps.store(options.targetFps);
        return std::unique_ptr<Presenter>(new Presenter(std::move(state)));
    }
}

Presenter::Presenter(std::unique_ptr<State> state) : state(std::move(state)) {}

Presenter::~Presenter() {
    if (state == nullptr) return;
    state->device.WaitIdle(5000);
    state->device.ReleaseBuffer(state->constants);
}

void Presenter::SetDrawableSize(std::uint32_t width, std::uint32_t height) {
    if (state->layer == nil || width == 0 || height == 0) return;
    const CGSize size = state->layer.drawableSize;
    if (static_cast<std::uint32_t>(size.width) == width && static_cast<std::uint32_t>(size.height) == height) return;
    state->layer.drawableSize = CGSizeMake(width, height);
}

void Presenter::SetOccluded(bool occluded) {
    state->occluded.store(occluded, std::memory_order_relaxed);
}

void Presenter::SetRefreshRate(double hz) {
    if (hz >= 24.0) state->refreshHz.store(hz, std::memory_order_relaxed);
}

void Presenter::SetTargetFps(double fps) {
    state->userFps.store(fps > 0.0 ? fps : 0.0, std::memory_order_relaxed);
}

PresenterStats Presenter::Stats() const {
    PresenterStats stats = state->stats;
    stats.refreshHz = state->refreshHz.load(std::memory_order_relaxed);
    stats.effectiveHz = state->pacer.EffectiveHz();
    stats.advice = state->policy.Current();
    stats.pacer = state->pacer.Stats();
    return stats;
}

PresentResult Presenter::Present(const PresentSource& source) {
    PresentResult result;
    auto& s = *state;
    if (source.texture == nullptr || source.width == 0 || source.height == 0 || s.layer == nil) return result;
    @autoreleasepool {
        const auto begin = nowNs();
        PowerInputs inputs = s.monitor.Sample(begin);
        inputs.occluded = s.occluded.load(std::memory_order_relaxed);
        const PowerAdvice advice = s.policy.Update(inputs, begin);
        if (advice.suspend) {
            ++s.stats.suspended;
            result.outcome = PresentOutcome::Suspended;
            return result;
        }
        const double refresh = s.refreshHz.load(std::memory_order_relaxed);
        const double target = s.targetFps(advice);
        if (refresh != s.configuredRefresh || target != s.configuredTarget) {
            s.pacer.Configure(refresh, target);
            s.configuredRefresh = refresh;
            s.configuredTarget = target;
        }
        const auto displayed = s.displayed->load(std::memory_order_relaxed);
        if (displayed > 0) s.pacer.NoteDisplayed(displayed);
        const auto slotNs = s.pacer.Schedule(begin);
        if (slotNs > begin) std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::nanoseconds(slotNs)));
        const auto paced = nowNs();
        id<CAMetalDrawable> drawable = [s.layer nextDrawable];
        if (drawable == nil) {
            ++s.stats.noDrawable;
            result.outcome = PresentOutcome::NoDrawable;
            return result;
        }
        const auto slot = s.acquireSlot();
        const auto acquired = nowNs();
        bool upscaled = false;
        if (!s.record(slot, source, drawable.texture, advice, true, upscaled)) return result;
        auto displayedSlot = s.displayed;
        [drawable addPresentedHandler:^(id<MTLDrawable> presented) {
            const double time = presented.presentedTime;
            if (time > 0.0) displayedSlot->store(static_cast<std::int64_t>(time * 1.0e9), std::memory_order_relaxed);
        }];
        void* list[1] = {(__bridge void*)s.commandBuffers[slot]};
        const auto serial = s.device.SubmitPresent(QueueKind::Graphics, (__bridge void*)drawable, list, 1);
        s.slotSerials[slot] = serial;
        s.lastSerial = serial;
        const auto end = nowNs();
        result.outcome = PresentOutcome::Presented;
        result.upscaled = upscaled;
        result.serial = serial;
        result.pacedMs = millis(paced - begin);
        result.waitMs = millis(acquired - paced);
        result.encodeMs = millis(end - acquired);
        ++s.stats.presented;
        if (upscaled) ++s.stats.upscaled;
        s.stats.pacedMs += result.pacedMs;
        s.stats.waitMs += result.waitMs;
        s.stats.encodeMs += result.encodeMs;
    }
    return result;
}

bool Presenter::RenderTo(const PresentSource& source, void* bgra8Target, bool allowUpscale) {
    auto& s = *state;
    if (source.texture == nullptr || bgra8Target == nullptr || source.width == 0 || source.height == 0) return false;
    @autoreleasepool {
        id<MTLTexture> target = (__bridge id<MTLTexture>)bgra8Target;
        if (target.pixelFormat != TargetFormat) return false;
        const auto slot = s.acquireSlot();
        bool upscaled = false;
        if (!s.record(slot, source, target, s.policy.Current(), allowUpscale, upscaled)) return false;
        void* list[1] = {(__bridge void*)s.commandBuffers[slot]};
        const auto serial = s.device.Submit(QueueKind::Graphics, list, 1);
        s.slotSerials[slot] = serial;
        s.lastSerial = serial;
        if (upscaled) ++s.stats.upscaled;
        return s.device.WaitSerial(QueueKind::Graphics, serial, 5000);
    }
}

}
