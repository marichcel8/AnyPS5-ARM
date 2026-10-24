#include "prx/libSceAgcDriver/Graphics/Metal/PipelineCache.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <mutex>
#include <unordered_map>

namespace AgcDriver::Graphics::Metal {

static_assert(static_cast<int>(BlendFactor::OneMinusSource1Alpha) == MTLBlendFactorOneMinusSource1Alpha);
static_assert(static_cast<int>(BlendFactor::SourceAlphaSaturated) == MTLBlendFactorSourceAlphaSaturated);
static_assert(static_cast<int>(BlendFactor::BlendAlpha) == MTLBlendFactorBlendAlpha);
static_assert(static_cast<int>(BlendOperation::Max) == MTLBlendOperationMax);
static_assert(static_cast<int>(CompareFunction::GreaterEqual) == MTLCompareFunctionGreaterEqual);
static_assert(static_cast<int>(CompareFunction::Always) == MTLCompareFunctionAlways);
static_assert(static_cast<int>(StencilOperation::IncrementWrap) == MTLStencilOperationIncrementWrap);
static_assert(static_cast<int>(StencilOperation::DecrementWrap) == MTLStencilOperationDecrementWrap);
static_assert(static_cast<int>(TopologyClass::Triangle) == MTLPrimitiveTopologyClassTriangle);
static_assert(ColorWriteRed == MTLColorWriteMaskRed && ColorWriteAlpha == MTLColorWriteMaskAlpha);
static_assert(static_cast<int>(PixelFormat::BGRA8Unorm) == MTLPixelFormatBGRA8Unorm);
static_assert(static_cast<int>(PixelFormat::Depth16Unorm) == MTLPixelFormatDepth16Unorm);
static_assert(static_cast<int>(PixelFormat::Depth32Float) == MTLPixelFormatDepth32Float);
static_assert(static_cast<int>(PixelFormat::Stencil8) == MTLPixelFormatStencil8);
static_assert(static_cast<int>(PixelFormat::Depth32FloatStencil8) == MTLPixelFormatDepth32Float_Stencil8);

namespace {

constexpr std::uint64_t RenderDomain = 0x4d52504cull;

NSString* nsString(const std::string& text) {
    return [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
}

MTL4FunctionDescriptor* functionDescriptor(id<MTLLibrary> library, const std::string& name, const std::vector<FunctionConstant>& constants) {
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = nsString(name);
    if (constants.empty()) return function;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    for (const auto& constant : constants) {
        const std::uint32_t scalar = constant.value;
        const bool flag = constant.value != 0;
        if (constant.boolean) [values setConstantValue:&flag type:MTLDataTypeBool atIndex:constant.index];
        else [values setConstantValue:&scalar type:MTLDataTypeUInt atIndex:constant.index];
    }
    MTL4SpecializedFunctionDescriptor* specialized = [MTL4SpecializedFunctionDescriptor new];
    specialized.functionDescriptor = function;
    specialized.constantValues = values;
    return specialized;
}

void addConstants(CompileKeyBuilder& builder, const std::vector<FunctionConstant>& constants) {
    builder.Add(constants.size());
    for (const auto& constant : constants) builder.Add(constant.index | (static_cast<std::uint64_t>(constant.boolean) << 32u) | (static_cast<std::uint64_t>(constant.value) << 33u));
}

std::uint64_t nanosSince(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

void setAttachmentFormat(MTL4RenderPipelineDescriptor* descriptor, NSString* key, PixelFormat format) {
    if (format == PixelFormat::Invalid) return;
    NSString* setter = [NSString stringWithFormat:@"set%@%@:", [[key substringToIndex:1] uppercaseString], [key substringFromIndex:1]];
    if (![descriptor respondsToSelector:NSSelectorFromString(setter)]) return;
    [descriptor setValue:@(static_cast<NSUInteger>(format)) forKey:key];
}

CompileOutcome buildRender(MetalWorker& worker, const RenderPipelineRequest& request, const CompileKey& key, double archiveHitMilliseconds) {
    CompileOutcome outcome;
    std::string error;
    const auto started = std::chrono::steady_clock::now();
    const auto vertexHolder = worker.Library(request.vertexLibrary, &error);
    id<MTLLibrary> vertexLibrary = (__bridge id<MTLLibrary>)vertexHolder.get();
    if (vertexLibrary == nil) {
        outcome.error = error;
        return outcome;
    }
    std::shared_ptr<void> fragmentHolder;
    id<MTLLibrary> fragmentLibrary = nil;
    if (!request.fragmentFunction.empty()) {
        fragmentHolder = request.fragmentLibrary.key == request.vertexLibrary.key ? vertexHolder : worker.Library(request.fragmentLibrary, &error);
        fragmentLibrary = (__bridge id<MTLLibrary>)fragmentHolder.get();
        if (fragmentLibrary == nil) {
            outcome.error = error;
            return outcome;
        }
    }
    MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = functionDescriptor(vertexLibrary, request.vertexFunction, request.vertexConstants);
    if (fragmentLibrary != nil) descriptor.fragmentFunctionDescriptor = functionDescriptor(fragmentLibrary, request.fragmentFunction, request.fragmentConstants);
    const auto& state = request.state;
    for (std::uint32_t index = 0; index < state.colorCount; ++index) {
        const auto& color = state.colors[index];
        MTL4RenderPipelineColorAttachmentDescriptor* attachment = descriptor.colorAttachments[index];
        attachment.pixelFormat = static_cast<MTLPixelFormat>(color.format);
        attachment.writeMask = color.format == PixelFormat::Invalid ? MTLColorWriteMaskNone : static_cast<MTLColorWriteMask>(color.writeMask);
        attachment.blendingState = color.blend ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
        if (!color.blend) continue;
        attachment.sourceRGBBlendFactor = static_cast<MTLBlendFactor>(color.sourceRgb);
        attachment.destinationRGBBlendFactor = static_cast<MTLBlendFactor>(color.destinationRgb);
        attachment.rgbBlendOperation = static_cast<MTLBlendOperation>(color.rgbOperation);
        attachment.sourceAlphaBlendFactor = static_cast<MTLBlendFactor>(color.sourceAlpha);
        attachment.destinationAlphaBlendFactor = static_cast<MTLBlendFactor>(color.destinationAlpha);
        attachment.alphaBlendOperation = static_cast<MTLBlendOperation>(color.alphaOperation);
    }
    if (!request.vertexAttributes.empty()) {
        MTLVertexDescriptor* vertex = [MTLVertexDescriptor vertexDescriptor];
        for (const auto& attribute : request.vertexAttributes) {
            vertex.attributes[attribute.location].format = static_cast<MTLVertexFormat>(attribute.format);
            vertex.attributes[attribute.location].offset = attribute.offset;
            vertex.attributes[attribute.location].bufferIndex = attribute.buffer;
        }
        for (const auto& buffer : request.vertexBuffers) {
            vertex.layouts[buffer.buffer].stride = buffer.stride;
            vertex.layouts[buffer.buffer].stepFunction = buffer.step == VertexStep::Instance ? MTLVertexStepFunctionPerInstance : buffer.step == VertexStep::Constant ? MTLVertexStepFunctionConstant : MTLVertexStepFunctionPerVertex;
            vertex.layouts[buffer.buffer].stepRate = buffer.step == VertexStep::Constant ? 0 : 1;
        }
        descriptor.vertexDescriptor = vertex;
    }
    setAttachmentFormat(descriptor, @"depthAttachmentPixelFormat", state.depthFormat);
    setAttachmentFormat(descriptor, @"stencilAttachmentPixelFormat", state.stencilFormat);
    descriptor.rasterSampleCount = state.sampleCount;
    descriptor.alphaToCoverageState = state.alphaToCoverage ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
    descriptor.inputPrimitiveTopology = static_cast<MTLPrimitiveTopologyClass>(state.topology);
    descriptor.rasterizationEnabled = state.rasterization;
    if (!request.label.empty()) descriptor.label = nsString(request.label);
    NSError* failure = nil;
    id<MTLRenderPipelineState> pipeline = [(__bridge id<MTL4Compiler>)worker.Compiler() newRenderPipelineStateWithDescriptor:descriptor dynamicLinkingDescriptor:nil compilerTaskOptions:(__bridge MTL4CompilerTaskOptions*)worker.TaskOptions() error:&failure];
    if (pipeline == nil) {
        outcome.error = std::string("render pipeline ") + request.vertexFunction + "/" + request.fragmentFunction + ": " + (failure.localizedDescription.UTF8String ?: "unknown");
        return outcome;
    }
    const double milliseconds = static_cast<double>(nanosSince(started)) / 1.0e6;
    auto product = std::make_shared<CompiledPipeline>();
    product->buildMilliseconds = milliseconds;
    product->archiveHit = milliseconds < archiveHitMilliseconds;
    product->native = (__bridge_retained void*)pipeline;
    worker.NoteKey(key);
    worker.NotePipeline(milliseconds);
    outcome.costBytes = 16ull << 10u;
    outcome.product = std::move(product);
    return outcome;
}

}

struct PipelineCache::State {
    Device& device;
    MetalCompiler& compiler;
    id<MTLDevice> native;
    mutable std::mutex mutex;
    std::unordered_map<CompileKey, std::shared_ptr<RenderPipeline>, CompileKeyHash> renders;
    std::unordered_map<std::uint64_t, id<MTLDepthStencilState>> depthStencils;
    PipelineCacheStats stats;

    State(Device& device, MetalCompiler& compiler) : device(device), compiler(compiler), native((__bridge id<MTLDevice>)device.NativeDevice()) {}

    std::shared_ptr<CompileTicket> submit(const RenderPipelineRequest& request, const CompileKey& key, CompilePriority priority) {
        return compiler.Run(key, priority, [request, key](MetalWorker& worker) { return buildRender(worker, request, key, 4.0); });
    }
};

const CompiledPipeline* RenderPipeline::Compiled() const {
    return ticket->Ready() ? static_cast<const CompiledPipeline*>(ticket->Product().get()) : nullptr;
}

void* RenderPipeline::Resolve() const {
    if (!ticket->Ready()) return nullptr;
    const auto* compiled = static_cast<const CompiledPipeline*>(ticket->Product().get());
    void* resolved = compiled != nullptr ? compiled->native : nullptr;
    native.store(resolved, std::memory_order_release);
    return resolved;
}

PipelineCache::PipelineCache(std::unique_ptr<State> state) : state(std::move(state)) {}

PipelineCache::~PipelineCache() {
    std::lock_guard lock(state->mutex);
    state->depthStencils.clear();
    state->renders.clear();
}

std::unique_ptr<PipelineCache> PipelineCache::Create(Device& device, MetalCompiler& compiler, std::string* failure) {
    if (device.NativeDevice() == nullptr) {
        if (failure != nullptr) *failure = "no Metal device";
        return nullptr;
    }
    return std::unique_ptr<PipelineCache>(new PipelineCache(std::make_unique<State>(device, compiler)));
}

CompileKey PipelineCache::RenderKey(const RenderPipelineRequest& request) {
    CompileKeyBuilder builder(RenderDomain);
    builder.Add(request.vertexLibrary.key).Add(request.vertexLibrary.languageVersion).Add(request.vertexLibrary.fastMath ? 1u : 0u).AddString(request.vertexFunction);
    addConstants(builder, request.vertexConstants);
    if (!request.fragmentFunction.empty()) {
        builder.Add(request.fragmentLibrary.key).Add(request.fragmentLibrary.languageVersion).Add(request.fragmentLibrary.fastMath ? 1u : 0u).AddString(request.fragmentFunction);
        addConstants(builder, request.fragmentConstants);
    }
    request.state.AddTo(builder);
    builder.Add(static_cast<std::uint64_t>(request.vertexAttributes.size()));
    for (const auto& attribute : request.vertexAttributes) builder.Add(static_cast<std::uint64_t>(attribute.location) | static_cast<std::uint64_t>(attribute.format) << 8u | static_cast<std::uint64_t>(attribute.buffer) << 24u | static_cast<std::uint64_t>(attribute.offset) << 32u);
    for (const auto& buffer : request.vertexBuffers) builder.Add(static_cast<std::uint64_t>(buffer.buffer) | static_cast<std::uint64_t>(buffer.stride) << 8u | static_cast<std::uint64_t>(buffer.step) << 40u);
    return builder.Finish();
}

std::shared_ptr<RenderPipeline> PipelineCache::Render(const RenderPipelineRequest& request) {
    const auto key = RenderKey(request);
    std::lock_guard lock(state->mutex);
    ++state->stats.renderRequests;
    if (const auto found = state->renders.find(key); found != state->renders.end()) {
        ++state->stats.renderHits;
        auto pipeline = found->second;
        if (request.priority > pipeline->ticket->Priority() && !pipeline->ticket->Done()) {
            ++state->stats.promotions;
            state->submit(pipeline->request, key, request.priority);
        }
        return pipeline;
    }
    auto pipeline = std::make_shared<RenderPipeline>();
    pipeline->key = key;
    pipeline->request = request;
    pipeline->ticket = state->submit(request, key, request.priority);
    state->renders.emplace(key, pipeline);
    ++state->stats.renderPipelines;
    return pipeline;
}

void PipelineCache::Promote(const RenderPipeline& pipeline, CompilePriority priority) {
    if (pipeline.ticket->Done() || pipeline.ticket->Priority() >= priority) return;
    std::lock_guard lock(state->mutex);
    ++state->stats.promotions;
    state->submit(pipeline.request, pipeline.key, priority);
}

void* PipelineCache::DepthStencil(const DepthStencilState& depthStencil) {
    const auto packed = depthStencil.Pack();
    std::lock_guard lock(state->mutex);
    if (const auto found = state->depthStencils.find(packed); found != state->depthStencils.end()) {
        ++state->stats.depthStencilHits;
        return (__bridge void*)found->second;
    }
    @autoreleasepool {
        MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
        descriptor.depthCompareFunction = static_cast<MTLCompareFunction>(depthStencil.depthCompare);
        descriptor.depthWriteEnabled = depthStencil.depthWrite;
        if (depthStencil.stencil) {
            auto face = [](const StencilFace& source) {
                MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
                stencil.stencilFailureOperation = static_cast<MTLStencilOperation>(source.fail);
                stencil.depthFailureOperation = static_cast<MTLStencilOperation>(source.depthFail);
                stencil.depthStencilPassOperation = static_cast<MTLStencilOperation>(source.pass);
                stencil.stencilCompareFunction = static_cast<MTLCompareFunction>(source.compare);
                stencil.readMask = source.readMask;
                stencil.writeMask = source.writeMask;
                return stencil;
            };
            descriptor.frontFaceStencil = face(depthStencil.front);
            descriptor.backFaceStencil = face(depthStencil.back);
        }
        id<MTLDepthStencilState> created = [state->native newDepthStencilStateWithDescriptor:descriptor];
        if (created == nil) return nullptr;
        state->depthStencils.emplace(packed, created);
        ++state->stats.depthStencilStates;
        return (__bridge void*)created;
    }
}

PipelineCacheStats PipelineCache::Stats() const {
    std::lock_guard lock(state->mutex);
    return state->stats;
}

}
