#import <Metal/Metal.h>
#include "MslGeometryPrograms.hpp"
#include "BdaAbi.hpp"
#include "MetalAbi.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ShaderRecompiler;
using MslGeometryPrograms::Mesh;
using MslGeometryPrograms::Tessellation;

constexpr std::array<std::uint8_t, 4> Background{16, 24, 40, 255};
constexpr std::uint32_t PushBytes = 128;
constexpr std::uint32_t GuestBufferEntries = 8;
constexpr std::uint32_t TessWidth = 256;
constexpr std::uint32_t TessHeight = 80;
constexpr std::uint32_t MeshWidth = 192;
constexpr std::uint32_t MeshHeight = 128;
constexpr std::uint32_t Columns = 8;
constexpr std::uint32_t Rows = 5;
constexpr std::uint32_t Triangles = Columns * Rows;

struct Vertex {
    std::array<float, 4> position;
    std::array<float, 4> color;
};

struct Compiled {
    RecompileResult result;
    std::string source;
    id<MTLFunction> function;
    MslGeometryLayout layout;
};

struct Image {
    std::uint32_t width;
    std::uint32_t height;
    std::vector<std::uint8_t> pixels;

    std::array<std::uint8_t, 4> At(std::uint32_t x, std::uint32_t y) const {
        const auto* pixel = &pixels[(static_cast<std::size_t>(y) * width + x) * 4u];
        return {pixel[0], pixel[1], pixel[2], pixel[3]};
    }
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Describe(NSError* error) {
    return error == nil ? std::string("unknown error") : std::string(error.localizedDescription.UTF8String);
}

std::string PixelText(const std::array<std::uint8_t, 4>& pixel) {
    char text[64];
    std::snprintf(text, sizeof(text), "(%u, %u, %u, %u)", pixel[0], pixel[1], pixel[2], pixel[3]);
    return text;
}

id<MTLBuffer> Zeroed(id<MTLDevice> device, std::size_t bytes) {
    id<MTLBuffer> buffer = [device newBufferWithLength:std::max<std::size_t>(bytes, 16u) options:MTLResourceStorageModeShared];
    Require(buffer != nil, "could not allocate a shared buffer");
    std::memset(buffer.contents, 0, buffer.length);
    return buffer;
}

Compiled Compile(id<MTLDevice> device, const RecompileRequest& request, const std::string& name, double& compileMs) {
    Compiled compiled{Recompile(request), {}, nil, {}};
    Require(compiled.result.backend == ShaderBackendKind::Msl, name + ": the result is not an MSL artifact");
    compiled.source = std::string(compiled.result.msl.Source());
    if (const auto layout = ParseMslGeometryLayout(compiled.source)) compiled.layout = *layout;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_1;
    if (@available(macOS 15.0, *)) {
        options.languageVersion = MTLLanguageVersion3_2;
        options.mathMode = MTLMathModeFast;
    }
    const auto start = std::chrono::steady_clock::now();
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:compiled.source.c_str()] options:options error:&error];
    if (library == nil) {
        std::fprintf(stderr, "%s\n", compiled.source.c_str());
        throw std::runtime_error(name + ": the MSL source does not compile: " + Describe(error));
    }
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    const auto declared = [&](std::uint32_t id) { return compiled.source.find("[[function_constant(" + std::to_string(id) + ")]]") != std::string::npos; };
    for (const auto& constant : compiled.result.specialization) {
        if (!declared(constant.id) || constant.id == MetalAbi::PositionFlipYConstant) continue;
        const std::uint32_t value = constant.value;
        [values setConstantValue:&value type:MTLDataTypeUInt atIndex:constant.id];
    }
    if (declared(MetalAbi::PositionFlipYConstant)) {
        const std::uint32_t keep = 0u;
        [values setConstantValue:&keep type:MTLDataTypeUInt atIndex:MetalAbi::PositionFlipYConstant];
    }
    compiled.function = [library newFunctionWithName:[NSString stringWithUTF8String:std::string(MslEntryPoint).c_str()] constantValues:values error:&error];
    if (compiled.function == nil) {
        std::fprintf(stderr, "%s\n", compiled.source.c_str());
        throw std::runtime_error(name + ": the entry point does not specialize: " + Describe(error));
    }
    compileMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return compiled;
}

id<MTLComputePipelineState> ComputePipeline(id<MTLDevice> device, const Compiled& compiled, const std::string& name) {
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:compiled.function error:&error];
    Require(pipeline != nil, name + ": the compute pipeline does not build: " + Describe(error));
    return pipeline;
}

MTLTessellationPartitionMode Partition(std::uint32_t partitioning) {
    switch (partitioning) {
    case 0u: return MTLTessellationPartitionModeInteger;
    case 1u: return MTLTessellationPartitionModePow2;
    case 2u: return MTLTessellationPartitionModeFractionalOdd;
    default: return MTLTessellationPartitionModeFractionalEven;
    }
}

id<MTLRenderPipelineState> TessellationPipeline(id<MTLDevice> device, const Compiled& domain, const Compiled& pixel, const std::string& name) {
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = domain.function;
    descriptor.fragmentFunction = pixel.function;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.tessellationPartitionMode = Partition(domain.layout.partitioning);
    descriptor.maxTessellationFactor = 64u;
    descriptor.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
    descriptor.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
    descriptor.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
    descriptor.tessellationOutputWindingOrder = domain.layout.outputTopology == 3u ? MTLWindingCounterClockwise : MTLWindingClockwise;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    Require(pipeline != nil, name + ": the tessellation render pipeline does not build: " + Describe(error));
    return pipeline;
}

id<MTLRenderPipelineState> MeshPipeline(id<MTLDevice> device, const Compiled& mesh, const Compiled& pixel, const std::string& name) {
    MTLMeshRenderPipelineDescriptor* descriptor = [MTLMeshRenderPipelineDescriptor new];
    descriptor.meshFunction = mesh.function;
    descriptor.fragmentFunction = pixel.function;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.maxTotalThreadsPerMeshThreadgroup = mesh.layout.meshThreads;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithMeshDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
    Require(pipeline != nil, name + ": the mesh render pipeline does not build: " + Describe(error));
    return pipeline;
}

struct Target {
    id<MTLTexture> texture;
    id<MTLBuffer> bytes;
    std::uint32_t width;
    std::uint32_t height;
};

Target MakeTarget(id<MTLDevice> device, std::uint32_t width, std::uint32_t height) {
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget;
    descriptor.storageMode = MTLStorageModePrivate;
    return {[device newTextureWithDescriptor:descriptor], Zeroed(device, static_cast<std::size_t>(width) * height * 4u), width, height};
}

id<MTLRenderCommandEncoder> BeginPass(id<MTLCommandBuffer> commands, const Target& target) {
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(Background[0] / 255.0, Background[1] / 255.0, Background[2] / 255.0, 1.0);
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<double>(target.width), static_cast<double>(target.height), 0.0, 1.0}];
    [encoder setCullMode:MTLCullModeNone];
    return encoder;
}

Image Finish(id<MTLCommandBuffer> commands, const Target& target, const std::string& name) {
    id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
    [blit copyFromTexture:target.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(target.width, target.height, 1) toBuffer:target.bytes destinationOffset:0 destinationBytesPerRow:target.width * 4u destinationBytesPerImage:target.width * target.height * 4u];
    [blit endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    Require(commands.status == MTLCommandBufferStatusCompleted, name + ": the command buffer failed: " + Describe(commands.error));
    Image image{target.width, target.height, std::vector<std::uint8_t>(static_cast<std::size_t>(target.width) * target.height * 4u)};
    std::memcpy(image.pixels.data(), target.bytes.contents, image.pixels.size());
    return image;
}

id<MTLBuffer> Push(id<MTLDevice> device, const RecompileResult& result) {
    id<MTLBuffer> push = Zeroed(device, PushBytes);
    std::memcpy(push.contents, result.pushConstants.data(), std::min<std::size_t>(result.pushConstants.size(), PushBytes));
    return push;
}

void SetPushFloat(id<MTLBuffer> push, std::uint32_t dword, float value) {
    std::memcpy(static_cast<std::uint8_t*>(push.contents) + dword * 4u, &value, sizeof(value));
}

void SetPushWord(id<MTLBuffer> push, std::uint32_t dword, std::uint32_t value) {
    std::memcpy(static_cast<std::uint8_t*>(push.contents) + dword * 4u, &value, sizeof(value));
}

void CompileQuadPatches(id<MTLDevice> device, double& compileMs) {
    const auto hull = Compile(device, Tessellation(ShaderStage::TessellationControl, MslGeometryPrograms::QuadHullCode, MslGeometryPrograms::QuadPatches), "quad hull", compileMs);
    const auto domain = Compile(device, Tessellation(ShaderStage::TessellationEvaluation, MslGeometryPrograms::QuadDomainCode, MslGeometryPrograms::QuadPatches), "quad domain", compileMs);
    const auto pixel = Compile(device, MslGeometryPrograms::Pixel(), "pixel", compileMs);
    static_cast<void>(ComputePipeline(device, hull, "quad hull"));
    static_cast<void>(TessellationPipeline(device, domain, pixel, "quad domain"));
}

std::uint32_t Column(std::uint32_t patch, float offset) {
    return static_cast<std::uint32_t>((-0.9f + 0.6f * static_cast<float>(patch) + offset + 1.0f) * 0.5f * TessWidth);
}

std::uint32_t Row(float y) {
    return static_cast<std::uint32_t>((1.0f - y) * 0.5f * TessHeight);
}

std::uint32_t CoveredPixels(const Image& image, std::uint32_t patch, bool interior) {
    std::uint32_t count = 0;
    for (std::uint32_t y = Row(0.5f); y <= Row(-0.5f) && y < image.height; ++y) {
        for (std::uint32_t x = Column(patch, 0.0f); x <= Column(patch, 0.5f) && x < image.width; ++x) {
            const auto pixel = image.At(x, y);
            if (pixel == Background) continue;
            Require(pixel[0] == 255 && pixel[2] == 0 && pixel[3] == 255, "tessellation: patch " + std::to_string(patch) + " pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + PixelText(pixel));
            if (!interior || pixel[1] > 32) ++count;
        }
    }
    return count;
}

void DrawTrianglePatches(id<MTLDevice> device, id<MTLCommandQueue> queue, double& compileMs) {
    const auto& patches = MslGeometryPrograms::TrianglePatches;
    const auto local = Compile(device, Tessellation(ShaderStage::Local, MslGeometryPrograms::LocalCode, patches), "local", compileMs);
    const auto hull = Compile(device, Tessellation(ShaderStage::TessellationControl, MslGeometryPrograms::HullCode, patches), "hull", compileMs);
    const auto domain = Compile(device, Tessellation(ShaderStage::TessellationEvaluation, MslGeometryPrograms::DomainCode, patches), "domain", compileMs);
    const auto pixel = Compile(device, MslGeometryPrograms::Pixel(), "pixel", compileMs);
    Require(local.layout.kind == MslGeometryKind::Local && hull.layout.kind == MslGeometryKind::Hull && domain.layout.kind == MslGeometryKind::Domain, "tessellation: a stage has no geometry layout");
    id<MTLComputePipelineState> localPipeline = ComputePipeline(device, local, "local");
    id<MTLComputePipelineState> hullPipeline = ComputePipeline(device, hull, "hull");
    id<MTLRenderPipelineState> domainPipeline = TessellationPipeline(device, domain, pixel, "domain");

    constexpr std::uint32_t PatchCount = 3;
    const auto points = PatchCount * patches.inputControlPoints;
    const auto halves = MetalAbi::TessellationFactorHalves(domain.layout.domain);
    id<MTLBuffer> indices = [device newBufferWithBytes:MslGeometryPrograms::CornerIndices.data() length:sizeof(MslGeometryPrograms::CornerIndices) options:MTLResourceStorageModeShared];
    id<MTLBuffer> localRing = Zeroed(device, static_cast<std::size_t>(points) * hull.layout.LocalWords() * 4u);
    id<MTLBuffer> controlRing = Zeroed(device, static_cast<std::size_t>(PatchCount) * patches.outputControlPoints * hull.layout.ControlWords() * 4u);
    id<MTLBuffer> patchRing = Zeroed(device, static_cast<std::size_t>(PatchCount) * std::max(hull.layout.PatchWords(), 1u) * 4u);
    id<MTLBuffer> factors = Zeroed(device, static_cast<std::size_t>(PatchCount) * halves * 2u);
    id<MTLBuffer> fault = Zeroed(device, BdaAbi::FaultBufferBytes);
    id<MTLBuffer> buffers = Zeroed(device, GuestBufferEntries * sizeof(MetalAbi::GuestBufferEntry));

    MetalAbi::GeometryParameters parameters{};
    parameters.localRing = localRing.gpuAddress;
    parameters.controlRing = controlRing.gpuAddress;
    parameters.patchRing = patchRing.gpuAddress;
    parameters.factors = factors.gpuAddress;
    parameters.indices = indices.gpuAddress;
    parameters.localWords = hull.layout.LocalWords();
    parameters.controlWords = hull.layout.ControlWords();
    parameters.patchWords = hull.layout.PatchWords();
    parameters.patchCount = PatchCount;
    parameters.instanceCount = 1u;
    parameters.indexBytes = 2u;
    Require(local.layout.LocalWords() == parameters.localWords && domain.layout.ControlWords() == parameters.controlWords, "tessellation: the stages disagree on their ring layout");

    id<MTLBuffer> localPush = Push(device, local.result);
    id<MTLBuffer> hullPush = Push(device, hull.result);
    SetPushFloat(hullPush, TessellationLevelPushOffsetBytes / 4u, 1.0f);
    SetPushFloat(hullPush, TessellationLevelPushOffsetBytes / 4u + 1u, 64.0f);
    id<MTLBuffer> domainPush = Push(device, domain.result);
    id<MTLBuffer> pixelPush = Push(device, pixel.result);

    const auto bindCompute = [&](id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> push) {
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:MetalAbi::GeometrySlot];
        [encoder setBuffer:push offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::PushConstants)];
        [encoder setBuffer:fault offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::FaultBuffer)];
        [encoder setBuffer:buffers offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestBuffers)];
        for (id<MTLBuffer> ring in @[indices, localRing, controlRing, patchRing, factors]) [encoder useResource:ring usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    };

    const auto start = std::chrono::steady_clock::now();
    const auto target = MakeTarget(device, TessWidth, TessHeight);
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> compute = [commands computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [compute setComputePipelineState:localPipeline];
    bindCompute(compute, localPush);
    [compute dispatchThreads:MTLSizeMake(points, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(points, localPipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [compute memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [compute setComputePipelineState:hullPipeline];
    bindCompute(compute, hullPush);
    [compute dispatchThreads:MTLSizeMake(patches.outputControlPoints, PatchCount, 1) threadsPerThreadgroup:MTLSizeMake(patches.outputControlPoints, 1, 1)];
    [compute endEncoding];

    id<MTLRenderCommandEncoder> render = BeginPass(commands, target);
    [render setRenderPipelineState:domainPipeline];
    [render setTessellationFactorBuffer:factors offset:0 instanceStride:0];
    [render setVertexBytes:&parameters length:sizeof(parameters) atIndex:MetalAbi::GeometrySlot];
    [render setVertexBuffer:domainPush offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::PushConstants)];
    [render setVertexBuffer:fault offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::FaultBuffer)];
    [render setVertexBuffer:buffers offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestBuffers)];
    [render setFragmentBuffer:pixelPush offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::PushConstants)];
    [render setFragmentBuffer:fault offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::FaultBuffer)];
    [render setFragmentBuffer:buffers offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestBuffers)];
    [render useResource:controlRing usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
    [render useResource:patchRing usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
    [render drawPatches:patches.outputControlPoints patchStart:0 patchCount:PatchCount patchIndexBuffer:nil patchIndexBufferOffset:0 instanceCount:1 baseInstance:0];
    [render endEncoding];
    const auto image = Finish(commands, target, "tessellation");
    const auto gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    const auto* faultWords = static_cast<const std::uint32_t*>(fault.contents);
    Require(faultWords[0] == 0u, "tessellation: a stage recorded fault reason " + std::to_string(faultWords[1]));
    const auto* factorHalves = static_cast<const std::uint16_t*>(factors.contents);
    for (std::uint32_t half = 0; half < halves; ++half) {
        Require(factorHalves[halves + half] == 0u, "tessellation: the level-0 patch has a nonzero factor at half " + std::to_string(half));
        Require(factorHalves[half] != 0u && factorHalves[2u * halves + half] != 0u, "tessellation: a drawn patch has a zero factor at half " + std::to_string(half));
    }
    const auto center = Row(-0.5f + 1.0f / 3.0f);
    Require(CoveredPixels(image, 0, false) > 100 && CoveredPixels(image, 0, true) == 0, "tessellation: the level-1 patch was not drawn as one flat triangle, centre " + PixelText(image.At(Column(0, 0.25f), center)));
    Require(CoveredPixels(image, 1, false) == 0, "tessellation: the patch with level 0 was not culled");
    Require(CoveredPixels(image, 2, false) > 100 && image.At(Column(2, 0.25f), center)[1] > 32, "tessellation: the level-5 patch has no interior domain points at its centre, " + PixelText(image.At(Column(2, 0.25f), center)));
    Require(image.At(0, 0) == Background && image.At(TessWidth - 1u, TessHeight - 1u) == Background, "tessellation: the corners changed");
    std::printf("MslGeometryMetal: local and hull kernels, factor buffer and post-tessellation vertex function draw the three patches (Metal compile %.1f ms, first frame %.2f ms)\n", compileMs, gpuMs);
}

std::array<std::uint8_t, 4> TriangleColor(std::uint32_t triangle) {
    return {static_cast<std::uint8_t>(40u + 29u * triangle), static_cast<std::uint8_t>(200u + 53u * triangle), static_cast<std::uint8_t>(triangle % 2u == 0u ? 60u : 200u), 255};
}

std::vector<Vertex> OrderedTriangles() {
    std::vector<Vertex> vertices;
    const float cellWidth = 2.0f / Columns;
    const float cellHeight = 2.0f / Rows;
    for (std::uint32_t triangle = 0; triangle < Triangles; ++triangle) {
        const float x = -1.0f + cellWidth * static_cast<float>(triangle % Columns);
        const float y = -1.0f + cellHeight * static_cast<float>(triangle / Columns);
        const auto color = TriangleColor(triangle);
        const std::array<float, 4> rgba{color[0] / 255.0f, color[1] / 255.0f, color[2] / 255.0f, 1.0f};
        vertices.push_back({{x + 0.1f * cellWidth, y + 0.1f * cellHeight, 0.5f, 1.0f}, rgba});
        vertices.push_back({{x + 0.9f * cellWidth, y + 0.1f * cellHeight, 0.5f, 1.0f}, rgba});
        vertices.push_back({{x + 0.5f * cellWidth, y + 0.9f * cellHeight, 0.5f, 1.0f}, rgba});
    }
    return vertices;
}

std::array<std::uint32_t, 2> CellPixel(std::uint32_t triangle) {
    const auto column = triangle % Columns;
    const auto row = triangle / Columns;
    return {(column * MeshWidth + MeshWidth / 2u) / Columns, MeshHeight - 1u - (row * MeshHeight + (MeshHeight * 4u) / 10u) / Rows};
}

bool Near(const std::array<std::uint8_t, 4>& got, const std::array<std::uint8_t, 4>& expected) {
    for (std::size_t c = 0; c < 4; ++c) {
        if (std::abs(static_cast<int>(got[c]) - static_cast<int>(expected[c])) > 1) return false;
    }
    return true;
}

void DrawMesh(id<MTLDevice> device, id<MTLCommandQueue> queue, double& compileMs) {
    const auto mesh = Compile(device, Mesh(MslGeometryPrograms::Wave32GeometryCode, MslGeometryPrograms::SmallSubgroup, 32u), "wave32 mesh", compileMs);
    const auto pixel = Compile(device, MslGeometryPrograms::Pixel(), "pixel", compileMs);
    Require(mesh.layout.kind == MslGeometryKind::Mesh && mesh.layout.meshThreads != 0u && mesh.layout.meshPrimitives != 0u, "mesh: the program has no mesh layout");
    id<MTLRenderPipelineState> pipeline = MeshPipeline(device, mesh, pixel, "wave32 mesh");

    const auto ordered = OrderedTriangles();
    id<MTLBuffer> vertices = [device newBufferWithBytes:ordered.data() length:ordered.size() * sizeof(Vertex) options:MTLResourceStorageModeShared];
    id<MTLBuffer> fault = Zeroed(device, BdaAbi::FaultBufferBytes);
    id<MTLBuffer> buffers = Zeroed(device, GuestBufferEntries * sizeof(MetalAbi::GuestBufferEntry));
    auto* entries = static_cast<MetalAbi::GuestBufferEntry*>(buffers.contents);
    for (std::uint32_t entry = 0; entry < GuestBufferEntries; ++entry) entries[entry] = {vertices.gpuAddress, static_cast<std::uint32_t>(vertices.length), 0u};
    id<MTLBuffer> push = Push(device, mesh.result);
    const auto draw = MeshDrawPushOffsetBytes / 4u;
    SetPushWord(push, draw, static_cast<std::uint32_t>(ordered.size()));
    for (std::uint32_t word = 1; word < MeshDrawPushBytes / 4u; ++word) SetPushWord(push, draw + word, 0u);
    id<MTLBuffer> pixelPush = Push(device, pixel.result);
    MetalAbi::GeometryParameters parameters{};
    parameters.instanceCount = 1u;
    const auto groups = (Triangles - 1u) / mesh.layout.meshPrimitives + 1u;

    const auto start = std::chrono::steady_clock::now();
    const auto target = MakeTarget(device, MeshWidth, MeshHeight);
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLRenderCommandEncoder> render = BeginPass(commands, target);
    [render setRenderPipelineState:pipeline];
    [render setMeshBytes:&parameters length:sizeof(parameters) atIndex:MetalAbi::GeometrySlot];
    [render setMeshBuffer:push offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::PushConstants)];
    [render setMeshBuffer:fault offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::FaultBuffer)];
    [render setMeshBuffer:buffers offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestBuffers)];
    [render setFragmentBuffer:pixelPush offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::PushConstants)];
    [render setFragmentBuffer:fault offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::FaultBuffer)];
    [render setFragmentBuffer:buffers offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestBuffers)];
    [render useResource:vertices usage:MTLResourceUsageRead stages:MTLRenderStageMesh];
    [render drawMeshThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(mesh.layout.meshThreads, 1, 1)];
    [render endEncoding];
    const auto image = Finish(commands, target, "mesh");
    const auto gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    const auto* faultWords = static_cast<const std::uint32_t*>(fault.contents);
    Require(faultWords[0] == 0u, "mesh: the mesh function recorded fault reason " + std::to_string(faultWords[1]));
    for (std::uint32_t triangle = 0; triangle < Triangles; ++triangle) {
        const auto [x, y] = CellPixel(triangle);
        Require(Near(image.At(x, y), TriangleColor(triangle)), "mesh: triangle " + std::to_string(triangle) + " was not rendered in its color, pixel " + PixelText(image.At(x, y)));
    }
    Require(image.At(0, 0) == Background && image.At(MeshWidth - 1u, MeshHeight - 1u) == Background, "mesh: the corners changed");
    std::printf("MslGeometryMetal: wave32 mesh function draws %u triangles in %u threadgroups (Metal compile %.1f ms, first frame %.2f ms)\n", Triangles, groups, compileMs, gpuMs);
}

void CompileFastLaunch(id<MTLDevice> device, double& compileMs) {
    const auto mesh = Compile(device, Mesh(MslGeometryPrograms::FastLaunchCode, MslGeometryPrograms::FastLaunchSubgroup, 32u), "fast-launch mesh", compileMs);
    const auto pixel = Compile(device, MslGeometryPrograms::Pixel(), "pixel", compileMs);
    static_cast<void>(MeshPipeline(device, mesh, pixel, "fast-launch mesh"));
}

}

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            Require(device != nil, "no Metal device");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            Require(queue != nil, "no Metal command queue");
            Require([device supportsFamily:MTLGPUFamilyApple7], "the device has no mesh shader support");

            double compileMs = 0.0;
            DrawTrianglePatches(device, queue, compileMs);
            compileMs = 0.0;
            CompileQuadPatches(device, compileMs);
            std::printf("MslGeometryMetal: quad hull kernel and quad post-tessellation vertex function build (Metal compile %.1f ms)\n", compileMs);
            compileMs = 0.0;
            DrawMesh(device, queue, compileMs);
            compileMs = 0.0;
            CompileFastLaunch(device, compileMs);
            std::printf("MslGeometryMetal: fast-launch mesh pipeline builds (Metal compile %.1f ms)\n", compileMs);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "MslGeometryMetal: %s\n", error.what());
            return 1;
        }
    }
    std::puts("MslGeometryMetal: tessellation and mesh programs compile, build Metal pipelines and draw the expected pixels");
    return 0;
}
