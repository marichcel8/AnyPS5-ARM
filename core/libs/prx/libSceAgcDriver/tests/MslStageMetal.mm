#import <Metal/Metal.h>
#include "MslStagePrograms.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint32_t Size = MslStagePrograms::TargetSize;
constexpr NSUInteger VertexBufferSlot = 16u;
constexpr float Clear = -1.0f;

struct Target {
    std::vector<float> color;
    std::vector<float> depth;
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Describe(NSError* error) {
    return error == nil ? std::string("unknown error") : std::string(error.localizedDescription.UTF8String);
}

id<MTLFunction> Function(id<MTLDevice> device, const RecompileResult& result, const std::string& name, double& compileMs) {
    const auto source = std::string(result.msl.Source());
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_1;
    if (@available(macOS 15.0, *)) {
        options.languageVersion = MTLLanguageVersion3_2;
        options.mathMode = MTLMathModeFast;
    }
    const auto start = std::chrono::steady_clock::now();
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
    if (library == nil) {
        std::fprintf(stderr, "%s\n", source.c_str());
        throw std::runtime_error(name + ": the MSL source does not compile: " + Describe(error));
    }
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    for (const auto& constant : result.specialization) {
        if (source.find("[[function_constant(" + std::to_string(constant.id) + ")]]") == std::string::npos) continue;
        const std::uint32_t value = constant.value;
        [values setConstantValue:&value type:MTLDataTypeUInt atIndex:constant.id];
    }
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:std::string(MslEntryPoint).c_str()] constantValues:values error:&error];
    if (function == nil) {
        std::fprintf(stderr, "%s\n", source.c_str());
        throw std::runtime_error(name + ": the entry point does not specialize: " + Describe(error));
    }
    compileMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return function;
}

Target Draw(id<MTLDevice> device, id<MTLCommandQueue> queue, const std::string& name, const RecompileRequest& vertexRequest, const RecompileRequest& pixelRequest, id<MTLBuffer> vertices, double& compileMs) {
    const auto vertexResult = Recompile(vertexRequest);
    const auto pixelResult = Recompile(pixelRequest);
    Require(vertexResult.backend == ShaderBackendKind::Msl && pixelResult.backend == ShaderBackendKind::Msl, name + ": a stage did not recompile to MSL");

    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = Function(device, vertexResult, name + " vertex", compileMs);
    descriptor.fragmentFunction = Function(device, pixelResult, name + " pixel", compileMs);
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    if (vertices != nil) {
        MTLVertexDescriptor* layout = [MTLVertexDescriptor vertexDescriptor];
        layout.attributes[0].format = MTLVertexFormatFloat4;
        layout.attributes[0].offset = 0;
        layout.attributes[0].bufferIndex = VertexBufferSlot;
        layout.layouts[VertexBufferSlot].stride = 16;
        descriptor.vertexDescriptor = layout;
    }
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    Require(pipeline != nil, name + ": the render pipeline does not build: " + Describe(error));

    MTLDepthStencilDescriptor* depthDescriptor = [MTLDepthStencilDescriptor new];
    depthDescriptor.depthCompareFunction = MTLCompareFunctionAlways;
    depthDescriptor.depthWriteEnabled = YES;
    id<MTLDepthStencilState> depthState = [device newDepthStencilStateWithDescriptor:depthDescriptor];

    MTLTextureDescriptor* colorDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:Size height:Size mipmapped:NO];
    colorDescriptor.usage = MTLTextureUsageRenderTarget;
    colorDescriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> color = [device newTextureWithDescriptor:colorDescriptor];
    MTLTextureDescriptor* depthTextureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:Size height:Size mipmapped:NO];
    depthTextureDescriptor.usage = MTLTextureUsageRenderTarget;
    depthTextureDescriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> depth = [device newTextureWithDescriptor:depthTextureDescriptor];
    id<MTLBuffer> colorBytes = [device newBufferWithLength:Size * Size * 16u options:MTLResourceStorageModeShared];
    id<MTLBuffer> depthBytes = [device newBufferWithLength:Size * Size * 4u options:MTLResourceStorageModeShared];

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = color;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(Clear, Clear, Clear, Clear);
    pass.depthAttachment.texture = depth;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    pass.depthAttachment.clearDepth = 1.0;

    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setDepthStencilState:depthState];
    [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<double>(Size), static_cast<double>(Size), 0.0, 1.0}];
    if (vertices != nil) [encoder setVertexBuffer:vertices offset:0 atIndex:VertexBufferSlot];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
    [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(Size, Size, 1) toBuffer:colorBytes destinationOffset:0 destinationBytesPerRow:Size * 16u destinationBytesPerImage:Size * Size * 16u];
    [blit copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(Size, Size, 1) toBuffer:depthBytes destinationOffset:0 destinationBytesPerRow:Size * 4u destinationBytesPerImage:Size * Size * 4u];
    [blit endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    Require(commands.status == MTLCommandBufferStatusCompleted, name + ": the draw failed: " + Describe(commands.error));

    Target target;
    target.color.resize(Size * Size * 4u);
    target.depth.resize(Size * Size);
    std::memcpy(target.color.data(), colorBytes.contents, target.color.size() * sizeof(float));
    std::memcpy(target.depth.data(), depthBytes.contents, target.depth.size() * sizeof(float));
    return target;
}

void Expect(const Target& target, const std::string& name, const std::function<bool(std::uint32_t, std::uint32_t, std::array<float, 4>&, float&)>& expected) {
    for (std::uint32_t y = 0; y < Size; ++y) {
        for (std::uint32_t x = 0; x < Size; ++x) {
            std::array<float, 4> color{Clear, Clear, Clear, Clear};
            float depth = 1.0f;
            static_cast<void>(expected(x, y, color, depth));
            const auto* got = &target.color[(y * Size + x) * 4u];
            for (std::uint32_t component = 0; component < 4u; ++component) {
                Require(std::fabs(got[component] - color[component]) <= 2e-3f, name + ": pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") component " + std::to_string(component) + " is " + std::to_string(got[component]) + ", expected " + std::to_string(color[component]));
            }
            Require(std::fabs(target.depth[y * Size + x] - depth) <= 1e-6f, name + ": pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") depth is " + std::to_string(target.depth[y * Size + x]) + ", expected " + std::to_string(depth));
        }
    }
}

float Coordinate(std::uint32_t pixel) {
    return (static_cast<float>(pixel) + 0.5f) / static_cast<float>(Size * 2u);
}

}

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            Require(device != nil, "no Metal device");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            Require(queue != nil, "no Metal command queue");

            double compileMs = 0.0;
            const auto interpolated = Draw(device, queue, "interpolated", MslStagePrograms::GridVertexRequest(0x10000u), MslStagePrograms::InterpolatedPixelRequest(0x20000u), nil, compileMs);
            Expect(interpolated, "interpolated", [](std::uint32_t x, std::uint32_t y, std::array<float, 4>& color, float& depth) {
                color = {Coordinate(x), Coordinate(y), 0.5f, 1.0f};
                depth = 0.0f;
                return true;
            });
            std::printf("MslStageMetal: interpolated parameters, y flip and export mapping ok (Metal compile %.1f ms)\n", compileMs);

            compileMs = 0.0;
            const auto killed = Draw(device, queue, "kill and depth", MslStagePrograms::GridVertexRequest(0x10000u), MslStagePrograms::KillDepthPixelRequest(0x21000u), nil, compileMs);
            Expect(killed, "kill and depth", [](std::uint32_t x, std::uint32_t, std::array<float, 4>& color, float& depth) {
                if (Coordinate(x) >= 0.25f) return false;
                color = {Coordinate(x), 1.0f, 0.0f, 1.0f};
                depth = 0.5f;
                return true;
            });
            std::printf("MslStageMetal: flat parameter, valid-mask discard and depth export ok (Metal compile %.1f ms)\n", compileMs);

            compileMs = 0.0;
            const float positions[] = {-1.0f, -1.0f, 0.25f, 1.0f, 3.0f, -1.0f, 0.25f, 1.0f, -1.0f, 3.0f, 0.25f, 1.0f};
            id<MTLBuffer> vertices = [device newBufferWithBytes:positions length:sizeof(positions) options:MTLResourceStorageModeShared];
            const auto fetched = Draw(device, queue, "vertex fetch", MslStagePrograms::FetchVertexRequest(0x11000u), MslStagePrograms::ConstantPixelRequest(0x22000u), vertices, compileMs);
            Expect(fetched, "vertex fetch", [](std::uint32_t, std::uint32_t, std::array<float, 4>& color, float& depth) {
                color = {1.0f, 0.5f, 0.0f, 1.0f};
                depth = 0.25f;
                return true;
            });
            std::printf("MslStageMetal: stage_in vertex attribute selected by function constants ok (Metal compile %.1f ms)\n", compileMs);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "MslStageMetal: %s\n", error.what());
            return 1;
        }
    }
    std::puts("MslStageMetal: vertex and pixel programs compile, draw and write the expected pixels and depths");
    return 0;
}
