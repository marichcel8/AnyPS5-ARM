#import <Metal/Metal.h>
#include "MslImagePrograms.hpp"
#include "BdaAbi.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "MetalAbi.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ShaderRecompiler;
using MslImagePrograms::Kind;

constexpr std::uint32_t OutputBytes = MslImagePrograms::Threads * sizeof(std::uint32_t);

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Describe(NSError* error) {
    return error == nil ? std::string("unknown error") : std::string(error.localizedDescription.UTF8String);
}

id<MTLBuffer> Zeroed(id<MTLDevice> device, std::size_t bytes) {
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    Require(buffer != nil, "could not allocate a shared buffer");
    std::memset(buffer.contents, 0, bytes);
    return buffer;
}

id<MTLComputePipelineState> Pipeline(id<MTLDevice> device, const std::string& source, const std::string& name) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_1;
    if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeFast;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
    if (library == nil) {
        std::fprintf(stderr, "%s\n", source.c_str());
        throw std::runtime_error(name + ": the MSL source does not compile: " + Describe(error));
    }
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:std::string(MslEntryPoint).c_str()] constantValues:constants error:&error];
    Require(function != nil, name + ": the library has no specialized entry point: " + Describe(error));
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    Require(pipeline != nil, name + ": the compute pipeline does not build: " + Describe(error));
    return pipeline;
}

std::uint32_t BindingIndex(IrTextureNumericClass numeric, ImageResourceClass resourceClass, bool atomic) {
    ImageResource image{};
    image.resourceClass = resourceClass;
    image.numericClass = numeric;
    image.dimension = RdnaImageDimension::Dim2D;
    image.atomic = atomic;
    return MetalAbi::GuestImageIndex(static_cast<std::uint32_t>(DescriptorBindingForImage(image)), 0u);
}

id<MTLTexture> MakeTexture(id<MTLDevice> device, Kind kind) {
    const bool storage = kind == Kind::Store || kind == Kind::AtomicAdd;
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kind == Kind::AtomicAdd ? MTLPixelFormatR32Uint : MTLPixelFormatRGBA8Unorm width:MslImagePrograms::Width height:1 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = kind == Kind::AtomicAdd ? (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageShaderAtomic) : storage ? (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite) : MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    Require(texture != nil, "could not create the test texture");
    if (!storage) {
        std::vector<std::uint8_t> texels(MslImagePrograms::Width * 4u);
        for (std::uint32_t x = 0; x < MslImagePrograms::Width; ++x) {
            texels[x * 4u + 0u] = static_cast<std::uint8_t>(x);
            texels[x * 4u + 1u] = static_cast<std::uint8_t>(x * 2u);
            texels[x * 4u + 2u] = static_cast<std::uint8_t>(x * 3u);
            texels[x * 4u + 3u] = 255u;
        }
        [texture replaceRegion:MTLRegionMake2D(0, 0, MslImagePrograms::Width, 1) mipmapLevel:0 withBytes:texels.data() bytesPerRow:texels.size()];
    } else {
        std::vector<std::uint32_t> texels(MslImagePrograms::Width, 0u);
        [texture replaceRegion:MTLRegionMake2D(0, 0, MslImagePrograms::Width, 1) mipmapLevel:0 withBytes:texels.data() bytesPerRow:texels.size() * sizeof(std::uint32_t)];
    }
    return texture;
}

void Run(id<MTLDevice> device, id<MTLCommandQueue> queue, const MslImagePrograms::Program& program, std::uint64_t address) {
    const auto name = std::string(program.name);
    const auto userData = MslImagePrograms::UserData(program.format);
    const auto result = Recompile(MslImagePrograms::Request(program, address, userData));
    Require(result.backend == ShaderBackendKind::Msl, name + ": the result is not an MSL artifact");
    const auto source = std::string(result.msl.Source());
    id<MTLComputePipelineState> pipeline = Pipeline(device, source, name);
    if (program.kind == Kind::CompileOnly) return;

    id<MTLTexture> texture = MakeTexture(device, program.kind);
    id<MTLBuffer> output = Zeroed(device, OutputBytes);
    id<MTLBuffer> push = Zeroed(device, std::max<std::size_t>(result.pushConstants.size(), 16u));
    if (!result.pushConstants.empty()) std::memcpy(push.contents, result.pushConstants.data(), result.pushConstants.size());
    id<MTLBuffer> buffers = Zeroed(device, sizeof(MetalAbi::GuestBufferEntry));
    const MetalAbi::GuestBufferEntry entry{output.gpuAddress, OutputBytes, 0u};
    std::memcpy(buffers.contents, &entry, sizeof(entry));

    id<MTLBuffer> images = Zeroed(device, MetalAbi::GuestImageTableBytes);
    const bool storage = program.kind == Kind::Store || program.kind == Kind::AtomicAdd;
    const auto imageIndex = program.kind == Kind::AtomicAdd ? BindingIndex(IrTextureNumericClass::Uint, ImageResourceClass::Storage, true) : storage ? BindingIndex(IrTextureNumericClass::Float, ImageResourceClass::Storage, false) : BindingIndex(IrTextureNumericClass::Float, ImageResourceClass::Sampled, false);
    const MTLResourceID textureId = texture.gpuResourceID;
    std::memcpy(static_cast<std::uint8_t*>(images.contents) + imageIndex * sizeof(std::uint64_t), &textureId, sizeof(textureId));

    MTLSamplerDescriptor* samplerDescriptor = [MTLSamplerDescriptor new];
    samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
    samplerDescriptor.magFilter = MTLSamplerMinMagFilterNearest;
    samplerDescriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDescriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDescriptor.supportArgumentBuffers = YES;
    id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:samplerDescriptor];
    id<MTLBuffer> samplers = Zeroed(device, MetalAbi::GuestSamplerTableBytes);
    const MTLResourceID samplerId = sampler.gpuResourceID;
    std::memcpy(samplers.contents, &samplerId, sizeof(samplerId));
    std::memcpy(static_cast<std::uint8_t*>(samplers.contents) + sizeof(std::uint64_t), &samplerId, sizeof(samplerId));

    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:push offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::PushConstants)];
    [encoder setBuffer:buffers offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestBuffers)];
    [encoder setBuffer:images offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestImages)];
    [encoder setBuffer:samplers offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestSamplers)];
    [encoder useResource:output usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder useResource:texture usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(MslImagePrograms::Threads, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    Require(commands.status == MTLCommandBufferStatusCompleted, name + ": the dispatch failed: " + Describe(commands.error));

    const auto* values = static_cast<const std::uint32_t*>(output.contents);
    switch (program.kind) {
    case Kind::Load:
    case Kind::Sample:
        for (std::uint32_t lane = 0; lane < MslImagePrograms::Threads; ++lane) {
            float red;
            std::memcpy(&red, &values[lane], sizeof(red));
            Require(std::fabs(red - MslImagePrograms::LoadedRed(lane)) < 1e-6f, name + ": thread " + std::to_string(lane) + " read " + std::to_string(red) + ", expected " + std::to_string(MslImagePrograms::LoadedRed(lane)));
        }
        break;
    case Kind::ResInfo:
        for (std::uint32_t lane = 0; lane < MslImagePrograms::Threads; ++lane) Require(values[lane] == MslImagePrograms::Width, name + ": thread " + std::to_string(lane) + " saw width " + std::to_string(values[lane]));
        break;
    case Kind::Store:
    case Kind::AtomicAdd: {
        std::vector<std::uint32_t> texels(MslImagePrograms::Width, 0u);
        [texture getBytes:texels.data() bytesPerRow:texels.size() * sizeof(std::uint32_t) fromRegion:MTLRegionMake2D(0, 0, MslImagePrograms::Width, 1) mipmapLevel:0];
        for (std::uint32_t x = 0; x < MslImagePrograms::Width; ++x) {
            const std::uint32_t expected = program.kind == Kind::Store ? MslImagePrograms::StoredTexel : (x < 4u ? MslImagePrograms::Threads / 4u : 0u);
            Require(texels[x] == expected, name + ": texel " + std::to_string(x) + " holds " + std::to_string(texels[x]) + ", expected " + std::to_string(expected));
        }
        break;
    }
    case Kind::CompileOnly: break;
    }
}

}

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            Require(device != nil, "no Metal device");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            Require(queue != nil, "no Metal command queue");
            std::uint64_t address = 0x90000u;
            for (const auto& program : MslImagePrograms::Programs()) {
                Run(device, queue, program, address);
                std::printf("MslImagesMetal: %s ok\n", std::string(program.name).c_str());
                address += 0x1000u;
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "MslImagesMetal: %s\n", error.what());
            return 1;
        }
    }
    std::puts("MslImagesMetal: every MSL image kernel compiles and the executed ones read, write, sample and count correctly");
    return 0;
}
