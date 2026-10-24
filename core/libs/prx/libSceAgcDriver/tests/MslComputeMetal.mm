#import <Metal/Metal.h>
#include "MslComputePrograms.hpp"
#include "BdaAbi.hpp"
#include "MetalAbi.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint32_t OutputBytes = MslComputePrograms::Threads * sizeof(std::uint32_t);

struct BdaTable {
    BdaAbi::Header header;
    BdaAbi::Range range;
};

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
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:std::string(MslEntryPoint).c_str()]];
    Require(function != nil, name + ": the library has no entry point");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    Require(pipeline != nil, name + ": the compute pipeline does not build: " + Describe(error));
    Require(pipeline.threadExecutionWidth == 32u, name + ": the pipeline does not run 32-wide SIMD-groups");
    return pipeline;
}

double Run(id<MTLDevice> device, id<MTLCommandQueue> queue, const MslComputePrograms::Program& program, std::uint64_t address, std::uint32_t waveSize) {
    const auto name = std::string(program.name) + " (wave" + std::to_string(waveSize) + ")";
    const auto userData = MslComputePrograms::UserData();
    const auto result = Recompile(MslComputePrograms::Request(program, address, userData, waveSize));
    Require(result.backend == ShaderBackendKind::Msl, name + ": the result is not an MSL artifact");
    const auto source = std::string(result.msl.Source());
    const auto compileStart = std::chrono::steady_clock::now();
    id<MTLComputePipelineState> pipeline = Pipeline(device, source, name);
    const auto compileMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compileStart).count();

    id<MTLBuffer> output = Zeroed(device, OutputBytes);
    id<MTLBuffer> push = Zeroed(device, std::max<std::size_t>(result.pushConstants.size(), 16u));
    if (!result.pushConstants.empty()) std::memcpy(push.contents, result.pushConstants.data(), result.pushConstants.size());
    id<MTLBuffer> fault = Zeroed(device, BdaAbi::FaultBufferBytes);
    id<MTLBuffer> buffers = Zeroed(device, sizeof(MetalAbi::GuestBufferEntry));
    const MetalAbi::GuestBufferEntry entry{output.gpuAddress, OutputBytes, 0u};
    std::memcpy(buffers.contents, &entry, sizeof(entry));
    id<MTLBuffer> table = Zeroed(device, sizeof(BdaTable));
    const BdaTable bda{{BdaAbi::Version, 1u, sizeof(BdaAbi::Range), 0u}, {MslComputePrograms::BdaGuestBase, MslComputePrograms::BdaGuestBase + OutputBytes, output.gpuAddress, BdaAbi::Read | BdaAbi::Write, 0u}};
    std::memcpy(table.contents, &bda, sizeof(bda));

    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:push offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::PushConstants)];
    [encoder setBuffer:table offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::BdaPagetable)];
    [encoder setBuffer:fault offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::FaultBuffer)];
    [encoder setBuffer:buffers offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestBuffers)];
    [encoder useResource:output usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(MslComputePrograms::HostThreads(waveSize), 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    Require(commands.status == MTLCommandBufferStatusCompleted, name + ": the dispatch failed: " + Describe(commands.error));

    const auto* faultWords = static_cast<const std::uint32_t*>(fault.contents);
    Require(faultWords[0] == 0u, name + ": the kernel recorded fault reason " + std::to_string(faultWords[1]));
    const auto* values = static_cast<const std::uint32_t*>(output.contents);
    for (std::uint32_t lane = 0; lane < MslComputePrograms::Threads; ++lane) {
        const auto expected = MslComputePrograms::Expected(program.expect, lane);
        Require(values[lane] == expected, name + ": thread " + std::to_string(lane) + " wrote " + std::to_string(values[lane]) + ", expected " + std::to_string(expected));
    }
    if (program.bda) Require(faultWords[BdaAbi::WrittenCountWord] != 0u, name + ": the BDA store did not note its written page");
    return compileMs;
}

}

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            Require(device != nil, "no Metal device");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            Require(queue != nil, "no Metal command queue");
            std::uint64_t address = 0x80000u;
            for (const auto& program : MslComputePrograms::Programs()) {
                for (const auto waveSize : {32u, 64u}) {
                    if ((program.waves & (waveSize == 32u ? MslComputePrograms::Wave32 : MslComputePrograms::Wave64)) == 0u) continue;
                    const auto compileMs = Run(device, queue, program, address, waveSize);
                    std::printf("MslComputeMetal: %s wave%u ok (Metal compile %.1f ms)\n", std::string(program.name).c_str(), waveSize, compileMs);
                    address += 0x1000u;
                }
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "MslComputeMetal: %s\n", error.what());
            return 1;
        }
    }
    std::puts("MslComputeMetal: every MSL kernel compiles, runs 32-wide and writes the expected values");
    return 0;
}
