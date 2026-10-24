#import <Metal/Metal.h>
#include "MslPrecisionPrograms.hpp"
#include "BdaAbi.hpp"
#include "MetalAbi.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint32_t OutputBytes = MslPrecisionPrograms::Threads * sizeof(std::uint32_t);
constexpr std::uint32_t TimingGroups = 4096u;
constexpr std::uint32_t TimingRuns = 5u;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Describe(NSError* error) {
    return error == nil ? std::string("unknown error") : std::string(error.localizedDescription.UTF8String);
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

id<MTLBuffer> Zeroed(id<MTLDevice> device, std::size_t bytes) {
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    Require(buffer != nil, "could not allocate a shared buffer");
    std::memset(buffer.contents, 0, bytes);
    return buffer;
}

id<MTLComputePipelineState> Pipeline(id<MTLDevice> device, const std::string& source, const std::string& entry, const std::string& name) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_1;
    if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeFast;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
    if (library == nil) {
        std::fprintf(stderr, "%s\n", source.c_str());
        throw std::runtime_error(name + ": the MSL source does not compile: " + Describe(error));
    }
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:entry.c_str()]];
    Require(function != nil, name + ": the library has no entry point");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    Require(pipeline != nil, name + ": the compute pipeline does not build: " + Describe(error));
    return pipeline;
}

double Run(id<MTLDevice> device, id<MTLCommandQueue> queue, const MslPrecisionPrograms::Program& program, std::uint64_t address) {
    const auto name = std::string(program.name);
    const auto compute = MslPrecisionPrograms::Compute(program);
    const auto result = Recompile(MslPrecisionPrograms::Request(program, compute, address));
    Require(result.backend == ShaderBackendKind::Msl, name + ": the result is not an MSL artifact");
    id<MTLComputePipelineState> pipeline = Pipeline(device, std::string(result.msl.Source()), std::string(MslEntryPoint), name);
    Require(pipeline.threadExecutionWidth == 32u, name + ": the pipeline does not run 32-wide SIMD-groups");

    id<MTLBuffer> output = Zeroed(device, OutputBytes);
    id<MTLBuffer> push = Zeroed(device, std::max<std::size_t>(result.pushConstants.size(), 16u));
    if (!result.pushConstants.empty()) std::memcpy(push.contents, result.pushConstants.data(), result.pushConstants.size());
    id<MTLBuffer> fault = Zeroed(device, BdaAbi::FaultBufferBytes);
    id<MTLBuffer> buffers = Zeroed(device, sizeof(MetalAbi::GuestBufferEntry));
    const MetalAbi::GuestBufferEntry entry{output.gpuAddress, OutputBytes, 0u};
    std::memcpy(buffers.contents, &entry, sizeof(entry));

    const bool timing = program.check == MslPrecisionPrograms::Check::Timing;
    const std::uint32_t runs = timing ? TimingRuns : 1u;
    const std::uint32_t groups = timing ? TimingGroups : 1u;
    double best = 0.0;
    std::vector<std::uint32_t> inputs(MslPrecisionPrograms::Threads);
    for (std::uint32_t lane = 0; lane < MslPrecisionPrograms::Threads; ++lane) inputs[lane] = MslPrecisionPrograms::Input(program.check, lane);
    for (std::uint32_t run = 0; run < runs; ++run) {
        std::memcpy(output.contents, inputs.data(), OutputBytes);
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:push offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::PushConstants)];
        [encoder setBuffer:fault offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::FaultBuffer)];
        [encoder setBuffer:buffers offset:0 atIndex:MetalAbi::SlotIndex(MetalAbi::BufferSlot::GuestBuffers)];
        [encoder useResource:output usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(MslPrecisionPrograms::Threads, 1, 1)];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        Require(commands.status == MTLCommandBufferStatusCompleted, name + ": the dispatch failed: " + Describe(commands.error));
        const double ms = (commands.GPUEndTime - commands.GPUStartTime) * 1000.0;
        best = run == 0u ? ms : std::min(best, ms);
    }

    const auto* faultWords = static_cast<const std::uint32_t*>(fault.contents);
    Require(faultWords[0] == 0u, name + ": the kernel recorded fault reason " + std::to_string(faultWords[1]));
    const auto* values = static_cast<const std::uint32_t*>(output.contents);
    for (std::uint32_t lane = 0; lane < MslPrecisionPrograms::Threads; ++lane) {
        Require(MslPrecisionPrograms::Matches(program.check, inputs[lane], values[lane]), name + ": input " + Hex(inputs[lane]) + " gave " + Hex(values[lane]));
    }
    return best;
}

constexpr const char* DenormalProbe = R"(#include <metal_stdlib>
using namespace metal;
kernel void probe_fast(device uint* data [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    data[index] = as_type<uint>(as_type<float>(data[index]) * as_type<float>(data[4]));
}
kernel void probe_safe(device uint* data [[buffer(0)]], uint index [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
    data[index] = as_type<uint>(as_type<float>(data[index]) * as_type<float>(data[4]));
}
)";

void ProbeDenormals(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    for (const char* entry : {"probe_fast", "probe_safe"}) {
        id<MTLComputePipelineState> pipeline = Pipeline(device, DenormalProbe, entry, entry);
        id<MTLBuffer> data = Zeroed(device, 8u * sizeof(std::uint32_t));
        const std::uint32_t words[8] = {0x00000001u, 0x007fffffu, 0x00800000u, 0x80000010u, 0x3f800000u, 0u, 0u, 0u};
        std::memcpy(data.contents, words, sizeof(words));
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:data offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        const auto* values = static_cast<const std::uint32_t*>(data.contents);
        const bool kept = values[0] == words[0] && values[1] == words[1] && values[3] == words[3];
        std::printf("MslPrecisionMetal: %s keeps fp32 denormals: %s (%s %s %s)\n", entry, kept ? "yes" : "no", Hex(values[0]).c_str(), Hex(values[1]).c_str(), Hex(values[3]).c_str());
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
            ProbeDenormals(device, queue);
            std::uint64_t address = 0x90000u;
            for (const auto& program : MslPrecisionPrograms::Programs()) {
                const auto ms = Run(device, queue, program, address);
                if (program.check == MslPrecisionPrograms::Check::Timing) {
                    std::printf("MslPrecisionMetal: %s ok (best GPU time %.3f ms for %u x 64 threads)\n", std::string(program.name).c_str(), ms, TimingGroups);
                } else {
                    std::printf("MslPrecisionMetal: %s ok\n", std::string(program.name).c_str());
                }
                address += 0x1000u;
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "MslPrecisionMetal: %s\n", error.what());
            return 1;
        }
    }
    std::puts("MslPrecisionMetal: NaN, fp64 and dot2 programs match the GCN reference on the GPU");
    return 0;
}
