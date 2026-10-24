#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

struct Probe {
    const char* name;
    const char* body;
    const char* parameter;
};

void Run(id<MTLDevice> device, id<MTLCommandQueue> queue, const Probe& probe) {
    std::string source = std::string("#include <metal_stdlib>\nusing namespace metal;\nkernel void probe(") + probe.parameter + ", uint tid [[thread_position_in_threadgroup]]) {\n" + probe.body + "\n}\n";
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_1;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
    if (library == nil) {
        std::printf("probe %s: compile failed: %s\n", probe.name, error.localizedDescription.UTF8String);
        return;
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
    if (pipeline == nil) {
        std::printf("probe %s: pipeline failed: %s\n", probe.name, error.localizedDescription.UTF8String);
        return;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    std::memset(buffer.contents, 0, 256);
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    const auto* words = static_cast<const unsigned*>(buffer.contents);
    std::printf("probe %s: status %ld words %u %u %u %u %u %u %u %u\n", probe.name, static_cast<long>(commands.status), words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
}

}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        std::printf("probe device %s\n", device.name.UTF8String);
        const Probe probes[] = {
            {"max on atomic_ulong buffer", "atomic_max_explicit(&out[0], ulong(tid) | (1ul << 32), memory_order_relaxed);", "device atomic_ulong* out [[buffer(0)]]"},
            {"max via cast from ulong buffer", "atomic_max_explicit(reinterpret_cast<device atomic_ulong*>(out) + 1, ulong(tid) | (1ul << 32), memory_order_relaxed);", "device ulong* out [[buffer(0)]]"},
            {"max via cast from atomic_uint buffer", "atomic_max_explicit(reinterpret_cast<device atomic_ulong*>(out + 4), ulong(tid) | (1ul << 32), memory_order_relaxed);", "device atomic_uint* out [[buffer(0)]]"},
            {"min on atomic_ulong buffer", "atomic_store_explicit(&out[0], ~0ul, memory_order_relaxed); threadgroup_barrier(mem_flags::mem_device); atomic_min_explicit(&out[0], ulong(tid) + 5ul, memory_order_relaxed);", "device atomic_ulong* out [[buffer(0)]]"},
            {"store and load on atomic_ulong buffer", "atomic_store_explicit(&out[tid & 1u], 7ul + tid, memory_order_relaxed); threadgroup_barrier(mem_flags::mem_device); out[2] = atomic_load_explicit(&out[0], memory_order_relaxed);", "device atomic_ulong* out [[buffer(0)]]"},
            {"compare exchange on atomic_ulong buffer", "ulong expected = 0ul; atomic_compare_exchange_weak_explicit(&out[0], &expected, 9ul, memory_order_relaxed, memory_order_relaxed);", "device atomic_ulong* out [[buffer(0)]]"},
            {"fetch add on atomic_ulong buffer", "atomic_fetch_add_explicit(&out[0], 3ul, memory_order_relaxed);", "device atomic_ulong* out [[buffer(0)]]"},
            {"exchange on atomic_ulong buffer", "atomic_exchange_explicit(&out[0], 11ul, memory_order_relaxed);", "device atomic_ulong* out [[buffer(0)]]"},
        };
        for (const auto& probe : probes) Run(device, queue, probe);
    }
    return 0;
}
