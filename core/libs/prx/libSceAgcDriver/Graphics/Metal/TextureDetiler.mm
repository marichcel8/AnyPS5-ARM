#include "prx/libSceAgcDriver/Graphics/Metal/TextureDetiler.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "TextureKernelsSource.inc"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace AgcDriver::Graphics::Metal {

static_assert(sizeof(AgcDetileParams) == 160);

namespace {

constexpr std::uint64_t ChunkBytes = 256 * 1024;
constexpr std::uint64_t ParamsAlign = 16;

constexpr const char* KernelNames[] = {
    "aps5_tiled_to_array2d",
    "aps5_tiled_to_array1d",
    "aps5_tiled_to_volume",
    "aps5_array2d_to_tiled",
    "aps5_volume_to_tiled",
    "aps5_tiled_to_linear",
    "aps5_linear_to_tiled",
};
static_assert(sizeof(KernelNames) / sizeof(KernelNames[0]) == static_cast<std::size_t>(DetileKernel::Count));

bool usesTexture(DetileKernel kernel) {
    return kernel != DetileKernel::TiledToLinear && kernel != DetileKernel::LinearToTiled;
}

struct Chunk {
    NativeBuffer buffer;
    std::uint64_t used = 0;
    QueueKind queue = QueueKind::Compute;
    std::uint64_t serial = 0;
    bool inflight = false;
};

MTLSize parseGroup(const char* text, MTLSize fallback) {
    if (text == nullptr) return fallback;
    unsigned width = 0, height = 0;
    if (std::sscanf(text, "%ux%u", &width, &height) != 2 || width == 0 || height == 0) return fallback;
    return MTLSizeMake(width, height, 1);
}

}

struct TextureDetiler::State {
    Device& device;
    id<MTLDevice> native = nil;
    id<MTLLibrary> library = nil;
    id<MTL4ArgumentTable> table = nil;
    std::mutex mutex;
    std::unordered_map<std::uint64_t, id<MTLComputePipelineState>> pipelines;
    std::vector<std::unique_ptr<Chunk>> chunks;
    Chunk* open = nullptr;
    std::vector<Chunk*> sealing;
    MTLSize group = MTLSizeMake(32, 8, 1);
    DetileCounters counters;

    explicit State(Device& device) : device(device) {}

    ~State() {
        for (auto& chunk : chunks) device.ReleaseBuffer(chunk->buffer);
    }

    id<MTLComputePipelineState> pipeline(DetileKernel kernel, std::uint32_t elementBytes, TextureTransform transform, std::uint32_t texelBytes) {
        const std::uint64_t key = static_cast<std::uint64_t>(kernel) | (static_cast<std::uint64_t>(elementBytes) << 8) | (static_cast<std::uint64_t>(transform) << 16) | (static_cast<std::uint64_t>(texelBytes) << 24);
        const auto found = pipelines.find(key);
        if (found != pipelines.end()) return found->second;
        MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
        const std::uint32_t values[3] = {elementBytes, static_cast<std::uint32_t>(transform), texelBytes};
        for (NSUInteger index = 0; index < 3; ++index) [constants setConstantValue:&values[index] type:MTLDataTypeUInt atIndex:index];
        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:KernelNames[static_cast<std::size_t>(kernel)]] constantValues:constants error:&error];
        if (function == nil) {
            std::fprintf(stderr, "AGC Metal: texture kernel %s: %s\n", KernelNames[static_cast<std::size_t>(kernel)], error.localizedDescription.UTF8String);
            return nil;
        }
        id<MTLComputePipelineState> state = [native newComputePipelineStateWithFunction:function error:&error];
        if (state == nil) {
            std::fprintf(stderr, "AGC Metal: texture pipeline %s: %s\n", KernelNames[static_cast<std::size_t>(kernel)], error.localizedDescription.UTF8String);
            return nil;
        }
        pipelines.emplace(key, state);
        ++counters.pipelines;
        return state;
    }

    Chunk* acquire() {
        for (auto& chunk : chunks) {
            if (!chunk->inflight) continue;
            if (device.CompletedSerial(chunk->queue) < chunk->serial) continue;
            chunk->inflight = false;
            chunk->used = 0;
            return chunk.get();
        }
        auto chunk = std::make_unique<Chunk>();
        chunk->buffer = device.NewSharedBuffer(ChunkBytes, "AnyPS5 texture constants");
        if (!chunk->buffer) return nullptr;
        ++counters.arenaChunks;
        counters.arenaBytes += ChunkBytes;
        chunks.push_back(std::move(chunk));
        return chunks.back().get();
    }

    std::uint64_t allocate(const AgcDetileParams& params) {
        if (open == nullptr || open->used + sizeof(AgcDetileParams) > ChunkBytes) {
            if (open != nullptr) sealing.push_back(open);
            open = acquire();
            if (open == nullptr) return 0;
        }
        const auto offset = open->used;
        std::memcpy(static_cast<std::uint8_t*>(open->buffer.contents) + offset, &params, sizeof(params));
        open->used = (offset + sizeof(params) + ParamsAlign - 1) & ~(ParamsAlign - 1);
        return open->buffer.gpuAddress + offset;
    }
};

std::unique_ptr<TextureDetiler> TextureDetiler::Create(Device& device, std::string* failure) {
    @autoreleasepool {
        auto state = std::make_unique<State>(device);
        state->native = (__bridge id<MTLDevice>)device.NativeDevice();
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        state->library = [state->native newLibraryWithSource:[NSString stringWithUTF8String:kTextureKernelsSource] options:options error:&error];
        if (state->library == nil) {
            if (failure != nullptr) *failure = std::string("texture kernels: ") + error.localizedDescription.UTF8String;
            return nullptr;
        }
        MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 1;
        descriptor.maxTextureBindCount = 1;
        state->table = [state->native newArgumentTableWithDescriptor:descriptor error:&error];
        if (state->table == nil) {
            if (failure != nullptr) *failure = "texture argument table";
            return nullptr;
        }
        state->group = parseGroup(std::getenv("APS5_METAL_DETILE_GROUP"), state->group);
        return std::unique_ptr<TextureDetiler>(new TextureDetiler(std::move(state)));
    }
}

TextureDetiler::TextureDetiler(std::unique_ptr<State> state) : state(std::move(state)) {}

TextureDetiler::~TextureDetiler() = default;

void TextureDetiler::Encode(void* computeEncoder, DetileKernel kernel, std::uint32_t elementBytes, TextureTransform transform, std::uint32_t texelBytes, const AgcDetileParams& params, void* texture, std::uint32_t columns, std::uint32_t rows) {
    auto& s = *state;
    if (columns == 0 || rows == 0) return;
    std::lock_guard lock(s.mutex);
    id<MTLComputePipelineState> pipeline = s.pipeline(kernel, elementBytes, transform, texelBytes);
    if (pipeline == nil) throw std::runtime_error("AGC graphics: Metal texture detile pipeline failed to build");
    const auto address = s.allocate(params);
    if (address == 0) throw std::runtime_error("AGC graphics: Metal texture constants could not be allocated");
    id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)computeEncoder;
    [encoder setComputePipelineState:pipeline];
    [s.table setAddress:address atIndex:0];
    if (usesTexture(kernel)) {
        id<MTLTexture> view = (__bridge id<MTLTexture>)texture;
        [s.table setTexture:view.gpuResourceID atIndex:0];
    }
    [encoder setArgumentTable:s.table];
    [encoder dispatchThreads:MTLSizeMake(columns, rows, 1) threadsPerThreadgroup:s.group];
    ++s.counters.dispatches;
}

void TextureDetiler::Seal(QueueKind queue, std::uint64_t serial) {
    auto& s = *state;
    std::lock_guard lock(s.mutex);
    if (s.open != nullptr) s.sealing.push_back(s.open);
    s.open = nullptr;
    for (auto* chunk : s.sealing) {
        chunk->inflight = true;
        chunk->queue = queue;
        chunk->serial = serial;
    }
    s.sealing.clear();
}

DetileCounters TextureDetiler::Counters() const {
    std::lock_guard lock(state->mutex);
    return state->counters;
}

}
