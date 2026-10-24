#include "prx/libSceAgcDriver/Graphics/Metal/TextureDetiler.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "TextureKernelsSource.inc"
#include <algorithm>
#include <atomic>
#include <chrono>
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
constexpr std::size_t MaxWarmReplay = 48;
constexpr std::uint32_t StallTimeoutMs = 60000;
constexpr const char* WarmPrefix = "detile";
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
    MetalCompiler& compiler;
    LibrarySource source;
    id<MTLDevice> native = nil;
    id<MTL4ArgumentTable> table = nil;
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<CompileTicket>> pipelines;
    std::vector<std::unique_ptr<Chunk>> chunks;
    Chunk* open = nullptr;
    std::vector<Chunk*> sealing;
    MTLSize group = MTLSizeMake(32, 8, 1);
    DetileCounters counters;
    std::atomic<std::uint64_t> stalls{0};
    std::atomic<std::uint64_t> stallNanos{0};

    State(Device& device, MetalCompiler& compiler) : device(device), compiler(compiler) {}

    ~State() {
        for (auto& chunk : chunks) device.ReleaseBuffer(chunk->buffer);
    }

    static std::uint64_t variantKey(DetileKernel kernel, std::uint32_t elementBytes, TextureTransform transform, std::uint32_t texelBytes) {
        return static_cast<std::uint64_t>(kernel) | (static_cast<std::uint64_t>(elementBytes) << 8) | (static_cast<std::uint64_t>(transform) << 16) | (static_cast<std::uint64_t>(texelBytes) << 24);
    }

    std::shared_ptr<CompileTicket> request(DetileKernel kernel, std::uint32_t elementBytes, TextureTransform transform, std::uint32_t texelBytes, CompilePriority priority, bool record) {
        const auto key = variantKey(kernel, elementBytes, transform, texelBytes);
        std::lock_guard lock(mutex);
        auto found = pipelines.find(key);
        if (found != pipelines.end() && (found->second->Ready() || found->second->Priority() >= priority)) return found->second;
        ComputePipelineRequest build;
        build.library = source;
        build.function = KernelNames[static_cast<std::size_t>(kernel)];
        build.constants = {{0, elementBytes, false}, {1, static_cast<std::uint32_t>(transform), false}, {2, texelBytes, false}};
        build.label = std::string("AnyPS5 ") + build.function;
        build.priority = priority;
        auto ticket = compiler.Compute(build);
        if (found == pipelines.end()) {
            pipelines.emplace(key, ticket);
            ++counters.pipelines;
            if (record) compiler.RecordWarm(WarmPrefix, std::to_string(key));
        } else {
            found->second = ticket;
        }
        return ticket;
    }

    void replayJournal() {
        std::size_t budget = MaxWarmReplay;
        for (const auto& entry : compiler.WarmEntries(WarmPrefix)) {
            if (budget == 0) break;
            char* end = nullptr;
            const auto key = std::strtoull(entry.c_str(), &end, 10);
            if (end == entry.c_str() || *end != '\0') continue;
            const auto kernel = static_cast<std::uint32_t>(key & 0xffu);
            if (kernel >= static_cast<std::uint32_t>(DetileKernel::Count)) continue;
            const auto transform = static_cast<std::uint32_t>((key >> 16) & 0xffu);
            request(static_cast<DetileKernel>(kernel), static_cast<std::uint32_t>((key >> 8) & 0xffu), static_cast<TextureTransform>(transform), static_cast<std::uint32_t>((key >> 24) & 0xffu), CompilePriority::Prefetch, false);
            --budget;
        }
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

std::unique_ptr<TextureDetiler> TextureDetiler::Create(Device& device, MetalCompiler& compiler, std::string* failure) {
    @autoreleasepool {
        auto state = std::make_unique<State>(device, compiler);
        state->native = (__bridge id<MTLDevice>)device.NativeDevice();
        state->source.text = kTextureKernelsSource;
        state->source.name = "AnyPS5 texture kernels";
        state->source.fastMath = true;
        state->source.key = CompileKeyBuilder(0x54455830ull).AddString(state->source.text).Finish();
        NSError* error = nil;
        MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 1;
        descriptor.maxTextureBindCount = 1;
        state->table = [state->native newArgumentTableWithDescriptor:descriptor error:&error];
        if (state->table == nil) {
            if (failure != nullptr) *failure = "texture argument table";
            return nullptr;
        }
        state->group = parseGroup(std::getenv("APS5_METAL_DETILE_GROUP"), state->group);
        state->replayJournal();
        return std::unique_ptr<TextureDetiler>(new TextureDetiler(std::move(state)));
    }
}

TextureDetiler::TextureDetiler(std::unique_ptr<State> state) : state(std::move(state)) {}

TextureDetiler::~TextureDetiler() = default;

void TextureDetiler::Encode(void* computeEncoder, DetileKernel kernel, std::uint32_t elementBytes, TextureTransform transform, std::uint32_t texelBytes, const AgcDetileParams& params, void* texture, std::uint32_t columns, std::uint32_t rows) {
    auto& s = *state;
    if (columns == 0 || rows == 0) return;
    auto ticket = s.request(kernel, elementBytes, transform, texelBytes, CompilePriority::Urgent, true);
    if (!ticket->Ready()) {
        const auto started = std::chrono::steady_clock::now();
        ticket->Wait(StallTimeoutMs);
        s.stalls.fetch_add(1, std::memory_order_relaxed);
        s.stallNanos.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    }
    if (!ticket->Ready()) throw std::runtime_error("AGC graphics: Metal texture detile pipeline failed to build: " + (ticket->Done() ? ticket->Error() : std::string("timed out")));
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)ticket->ProductAs<CompiledPipeline>()->native;
    std::lock_guard lock(s.mutex);
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

void TextureDetiler::Warm(DetileKernel kernel, std::uint32_t elementBytes, TextureTransform transform, std::uint32_t texelBytes) {
    state->request(kernel, elementBytes, transform, texelBytes, CompilePriority::Needed, true);
}

DetileCounters TextureDetiler::Counters() const {
    std::lock_guard lock(state->mutex);
    auto counters = state->counters;
    counters.stalls = state->stalls.load(std::memory_order_relaxed);
    counters.stallNanos = state->stallNanos.load(std::memory_order_relaxed);
    return counters;
}

}
