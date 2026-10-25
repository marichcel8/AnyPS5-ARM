#include "prx/libSceAgcDriver/Graphics/Metal/MetalCompiler.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/sysctl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr std::uint32_t CompileCacheVersion = 2;
constexpr MTLLanguageVersion DefaultLanguageVersion = MTLLanguageVersion3_2;
constexpr std::uint64_t PipelineDomain = 0x4d504c31ull;
constexpr std::uint64_t FlushDomain = 0x4d464c31ull;

std::uint64_t nanosSince(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

std::string osBuild() {
    char buffer[64] = {};
    std::size_t size = sizeof(buffer);
    if (sysctlbyname("kern.osversion", buffer, &size, nullptr, 0) != 0) return "unknown";
    return buffer;
}

bool cacheDisabled() {
    const char* value = std::getenv("ANYPS5_NO_SHADER_CACHE");
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

std::filesystem::path defaultRoot() {
    const char* directory = std::getenv("ANYPS5_SHADER_CACHE_DIR");
    if (directory != nullptr && *directory != '\0') return std::filesystem::path(directory) / "metal";
    NSArray<NSString*>* caches = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
    if (caches.count == 0) return {};
    return std::filesystem::path(caches.firstObject.UTF8String) / "AnyPS5" / "metal";
}

NSString* nsString(std::string_view text) {
    return [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding] ?: @"";
}

struct LibraryEntry {
    std::mutex mutex;
    bool built = false;
    id<MTLLibrary> library = nil;
    std::string error;
    std::uint64_t stamp = 0;
};

}

struct MetalCompiler::State {
    Device& device;
    id<MTLDevice> native = nil;
    CompilerConfig config;
    std::unique_ptr<CompileCacheDirectory> directory;
    std::unique_ptr<WarmJournal> journal;
    id<MTL4PipelineDataSetSerializer> serializer = nil;
    NSArray<id<MTL4Archive>>* archives = nil;
    MTL4CompilerTaskOptions* taskOptions = nil;
    std::shared_ptr<CompileBackend> backend;
    std::unique_ptr<ShaderCompileService> service;

    std::mutex libraryMutex;
    std::unordered_map<CompileKey, std::shared_ptr<LibraryEntry>, CompileKeyHash> libraries;
    std::uint64_t libraryClock = 0;

    std::mutex flushMutex;
    std::filesystem::path sessionShard;
    std::mutex keysMutex;
    std::unordered_set<CompileKey, CompileKeyHash> sessionKeys;
    std::atomic<std::uint64_t> shardsCompacted{0};
    std::atomic<std::uint64_t> newSinceFlush{0};
    std::atomic<std::uint64_t> nextFlushAt{0};
    std::atomic<bool> flushQueued{false};
    std::atomic<std::uint64_t> flushSerial{0};

    std::atomic<std::uint64_t> librariesBuilt{0};
    std::atomic<std::uint64_t> libraryHits{0};
    std::atomic<std::uint64_t> libraryNanos{0};
    std::atomic<std::uint64_t> pipelinesBuilt{0};
    std::atomic<std::uint64_t> pipelinesFromArchive{0};
    std::atomic<std::uint64_t> pipelineNanos{0};
    std::atomic<std::uint64_t> flushes{0};
    std::atomic<std::uint64_t> flushNanos{0};
    std::uint64_t shardsLoaded = 0;
    std::uint64_t shardsRejected = 0;

    explicit State(Device& device) : device(device) {}

    std::shared_ptr<void> library(const LibrarySource& source, id<MTL4Compiler> compiler, std::string* error) {
        std::shared_ptr<LibraryEntry> entry;
        {
            std::lock_guard lock(libraryMutex);
            auto& slot = libraries[source.key];
            if (slot == nullptr) slot = std::make_shared<LibraryEntry>();
            slot->stamp = ++libraryClock;
            entry = slot;
            while (libraries.size() > config.libraryEntries) {
                auto oldest = libraries.end();
                for (auto it = libraries.begin(); it != libraries.end(); ++it) {
                    if (it->second == entry) continue;
                    if (oldest == libraries.end() || it->second->stamp < oldest->second->stamp) oldest = it;
                }
                if (oldest == libraries.end()) break;
                libraries.erase(oldest);
            }
        }
        std::lock_guard lock(entry->mutex);
        if (entry->built) {
            ++libraryHits;
            if (entry->library == nil) {
                if (error != nullptr) *error = entry->error;
                return nullptr;
            }
            return std::shared_ptr<void>(entry, (__bridge void*)entry->library);
        }
        @autoreleasepool {
            const auto started = std::chrono::steady_clock::now();
            MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
            descriptor.source = nsString(source.text);
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.mathMode = source.fastMath ? MTLMathModeFast : MTLMathModeSafe;
            options.languageVersion = source.languageVersion != 0 ? static_cast<MTLLanguageVersion>(source.languageVersion) : DefaultLanguageVersion;
            descriptor.options = options;
            if (!source.name.empty()) descriptor.name = nsString(source.name);
            NSError* failure = nil;
            entry->library = [compiler newLibraryWithDescriptor:descriptor error:&failure];
            if (entry->library == nil) entry->error = std::string("library ") + source.name + ": " + (failure.localizedDescription.UTF8String ?: "unknown");
            entry->built = true;
            libraryNanos += nanosSince(started);
            ++librariesBuilt;
        }
        if (entry->library == nil) {
            if (error != nullptr) *error = entry->error;
            return nullptr;
        }
        return std::shared_ptr<void>(entry, (__bridge void*)entry->library);
    }

    void flush(bool final) {
        std::lock_guard lock(flushMutex);
        if (serializer == nil || directory == nullptr) return;
        const auto pending = newSinceFlush.exchange(0);
        if (pending == 0) {
            if (final && journal != nullptr) journal->Save();
            return;
        }
        const auto started = std::chrono::steady_clock::now();
        std::vector<CompileKey> keys;
        {
            std::lock_guard keysLock(keysMutex);
            keys.assign(sessionKeys.begin(), sessionKeys.end());
        }
        const auto partial = directory->NewShardPath();
        bool committed = false;
        @autoreleasepool {
            NSError* failure = nil;
            NSURL* url = [NSURL fileURLWithPath:nsString(partial.string())];
            if ([serializer serializeAsArchiveAndFlushToURL:url error:&failure]) {
                committed = directory->CommitShard(partial);
            } else {
                std::fprintf(stderr, "AGC Metal: archive serialization to %s failed: %s (domain %s code %ld)\n", partial.c_str(), failure.localizedDescription.UTF8String ?: "no error object", failure.domain.UTF8String ?: "-", static_cast<long>(failure.code));
                directory->AbandonShard(partial);
            }
        }
        if (committed) {
            auto name = partial.string();
            const auto shard = directory->Directory() / partial.filename();
            std::error_code error;
            if (!sessionShard.empty() && sessionShard != shard) {
                std::filesystem::remove(sessionShard, error);
                std::filesystem::remove(sessionShard.string() + ".keys", error);
            }
            sessionShard = shard;
            directory->WriteKeys(shard, keys);
            shardsCompacted += directory->Compact();
        } else {
            newSinceFlush += pending;
        }
        if (journal != nullptr) journal->Save();
        ++flushes;
        flushNanos += nanosSince(started);
    }

    void scheduleFlush() {
        if (serializer == nil || flushQueued.exchange(true)) return;
        const auto key = CompileKeyBuilder(FlushDomain).Add(flushSerial++).Finish();
        service->Submit(key, CompilePriority::Prefetch, [this](CompileWorker&) {
            nextFlushAt.store(std::min<std::uint64_t>(nextFlushAt.load() * 4, 4096), std::memory_order_relaxed);
            flush(false);
            flushQueued.store(false);
            CompileOutcome outcome;
            outcome.product = std::make_shared<int>(0);
            return outcome;
        });
    }
};

namespace {

class MetalBackend final : public CompileBackend {
public:
    explicit MetalBackend(MetalCompiler& owner, MetalCompiler::State& state) : owner(owner), state(state) {}

    std::unique_ptr<CompileWorker> CreateWorker(std::uint32_t index) override {
        @autoreleasepool {
            MTL4CompilerDescriptor* descriptor = [MTL4CompilerDescriptor new];
            descriptor.label = [NSString stringWithFormat:@"AnyPS5 compile %u", index];
            if (state.serializer != nil) descriptor.pipelineDataSetSerializer = state.serializer;
            NSError* failure = nil;
            id<MTL4Compiler> compiler = [state.native newCompilerWithDescriptor:descriptor error:&failure];
            if (compiler == nil) {
                std::fprintf(stderr, "AGC Metal: compiler %u: %s\n", index, failure.localizedDescription.UTF8String ?: "unknown");
                throw std::runtime_error("Metal 4 compiler creation failed");
            }
            return std::make_unique<MetalWorker>(owner, (__bridge_retained void*)compiler);
        }
    }

    void Prepare(CompileWorker&, CompilePriority priority) override {
        const auto qos = priority == CompilePriority::Urgent ? QOS_CLASS_USER_INITIATED : priority == CompilePriority::Needed ? QOS_CLASS_DEFAULT : QOS_CLASS_UTILITY;
        pthread_set_qos_class_self_np(qos, 0);
    }

private:
    MetalCompiler& owner;
    MetalCompiler::State& state;
};

}

CompiledPipeline::~CompiledPipeline() {
    if (native != nullptr) CFRelease(native);
}

MetalWorker::~MetalWorker() {
    if (compiler != nullptr) CFRelease(compiler);
}

void* MetalWorker::TaskOptions() const {
    return (__bridge void*)owner.state->taskOptions;
}

std::shared_ptr<void> MetalWorker::Library(const LibrarySource& source, std::string* error) {
    return owner.state->library(source, (__bridge id<MTL4Compiler>)compiler, error);
}

void MetalWorker::NoteKey(const CompileKey& key) {
    std::lock_guard keysLock(owner.state->keysMutex);
    owner.state->sessionKeys.insert(key);
}

void MetalWorker::NotePipeline(double milliseconds) {
    auto& state = *owner.state;
    ++state.pipelinesBuilt;
    state.pipelineNanos += static_cast<std::uint64_t>(milliseconds * 1.0e6);
    if (milliseconds < state.config.archiveHitMilliseconds) {
        ++state.pipelinesFromArchive;
        if (state.shardsLoaded != 0) return;
    }
    NoteFresh();
}

void MetalWorker::NoteFresh() {
    auto& state = *owner.state;
    const auto fresh = ++state.newSinceFlush;
    if (fresh >= state.nextFlushAt.load(std::memory_order_relaxed)) state.scheduleFlush();
}

std::unique_ptr<MetalCompiler> MetalCompiler::Create(Device& device, const CompilerConfig& requested, std::string* failure) {
    @autoreleasepool {
        auto state = std::make_unique<State>(device);
        state->native = (__bridge id<MTLDevice>)device.NativeDevice();
        state->config = requested;
        auto& config = state->config;
        if (config.workers == 0) config.workers = std::clamp<std::uint32_t>(static_cast<std::uint32_t>([NSProcessInfo processInfo].activeProcessorCount) / 3u, 1u, 3u);
        state->nextFlushAt.store(std::max<std::uint32_t>(config.firstFlushAfter, 1));

        const bool useCache = config.archives && !cacheDisabled();
        if (useCache) {
            const auto root = config.cacheRoot.empty() ? defaultRoot() : config.cacheRoot;
            if (!root.empty()) {
                CacheEnvironment environment;
                environment.osBuild = osBuild();
                environment.gpuName = device.Limits().name;
                environment.metalVersion = 4;
                environment.abiVersion = config.abiVersion;
                environment.compilerVersion = CompileCacheVersion;
                state->directory = std::make_unique<CompileCacheDirectory>(root, environment, config.archiveBudgetBytes, config.maxShards);
                state->directory->Prune();
                state->shardsCompacted += state->directory->Compact();
                state->journal = std::make_unique<WarmJournal>(state->directory->Directory() / "warm.txt", config.journalEntries);
                NSMutableArray<id<MTL4Archive>>* archives = [NSMutableArray new];
                for (const auto& shard : state->directory->Shards()) {
                    NSError* error = nil;
                    id<MTL4Archive> archive = [state->native newArchiveWithURL:[NSURL fileURLWithPath:nsString(shard.string())] error:&error];
                    if (archive == nil) {
                        std::fprintf(stderr, "AGC Metal: rejected shader archive %s: %s\n", shard.filename().string().c_str(), error.localizedDescription.UTF8String ?: "unknown");
                        state->directory->Quarantine(shard);
                        ++state->shardsRejected;
                        continue;
                    }
                    [archives addObject:archive];
                    ++state->shardsLoaded;
                }
                state->archives = archives;
                MTL4PipelineDataSetSerializerDescriptor* serializerDescriptor = [MTL4PipelineDataSetSerializerDescriptor new];
                const char* mode = std::getenv("APS5_METAL_SERIALIZER");
                if (mode != nullptr && std::strcmp(mode, "descriptors") == 0) serializerDescriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureDescriptors;
                else if (mode != nullptr && std::strcmp(mode, "both") == 0) serializerDescriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureDescriptors | MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
                else serializerDescriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
                state->serializer = [state->native newPipelineDataSetSerializerWithDescriptor:serializerDescriptor];
            }
        }
        state->taskOptions = [MTL4CompilerTaskOptions new];
        if (state->archives.count != 0) state->taskOptions.lookupArchives = state->archives;

        auto compiler = std::unique_ptr<MetalCompiler>(new MetalCompiler(std::move(state)));
        auto& s = *compiler->state;
        s.backend = std::make_shared<MetalBackend>(*compiler, s);
        CompileServiceConfig serviceConfig;
        serviceConfig.workers = s.config.workers;
        serviceConfig.prefetchWorkers = s.config.prefetchWorkers;
        serviceConfig.retainedBytes = s.config.retainedBytes;
        s.service = std::make_unique<ShaderCompileService>(s.backend, serviceConfig);
        (void)failure;
        return compiler;
    }
}

MetalCompiler::MetalCompiler(std::unique_ptr<State> state) : state(std::move(state)) {}

MetalCompiler::~MetalCompiler() {
    state->service->Shutdown();
    state->flush(true);
}

ShaderCompileService& MetalCompiler::Service() {
    return *state->service;
}

CompileKey MetalCompiler::ComputeKey(const ComputePipelineRequest& request) const {
    CompileKeyBuilder builder(PipelineDomain);
    builder.Add(request.library.key).Add(request.library.languageVersion).Add(request.library.fastMath ? 1u : 0u).AddString(request.function);
    builder.Add(request.constants.size());
    for (const auto& constant : request.constants) builder.Add(constant.index | (static_cast<std::uint64_t>(constant.boolean) << 32u) | (static_cast<std::uint64_t>(constant.value) << 33u));
    return builder.Finish();
}

std::shared_ptr<CompileTicket> MetalCompiler::Compute(const ComputePipelineRequest& request) {
    auto* s = state.get();
    const auto key = ComputeKey(request);
    return s->service->Submit(key, request.priority, [s, request, key](CompileWorker& base) {
        auto& worker = static_cast<MetalWorker&>(base);
        CompileOutcome outcome;
        @autoreleasepool {
            std::string error;
            const auto started = std::chrono::steady_clock::now();
            const auto holder = worker.Library(request.library, &error);
            id<MTLLibrary> library = (__bridge id<MTLLibrary>)holder.get();
            if (library == nil) {
                outcome.error = error;
                return outcome;
            }
            MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
            function.library = library;
            function.name = nsString(request.function);
            MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
            MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
            for (const auto& constant : request.constants) {
                const std::uint32_t scalar = constant.value;
                const bool flag = constant.value != 0;
                if (constant.boolean) [values setConstantValue:&flag type:MTLDataTypeBool atIndex:constant.index];
                else [values setConstantValue:&scalar type:MTLDataTypeUInt atIndex:constant.index];
            }
            MTL4SpecializedFunctionDescriptor* specialized = [MTL4SpecializedFunctionDescriptor new];
            specialized.functionDescriptor = function;
            specialized.constantValues = values;
            descriptor.computeFunctionDescriptor = specialized;
            if (!request.label.empty()) descriptor.label = nsString(request.label);
            NSError* failure = nil;
            id<MTLComputePipelineState> pipeline = [(__bridge id<MTL4Compiler>)worker.Compiler() newComputePipelineStateWithDescriptor:descriptor dynamicLinkingDescriptor:nil compilerTaskOptions:(__bridge MTL4CompilerTaskOptions*)worker.TaskOptions() error:&failure];
            if (pipeline == nil) {
                outcome.error = std::string("pipeline ") + request.function + ": " + (failure.localizedDescription.UTF8String ?: "unknown");
                return outcome;
            }
            const double milliseconds = static_cast<double>(nanosSince(started)) / 1.0e6;
            auto product = std::make_shared<CompiledPipeline>();
            product->maxTotalThreadsPerThreadgroup = static_cast<std::uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
            product->threadExecutionWidth = static_cast<std::uint32_t>(pipeline.threadExecutionWidth);
            product->staticThreadgroupMemoryBytes = static_cast<std::uint32_t>(pipeline.staticThreadgroupMemoryLength);
            product->buildMilliseconds = milliseconds;
            product->archiveHit = milliseconds < s->config.archiveHitMilliseconds;
            product->native = (__bridge_retained void*)pipeline;
            {
                std::lock_guard keysLock(s->keysMutex);
                s->sessionKeys.insert(key);
            }
            worker.NotePipeline(milliseconds);
            outcome.costBytes = s->config.pipelineCostBytes;
            outcome.product = std::move(product);
        }
        return outcome;
    });
}

std::shared_ptr<CompileTicket> MetalCompiler::Run(const CompileKey& key, CompilePriority priority, std::function<CompileOutcome(MetalWorker&)> task) {
    return state->service->Submit(key, priority, [task = std::move(task)](CompileWorker& worker) {
        @autoreleasepool {
            return task(static_cast<MetalWorker&>(worker));
        }
    });
}

std::vector<std::string> MetalCompiler::WarmEntries(std::string_view prefix) {
    std::vector<std::string> matches;
    if (state->journal == nullptr) return matches;
    auto entries = state->journal->Load();
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        if (it->size() > prefix.size() && it->compare(0, prefix.size(), prefix) == 0 && (*it)[prefix.size()] == ' ') matches.push_back(it->substr(prefix.size() + 1));
    }
    return matches;
}

void MetalCompiler::RecordWarm(std::string_view prefix, std::string_view arguments) {
    if (state->journal == nullptr) return;
    state->journal->Record(std::string(prefix) + " " + std::string(arguments));
}

void MetalCompiler::Flush(bool wait) {
    if (wait) {
        state->flush(false);
        return;
    }
    state->scheduleFlush();
}

bool MetalCompiler::WaitIdle(std::uint32_t timeoutMs) {
    return state->service->WaitIdle(timeoutMs);
}

CompilerReport MetalCompiler::Report() const {
    CompilerReport report;
    report.service = state->service->Stats();
    report.librariesBuilt = state->librariesBuilt;
    report.libraryHits = state->libraryHits;
    report.libraryNanos = state->libraryNanos;
    report.pipelinesBuilt = state->pipelinesBuilt;
    report.pipelinesFromArchive = state->pipelinesFromArchive;
    report.pipelineNanos = state->pipelineNanos;
    report.archiveShardsLoaded = state->shardsLoaded;
    report.archiveShardsRejected = state->shardsRejected;
    report.archiveShardsCompacted = state->shardsCompacted;
    report.flushes = state->flushes;
    report.flushNanos = state->flushNanos;
    report.pendingPipelines = state->newSinceFlush;
    if (state->directory != nullptr) {
        report.archiveBytes = state->directory->Bytes();
        report.cacheDirectory = state->directory->Directory().string();
    }
    return report;
}

}
