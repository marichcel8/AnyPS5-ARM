#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_METALCOMPILER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_METALCOMPILER_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/CompileCache.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/ShaderCompileService.hpp"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace AgcDriver::Graphics::Metal {

struct CompilerConfig {
    std::filesystem::path cacheRoot;
    std::uint32_t workers = 0;
    std::uint32_t prefetchWorkers = 1;
    std::uint64_t retainedBytes = 64ull << 20u;
    std::uint64_t pipelineCostBytes = 16ull << 10u;
    std::uint64_t archiveBudgetBytes = 256ull << 20u;
    std::uint32_t maxShards = 16;
    std::uint32_t firstFlushAfter = 128;
    std::uint32_t libraryEntries = 128;
    std::uint32_t journalEntries = 1024;
    std::uint32_t abiVersion = 0;
    double archiveHitMilliseconds = 4.0;
    bool archives = true;
};

struct LibrarySource {
    CompileKey key;
    std::shared_ptr<const void> owner;
    std::string_view text;
    std::string name;
    std::uint32_t languageVersion = 0;
    bool fastMath = true;
};

struct FunctionConstant {
    std::uint32_t index = 0;
    std::uint32_t value = 0;
    bool boolean = false;
};

struct ComputePipelineRequest {
    LibrarySource library;
    std::string function;
    std::vector<FunctionConstant> constants;
    std::string label;
    CompilePriority priority = CompilePriority::Needed;
};

struct CompiledPipeline {
    void* native = nullptr;
    std::uint32_t maxTotalThreadsPerThreadgroup = 0;
    std::uint32_t threadExecutionWidth = 0;
    std::uint32_t staticThreadgroupMemoryBytes = 0;
    double buildMilliseconds = 0.0;
    bool archiveHit = false;

    CompiledPipeline() = default;
    CompiledPipeline(const CompiledPipeline&) = delete;
    CompiledPipeline& operator=(const CompiledPipeline&) = delete;
    ~CompiledPipeline();
};

struct CompilerReport {
    CompileStats service;
    std::uint64_t librariesBuilt = 0;
    std::uint64_t libraryHits = 0;
    std::uint64_t libraryNanos = 0;
    std::uint64_t pipelinesBuilt = 0;
    std::uint64_t pipelinesFromArchive = 0;
    std::uint64_t pipelineNanos = 0;
    std::uint64_t archiveShardsLoaded = 0;
    std::uint64_t archiveShardsRejected = 0;
    std::uint64_t archiveShardsCompacted = 0;
    std::uint64_t archiveBytes = 0;
    std::uint64_t flushes = 0;
    std::uint64_t flushNanos = 0;
    std::uint64_t pendingPipelines = 0;
    std::string cacheDirectory;
};

class MetalCompiler;

class MetalWorker final : public CompileWorker {
public:
    MetalWorker(MetalCompiler& owner, void* compiler) : owner(owner), compiler(compiler) {}
    ~MetalWorker() override;

    [[nodiscard]] void* Compiler() const { return compiler; }
    [[nodiscard]] void* TaskOptions() const;
    std::shared_ptr<void> Library(const LibrarySource& source, std::string* error);
    void NotePipeline(double milliseconds);
    void NoteFresh();

private:
    MetalCompiler& owner;
    void* compiler;
};

class MetalCompiler {
public:
    static std::unique_ptr<MetalCompiler> Create(Device& device, const CompilerConfig& config, std::string* failure);
    ~MetalCompiler();
    MetalCompiler(const MetalCompiler&) = delete;
    MetalCompiler& operator=(const MetalCompiler&) = delete;

    CompileKey ComputeKey(const ComputePipelineRequest& request) const;
    std::shared_ptr<CompileTicket> Compute(const ComputePipelineRequest& request);
    std::shared_ptr<CompileTicket> Run(const CompileKey& key, CompilePriority priority, std::function<CompileOutcome(MetalWorker&)> task);

    std::vector<std::string> WarmEntries(std::string_view prefix);
    void RecordWarm(std::string_view prefix, std::string_view arguments);

    void Flush(bool wait);
    bool WaitIdle(std::uint32_t timeoutMs);
    CompilerReport Report() const;
    ShaderCompileService& Service();

    struct State;

private:
    friend class MetalWorker;
    explicit MetalCompiler(std::unique_ptr<State> state);

    std::unique_ptr<State> state;
};

}

#endif
