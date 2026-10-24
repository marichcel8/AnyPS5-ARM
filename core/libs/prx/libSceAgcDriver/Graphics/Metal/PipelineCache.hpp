#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_PIPELINECACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_PIPELINECACHE_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/MetalCompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/RenderState.hpp"
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace AgcDriver::Graphics::Metal {

enum class VertexStep : std::uint8_t { Vertex, Instance, Constant };

struct VertexAttributeLayout {
    std::uint32_t location = 0;
    std::uint32_t format = 0;
    std::uint32_t buffer = 0;
    std::uint32_t offset = 0;
};

struct VertexBufferLayout {
    std::uint32_t buffer = 0;
    std::uint32_t stride = 0;
    VertexStep step = VertexStep::Vertex;
};

struct RenderPipelineRequest {
    LibrarySource vertexLibrary;
    std::string vertexFunction;
    LibrarySource fragmentLibrary;
    std::string fragmentFunction;
    std::vector<FunctionConstant> vertexConstants;
    std::vector<FunctionConstant> fragmentConstants;
    RenderPipelineState state;
    std::vector<VertexAttributeLayout> vertexAttributes;
    std::vector<VertexBufferLayout> vertexBuffers;
    std::string label;
    CompilePriority priority = CompilePriority::Needed;
};

class RenderPipeline {
public:
    [[nodiscard]] void* Native() const {
        if (auto* cached = native.load(std::memory_order_acquire)) return cached;
        return Resolve();
    }

    [[nodiscard]] CompileState State() const { return ticket->State(); }
    [[nodiscard]] bool Failed() const { return ticket->State() >= CompileState::Failed; }
    [[nodiscard]] const std::string& Error() const { return ticket->Error(); }
    [[nodiscard]] const CompileKey& Key() const { return key; }
    [[nodiscard]] const RenderPipelineState& StateKey() const { return request.state; }
    [[nodiscard]] const CompiledPipeline* Compiled() const;
    bool Wait(std::uint32_t timeoutMs) const { return ticket->Wait(timeoutMs) && Native() != nullptr; }

private:
    friend class PipelineCache;

    void* Resolve() const;

    CompileKey key;
    RenderPipelineRequest request;
    std::shared_ptr<CompileTicket> ticket;
    mutable std::atomic<void*> native{nullptr};
};

struct PipelineCacheStats {
    std::uint64_t renderRequests = 0;
    std::uint64_t renderHits = 0;
    std::uint64_t renderPipelines = 0;
    std::uint64_t promotions = 0;
    std::uint64_t depthStencilStates = 0;
    std::uint64_t depthStencilHits = 0;
};

class PipelineCache {
public:
    static std::unique_ptr<PipelineCache> Create(Device& device, MetalCompiler& compiler, std::string* failure);
    ~PipelineCache();
    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    static CompileKey RenderKey(const RenderPipelineRequest& request);
    std::shared_ptr<RenderPipeline> Render(const RenderPipelineRequest& request);
    void Promote(const RenderPipeline& pipeline, CompilePriority priority);
    void* DepthStencil(const DepthStencilState& state);
    PipelineCacheStats Stats() const;

    struct State;

private:
    explicit PipelineCache(std::unique_ptr<State> state);

    std::unique_ptr<State> state;
};

}

#endif
