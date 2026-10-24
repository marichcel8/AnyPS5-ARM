#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_RECORDER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_RECORDER_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/RenderPassPlan.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/RenderState.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/UploadRing.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace AgcDriver::Graphics::Metal {

inline constexpr std::uint8_t StageVertex = 1u;
inline constexpr std::uint8_t StageFragment = 2u;
inline constexpr std::uint8_t StageAll = StageVertex | StageFragment;
inline constexpr std::uint32_t RecorderBufferSlots = 14;

struct ColorAttachmentDesc {
    void* texture = nullptr;
    void* resolveTexture = nullptr;
    std::uint32_t level = 0;
    std::uint32_t slice = 0;
    std::uint32_t depthPlane = 0;
    std::array<double, 4> clearColor{};
    AttachmentIntent intent;
};

struct DepthAttachmentDesc {
    void* texture = nullptr;
    std::uint32_t level = 0;
    std::uint32_t slice = 0;
    double clearDepth = 1.0;
    std::uint32_t clearStencil = 0;
    bool stencil = false;
    AttachmentIntent depthIntent;
    AttachmentIntent stencilIntent;
};

struct RenderPassDesc {
    std::array<ColorAttachmentDesc, MaxColorAttachments> colors{};
    std::uint32_t colorCount = 0;
    DepthAttachmentDesc depth;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t layers = 1;
    std::uint32_t samples = 1;
};

struct DrawArgs {
    PrimitiveType primitive = PrimitiveType::Triangle;
    std::uint32_t vertexStart = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t instanceCount = 1;
    std::uint32_t baseInstance = 0;
};

struct IndexedDrawArgs {
    PrimitiveType primitive = PrimitiveType::Triangle;
    IndexType indexType = IndexType::UInt16;
    std::uint64_t indexAddress = 0;
    std::uint64_t indexBytes = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t instanceCount = 1;
    std::int32_t baseVertex = 0;
    std::uint32_t baseInstance = 0;
};

struct IndirectDrawArgs {
    PrimitiveType primitive = PrimitiveType::Triangle;
    std::uint64_t argumentsAddress = 0;
    bool indexed = false;
    IndexType indexType = IndexType::UInt16;
    std::uint64_t indexAddress = 0;
    std::uint64_t indexBytes = 0;
};

enum class DrawOutcome : std::uint8_t { Drawn, NoPass, NoPipeline, PipelinePending, PipelineFailed };
enum class PipelineMissPolicy : std::uint8_t { Wait, Skip };

struct RecorderOptions {
    std::uint32_t maxSlots = 6;
    std::uint64_t uploadChunkBytes = 256ull << 10u;
    std::uint32_t uploadAlignment = 16;
    PipelineMissPolicy missPolicy = PipelineMissPolicy::Wait;
    std::uint32_t pipelineWaitMs = 30000;
    bool deferredStores = true;
    bool passBarriers = true;
    QueueKind queue = QueueKind::Graphics;
    const char* label = nullptr;

    static RecorderOptions FromEnvironment();
};

struct RecorderStats {
    std::uint64_t draws = 0;
    std::uint64_t skippedDraws = 0;
    std::uint64_t pipelineBinds = 0;
    std::uint64_t pipelineWaits = 0;
    std::uint64_t pipelineWaitNanos = 0;
    std::uint64_t depthStencilBinds = 0;
    std::uint64_t dynamicSets = 0;
    std::uint64_t slotSets = 0;
    std::uint64_t slotSetsSkipped = 0;
    std::uint64_t uploadBytes = 0;
    std::uint64_t barriers = 0;
    std::uint64_t submits = 0;
    std::uint64_t slotWaits = 0;
    std::uint64_t slots = 0;
    std::uint64_t storesDeferred = 0;
    std::uint64_t storesDiscarded = 0;
    PassCounters passes;
    bool deferredStoreSupport = false;
};

class Recorder {
public:
    static std::unique_ptr<Recorder> Create(Device& device, PipelineCache& pipelines, const RecorderOptions& options, std::string* failure);
    ~Recorder();
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    void Begin();
    bool BeginPass(const RenderPassDesc& pass);
    void EndPass();
    void DiscardAttachment(std::uint32_t attachment);

    void SetPipeline(const RenderPipeline* pipeline);
    void SetDepthStencil(const DepthStencilState& depthStencil);
    void SetDynamic(const DynamicState& dynamic);
    void SetSlot(std::uint8_t stages, std::uint32_t slot, std::uint64_t address);
    std::uint64_t PushData(std::uint8_t stages, std::uint32_t slot, const void* data, std::uint32_t bytes);
    UploadAllocation Upload(std::uint32_t bytes);

    DrawOutcome Draw(const DrawArgs& args);
    DrawOutcome DrawIndexed(const IndexedDrawArgs& args);
    DrawOutcome DrawIndirect(const IndirectDrawArgs& args);

    void* NativeCommandBuffer();
    void* NativeRenderEncoder() const;
    std::uint64_t Submit();
    RecorderStats Stats() const;

    struct State;

private:
    explicit Recorder(std::unique_ptr<State> state);

    std::unique_ptr<State> state;
};

}

#endif
