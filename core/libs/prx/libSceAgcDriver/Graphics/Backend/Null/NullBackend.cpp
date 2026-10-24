#include "prx/libSceAgcDriver/Graphics/Backend/Backend.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelCompare.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineJournal.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineLibrary.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"

namespace AgcDriver::Graphics {

std::vector<RegisterRead>*& RegisterReadLog() {
    static thread_local std::vector<RegisterRead>* log = nullptr;
    return log;
}

std::atomic<std::uint32_t>& RegisterReadLogsOpen() {
    static std::atomic<std::uint32_t> open{0};
    return open;
}

const char* DccKeysName(DccKeys) { return ""; }

std::size_t DccKeyCount(TextureTileMode, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t) { return 0; }

bool IsDccClear(DccKeys) { return false; }

DccKeys CurrentDccKeys(std::uint64_t, std::uint64_t, std::size_t) { return {}; }

bool FillDccClear(VkFormat, DccKeys, bool, std::span<std::byte>) { return false; }

DepthPlaneClear::DepthPlaneClear(const StorageImageClear&) {}

DepthPlaneClear::~DepthPlaneClear() = default;

}

namespace AgcDriver::Graphics {
bool AnyShadowedOverlaps(std::span<const std::pair<std::uint64_t, std::uint64_t>>) { return false; }

bool AnyShadowedOverlaps(std::uint64_t, std::size_t) { return false; }

std::uint32_t ColorWriteMask(const Registers&) { return 0; }

void CountDrawSkip(DrawSkip, double) {}

void CountIndirectDraw(IndirectDrawPath, double, bool) {}

std::optional<ColorMetadataPass> DecodeColorMetadataPass(const QueueState&) { return {}; }

ShaderRecompiler::ShaderComputeStageInfo DecodeComputeStageInfo(const Registers&, std::span<const std::byte>) { return {}; }

ShaderRecompiler::ShaderPixelStageInfo DecodePixelStageInfo(const Registers&, const std::array<std::uint8_t, 8>&, bool) { return {}; }

ShaderStages DecodeShaderStages(const QueueState&, bool) { return {}; }

State DecodeState(const QueueState&, bool) { return {}; }

ShaderRecompiler::ShaderVertexStageInfo DecodeVertexStageInfo(std::span<const std::byte>, std::uint64_t, std::span<const std::uint32_t>, std::vector<DecodeRead>*, bool, std::uint32_t) { return {}; }

ShaderRecompiler::ShaderPixelStageInfo DefaultUnexportedInputs(ShaderRecompiler::ShaderPixelStageInfo, std::span<const std::uint32_t>) { return {}; }

std::string DepthMaintenanceRejection(const QueueState&) { return {}; }

bool DepthResummarizeDrawsNothing(const QueueState&) { return false; }

std::uint64_t DepthSliceBytes(VkExtent2D, std::uint32_t) { return 0; }

bool DepthSurfaceAt(std::uint64_t) { return false; }

bool DrawKeyCovers(RegisterRead) { return false; }

bool DrawRecipes() { return false; }

std::string DrawRejection(const QueueState&, bool) { return {}; }

std::array<std::uint8_t, 8> ExportMappings(const State&) { return {}; }

std::array<ShaderRecompiler::ColorExportPacking, 8> ExportPackings(const State&) { return {}; }

void FlushPipelineJournal() {}

std::uint64_t HtileDepthClearAddress(std::span<const std::uint32_t>, std::span<const std::uint32_t>, const std::array<std::uint32_t, 3>&) { return 0; }

std::size_t HtileDepthClearBytes_nid_no_patch(std::span<const std::uint32_t>, std::span<const std::uint32_t>, const ShaderRecompiler::ShaderComputeStageInfo&, std::span<const std::uint32_t>, std::uint64_t) { return 0; }

const char* IndirectDrawPathName(IndirectDrawPath) { return "none"; }

std::optional<MemoryValueFill> MatchMemoryValueFill(std::span<const std::uint32_t>, std::span<const std::uint32_t>, const std::array<std::uint32_t, 3>&, const std::array<std::uint32_t, 3>&) { return {}; }

std::optional<StorageImageClear> MatchStorageImageClear(std::span<const std::uint32_t>, std::span<const std::uint32_t>, const ShaderRecompiler::ShaderComputeStageInfo&, const std::array<std::uint32_t, 3>&) { return {}; }

std::array<std::uint32_t, 4> MeshIndexBufferDescriptor(const Pm4::DrawParameters&) { return {}; }

bool MultisampledColorFillAbsorbs(std::uint64_t, std::size_t, std::uint32_t) { return false; }

void NoteColorMetadataFill(std::uint64_t, std::size_t, std::uint32_t) {}

void NoteDepthMetadataFill(std::uint64_t, std::size_t, std::uint32_t) {}

void NoteHtileDepthClear(std::uint64_t, std::size_t, std::size_t) {}

void NoteKeysFillOnGpu(std::uint64_t, std::size_t, DccKeys) {}

void NoteMemoryValueFill(const MemoryValueFill&) {}

std::string NullPixelProgramRejection(const QueueState&) { return {}; }

bool PixelProgramSkipped(const QueueState&) { return false; }

std::uint8_t QuadPixelMask(const State&) { return 0; }

std::array<std::uint16_t, 4> QuadSampleMasks(const State&) { return {}; }

ShaderRecompiler::RasterPrimitive RasterPrimitiveOf(const State&) { return {}; }

auto Recorder::Active() -> Recorder* { return nullptr; }

auto Recorder::CloseLabelGroup(std::uint64_t) -> void {}

auto Recorder::CountRecordedWork() -> void {}

auto Recorder::CountSamples() -> void {}

auto Recorder::DescribePendingWrite(std::uint64_t, std::size_t) const -> std::optional<PendingWriteInfo> { return {}; }

auto Recorder::ForgetQueuedLabels() -> void {}

auto Recorder::LateCounts() -> LateStatistics { return {}; }

auto Recorder::LookupLabel(std::uint64_t, std::size_t, std::uint64_t, LabelRefusal*) -> std::optional<LabelHit> { return {}; }

auto Recorder::LookupLabelValue(std::uint64_t, std::size_t, std::uint64_t) -> std::optional<std::uint64_t> { return {}; }

auto Recorder::NoteQueuedLabel(std::uint64_t, std::span<const std::byte>, std::uint64_t, std::uint32_t) -> void {}

auto Recorder::PendingCompletionLabels() -> std::uint64_t { return 0; }

auto Recorder::PendingLabelSince() -> std::optional<std::chrono::steady_clock::time_point> { return {}; }

auto Recorder::PendingWriteBackCompletions() -> std::uint64_t { return 0; }

auto Recorder::PendingWriteSnapshot() -> std::shared_ptr<const WriteRanges> { return {}; }

auto Recorder::PublishGeneration() -> std::uint64_t { return 0; }

auto Recorder::ReapsWithWork() -> std::uint64_t { return 0; }

auto Recorder::RecordedWorkSinceSubmit() -> std::uint64_t { return 0; }

auto Recorder::SamplesTotal() -> std::uint64_t { return 0; }

auto Recorder::SetQueuedLabelRecorder(void (*)()) -> void {}

auto Recorder::SnapshotOverlaps(const WriteRanges*, std::uint64_t, std::size_t) -> bool { return false; }

auto Recorder::SnapshotWriteOverlaps(std::uint64_t, std::size_t) -> bool { return false; }

auto Recorder::StoreCounts() -> StoreStatistics { return {}; }

auto Recorder::SyncThrough(std::uint64_t, std::size_t, bool) -> void {}

auto Recorder::ThreadHookWaits() -> std::uint64_t { return 0; }

auto Recorder::ThreadWaitedMs() -> double { return 0; }

auto Recorder::WideLabelIn(std::uint64_t, std::size_t) -> bool { return false; }

auto Recorder::WriteGeneration() -> std::uint64_t { return 0; }

bool RegisteredReadableCovers(std::uint64_t, std::size_t, std::vector<RegisteredPiece>*) { return false; }

void RequireNoMultisampleTarget(std::uint64_t, std::size_t, const char*) {}

auto ShaderResources::DataWordsHash(const CompiledShader&) -> std::uint64_t { return 0; }

void ShutdownGuestBufferWorkers() {}

bool StencilReferenceUnused(const State&) { return false; }

void StopCompareHelpers() {}

void StopPipelineLibraryWorker() {}

std::uint64_t StorageImageClearAddress(const StorageImageClear&) { return 0; }

auto StorageTexture::AnyPending() -> bool { return false; }

auto StorageTexture::AnyPendingOverlaps(std::span<const std::pair<std::uint64_t, std::uint64_t>>) -> bool { return false; }

auto StorageTexture::ClassifyFill(std::uint64_t, std::size_t) -> FillCoverage { return {}; }

auto StorageTexture::ClearByKeysFill(std::uint64_t, std::size_t, std::uint8_t) -> std::size_t { return 0; }

auto StorageTexture::DiscardPendingInside(std::uint64_t, std::size_t) -> std::size_t { return 0; }

auto StorageTexture::FillClear(std::span<const std::uint32_t, 4>, std::uint32_t, const char*&, bool) -> bool { return false; }

auto StorageTexture::FlushPending(std::uint64_t, std::size_t, const StorageTexture*, const char*, PublishScope, bool*) -> bool { return false; }

auto StorageTexture::NoteKeysFill(std::uint64_t, std::size_t, std::uint8_t) -> std::size_t { return 0; }

auto StorageTexture::PendingSerial() -> std::uint64_t { return 0; }

auto StorageTexture::StoreSmallPending(const char*) -> std::size_t { return 0; }

}
