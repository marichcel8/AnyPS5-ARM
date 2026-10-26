#include "prx/libSceAgcDriver/Graphics/Backend/Backend.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceOverlay.hpp"
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
#include <algorithm>
#include <deque>
#include <mutex>

namespace AgcDriver::Graphics {

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

void CountDrawSkip(DrawSkip, double) {}

void CountIndirectDraw(IndirectDrawPath, double, bool) {}

std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel) {
    const std::uint32_t blockWidth = bytesPerTexel == 4 ? 128u : 256u;
    const std::uint32_t blockHeight = bytesPerTexel == 1 ? 256u : 128u;
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1) / blockHeight * blockHeight);
    return width * height * bytesPerTexel;
}

std::uint64_t HtileMetadataBytes(VkExtent2D extent) {
    const auto blocks = static_cast<std::uint64_t>((extent.width + 1023u) / 1024u) * ((extent.height + 511u) / 512u);
    return blocks * 32768u;
}

bool DccAlphaOnMsb(VkFormat format, std::uint32_t componentSwap) {
    constexpr std::uint32_t standardReversed = 2;
    constexpr std::uint32_t alternateReversed = 3;
    switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
        return componentSwap == alternateReversed;
    default:
        return componentSwap != standardReversed && componentSwap != alternateReversed;
    }
}

bool DepthSurfaceAt(std::uint64_t) { return false; }

bool DrawRecipes() { return false; }

void FlushPipelineJournal() {}

std::uint64_t HtileDepthClearAddress(std::span<const std::uint32_t>, std::span<const std::uint32_t>, const std::array<std::uint32_t, 3>&) { return 0; }

std::size_t HtileDepthClearBytes_nid_no_patch(std::span<const std::uint32_t>, std::span<const std::uint32_t>, const ShaderRecompiler::ShaderComputeStageInfo&, std::span<const std::uint32_t>, std::uint64_t) { return 0; }

const char* IndirectDrawPathName(IndirectDrawPath) { return "none"; }

std::optional<MemoryValueFill> MatchMemoryValueFill(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& numThreads, const std::array<std::uint32_t, 3>& groups) {
    static constexpr std::array<std::uint32_t, 9> kernel{0xd7460000u, 0x04010c08u, 0xf4201a82u, 0xfa000000u, 0xbf8cc07fu, 0x7e02026au, 0xe0102000u, 0x80000100u, 0xbf810000u};
    if (userData.size() < 8 || numThreads != std::array<std::uint32_t, 3>{64u, 1u, 1u} || groups[1] != 1u || groups[2] != 1u) return std::nullopt;
    if (code.size() < kernel.size() || !std::equal(kernel.begin(), kernel.end(), code.begin())) return std::nullopt;
    const auto stride = (userData[1] >> 16u) & 0x3fffu;
    const bool swizzled = ((userData[1] >> 31u) & 1u) != 0;
    const bool addTid = ((userData[3] >> 23u) & 1u) != 0;
    const auto type = userData[3] >> 30u;
    const auto format = (userData[3] >> 12u) & 0x7fu;
    const auto valueStride = (userData[5] >> 16u) & 0x3fffu;
    const auto valueBytes = valueStride == 0 ? static_cast<std::uint64_t>(userData[6]) : static_cast<std::uint64_t>(userData[6]) * valueStride;
    if (type != 0 || stride != 4 || swizzled || addTid || format < 20u || format > 22u || valueBytes < 4u) return std::nullopt;
    const auto dwords = std::min<std::uint64_t>(static_cast<std::uint64_t>(groups[0]) * 64u, userData[2]);
    const auto address = userData[0] | (static_cast<std::uint64_t>(userData[1] & 0xffffu) << 32u);
    const auto valueAddress = userData[4] | (static_cast<std::uint64_t>(userData[5] & 0xffffu) << 32u);
    if (dwords == 0 || valueAddress == 0 || (valueAddress & 3u) != 0) return std::nullopt;
    return MemoryValueFill{address, static_cast<std::size_t>(dwords * 4u), valueAddress};
}

void NoteMemoryValueFill(const MemoryValueFill& fill) {
    std::uint32_t value = 0;
    GuestMemory::Read(fill.valueAddress, std::as_writable_bytes(std::span(&value, 1)), 4);
    NoteDepthMetadataFill(fill.address, fill.bytes, value);
}

std::optional<StorageImageClear> MatchStorageImageClear(std::span<const std::uint32_t>, std::span<const std::uint32_t>, const ShaderRecompiler::ShaderComputeStageInfo&, const std::array<std::uint32_t, 3>&) { return {}; }

std::array<std::uint32_t, 4> MeshIndexBufferDescriptor(const Pm4::DrawParameters&) { return {}; }

bool MultisampledColorFillAbsorbs(std::uint64_t, std::size_t, std::uint32_t) { return false; }

void NoteColorMetadataFill(std::uint64_t, std::size_t, std::uint32_t) {}

namespace {
constexpr std::size_t MaxDepthMetadataFills = 1024;
std::mutex depthMetadataFillMutex;
std::deque<DepthMetadataFillRecord> depthMetadataFills;
std::uint64_t depthMetadataFillCount = 0;
}

void NoteDepthMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern) {
    std::lock_guard lock(depthMetadataFillMutex);
    depthMetadataFills.push_back({++depthMetadataFillCount, address, bytes, pattern});
    if (depthMetadataFills.size() > MaxDepthMetadataFills) depthMetadataFills.pop_front();
}

std::uint64_t DepthMetadataFillGeneration() {
    std::lock_guard lock(depthMetadataFillMutex);
    return depthMetadataFillCount;
}

std::vector<DepthMetadataFillRecord> DepthMetadataFillsSince(std::uint64_t generation) {
    std::lock_guard lock(depthMetadataFillMutex);
    std::vector<DepthMetadataFillRecord> fills;
    for (const auto& fill : depthMetadataFills) if (fill.generation > generation) fills.push_back(fill);
    return fills;
}

bool NoteHtileDepthClear(std::uint64_t, std::size_t, std::size_t) { return false; }

void NoteKeysFillOnGpu(std::uint64_t, std::size_t, DccKeys) {}


auto Recorder::Active() -> Recorder* { return nullptr; }

auto Recorder::CloseLabelGroup(std::uint64_t) -> void {}

auto Recorder::LastWriteNote(std::uint64_t, std::size_t) const -> std::uint64_t { return 0; }

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

void StopCompareHelpers() {}

void StopPipelineLibraryWorker() {}

std::uint64_t StorageImageClearAddress(const StorageImageClear&) { return 0; }

auto StorageTexture::AnyPending() -> bool { return false; }

auto StorageTexture::AnyPendingOverlaps(std::span<const std::pair<std::uint64_t, std::uint64_t>>) -> bool { return false; }

auto StorageTexture::ClassifyFill(std::uint64_t, std::size_t) -> FillCoverage { return {}; }

auto StorageTexture::ClearByKeysFill(std::uint64_t, std::size_t, std::uint8_t) -> std::size_t { return 0; }

auto StorageTexture::DiscardPendingInside(std::uint64_t, std::size_t) -> std::size_t { return 0; }

auto StorageTexture::FillClear(std::span<const std::uint32_t, 4>, std::uint32_t, const char*&, bool) -> bool { return false; }

auto StorageTexture::FlushPending(std::uint64_t, std::size_t, const StorageTexture*, const char*, PublishScope, bool*, bool) -> bool { return false; }

auto StorageTexture::NoteKeysFill(std::uint64_t, std::size_t, std::uint8_t) -> std::size_t { return 0; }

auto StorageTexture::PendingSerial() -> std::uint64_t { return 0; }

auto StorageTexture::StoreSmallPending(const char*) -> std::size_t { return 0; }

auto StorageTexture::StoreReadbackBeforeLabel(const char*) -> void {}

bool DrawPipelineEnabled() { return false; }

std::optional<MultisampledSingleClear> MatchMultisampledSingleClear(std::span<const std::uint32_t>, std::span<const std::uint32_t>, const std::array<std::uint32_t, 3>&) { return std::nullopt; }

bool TakeMultisampledSingleClear(std::span<const std::uint32_t>, std::span<const std::uint32_t>, const std::array<std::uint32_t, 3>&) { return false; }

std::vector<std::pair<std::uint64_t, std::uint64_t>> PipelinedColorMetadataPassWrites(const ColorMetadataPass&) { return {}; }

bool ShaderStoresThroughAddresses(const ShaderRecompiler::SharedSpirv&) { return false; }

void CountEmptyIndirectDraw(double) {}

std::array<std::uint32_t, 8> DepthCopySourceWords(const DepthTarget&) { return {}; }

void FinishLeaseWork() {}

void ReportHostImports() {}

PipelineLibraryCounters PipelineLibraryCountersOf(VkDevice) { return {}; }

}

namespace AgcDriver {

void PerformanceCounters::AddDriverNanoseconds(std::uint64_t) {}

PacketCounters& PacketCounters::ThisThread() {
    thread_local PacketCounters counters;
    return counters;
}

}

extern "C" void AgcDriverTogglePerformanceOverlay_nid_postfix() {}

extern "C" bool AgcDriverMinimalOverlayShown_nid_postfix() { return false; }

extern "C" int AgcDriverOverlayCorner_nid_postfix() { return 0; }

extern "C" std::uint64_t AgcDriverProcessFootprint_nid_postfix() { return 0; }

extern "C" std::uint64_t AgcDriverGeneratedFrames_nid_postfix() { return 0; }
