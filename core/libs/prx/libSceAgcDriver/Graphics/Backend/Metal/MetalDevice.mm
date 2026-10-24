#include "prx/libSceAgcDriver/Graphics/Backend/Backend.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/GuestRegistry.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/MetalCompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Sync.hpp"
#include "BdaAbi.hpp"
#include "MetalAbi.hpp"
#include "MslBackend/MslEmitter.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace AgcDriver {

namespace {

namespace Metal = Graphics::Metal;
using ShaderRecompiler::MetalAbi::BufferSlot;
using ShaderRecompiler::MetalAbi::SlotIndex;

constexpr std::uint64_t ChunkBytes = 1ull << 20u;
constexpr std::uint64_t UploadAlignment = 256;
constexpr std::uint32_t WaitTimeoutMs = 120000;
constexpr std::uint32_t PipelineTimeoutMs = 120000;
constexpr Metal::QueueKind Queue = Metal::QueueKind::Graphics;
constexpr std::uint32_t ArgumentSlots = static_cast<std::uint32_t>(BufferSlot::Count);

struct UploadChunk {
    Metal::NativeBuffer buffer;
    std::uint64_t used = 0;
    std::uint64_t serial = 0;
    bool inflight = false;
};

struct Upload {
    std::byte* data = nullptr;
    std::uint64_t gpuAddress = 0;
};

struct HostImport {
    Metal::HostSpan span;
    id<MTLBuffer> buffer = nil;
};

struct WriteBack {
    void* host = nullptr;
    const void* staged = nullptr;
    std::size_t bytes = 0;
};

struct PipelineEntry {
    std::shared_ptr<const std::string> source;
    std::shared_ptr<Metal::CompileTicket> ticket;
};

struct IdentityHash {
    std::size_t operator()(const ShaderRecompiler::PipelineIdentity& identity) const {
        return static_cast<std::size_t>(identity.low ^ (identity.high * 0x9e3779b97f4a7c15ull));
    }
};

bool Verbose() {
    static const bool verbose = std::getenv("APS5_METAL_VERBOSE") != nullptr;
    return verbose;
}

std::set<std::uint32_t> DeclaredConstants(std::string_view source) {
    std::set<std::uint32_t> ids;
    constexpr std::string_view marker = "[[function_constant(";
    for (auto position = source.find(marker); position != std::string_view::npos; position = source.find(marker, position + marker.size())) {
        std::uint32_t value = 0;
        bool digits = false;
        for (auto index = position + marker.size(); index < source.size() && source[index] >= '0' && source[index] <= '9'; ++index) {
            value = value * 10u + static_cast<std::uint32_t>(source[index] - '0');
            digits = true;
        }
        if (digits) ids.insert(value);
    }
    return ids;
}

ShaderRecompiler::MslTarget BuildMslTarget(id<MTLDevice> device) {
    ShaderRecompiler::MslTarget target;
    target.metalVersion = 320;
    target.simdgroupSize = 32;
    target.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
    const MTLSize threads = device.maxThreadsPerThreadgroup;
    target.maxThreadgroupSize = {static_cast<std::uint32_t>(threads.width), static_cast<std::uint32_t>(threads.height), static_cast<std::uint32_t>(threads.depth)};
    target.maxThreadsPerThreadgroup = static_cast<std::uint32_t>(std::max({threads.width, threads.height, threads.depth}));
    target.maxThreadgroupMemoryBytes = static_cast<std::uint32_t>(device.maxThreadgroupMemoryLength);
    target.relaxedFloat = true;
    return target;
}

}

struct VulkanDevice::State {
    std::unique_ptr<Metal::Device> device;
    std::unique_ptr<Metal::Sync> sync;
    std::unique_ptr<Metal::MetalCompiler> compiler;
    std::unique_ptr<Metal::GuestRegistrySync> registry;
    id<MTLDevice> native = nil;
    id<MTL4ArgumentTable> table = nil;
    ShaderRecompiler::MslTarget msl;
    ShaderRecompiler::SpirvTarget target{};
    std::array<std::uint32_t, 3> groupLimit{};
    bool compileOnly = false;

    std::recursive_mutex mutex;
    Metal::Recording open;
    id<MTL4ComputeCommandEncoder> encoder = nil;
    std::uint32_t encodedDispatches = 0;
    std::uint64_t recordedSinceSubmit = 0;
    std::uint64_t lastSerial = 0;

    std::vector<std::unique_ptr<UploadChunk>> chunks;
    UploadChunk* chunk = nullptr;
    std::vector<UploadChunk*> sealing;

    std::vector<HostImport> imports;
    std::vector<Metal::NativeBuffer> staged;
    std::vector<WriteBack> writeBacks;
    std::map<std::uint64_t, std::pair<std::vector<WriteBack>, std::vector<Metal::NativeBuffer>>> completions;

    Metal::NativeBuffer fault;

    std::mutex pipelineMutex;
    std::unordered_map<ShaderRecompiler::PipelineIdentity, PipelineEntry, IdentityHash> pipelines;
    std::atomic<std::uint64_t> pipelinesCreated{0};
    std::atomic<std::uint64_t> dispatches{0};

    ~State() {
        if (device == nullptr) return;
        try {
            flush();
            drain(lastSerial);
        } catch (...) {
        }
        for (auto& item : chunks) device->ReleaseBuffer(item->buffer);
        if (fault) device->ReleaseBuffer(fault);
        for (auto& import : imports) {
            id<MTLBuffer> buffer = import.buffer;
            import.buffer = nil;
            device->RetireAllocation((__bridge void*)buffer, {});
        }
        device->Collect();
    }

    UploadChunk* acquireChunk() {
        const auto completed = device->CompletedSerial(Queue);
        for (auto& item : chunks) {
            if (!item->inflight || item->serial > completed) continue;
            item->inflight = false;
            item->used = 0;
            return item.get();
        }
        auto created = std::make_unique<UploadChunk>();
        created->buffer = device->NewSharedBuffer(ChunkBytes, "AnyPS5 dispatch uploads");
        if (!created->buffer) throw std::runtime_error("AGC graphics: Metal upload chunk could not be allocated");
        chunks.push_back(std::move(created));
        return chunks.back().get();
    }

    Upload upload(std::uint64_t bytes) {
        const auto size = std::max<std::uint64_t>((bytes + UploadAlignment - 1) & ~(UploadAlignment - 1), UploadAlignment);
        if (size > ChunkBytes) {
            auto buffer = device->NewSharedBuffer(size, "AnyPS5 dispatch upload");
            if (!buffer) throw std::runtime_error("AGC graphics: Metal upload could not be allocated");
            staged.push_back(buffer);
            return {static_cast<std::byte*>(buffer.contents), buffer.gpuAddress};
        }
        if (chunk == nullptr || chunk->used + size > ChunkBytes) {
            if (chunk != nullptr) sealing.push_back(chunk);
            chunk = acquireChunk();
        }
        const auto offset = chunk->used;
        chunk->used += size;
        return {static_cast<std::byte*>(chunk->buffer.contents) + offset, chunk->buffer.gpuAddress + offset};
    }

    std::uint64_t uploadBytes(const void* data, std::size_t bytes, std::size_t minimum = 16) {
        const auto allocation = upload(std::max(bytes, minimum));
        if (bytes != 0) std::memcpy(allocation.data, data, bytes);
        if (bytes < minimum) std::memset(allocation.data + bytes, 0, minimum - bytes);
        return allocation.gpuAddress;
    }

    const HostImport* findImport(std::uint64_t address, std::uint64_t bytes) const {
        for (const auto& import : imports) {
            if (import.span.Contains(address, bytes)) return &import;
        }
        return nullptr;
    }

    const HostImport* importHost(std::uint64_t address, std::uint64_t bytes) {
        if (const auto* found = findImport(address, bytes)) return found;
        const auto span = Metal::HostPagesCovering(address, bytes, device->Limits().hostPageBytes);
        if (span.bytes == 0 || span.bytes > device->Limits().maxBufferBytes) return nullptr;
        id<MTLBuffer> buffer = [native newBufferWithBytesNoCopy:reinterpret_cast<void*>(span.base) length:span.bytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
        if (buffer == nil) return nullptr;
        buffer.label = @"AnyPS5 host import";
        device->AddResidentAllocation((__bridge void*)buffer);
        imports.push_back({span, buffer});
        return &imports.back();
    }

    std::uint64_t resolve(std::uint64_t address, std::uint64_t bytes, bool writable, std::span<const Graphics::GuestMemorySnapshot> snapshots) {
        if (address == 0 || bytes == 0) return 0;
        if (!writable) {
            for (const auto& snapshot : snapshots) {
                if (address >= snapshot.address && bytes <= snapshot.bytes.size() && address - snapshot.address <= snapshot.bytes.size() - bytes) {
                    return uploadBytes(snapshot.bytes.data() + (address - snapshot.address), bytes, bytes);
                }
            }
        }
        const auto windows = device->Guest().Snapshot();
        if (windows != nullptr) {
            if (const auto resolved = Metal::Resolve(*windows, address, bytes); resolved && (!writable || (resolved.window->writable && !resolved.window->copied))) return resolved.deviceAddress;
        }
        if (const auto* import = importHost(address, bytes)) return import->buffer.gpuAddress + (address - import->span.base);
        auto buffer = device->NewSharedBuffer(bytes, "AnyPS5 staged guest range");
        if (!buffer) throw std::runtime_error("AGC graphics: Metal staging buffer could not be allocated");
        std::memcpy(buffer.contents, reinterpret_cast<const void*>(address), bytes);
        staged.push_back(buffer);
        if (writable) writeBacks.push_back({reinterpret_cast<void*>(address), buffer.contents, static_cast<std::size_t>(bytes)});
        return buffer.gpuAddress;
    }

    std::uint64_t bdaTable(std::span<const Graphics::GuestMemorySnapshot>) {
        std::vector<ShaderRecompiler::BdaAbi::Range> ranges;
        if (const auto windows = device->Guest().Snapshot(); windows != nullptr) ranges = windows->table;
        for (const auto& import : imports) {
            ranges.push_back({import.span.base, import.span.End(), import.buffer.gpuAddress, ShaderRecompiler::BdaAbi::Read | ShaderRecompiler::BdaAbi::Write, 0});
        }
        std::sort(ranges.begin(), ranges.end(), [](const auto& left, const auto& right) { return left.begin < right.begin; });
        const auto bytes = Metal::BdaTableBytes(ranges.size());
        const auto allocation = upload(bytes);
        Metal::WriteBdaTable(std::span<std::byte>(allocation.data, bytes), ranges);
        return allocation.gpuAddress;
    }

    void begin() {
        if (open) return;
        open = sync->Begin(Queue, "AnyPS5 compute");
        id<MTL4CommandBuffer> commands = (__bridge id<MTL4CommandBuffer>)open.commands;
        encoder = [commands computeCommandEncoder];
        if (encoder == nil) throw std::runtime_error("AGC graphics: Metal compute encoder could not be created");
        encodedDispatches = 0;
    }

    void flush() {
        if (!open) return;
        @autoreleasepool {
            [encoder endEncoding];
            encoder = nil;
            sync->EncodeLabels(open);
            const auto serial = sync->Submit(&open, 1);
            open = {};
            if (chunk != nullptr) sealing.push_back(chunk);
            chunk = nullptr;
            for (auto* item : sealing) {
                item->inflight = true;
                item->serial = serial;
            }
            sealing.clear();
            auto& completion = completions[serial];
            completion.first = std::move(writeBacks);
            completion.second = std::move(staged);
            writeBacks.clear();
            staged.clear();
            lastSerial = serial;
            recordedSinceSubmit = 0;
        }
    }

    void complete(std::uint64_t through) {
        while (!completions.empty() && completions.begin()->first <= through) {
            auto entry = std::move(completions.begin()->second);
            completions.erase(completions.begin());
            for (const auto& item : entry.first) std::memcpy(item.host, item.staged, item.bytes);
            for (const auto& buffer : entry.second) device->ReleaseBuffer(buffer);
        }
        checkFault();
    }

    void drain(std::uint64_t serial) {
        if (serial == 0) return;
        if (!device->WaitSerial(Queue, serial, WaitTimeoutMs)) throw std::runtime_error("AGC graphics: Metal queue did not finish in time");
        complete(serial);
    }

    void reap() {
        complete(device->CompletedSerial(Queue));
    }

    void checkFault() {
        if (!fault) return;
        auto* record = static_cast<ShaderRecompiler::BdaAbi::Fault*>(fault.contents);
        if (record->state != ShaderRecompiler::BdaAbi::FaultState::Ready) return;
        std::fprintf(stderr, "[metal] shader fault reason %u at 0x%llx (%u bytes, stage %u, instruction %u)\n", static_cast<unsigned>(record->reason), static_cast<unsigned long long>(record->address), record->bytes, record->stage, record->instruction);
        std::memset(fault.contents, 0, ShaderRecompiler::BdaAbi::FaultBufferBytes);
    }

    id<MTLComputePipelineState> pipeline(const ShaderRecompiler::RecompileResult& shader) {
        const auto identity = shader.Identity();
        std::shared_ptr<Metal::CompileTicket> ticket;
        {
            std::lock_guard lock(pipelineMutex);
            auto found = pipelines.find(identity);
            if (found == pipelines.end()) {
                auto source = std::make_shared<const std::string>(shader.msl.Source());
                Metal::ComputePipelineRequest request;
                request.library.text = *source;
                request.library.owner = source;
                request.library.name = "AnyPS5 guest compute";
                request.library.fastMath = true;
                const auto hash = shader.msl.ContentHash();
                request.library.key = Metal::CompileKeyBuilder(0x4d534c43ull).Add(hash.low).Add(hash.high).Finish();
                request.function = std::string(ShaderRecompiler::MslEntryPoint);
                const auto declared = DeclaredConstants(*source);
                for (const auto& constant : shader.specialization) {
                    if (declared.contains(constant.id)) request.constants.push_back({constant.id, constant.value, false});
                }
                request.label = "AnyPS5 guest compute";
                request.priority = Metal::CompilePriority::Urgent;
                ticket = compiler->Compute(request);
                pipelines.emplace(identity, PipelineEntry{std::move(source), ticket});
                pipelinesCreated.fetch_add(1, std::memory_order_relaxed);
            } else {
                ticket = found->second.ticket;
            }
        }
        if (!ticket->Ready()) ticket->Wait(PipelineTimeoutMs);
        if (!ticket->Ready()) {
            if (Verbose()) std::fprintf(stderr, "%.*s\n", static_cast<int>(shader.msl.Source().size()), shader.msl.Source().data());
            throw std::runtime_error("AGC graphics: Metal compute pipeline failed to build: " + (ticket->Done() ? ticket->Error() : std::string("timed out")));
        }
        return (__bridge id<MTLComputePipelineState>)ticket->ProductAs<Metal::CompiledPipeline>()->native;
    }

    void dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::span<const Graphics::GuestMemorySnapshot> snapshots) {
        if (shader.backend != ShaderRecompiler::ShaderBackendKind::Msl) throw std::runtime_error("AGC graphics: the Metal backend dispatches only MSL results");
        if (shader.msl.empty()) throw std::runtime_error("AGC graphics: the MSL result has no source");
        if (shader.workgroupMemoryDwords != 0) throw std::runtime_error("AGC graphics: Metal dispatch with device-memory LDS is not wired yet");
        if (x == 0 || y == 0 || z == 0) return;
        const auto threads = shader.hostThreadgroup;
        if (threads[0] == 0 || threads[1] == 0 || threads[2] == 0) throw std::runtime_error("AGC graphics: the MSL result has no threadgroup size");
        id<MTLComputePipelineState> state = pipeline(shader);
        @autoreleasepool {
            std::lock_guard lock(mutex);
            registry->Refresh();
            begin();
            std::array<std::uint64_t, ArgumentSlots> slots{};
            slots[SlotIndex(BufferSlot::PushConstants)] = uploadBytes(shader.pushConstants.data(), shader.pushConstants.size(), ShaderRecompiler::MetalAbi::PushConstantBytes);
            slots[SlotIndex(BufferSlot::FaultBuffer)] = fault.gpuAddress;
            if (shader.bdaAbiVersion != 0) slots[SlotIndex(BufferSlot::BdaPagetable)] = bdaTable(snapshots);
            for (const auto& binding : shader.bindings) {
                switch (binding.role) {
                case ShaderRecompiler::DescriptorRole::ShaderData:
                    slots[SlotIndex(BufferSlot::ShaderData)] = uploadBytes(binding.guestDescriptor.data(), binding.guestDescriptor.size() * sizeof(std::uint32_t));
                    break;
                case ShaderRecompiler::DescriptorRole::FlattenedSrt:
                    slots[SlotIndex(BufferSlot::FlattenedSrt)] = uploadBytes(binding.guestDescriptor.data(), binding.guestDescriptor.size() * sizeof(std::uint32_t));
                    break;
                case ShaderRecompiler::DescriptorRole::GuestBuffers: {
                    if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) throw std::runtime_error("AGC graphics: guest buffer descriptor must contain four DWORDs per element");
                    std::vector<ShaderRecompiler::MetalAbi::GuestBufferEntry> entries(std::max<std::uint32_t>(binding.count, 1u), {0, 0, 0});
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const ShaderRecompiler::ShaderBufferResource descriptor{{binding.guestDescriptor[element * 4u], binding.guestDescriptor[element * 4u + 1u], binding.guestDescriptor[element * 4u + 2u], binding.guestDescriptor[element * 4u + 3u]}};
                        const auto base = descriptor.Base48();
                        const auto bytes = std::min<std::uint64_t>(descriptor.GetSize(), 0xffffffffull);
                        if (base == 0 || bytes == 0) continue;
                        const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                        entries[element] = {resolve(base, bytes, written, snapshots), static_cast<std::uint32_t>(bytes), 0};
                    }
                    slots[SlotIndex(BufferSlot::GuestBuffers)] = uploadBytes(entries.data(), entries.size() * sizeof(entries[0]));
                    break;
                }
                case ShaderRecompiler::DescriptorRole::BdaPagetable:
                    if (slots[SlotIndex(BufferSlot::BdaPagetable)] == 0) slots[SlotIndex(BufferSlot::BdaPagetable)] = bdaTable(snapshots);
                    break;
                case ShaderRecompiler::DescriptorRole::FaultBuffer:
                    break;
                default:
                    throw std::runtime_error("AGC graphics: Metal dispatch does not bind descriptor role " + std::to_string(static_cast<int>(binding.role)) + " yet");
                }
            }
            for (std::uint32_t slot = 0; slot < ArgumentSlots; ++slot) {
                if (slots[slot] != 0) [table setAddress:slots[slot] atIndex:slot];
            }
            if (encodedDispatches != 0) [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            [encoder setComputePipelineState:state];
            [encoder setArgumentTable:table];
            if (y > groupLimit[1] || z > groupLimit[2]) throw std::runtime_error("AGC graphics: Metal dispatch exceeds the threadgroup count limit");
            [encoder dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(threads[0], threads[1], threads[2])];
            ++encodedDispatches;
            ++recordedSinceSubmit;
            dispatches.fetch_add(1, std::memory_order_relaxed);
        }
    }
};

VulkanDevice::VulkanDevice(const PresentationWindow* window) : VulkanDevice(window, false) {}

VulkanDevice::VulkanDevice(CompileOnly) : VulkanDevice(nullptr, true) {}

VulkanDevice::VulkanDevice(const PresentationWindow*, bool compileOnly) : state(std::make_unique<State>()), serial(0) {
    static std::atomic<std::uint64_t> serials{0};
    serial = serials.fetch_add(1, std::memory_order_relaxed) + 1;
    @autoreleasepool {
        auto& s = *state;
        s.compileOnly = compileOnly;
        std::string failure;
        s.device = Metal::Device::Create(&failure);
        if (s.device == nullptr) throw std::runtime_error("AGC graphics: no usable Metal 4 device: " + failure);
        s.native = (__bridge id<MTLDevice>)s.device->NativeDevice();
        s.msl = BuildMslTarget(s.native);
        s.target = ShaderRecompiler::MslFrontendTarget(s.msl);
        s.groupLimit = {0xffffffffu, 0xffffu, 0xffffu};
        ShaderRecompiler::SetDefaultShaderBackend(ShaderRecompiler::ShaderBackendKind::Msl, s.msl);
        if (compileOnly) return;
        s.sync = Metal::Sync::Create(*s.device, &failure);
        if (s.sync == nullptr) throw std::runtime_error("AGC graphics: Metal sync could not be created: " + failure);
        Metal::CompilerConfig config;
        config.abiVersion = ShaderRecompiler::MetalAbi::Version;
        s.compiler = Metal::MetalCompiler::Create(*s.device, config, &failure);
        if (s.compiler == nullptr) throw std::runtime_error("AGC graphics: Metal compiler could not be created: " + failure);
        s.registry = std::make_unique<Metal::GuestRegistrySync>(s.device->Guest());
        s.registry->Refresh();
        NSError* error = nil;
        MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = ArgumentSlots;
        s.table = [s.native newArgumentTableWithDescriptor:descriptor error:&error];
        if (s.table == nil) throw std::runtime_error("AGC graphics: Metal argument table could not be created");
        s.fault = s.device->NewSharedBuffer(ShaderRecompiler::BdaAbi::FaultBufferBytes, "AnyPS5 fault buffer");
        if (!s.fault) throw std::runtime_error("AGC graphics: Metal fault buffer could not be allocated");
        std::memset(s.fault.contents, 0, ShaderRecompiler::BdaAbi::FaultBufferBytes);
    }
}

VulkanDevice::~VulkanDevice() = default;

auto VulkanDevice::ReplayComputePipeline(const ShaderRecompiler::RecompileResult&, bool) -> bool { return false; }

auto VulkanDevice::WaitForPipelineJournalReplay() -> void {}

auto VulkanDevice::ComputePipelinesCreated() const -> std::uint64_t { return state->pipelinesCreated.load(std::memory_order_relaxed); }

auto VulkanDevice::startJournalReplay() -> void {}

auto VulkanDevice::DeviceName() const -> std::string { return state->device->Limits().name; }

auto VulkanDevice::DriverInfo() const -> std::string { return "Metal 4"; }

auto VulkanDevice::CalibrateClock(double&, std::chrono::steady_clock::time_point&) const -> bool { return false; }

auto VulkanDevice::Target() const -> ShaderRecompiler::SpirvTarget { return state->target; }

auto VulkanDevice::ComputeTarget(std::uint32_t) const -> ShaderRecompiler::SpirvTarget { return state->target; }

auto VulkanDevice::ComputeFullSubgroups() const -> bool { return true; }

auto VulkanDevice::GeometryLimits() const -> std::optional<ShaderRecompiler::GeometryStageLimits> { return {}; }

auto VulkanDevice::MaxFragmentInputComponents() const -> std::uint32_t { return 124; }

auto VulkanDevice::WorkGroupCountLimit() const -> const std::uint32_t* { return state->groupLimit.data(); }

auto VulkanDevice::VendorId() const -> std::uint32_t { return 0x106b; }

auto VulkanDevice::MeshShaders() const -> bool { return false; }

auto VulkanDevice::NormalizeUnobservedViewportDepth(Graphics::State&, const ShaderRecompiler::ShaderPixelStageInfo&) const -> bool { return false; }

auto VulkanDevice::Device() const -> VkDevice { return VK_NULL_HANDLE; }

auto VulkanDevice::WaitIdle() -> void {
    auto& s = *state;
    if (s.sync == nullptr) return;
    std::lock_guard lock(s.mutex);
    s.flush();
    s.drain(s.lastSerial);
}

auto VulkanDevice::PrepareForReplacement() -> void { WaitIdle(); }

auto VulkanDevice::ImportGuestMemory(const GuestAllocations::Mapped&, std::uint64_t, bool) -> bool {
    if (state->registry != nullptr) state->registry->Refresh();
    return true;
}

auto VulkanDevice::SubmitRecorded(bool reapFirst) -> void {
    auto& s = *state;
    if (s.sync == nullptr) return;
    std::lock_guard lock(s.mutex);
    if (reapFirst) s.reap();
    s.flush();
}

auto VulkanDevice::StoreIdleResults(bool reapFirst) -> void { SubmitRecorded(reapFirst); }

auto VulkanDevice::CanWaitUnlocked() const -> bool { return true; }

auto VulkanDevice::SubmitAndEpoch() -> std::uint64_t {
    auto& s = *state;
    if (s.sync == nullptr) return 0;
    std::lock_guard lock(s.mutex);
    s.flush();
    return s.lastSerial > s.device->CompletedSerial(Queue) ? s.lastSerial : 0;
}

auto VulkanDevice::WaitRecorded(std::uint64_t value) -> void {
    if (value == 0 || state->device == nullptr) return;
    if (!state->device->WaitSerial(Queue, value, WaitTimeoutMs)) throw std::runtime_error("AGC graphics: Metal queue did not finish in time");
}

auto VulkanDevice::ReapRecorded(std::uint64_t value) -> void {
    auto& s = *state;
    if (s.sync == nullptr) return;
    std::lock_guard lock(s.mutex);
    s.complete(std::min(value, s.device->CompletedSerial(Queue)));
}

auto VulkanDevice::ReapRecorded() -> void {
    auto& s = *state;
    if (s.sync == nullptr) return;
    std::lock_guard lock(s.mutex);
    s.reap();
}

auto VulkanDevice::WriteLabelOnGpu(std::uint64_t, std::span<const std::byte>, std::uint64_t, std::uint32_t, bool) -> int { return 3; }

auto VulkanDevice::AfterRecordedWork(std::function<void()>, bool) -> bool { return false; }

auto VulkanDevice::PendingLabel(std::uint64_t, std::size_t, std::uint64_t, Graphics::Recorder::LabelRefusal*) const -> std::optional<Graphics::Recorder::LabelHit> { return {}; }

auto VulkanDevice::OpenWriteOverlaps(std::uint64_t, std::size_t) const -> bool { return false; }

auto VulkanDevice::RecordedWritesSettled(std::uint64_t, std::size_t) const -> bool { return state->lastSerial <= state->device->CompletedSerial(Queue) && state->recordedSinceSubmit == 0; }

auto VulkanDevice::StoresPendingOver(std::uint64_t, std::size_t) const -> bool { return false; }

auto VulkanDevice::FillBuffer(std::uint64_t, std::size_t, std::span<const std::uint32_t, 4>) -> bool { return false; }

auto VulkanDevice::DumpSamplesOnGpu(std::uint64_t) -> bool { return false; }

auto VulkanDevice::CopyBuffer(std::uint64_t, std::uint64_t, std::size_t, std::size_t, std::size_t, std::size_t, std::uint64_t, std::uint32_t, const CopyWriterNote&) -> CopyOutcome { return CopyOutcome{2, false, 0.0, CopyOutcome::None, false, false, 0.0, 0, 0, 0, false}; }

auto VulkanDevice::CopyVerifyCounts() -> CopyVerification { return {}; }

auto VulkanDevice::CopyAliasCounts() -> CopyAliasing { return {}; }

auto VulkanDevice::ComputePipelineCounts() -> ComputePipelineCreation { return {}; }

auto VulkanDevice::DispatchMoveCounts() -> DispatchMoveCount { return {}; }

auto VulkanDevice::ResolveMemory(std::uint64_t, std::size_t, bool) -> void {}

auto VulkanDevice::Window() const -> void* { return nullptr; }

auto VulkanDevice::Resize(std::uint32_t, std::uint32_t) -> void {}

auto VulkanDevice::SelectDynamicRange(const DisplayBuffer*) -> void {}

auto VulkanDevice::Presentable() const -> bool { return false; }

auto VulkanDevice::DualSrcBlend() const -> bool { return false; }

auto VulkanDevice::PrimitiveListRestart() const -> bool { return false; }

auto VulkanDevice::SamplerFilterMinmax() const -> bool { return false; }

auto VulkanDevice::ConservativeRasterization() const -> bool { return false; }

auto VulkanDevice::SubgroupStages() const -> VkShaderStageFlags { return VK_SHADER_STAGE_COMPUTE_BIT; }

auto VulkanDevice::ProvokingVertexLast() const -> bool { return false; }

auto VulkanDevice::GraphicsPipelineLibraries() const -> bool { return false; }

auto VulkanDevice::FlipInFlight() -> std::size_t { return 0; }

auto VulkanDevice::RetirePresents(std::size_t) -> double { return 0; }

auto VulkanDevice::PresentWaitsForSlots(const DisplayBuffer*) const -> bool { return false; }

auto VulkanDevice::AcquireImage() -> bool { return false; }

auto VulkanDevice::PresentClear(std::uint32_t, std::uint32_t, bool) -> bool { return false; }

auto VulkanDevice::PresentPixels(std::uint32_t, std::uint32_t, std::span<const std::byte>) -> void {}

auto VulkanDevice::PresentDisplayBuffer(const DisplayBuffer&) -> bool { return false; }

auto VulkanDevice::FinishPresent() -> double { return 0; }

auto VulkanDevice::QueuePresent() -> void {}

auto VulkanDevice::FlipBatches(std::uint64_t& submissions, std::uint64_t& unsignaled) const -> void {
    submissions = 0;
    unsignaled = 0;
}

auto VulkanDevice::PresentCounts() -> PresentStatistics { return {}; }

auto VulkanDevice::PresyncCounts() -> PresyncStatistics { return {}; }

auto VulkanDevice::PrepareDispatch(const ShaderRecompiler::RecompileResult&, std::span<const Graphics::GuestMemorySnapshot>, std::uint64_t, std::uint64_t) -> std::shared_ptr<PreparedDispatch> { return {}; }

auto VulkanDevice::PreparePhaseMs(const PreparedDispatch&) -> std::span<const double, 5> {
    static constexpr double zeros[5] = {};
    return std::span<const double, 5>(zeros);
}

auto VulkanDevice::Dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>*) -> void {
    state->dispatch(shader, x, y, z, snapshots);
}

auto VulkanDevice::DispatchIndirect(const ShaderRecompiler::RecompileResult& shader, std::uint64_t arguments, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>*) -> IndirectOutcome {
    const auto started = std::chrono::steady_clock::now();
    WaitIdle();
    std::array<std::uint32_t, 3> groups{};
    std::memcpy(groups.data(), reinterpret_cast<const void*>(arguments), sizeof(groups));
    const auto readMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    state->dispatch(shader, groups[0], groups[1], groups[2], snapshots);
    return {1, readMs};
}

auto VulkanDevice::PrepareRecipe(const std::shared_ptr<const Recipe>&, bool) -> std::shared_ptr<RecipeHit> { return {}; }

auto VulkanDevice::DispatchRecipe(const ShaderRecompiler::RecompileResult&, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t, std::uint64_t, const std::shared_ptr<RecipeHit>&, IndirectOutcome&, const std::shared_ptr<PreparedDispatch>&, bool) -> RecipeOutcome { return RecipeOutcome::Rebuild; }

auto VulkanDevice::DispatchRecipes() -> bool { return false; }

auto VulkanDevice::VerifyRecipes() -> bool { return false; }

auto VulkanDevice::TemplateDataRefresh() -> bool { return false; }

auto VulkanDevice::NoteRecipe(RecipeEvent, bool) -> void {}

auto VulkanDevice::NoteRecipe(RecipeEvent, RecipeKind) -> void {}

auto VulkanDevice::NoteDrawRecipeMiss(DrawRecipePrecheck) -> void {}

auto VulkanDevice::Draw(const Graphics::State&, const Pm4::DrawParameters&, std::span<const Graphics::CompiledShader>, std::span<const Graphics::GuestMemorySnapshot>, std::shared_ptr<const DrawRecipe>*) -> void {}

auto VulkanDevice::KnownDrawRejection(const Graphics::State&, std::span<const Graphics::CompiledShader>) const -> std::optional<std::string> { return std::string("Metal draws are not wired yet"); }

auto VulkanDevice::ColorMetadataPass(const Graphics::ColorMetadataPass&) -> void {}

auto VulkanDevice::DrawFromRecipe(const Graphics::State&, const Pm4::DrawParameters&, std::span<const Graphics::CompiledShader>, std::span<const Graphics::GuestMemorySnapshot>, const std::shared_ptr<const DrawRecipe>&) -> RecipeOutcome { return RecipeOutcome::Rebuild; }

auto VulkanDevice::DrawIndirectSupport() const -> IndirectDrawSupport { return {false, false, false}; }

auto VulkanDevice::EnqueueDraw(const Graphics::State&, const Pm4::DrawParameters&, std::span<const Graphics::CompiledShader>, std::span<const Graphics::GuestMemorySnapshot>) -> void {}

auto VulkanDevice::PrebuildGraphicsLibraries(const Graphics::ShaderStages&, std::span<const Graphics::CompiledShader>) -> bool { return false; }

}
