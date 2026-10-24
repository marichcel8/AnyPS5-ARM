#include "prx/libSceAgcDriver/Graphics/Backend/Backend.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/GuestRegistry.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/MetalCompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Sync.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureJobs.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Textures.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestSamplerResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "BdaAbi.hpp"
#include "MetalAbi.hpp"
#include "RuntimeAbi.hpp"
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

enum class Numeric : std::uint8_t { Float, Uint, Sint };

struct ImageKind {
    Numeric numeric = Numeric::Float;
    bool storage = false;
};

using SurfaceKey = std::array<std::uint64_t, 5>;
using ViewKey = std::array<std::uint32_t, 7>;
using SamplerKey = std::array<std::uint32_t, 5>;

struct CachedSurface {
    Metal::TextureHandle texture;
    std::map<ViewKey, Metal::TextureHandle> views;
};

struct ImageUse {
    CachedSurface* surface = nullptr;
    Graphics::GuestTextureResource resource{};
    std::uint64_t address = 0;
    bool written = false;
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

ImageKind ImageKindOf(std::uint32_t binding) {
    using namespace ShaderRecompiler::RuntimeAbi;
    const auto kind = binding % static_cast<std::uint32_t>(Binding::Count);
    if (kind < FirstImageBinding || kind >= static_cast<std::uint32_t>(Binding::Samplers)) throw std::runtime_error("AGC graphics: Metal image binding " + std::to_string(binding) + " is not an image binding");
    if (kind >= FirstStorageImageBinding) {
        const auto group = (kind - FirstStorageImageBinding) / 7u;
        return {group == 0u ? Numeric::Float : Numeric::Uint, true};
    }
    const auto group = (kind - FirstImageBinding) / 7u;
    return {group == 1u ? Numeric::Uint : group == 2u ? Numeric::Sint : Numeric::Float, false};
}

Metal::PixelFormat NumericView(Metal::PixelFormat format, Numeric numeric) {
    using enum Metal::PixelFormat;
    static constexpr std::array<std::array<Metal::PixelFormat, 6>, 11> families{{
        {R8Unorm, R8UnormSrgb, R8Snorm, R8Uint, R8Sint, Invalid},
        {RG8Unorm, RG8UnormSrgb, RG8Snorm, RG8Uint, RG8Sint, Invalid},
        {RGBA8Unorm, RGBA8UnormSrgb, RGBA8Snorm, RGBA8Uint, RGBA8Sint, Invalid},
        {R16Unorm, Invalid, R16Snorm, R16Uint, R16Sint, R16Float},
        {RG16Unorm, Invalid, RG16Snorm, RG16Uint, RG16Sint, RG16Float},
        {RGBA16Unorm, Invalid, RGBA16Snorm, RGBA16Uint, RGBA16Sint, RGBA16Float},
        {Invalid, Invalid, Invalid, R32Uint, R32Sint, R32Float},
        {Invalid, Invalid, Invalid, RG32Uint, RG32Sint, RG32Float},
        {Invalid, Invalid, Invalid, RGBA32Uint, RGBA32Sint, RGBA32Float},
        {RGB10A2Unorm, Invalid, Invalid, RGB10A2Uint, Invalid, Invalid},
        {Invalid, Invalid, Invalid, Invalid, Invalid, Invalid},
    }};
    for (const auto& family : families) {
        const auto found = std::find(family.begin(), family.end(), format);
        if (found == family.end() || format == Invalid) continue;
        const auto column = static_cast<std::size_t>(found - family.begin());
        const bool integer = column == 3u || column == 4u;
        Metal::PixelFormat wanted = Invalid;
        if (numeric == Numeric::Uint) wanted = family[3];
        else if (numeric == Numeric::Sint) wanted = family[4];
        else if (integer) wanted = family[5] != Invalid ? family[5] : family[0];
        else return format;
        return wanted != Invalid ? wanted : format;
    }
    return format;
}

Metal::TextureKind ViewKindOf(ShaderRecompiler::DescriptorImageShape shape) {
    switch (shape) {
    case ShaderRecompiler::DescriptorImageShape::Image1D:
    case ShaderRecompiler::DescriptorImageShape::Image2D: return Metal::TextureKind::Tex2D;
    case ShaderRecompiler::DescriptorImageShape::Image1DArray:
    case ShaderRecompiler::DescriptorImageShape::Image2DArray:
    case ShaderRecompiler::DescriptorImageShape::ImageCube: return Metal::TextureKind::Tex2DArray;
    case ShaderRecompiler::DescriptorImageShape::Image3D: return Metal::TextureKind::Tex3D;
    case ShaderRecompiler::DescriptorImageShape::Image2DMultisample: return Metal::TextureKind::Tex2DMsaa;
    }
    throw std::runtime_error("AGC graphics: unknown descriptor image shape");
}

MTLSamplerMinMagFilter NativeFilter(VkFilter filter) {
    return filter == VK_FILTER_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
}

MTLSamplerAddressMode NativeAddress(VkSamplerAddressMode mode, bool unnormalized) {
    if (unnormalized) return mode == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ? MTLSamplerAddressModeClampToZero : MTLSamplerAddressModeClampToEdge;
    switch (mode) {
    case VK_SAMPLER_ADDRESS_MODE_REPEAT: return MTLSamplerAddressModeRepeat;
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return MTLSamplerAddressModeMirrorRepeat;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return MTLSamplerAddressModeClampToBorderColor;
    case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: return MTLSamplerAddressModeMirrorClampToEdge;
    default: return MTLSamplerAddressModeClampToEdge;
    }
}

MTLSamplerBorderColor NativeBorder(VkBorderColor color) {
    switch (color) {
    case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
    case VK_BORDER_COLOR_INT_OPAQUE_BLACK: return MTLSamplerBorderColorOpaqueBlack;
    case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
    case VK_BORDER_COLOR_INT_OPAQUE_WHITE: return MTLSamplerBorderColorOpaqueWhite;
    default: return MTLSamplerBorderColorTransparentBlack;
    }
}

ShaderRecompiler::MslTarget BuildMslTarget(id<MTLDevice> device, bool presenting) {
    ShaderRecompiler::MslTarget target;
    target.metalVersion = 320;
    target.simdgroupSize = 32;
    target.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
    const MTLSize threads = device.maxThreadsPerThreadgroup;
    target.maxThreadgroupSize = {static_cast<std::uint32_t>(threads.width), static_cast<std::uint32_t>(threads.height), static_cast<std::uint32_t>(threads.depth)};
    target.maxThreadsPerThreadgroup = static_cast<std::uint32_t>(std::max({threads.width, threads.height, threads.depth}));
    target.maxThreadgroupMemoryBytes = static_cast<std::uint32_t>(device.maxThreadgroupMemoryLength);
    static const bool exactFloat = std::getenv("APS5_EXACT_FLOAT") != nullptr;
    static const bool relaxedFloat = std::getenv("APS5_RELAXED_FLOAT") != nullptr;
    target.relaxedFloat = !exactFloat && (relaxedFloat || presenting);
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

    std::unique_ptr<Metal::TextureManager> textures;
    std::map<SurfaceKey, CachedSurface> surfaces;
    std::map<SamplerKey, id<MTLSamplerState>> samplers;
    std::array<id<MTLTexture>, 4> nullTextures{};

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
        for (auto& [key, surface] : surfaces) {
            for (auto& [view, handle] : surface.views) textures->Release(handle);
            textures->Release(surface.texture);
        }
        surfaces.clear();
        textures.reset();
        for (auto& texture : nullTextures) {
            if (texture == nil) continue;
            id<MTLTexture> retired = texture;
            texture = nil;
            device->RetireAllocation((__bridge void*)retired, {});
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
            if (textures != nullptr) textures->Seal(Queue, serial);
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

    Metal::TextureManager& textureManager() {
        if (textures != nullptr) return *textures;
        std::string failure;
        textures = Metal::TextureManager::Create(*device, *compiler, &failure);
        if (textures == nullptr) throw std::runtime_error("AGC graphics: Metal texture manager could not be created: " + failure);
        return *textures;
    }

    id<MTLTexture> nullTexture(Metal::TextureKind kind) {
        const std::size_t index = kind == Metal::TextureKind::Tex2DArray ? 1u : kind == Metal::TextureKind::Tex3D ? 2u : kind == Metal::TextureKind::Tex2DMsaa ? 3u : 0u;
        if (nullTextures[index] != nil) return nullTextures[index];
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
        descriptor.textureType = index == 1u ? MTLTextureType2DArray : index == 2u ? MTLTextureType3D : index == 3u ? MTLTextureType2DMultisample : MTLTextureType2D;
        descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
        descriptor.width = 1;
        descriptor.height = 1;
        descriptor.sampleCount = index == 3u ? 4 : 1;
        descriptor.storageMode = index == 3u ? MTLStorageModePrivate : MTLStorageModeShared;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        descriptor.usage = index == 3u ? MTLTextureUsageShaderRead : MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
        id<MTLTexture> texture = [native newTextureWithDescriptor:descriptor];
        if (texture == nil) throw std::runtime_error("AGC graphics: Metal null texture could not be created");
        texture.label = @"AnyPS5 null texture";
        if (index != 3u) {
            const std::uint32_t zero = 0;
            [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:0 withBytes:&zero bytesPerRow:4 bytesPerImage:4];
        }
        device->AddResidentAllocation((__bridge void*)texture);
        nullTextures[index] = texture;
        return texture;
    }

    std::uint64_t image(const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t element, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::vector<ImageUse>& uses) {
        if (!binding.imageShape) throw std::runtime_error("AGC graphics: Metal image binding has no image shape");
        const auto viewKind = ViewKindOf(*binding.imageShape);
        const std::span<const std::uint32_t> words(binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 8u, 8u);
        if (std::all_of(words.begin(), words.end(), [](std::uint32_t word) { return word == 0u; }) || Graphics::IsNullTextureResource(words)) return nullTexture(viewKind).gpuResourceID._impl;
        const auto resource = Graphics::DecodeTextureResource(words);
        const bool written = element < binding.imageWritten.size() && binding.imageWritten[element];
        const auto layers = resource.depthOrLastArray + 1u;
        auto info = Metal::DescribeTexture(resource);
        if (info.kind != Metal::TextureKind::Tex3D && info.kind != Metal::TextureKind::Tex2DMsaa) {
            info.kind = Metal::TextureKind::Tex2DArray;
            info.arrayLength = layers;
        }
        if ((viewKind == Metal::TextureKind::Tex3D) != (info.kind == Metal::TextureKind::Tex3D) || (viewKind == Metal::TextureKind::Tex2DMsaa) != (info.kind == Metal::TextureKind::Tex2DMsaa)) throw std::runtime_error("AGC graphics: Metal image dimension does not match the shader image type");
        info.shaderWrite = true;
        info.label = "AnyPS5 guest texture";
        const SurfaceKey key{resource.baseAddress,
                             static_cast<std::uint64_t>(resource.format) | static_cast<std::uint64_t>(info.kind) << 32u | static_cast<std::uint64_t>(resource.tileMode) << 40u,
                             static_cast<std::uint64_t>(info.width) | static_cast<std::uint64_t>(info.height) << 32u,
                             static_cast<std::uint64_t>(layers) | static_cast<std::uint64_t>(info.mipLevels) << 32u,
                             static_cast<std::uint64_t>(resource.pipeBankXor) | static_cast<std::uint64_t>(info.samples) << 32u};
        auto& manager = textureManager();
        auto found = surfaces.find(key);
        if (found == surfaces.end()) {
            std::string failure;
            auto created = manager.Create(info, &failure);
            if (!created) throw std::runtime_error("AGC graphics: Metal texture could not be created: " + failure);
            found = surfaces.emplace(key, CachedSurface{created, {}}).first;
        }
        auto& surface = found->second;
        const auto bytes = Graphics::DescribeSurface(resource).guestBytes;
        const auto address = resolve(resource.baseAddress, bytes, written, snapshots);
        uses.push_back({&surface, resource, address, written});
        Metal::ViewInfo view;
        view.kind = viewKind;
        view.changeKind = true;
        view.pixelFormat = NumericView(surface.texture.pixelFormat, ImageKindOf(binding.binding).numeric);
        view.baseLevel = std::min(resource.baseLevel, info.mipLevels - 1u);
        view.levelCount = std::min(std::max(resource.lastLevel, view.baseLevel), info.mipLevels - 1u) - view.baseLevel + 1u;
        if (viewKind == Metal::TextureKind::Tex3D || viewKind == Metal::TextureKind::Tex2DMsaa) {
            view.baseSlice = 0;
            view.sliceCount = 1;
        } else {
            view.baseSlice = std::min(resource.baseArray, layers - 1u);
            view.sliceCount = viewKind == Metal::TextureKind::Tex2D ? 1u : layers - view.baseSlice;
        }
        if (!written) view.swizzle = Metal::ViewSwizzle(resource.format, {resource.dstSelX, resource.dstSelY, resource.dstSelZ, resource.dstSelW});
        const ViewKey viewKey{static_cast<std::uint32_t>(view.kind), static_cast<std::uint32_t>(view.pixelFormat), view.baseLevel, view.levelCount, view.baseSlice, view.sliceCount,
                              static_cast<std::uint32_t>(view.swizzle[0]) | static_cast<std::uint32_t>(view.swizzle[1]) << 8u | static_cast<std::uint32_t>(view.swizzle[2]) << 16u | static_cast<std::uint32_t>(view.swizzle[3]) << 24u};
        auto existing = surface.views.find(viewKey);
        if (existing == surface.views.end()) {
            auto created = manager.View(surface.texture, view);
            if (!created) throw std::runtime_error("AGC graphics: Metal texture view could not be created");
            existing = surface.views.emplace(viewKey, created).first;
        }
        return existing->second.resourceId;
    }

    std::uint64_t sampler(const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t element) {
        const std::span<const std::uint32_t> words(binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u, 4u);
        const bool unnormalized = element < binding.samplerUnnormalized.size() && binding.samplerUnnormalized[element];
        const bool integerBorder = element < binding.samplerIntegerBorder.size() && binding.samplerIntegerBorder[element];
        const SamplerKey key{words[0], words[1], words[2], words[3], (unnormalized ? 1u : 0u) | (integerBorder ? 2u : 0u)};
        if (const auto found = samplers.find(key); found != samplers.end()) return found->second.gpuResourceID._impl;
        const auto decoded = Graphics::DecodeSamplerResource(words, unnormalized, false, integerBorder);
        MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
        descriptor.minFilter = NativeFilter(decoded.minFilter);
        descriptor.magFilter = decoded.unnormalizedCoordinates ? descriptor.minFilter : NativeFilter(decoded.magFilter);
        descriptor.mipFilter = decoded.unnormalizedCoordinates ? MTLSamplerMipFilterNotMipmapped : decoded.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
        descriptor.sAddressMode = NativeAddress(decoded.addressModeU, decoded.unnormalizedCoordinates);
        descriptor.tAddressMode = NativeAddress(decoded.addressModeV, decoded.unnormalizedCoordinates);
        descriptor.rAddressMode = NativeAddress(decoded.addressModeW, decoded.unnormalizedCoordinates);
        descriptor.borderColor = NativeBorder(decoded.borderColor);
        descriptor.normalizedCoordinates = !decoded.unnormalizedCoordinates;
        descriptor.maxAnisotropy = decoded.anisotropyEnable && !decoded.unnormalizedCoordinates ? static_cast<NSUInteger>(std::clamp(decoded.maxAnisotropy, 1.0f, 16.0f)) : 1u;
        if (!decoded.unnormalizedCoordinates) {
            descriptor.lodMinClamp = std::max(decoded.minLod, 0.0f);
            descriptor.lodMaxClamp = std::max(decoded.maxLod, descriptor.lodMinClamp);
        }
        if (decoded.compareEnable) descriptor.compareFunction = static_cast<MTLCompareFunction>(decoded.compareOp);
        descriptor.supportArgumentBuffers = YES;
        descriptor.label = @"AnyPS5 guest sampler";
        id<MTLSamplerState> state = [native newSamplerStateWithDescriptor:descriptor];
        if (state == nil) throw std::runtime_error("AGC graphics: Metal sampler state could not be created");
        samplers.emplace(key, state);
        return state.gpuResourceID._impl;
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
            std::vector<std::pair<std::uint32_t, std::uint64_t>> imageEntries;
            std::vector<ImageUse> uses;
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
                case ShaderRecompiler::DescriptorRole::GuestImages: {
                    if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 8u) throw std::runtime_error("AGC graphics: guest image descriptor must contain eight DWORDs per element");
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        imageEntries.emplace_back(ShaderRecompiler::MetalAbi::GuestImageIndex(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count), element), image(binding, element, snapshots, uses));
                    }
                    break;
                }
                case ShaderRecompiler::DescriptorRole::GuestSamplers: {
                    if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) throw std::runtime_error("AGC graphics: guest sampler descriptor must contain four DWORDs per element");
                    std::vector<ShaderRecompiler::MetalAbi::ResourceIdEntry> entries(std::max<std::uint32_t>(binding.count, 1u), {0});
                    for (std::uint32_t element = 0; element < binding.count; ++element) entries[element].resourceId = sampler(binding, element);
                    slots[SlotIndex(BufferSlot::GuestSamplers)] = uploadBytes(entries.data(), entries.size() * sizeof(entries[0]));
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
            if (!imageEntries.empty()) {
                const auto [lowest, highest] = std::minmax_element(imageEntries.begin(), imageEntries.end());
                const auto first = lowest->first;
                std::vector<ShaderRecompiler::MetalAbi::ResourceIdEntry> entries(highest->first - first + 1u, {0});
                for (const auto& [index, resourceId] : imageEntries) entries[index - first].resourceId = resourceId;
                slots[SlotIndex(BufferSlot::GuestImages)] = uploadBytes(entries.data(), entries.size() * sizeof(entries[0])) - static_cast<std::uint64_t>(first) * sizeof(entries[0]);
            }
            if (encodedDispatches != 0) [encoder barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
            if (!uses.empty()) {
                std::vector<const CachedSurface*> uploaded;
                for (const auto& use : uses) {
                    if (std::find(uploaded.begin(), uploaded.end(), use.surface) != uploaded.end()) continue;
                    uploaded.push_back(use.surface);
                    textures->EncodeUpload((__bridge void*)encoder, use.surface->texture, use.resource, use.address, textures->DefaultUploadPath());
                }
                [encoder barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            }
            for (std::uint32_t slot = 0; slot < ArgumentSlots; ++slot) {
                if (slots[slot] != 0) [table setAddress:slots[slot] atIndex:slot];
            }
            [encoder setComputePipelineState:state];
            [encoder setArgumentTable:table];
            if (y > groupLimit[1] || z > groupLimit[2]) throw std::runtime_error("AGC graphics: Metal dispatch exceeds the threadgroup count limit");
            [encoder dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(threads[0], threads[1], threads[2])];
            if (std::any_of(uses.begin(), uses.end(), [](const ImageUse& use) { return use.written; })) {
                [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
                std::vector<const CachedSurface*> stored;
                for (const auto& use : uses) {
                    if (!use.written || std::find(stored.begin(), stored.end(), use.surface) != stored.end()) continue;
                    stored.push_back(use.surface);
                    textures->EncodeWriteback((__bridge void*)encoder, use.surface->texture, use.resource, use.address);
                }
            }
            ++encodedDispatches;
            ++recordedSinceSubmit;
            dispatches.fetch_add(1, std::memory_order_relaxed);
        }
    }
};

VulkanDevice::VulkanDevice(const PresentationWindow* window) : VulkanDevice(window, false) {}

VulkanDevice::VulkanDevice(CompileOnly) : VulkanDevice(nullptr, true) {}

VulkanDevice::VulkanDevice(const PresentationWindow* window, bool compileOnly) : state(std::make_unique<State>()), serial(0) {
    static std::atomic<std::uint64_t> serials{0};
    serial = serials.fetch_add(1, std::memory_order_relaxed) + 1;
    @autoreleasepool {
        auto& s = *state;
        s.compileOnly = compileOnly;
        std::string failure;
        s.device = Metal::Device::Create(&failure);
        if (s.device == nullptr) throw std::runtime_error("AGC graphics: no usable Metal 4 device: " + failure);
        s.native = (__bridge id<MTLDevice>)s.device->NativeDevice();
        s.msl = BuildMslTarget(s.native, window != nullptr);
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
