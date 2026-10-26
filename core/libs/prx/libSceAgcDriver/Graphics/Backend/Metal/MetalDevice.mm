#include "prx/libSceAgcDriver/Graphics/Backend/Backend.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/GuestRegistry.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/MetalCompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Presenter.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/RenderStateTranslate.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Sync.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureJobs.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/Textures.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestSamplerResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "BdaAbi.hpp"
#include "ImageTableAbi.hpp"
#include "MetalAbi.hpp"
#include "RuntimeAbi.hpp"
#include "MslBackend/MslEmitter.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <SDL_metal.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" void Aps5RunOnMainThread_nid_no_patch(void (*function)(void*), void* context);

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
constexpr std::size_t PresentSlots = 3;
constexpr std::uint32_t RenderSlots = ShaderRecompiler::MetalAbi::MaxBufferSlots;
constexpr std::uint32_t FirstVertexBuffer = ShaderRecompiler::MetalAbi::FirstVertexBufferSlot;

constexpr const char* ScanoutSource = R"(#include <metal_stdlib>
using namespace metal;
struct ScanoutParams { uint width; uint height; uint pitch; uint blocksPerRow; uint tiled; uint redLow; uint tenBit; };
uint scanout_tile(uint x, uint y) {
    return ((y << 4u) & 0x0070u) ^ ((y << 5u) & 0x0f00u) ^ ((y << 9u) & 0x1000u) ^ ((y << 8u) & 0x4000u) ^ ((x << 2u) & 0x000cu) ^ ((x << 5u) & 0x0380u) ^ ((x << 4u) & 0x0400u) ^ ((x << 6u) & 0x0800u) ^ ((x << 9u) & 0xa000u);
}
kernel void aps5_scanout(device const uint* source [[buffer(0)]], constant ScanoutParams& p [[buffer(1)]], texture2d<float, access::write> target [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) return;
    const uint offset = p.tiled != 0u ? ((gid.y / 128u) * p.blocksPerRow + gid.x / 128u) * 65536u + scanout_tile(gid.x, gid.y) : (gid.y * p.pitch + gid.x) * 4u;
    const uint word = source[offset / 4u];
    float4 color;
    if (p.tenBit != 0u) {
        const float low = float((word >> 2u) & 0xffu);
        const float high = float((word >> 22u) & 0xffu);
        color = float4(p.redLow != 0u ? low : high, float((word >> 12u) & 0xffu), p.redLow != 0u ? high : low, 255.0f);
    } else {
        const float byte0 = float(word & 0xffu);
        const float byte2 = float((word >> 16u) & 0xffu);
        color = float4(p.redLow != 0u ? byte0 : byte2, float((word >> 8u) & 0xffu), p.redLow != 0u ? byte2 : byte0, float(word >> 24u));
    }
    target.write(color / 255.0f, gid);
}
kernel void aps5_scanout_texture(constant uint& swapRedBlue [[buffer(0)]], texture2d<float, access::read> source [[texture(0)]], texture2d<float, access::write> target [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= target.get_width() || gid.y >= target.get_height()) return;
    const float4 color = source.read(gid);
    target.write(float4(swapRedBlue != 0u ? color.bgr : color.rgb, 1.0f), gid);
}
)";

struct ScanoutParams {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pitch;
    std::uint32_t blocksPerRow;
    std::uint32_t tiled;
    std::uint32_t redLow;
    std::uint32_t tenBit;
};

bool GpuScanout(const DisplayBuffer& buffer) {
    static const bool disabled = std::getenv("APS5_METAL_CPU_SCANOUT") != nullptr;
    return !disabled && buffer.dccAddress == 0 && !DisplayHalfFloat(buffer.pixelFormat) && !DisplayPq(buffer.pixelFormat) && DisplayTexelBytes(buffer.pixelFormat) == 4u && buffer.tilingMode <= 1u;
}

bool gpuTimeEnabled() {
    static const bool enabled = std::getenv("APS5_METAL_GPU_TIME") != nullptr;
    return enabled;
}

thread_local bool encodingNow = false;

struct EncodingScope {
    bool previous = encodingNow;
    EncodingScope() { encodingNow = true; }
    ~EncodingScope() { encodingNow = previous; }
    EncodingScope(const EncodingScope&) = delete;
    EncodingScope& operator=(const EncodingScope&) = delete;
};

struct PendingWrite {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint64_t serial = 0;
};

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

bool PagesAreDataMemory(std::uint64_t base, std::uint64_t bytes) {
    const std::uint64_t end = base + bytes;
    std::uint64_t cursor = base;
    while (cursor < end) {
        mach_vm_address_t region = cursor;
        mach_vm_size_t regionBytes = 0;
        vm_region_basic_info_data_64_t info{};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &region, &regionBytes, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || region > cursor) return false;
        constexpr vm_prot_t Required = VM_PROT_READ | VM_PROT_WRITE;
        if ((info.protection & Required) != Required || (info.protection & VM_PROT_EXECUTE) != 0) return false;
        cursor = region + regionBytes;
    }
    return true;
}

constexpr std::uint32_t MaxTextureArrayLayers = 2048;

struct HostImport {
    Metal::HostSpan span;
    id<MTLBuffer> buffer = nil;
};

struct WriteBack {
    void* host = nullptr;
    const void* staged = nullptr;
    std::size_t bytes = 0;
};

struct LabelStore {
    std::uint64_t address = 0;
    std::vector<std::byte> bytes;
};

struct Completion {
    std::vector<WriteBack> writeBacks;
    std::vector<Metal::NativeBuffer> staged;
    std::vector<LabelStore> labels;
    std::vector<std::function<void()>> actions;
    bool waking = false;
    std::chrono::steady_clock::time_point submitted;
};

struct Waker {
    std::mutex mutex;
    std::condition_variable ready;
    bool pending = false;
    bool stop = false;

    void Notify() {
        {
            std::lock_guard lock(mutex);
            pending = true;
        }
        ready.notify_one();
    }
};

enum class Numeric : std::uint8_t { Float, Uint, Sint };

constexpr std::uint32_t RawPackedFormat = 20;
constexpr std::uint32_t ReleaseMemOpcode = 0x49;

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
    Graphics::GuestTextureResource resource{};
    std::uint64_t address = 0;
    std::uint64_t guestAddress = 0;
    std::uint64_t guestBytes = 0;
    std::uint64_t validEpoch = 0;
    bool dirty = false;
    std::uint64_t drawnFrame = 0;
    bool scanned = false;
};

struct SurfaceRef {
    CachedSurface* surface = nullptr;
    Metal::TextureCreateInfo info;
    std::uint32_t layers = 1;
};

struct StageSource {
    std::shared_ptr<const std::string> source;
    Metal::CompileKey key;
    std::vector<Metal::FunctionConstant> constants;
};

struct DepthSurface {
    id<MTLTexture> texture = nil;
    bool initialized = false;
    bool stencilInitialized = false;
    std::uint64_t seenFills = 0;
    std::optional<float> fillDepth;
    std::optional<std::uint8_t> fillStencil;
    std::optional<std::uint32_t> htilePattern;
    std::vector<std::uint32_t> htileWords;
};

std::atomic<std::uint64_t>& GlobalComputePipelinesCreated() {
    static std::atomic<std::uint64_t> created{0};
    return created;
}

bool DepthFillCovers(std::uint64_t base, std::uint64_t baseBytes, std::uint64_t address, std::size_t bytes) {
    return base != 0 && base >= address && base + baseBytes <= address + bytes;
}

std::uint32_t ExpandedHtileWord(std::uint32_t pattern, bool stencilInHtile) {
    const bool depthCleared = (pattern & 0xfu) == 0u;
    const bool stencilCleared = stencilInHtile && ((pattern >> 8u) & 0x3u) == 0u;
    if (!stencilInHtile) return depthCleared ? 0xfffc000fu : pattern;
    std::uint32_t word = pattern;
    if (depthCleared) word = (word & 0x3f0u) | 0xfffff00fu;
    if (stencilCleared) word |= 0x3f0u;
    return word;
}

std::uint64_t HtileBytes(VkExtent2D extent) {
    return static_cast<std::uint64_t>((extent.width + 1023u) / 1024u) * ((extent.height + 511u) / 512u) * 32768u;
}

void ScanHtile(DepthSurface& surface, const Graphics::DepthTarget& target) {
    if (target.samples != 1u || target.htileAddress == 0u) return;
    std::vector<std::uint32_t> words(HtileBytes(target.extent) / 4u);
    try {
        GuestMemory::Read(target.htileAddress, std::as_writable_bytes(std::span(words)), 4);
    } catch (const std::exception&) {
        return;
    }
    const bool depthCleared = std::all_of(words.begin(), words.end(), [](std::uint32_t word) { return (word & 0xfu) == 0u; });
    const bool stencilCleared = target.htileStencil && std::all_of(words.begin(), words.end(), [](std::uint32_t word) { return ((word >> 8u) & 0x3u) == 0u; });
    if (!depthCleared && !stencilCleared) return;
    surface.htilePattern.reset();
    surface.htileWords = std::move(words);
    if (depthCleared) {
        surface.initialized = false;
        surface.fillDepth.reset();
    }
    if (stencilCleared) {
        surface.stencilInitialized = false;
        surface.fillStencil.reset();
    }
}

void ApplyDepthFills(DepthSurface& surface, const Graphics::DepthTarget& target) {
    const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
    for (const auto& fill : Graphics::DepthMetadataFillsSince(surface.seenFills)) {
        surface.seenFills = fill.generation;
        if (target.samples != 1u) continue;
        if (target.htileAddress != 0u && target.htileAddress >= fill.address && target.htileAddress < fill.address + fill.bytes && fill.address + fill.bytes - target.htileAddress >= HtileBytes(target.extent)) {
            const bool depthCleared = (fill.pattern & 0xfu) == 0u;
            const bool stencilCleared = target.htileStencil && ((fill.pattern >> 8u) & 0x3u) == 0u;
            surface.htileWords.clear();
            if (depthCleared || stencilCleared) surface.htilePattern = fill.pattern;
            else surface.htilePattern.reset();
            if (depthCleared) {
                surface.initialized = false;
                surface.fillDepth.reset();
            }
            if (stencilCleared) {
                surface.stencilInitialized = false;
                surface.fillStencil.reset();
            }
            continue;
        }
        if (DepthFillCovers(target.address, static_cast<std::uint64_t>(target.extent.width) * target.extent.height * (d16 ? 2u : 4u), fill.address, fill.bytes)) {
            std::optional<float> value;
            if (d16) {
                if ((fill.pattern >> 16u) == (fill.pattern & 0xffffu)) value = static_cast<float>(fill.pattern & 0xffffu) / 65535.0f;
            } else if (const auto depth = std::bit_cast<float>(fill.pattern); depth >= 0.0f && depth <= 1.0f) {
                value = depth;
            }
            if (value) {
                surface.initialized = false;
                surface.fillDepth = value;
            }
        }
        if (DepthFillCovers(target.stencilAddress, static_cast<std::uint64_t>(target.extent.width) * target.extent.height, fill.address, fill.bytes)) {
            const auto byte = static_cast<std::uint8_t>(fill.pattern & 0xffu);
            if (fill.pattern == byte * 0x01010101u) {
                surface.stencilInitialized = false;
                surface.fillStencil = byte;
            }
        }
    }
}

using DepthKey = std::array<std::uint64_t, 3>;

struct ColorAttachment {
    id<MTLTexture> texture = nil;
    std::uint32_t level = 0;
    std::uint32_t depthPlane = 0;
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

bool Trace() {
    static const bool trace = std::getenv("APS5_METAL_TRACE") != nullptr;
    return trace;
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
    if ((format == RG11B10Float || format == RGB9E5Float) && numeric != Numeric::Float) return numeric == Numeric::Uint ? R32Uint : R32Sint;
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

bool IntegerFormat(Metal::PixelFormat format) {
    using enum Metal::PixelFormat;
    switch (format) {
    case R8Uint: case R8Sint: case RG8Uint: case RG8Sint: case RGBA8Uint: case RGBA8Sint:
    case R16Uint: case R16Sint: case RG16Uint: case RG16Sint: case RGBA16Uint: case RGBA16Sint:
    case R32Uint: case R32Sint: case RG32Uint: case RG32Sint: case RGBA32Uint: case RGBA32Sint:
    case RGB10A2Uint: return true;
    default: return false;
    }
}

struct SamplerUse {
    std::uint32_t integer = 0;
    std::uint32_t other = 0;
    std::uint32_t singleLevel = 0;
    std::uint32_t mipmapped = 0;
};

SamplerUse SamplerUseOf(const ShaderRecompiler::DescriptorBinding& binding) {
    if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.imageSamplers.size() < binding.count || ImageKindOf(binding.binding).storage) return {};
    const bool integerView = ImageKindOf(binding.binding).numeric != Numeric::Float;
    SamplerUse use;
    for (std::uint32_t element = 0; element < binding.count; ++element) {
        const std::span<const std::uint32_t> words(binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 8u, 8u);
        bool integer = integerView;
        const bool empty = std::all_of(words.begin(), words.end(), [](std::uint32_t word) { return word == 0u; }) || Graphics::IsNullTextureResource(words);
        if (!integer && !empty) {
            const auto* format = Metal::FindFormat(Graphics::DecodeTextureResource(words).format);
            integer = format != nullptr && IntegerFormat(format->pixelFormat);
        }
        if (!integer && empty) continue;
        (integer ? use.integer : use.other) |= binding.imageSamplers[element];
        if (!empty) (Graphics::DescriptorSingleLevel(words) ? use.singleLevel : use.mipmapped) |= binding.imageSamplers[element];
    }
    return use;
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

MTLVertexFormat NativeVertexFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM: return MTLVertexFormatUCharNormalized;
    case VK_FORMAT_R8G8_UNORM: return MTLVertexFormatUChar2Normalized;
    case VK_FORMAT_R8G8B8_UNORM: return MTLVertexFormatUChar3Normalized;
    case VK_FORMAT_R8G8B8A8_UNORM: return MTLVertexFormatUChar4Normalized;
    case VK_FORMAT_R8_SNORM: return MTLVertexFormatCharNormalized;
    case VK_FORMAT_R8G8_SNORM: return MTLVertexFormatChar2Normalized;
    case VK_FORMAT_R8G8B8_SNORM: return MTLVertexFormatChar3Normalized;
    case VK_FORMAT_R8G8B8A8_SNORM: return MTLVertexFormatChar4Normalized;
    case VK_FORMAT_R8_UINT: return MTLVertexFormatUChar;
    case VK_FORMAT_R8G8_UINT: return MTLVertexFormatUChar2;
    case VK_FORMAT_R8G8B8_UINT: return MTLVertexFormatUChar3;
    case VK_FORMAT_R8G8B8A8_UINT: return MTLVertexFormatUChar4;
    case VK_FORMAT_R8_SINT: return MTLVertexFormatChar;
    case VK_FORMAT_R8G8_SINT: return MTLVertexFormatChar2;
    case VK_FORMAT_R8G8B8_SINT: return MTLVertexFormatChar3;
    case VK_FORMAT_R8G8B8A8_SINT: return MTLVertexFormatChar4;
    case VK_FORMAT_R16_UNORM: return MTLVertexFormatUShortNormalized;
    case VK_FORMAT_R16G16_UNORM: return MTLVertexFormatUShort2Normalized;
    case VK_FORMAT_R16G16B16_UNORM: return MTLVertexFormatUShort3Normalized;
    case VK_FORMAT_R16G16B16A16_UNORM: return MTLVertexFormatUShort4Normalized;
    case VK_FORMAT_R16_SNORM: return MTLVertexFormatShortNormalized;
    case VK_FORMAT_R16G16_SNORM: return MTLVertexFormatShort2Normalized;
    case VK_FORMAT_R16G16B16_SNORM: return MTLVertexFormatShort3Normalized;
    case VK_FORMAT_R16G16B16A16_SNORM: return MTLVertexFormatShort4Normalized;
    case VK_FORMAT_R16_UINT: return MTLVertexFormatUShort;
    case VK_FORMAT_R16G16_UINT: return MTLVertexFormatUShort2;
    case VK_FORMAT_R16G16B16_UINT: return MTLVertexFormatUShort3;
    case VK_FORMAT_R16G16B16A16_UINT: return MTLVertexFormatUShort4;
    case VK_FORMAT_R16_SINT: return MTLVertexFormatShort;
    case VK_FORMAT_R16G16_SINT: return MTLVertexFormatShort2;
    case VK_FORMAT_R16G16B16_SINT: return MTLVertexFormatShort3;
    case VK_FORMAT_R16G16B16A16_SINT: return MTLVertexFormatShort4;
    case VK_FORMAT_R16_SFLOAT: return MTLVertexFormatHalf;
    case VK_FORMAT_R16G16_SFLOAT: return MTLVertexFormatHalf2;
    case VK_FORMAT_R16G16B16_SFLOAT: return MTLVertexFormatHalf3;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return MTLVertexFormatHalf4;
    case VK_FORMAT_R32_UINT: return MTLVertexFormatUInt;
    case VK_FORMAT_R32G32_UINT: return MTLVertexFormatUInt2;
    case VK_FORMAT_R32G32B32_UINT: return MTLVertexFormatUInt3;
    case VK_FORMAT_R32G32B32A32_UINT: return MTLVertexFormatUInt4;
    case VK_FORMAT_R32_SINT: return MTLVertexFormatInt;
    case VK_FORMAT_R32G32_SINT: return MTLVertexFormatInt2;
    case VK_FORMAT_R32G32B32_SINT: return MTLVertexFormatInt3;
    case VK_FORMAT_R32G32B32A32_SINT: return MTLVertexFormatInt4;
    case VK_FORMAT_R32_SFLOAT: return MTLVertexFormatFloat;
    case VK_FORMAT_R32G32_SFLOAT: return MTLVertexFormatFloat2;
    case VK_FORMAT_R32G32B32_SFLOAT: return MTLVertexFormatFloat3;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return MTLVertexFormatFloat4;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return MTLVertexFormatFloatRG11B10;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return MTLVertexFormatUInt1010102Normalized;
    case VK_FORMAT_A2B10G10R10_SNORM_PACK32: return MTLVertexFormatInt1010102Normalized;
    default: return MTLVertexFormatInvalid;
    }
}

bool HasFlatInputs(std::span<const Graphics::CompiledShader> shaders) {
    for (const auto& shader : shaders) {
        if (shader.stage != ShaderRecompiler::ShaderStage::Fragment) continue;
        for (const auto& parameter : shader.program->fragmentParameters) {
            if (parameter.flat || parameter.flatP10 || parameter.flatP20 || parameter.vertexOrder || parameter.perVertex) return true;
        }
    }
    return false;
}

std::optional<std::string> DrawRejection(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> shaders) {
    for (const auto& shader : shaders) {
        if (shader.stage != ShaderRecompiler::ShaderStage::Vertex && shader.stage != ShaderRecompiler::ShaderStage::Fragment) return std::string("Metal draws support only vertex and pixel stages");
        if (shader.stage != ShaderRecompiler::ShaderStage::Vertex) continue;
        for (const auto& attribute : shader.program->vertexAttributes) {
            const auto format = Graphics::DecodeVertexFormat(attribute).format;
            if (NativeVertexFormat(format) == MTLVertexFormatInvalid) return "vertex format " + std::to_string(static_cast<int>(format)) + " has no Metal vertex format";
        }
    }
    const auto translated = Metal::TranslateState(graphics, HasFlatInputs(shaders));
    if (translated.rejection != nullptr) return std::string(translated.rejection);
    if (graphics.samples > 1 || std::any_of(graphics.colors.begin(), graphics.colors.end(), [](const Graphics::ColorTarget& color) { return color.samples > 1; }) || (graphics.depth && graphics.depth->samples > 1)) return std::string("multisampled Metal draws are not wired yet");
    return std::nullopt;
}

std::uint32_t MslExperiment() {
    static const std::uint32_t mask = [] {
        const char* text = std::getenv("APS5_MSL_EXPERIMENT");
        if (text == nullptr) return 0u;
        const std::string list = text;
        std::uint32_t result = 0;
        if (list.find("nodiscard") != std::string::npos) result |= ShaderRecompiler::MslExperimentNoDiscard;
        if (list.find("nopow") != std::string::npos) result |= ShaderRecompiler::MslExperimentNoPow;
        if (list.find("nouniform") != std::string::npos) result |= ShaderRecompiler::MslExperimentNoUniformLoad;
        if (list.find("constout") != std::string::npos) result |= ShaderRecompiler::MslExperimentConstantOutput;
        std::fprintf(stderr, "[metal] msl experiment %s (mask %u): shader output is wrong on purpose\n", text, result);
        return result;
    }();
    return mask;
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
    target.experiment = MslExperiment();
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
    id<MTL4ArgumentTable> vertexTable = nil;
    id<MTL4ArgumentTable> fragmentTable = nil;
    ShaderRecompiler::MslTarget msl;
    ShaderRecompiler::SpirvTarget target{};
    std::array<std::uint32_t, 3> groupLimit{};
    bool compileOnly = false;

    std::recursive_mutex mutex;
    Metal::Recording open;
    id<MTL4ComputeCommandEncoder> encoder = nil;
    id<MTL4RenderCommandEncoder> render = nil;
    std::vector<std::uint64_t> passKey;
    std::uint32_t passWidth = 0;
    std::uint32_t passHeight = 0;
    void* boundPipeline = nullptr;
    void* boundDepthStencil = nullptr;
    Metal::DynamicState boundDynamic;
    bool dynamicValid = false;
    std::uint64_t epoch = 1;
    std::uint32_t encodedDispatches = 0;
    std::uint64_t recordedSinceSubmit = 0;
    std::uint32_t cbDraws = 0;
    std::uint32_t cbPasses = 0;
    std::uint32_t cbDispatches = 0;
    std::uint32_t cbWritebacks = 0;
    std::uint32_t cbScanouts = 0;
    std::uint32_t cbUploads = 0;
    std::uint64_t cbMovedBytes = 0;
    std::string cbLastDraw;
    std::uint64_t lastSerial = 0;
    const char* resolvedBy = "";
    std::vector<PendingWrite> pendingWrites;
    std::atomic<std::size_t> pendingCount{0};
    std::atomic<std::uint32_t> hookUsers{0};
    std::atomic<std::uint64_t> hookSubmits{0};
    std::atomic<std::uint64_t> hookWaits{0};
    std::uint64_t reportSubmits = 0;
    std::uint64_t reportWaits = 0;
    std::array<std::uint64_t, 3> reportLabels{};
    std::uint64_t snapshotBytes = 0;
    std::uint64_t reportSnapshotBytes = 0;

    std::vector<std::unique_ptr<UploadChunk>> chunks;
    UploadChunk* chunk = nullptr;
    std::vector<UploadChunk*> sealing;

    std::vector<HostImport> imports;
    std::vector<Metal::NativeBuffer> staged;
    std::vector<WriteBack> writeBacks;
    std::map<std::uint64_t, Completion> completions;
    std::shared_ptr<Waker> waker;
    std::thread completer;
    std::atomic<std::uint64_t> labelsOnCompletion{0};
    std::atomic<std::uint64_t> labelsOnCpu{0};
    std::atomic<std::uint64_t> actionsOnCompletion{0};

    Metal::NativeBuffer fault;

    std::unique_ptr<Metal::TextureManager> textures;
    std::map<SurfaceKey, CachedSurface> surfaces;
    std::map<SamplerKey, id<MTLSamplerState>> samplers;
    std::array<id<MTLTexture>, 4> nullTextures{};
    std::vector<CachedSurface*> dirtySurfaces;
    std::map<DepthKey, DepthSurface> depthSurfaces;
    std::unique_ptr<Metal::PipelineCache> renderPipelines;
    std::unordered_map<ShaderRecompiler::PipelineIdentity, StageSource, IdentityHash> stageSources;
    std::unordered_map<Metal::CompileKey, std::shared_ptr<Metal::RenderPipeline>, Metal::CompileKeyHash> drawPipelines;

    void* window = nullptr;
    std::unique_ptr<Metal::Presenter> presenter;
    std::array<id<MTLTexture>, PresentSlots> presentTextures{};
    std::array<std::uint64_t, PresentSlots> presentSerials{};
    std::size_t presentSlot = 0;
    std::optional<std::size_t> pendingPresent;
    std::uint32_t presentWidth = 0;
    std::uint32_t presentHeight = 0;
    std::uint64_t presentAfter = 0;
    id<MTLComputePipelineState> scanoutPipeline = nil;
    id<MTLComputePipelineState> scanoutTexturePipeline = nil;
    id<MTL4ArgumentTable> scanoutTable = nil;
    std::chrono::steady_clock::time_point reportAt{};
    std::uint64_t reportFrames = 0;
    std::uint64_t scanoutFrames = 1;
    std::uint64_t directDeclined = 0;
    std::array<double, 5> presentMs{};
    std::uint64_t writebackSurfaces = 0;
    std::uint64_t flushes = 0;
    std::uint64_t completedFlushes = 0;
    double completionMsTotal = 0.0;
    double completionMsMax = 0.0;
    std::uint64_t reportDraws = 0;
    std::uint64_t reportDispatches = 0;
    Metal::PresenterStats reportStats{};

    std::mutex pipelineMutex;
    std::unordered_map<ShaderRecompiler::PipelineIdentity, PipelineEntry, IdentityHash> pipelines;
    std::atomic<std::uint64_t> pipelinesCreated{0};
    std::atomic<std::uint64_t> dispatches{0};
    std::atomic<std::uint64_t> draws{0};

    ~State() {
        if (waker != nullptr) {
            {
                std::lock_guard lock(waker->mutex);
                waker->stop = true;
            }
            waker->ready.notify_one();
            if (completer.joinable()) completer.join();
        }
        if (device == nullptr) return;
        {
            std::lock_guard lock(targetsMutex());
            std::erase(targets(), this);
        }
        while (hookUsers.load(std::memory_order_acquire) != 0) std::this_thread::yield();
        try {
            flush();
            drain(lastSerial);
        } catch (...) {
        }
        pendingTotal().fetch_sub(pendingWrites.size(), std::memory_order_acq_rel);
        pendingWrites.clear();
        drawPipelines.clear();
        renderPipelines.reset();
        for (auto& [key, depth] : depthSurfaces) {
            id<MTLTexture> retired = depth.texture;
            depth.texture = nil;
            device->RetireAllocation((__bridge void*)retired, {});
        }
        depthSurfaces.clear();
        for (auto& [key, surface] : surfaces) {
            for (auto& [view, handle] : surface.views) textures->Release(handle);
            textures->Release(surface.texture);
        }
        surfaces.clear();
        textures.reset();
        presenter.reset();
        for (auto& texture : presentTextures) {
            if (texture == nil) continue;
            id<MTLTexture> retired = texture;
            texture = nil;
            device->RetireAllocation((__bridge void*)retired, {});
        }
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
        if (span.bytes == 0 || span.bytes > device->Limits().maxBufferBytes || !PagesAreDataMemory(span.base, span.bytes)) return nullptr;
        id<MTLBuffer> buffer = [native newBufferWithBytesNoCopy:reinterpret_cast<void*>(span.base) length:span.bytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
        if (buffer == nil) return nullptr;
        buffer.label = @"AnyPS5 host import";
        device->AddResidentAllocation((__bridge void*)buffer);
        imports.push_back({span, buffer});
        return &imports.back();
    }

    std::uint64_t resolve(std::uint64_t address, std::uint64_t bytes, bool writable, std::span<const Graphics::GuestMemorySnapshot> snapshots, const CachedSurface* self = nullptr, bool settle = true) {
        if (address == 0 || bytes == 0) return 0;
        if (settle) writebackOverlapping(address, bytes, self);
        if (!writable) {
            for (const auto& snapshot : snapshots) {
                if (address >= snapshot.address && bytes <= snapshot.bytes.size() && address - snapshot.address <= snapshot.bytes.size() - bytes) {
                    resolvedBy = "snapshot";
                    snapshotBytes += bytes;
                    return uploadBytes(snapshot.bytes.data() + (address - snapshot.address), bytes, bytes);
                }
            }
        }
        const auto windows = device->Guest().Snapshot();
        if (windows != nullptr) {
            if (const auto resolved = Metal::Resolve(*windows, address, bytes); resolved && (!writable || (resolved.window->writable && !resolved.window->copied))) {
                resolvedBy = "window";
                if (writable) noteWrite(address, bytes);
                return resolved.deviceAddress;
            }
        }
        if (const auto* import = importHost(address, bytes)) {
            resolvedBy = "import";
            if (writable) noteWrite(address, bytes);
            return import->buffer.gpuAddress + (address - import->span.base);
        }
        resolvedBy = writable ? "staged, written back" : "staged";
        if (writable) noteWrite(address, bytes);
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
        open = sync->Begin(Queue, "AnyPS5 commands");
    }

    void endRender() {
        if (render == nil) return;
        [render endEncoding];
        render = nil;
        passKey.clear();
    }

    void endCompute() {
        if (encoder == nil) return;
        [encoder endEncoding];
        encoder = nil;
    }

    id<MTL4ComputeCommandEncoder> computeEncoder() {
        begin();
        endRender();
        if (encoder != nil) return encoder;
        id<MTL4CommandBuffer> commands = (__bridge id<MTL4CommandBuffer>)open.commands;
        encoder = [commands computeCommandEncoder];
        if (encoder == nil) throw std::runtime_error("AGC graphics: Metal compute encoder could not be created");
        [encoder barrierAfterQueueStages:MTLStageDispatch | MTLStageBlit | MTLStageVertex | MTLStageFragment beforeStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        encodedDispatches = 0;
        return encoder;
    }

    void encodeWritebacks(const std::vector<CachedSurface*>& surfaces) {
        auto compute = computeEncoder();
        [compute barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        writebackSurfaces += surfaces.size();
        cbWritebacks += static_cast<std::uint32_t>(surfaces.size());
        for (auto* surface : surfaces) {
            textures->EncodeWriteback((__bridge void*)compute, surface->texture, surface->resource, surface->address);
            cbMovedBytes += surface->guestBytes;
            surface->dirty = false;
        }
        [compute barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    }

    void writebackDirty(bool scannedToo = true) {
        if (dirtySurfaces.empty()) return;
        if (scannedToo) {
            encodeWritebacks(dirtySurfaces);
            dirtySurfaces.clear();
            return;
        }
        std::vector<CachedSurface*> settle;
        for (auto* surface : dirtySurfaces) {
            if (!surface->scanned) settle.push_back(surface);
        }
        if (settle.empty()) return;
        encodeWritebacks(settle);
        std::erase_if(dirtySurfaces, [](const CachedSurface* surface) { return !surface->dirty; });
    }

    bool writebackOverlapping(std::uint64_t address, std::uint64_t bytes, const CachedSurface* self) {
        if (dirtySurfaces.empty() || bytes == 0) return false;
        std::vector<CachedSurface*> hit;
        for (auto* surface : dirtySurfaces) {
            if (surface != self && surface->guestAddress < address + bytes && address < surface->guestAddress + surface->guestBytes) hit.push_back(surface);
        }
        if (hit.empty()) return false;
        encodeWritebacks(hit);
        std::erase_if(dirtySurfaces, [](const CachedSurface* surface) { return !surface->dirty; });
        return true;
    }

    void flush(bool settleSurfaces = true) {
        if (!open) return;
        @autoreleasepool {
            if (settleSurfaces) writebackDirty(false);
            endRender();
            endCompute();
            sync->EncodeLabels(open);
            if (gpuTimeEnabled()) {
                char tag[160];
                std::snprintf(tag, sizeof(tag), "draws %u, passes %u, dispatches %u, writebacks %u, uploads %u, scanouts %u, %llu MB moved", cbDraws, cbPasses, cbDispatches, cbWritebacks, cbUploads, cbScanouts, static_cast<unsigned long long>(cbMovedBytes >> 20u));
                device->TagNextSubmit(std::string(tag) + ", " + cbLastDraw);
                cbLastDraw.clear();
            }
            cbDraws = cbPasses = cbDispatches = cbWritebacks = cbScanouts = cbUploads = 0;
            cbMovedBytes = 0;
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
            completion.submitted = std::chrono::steady_clock::now();
            ++flushes;
            completion.writeBacks = std::move(writeBacks);
            completion.staged = std::move(staged);
            writeBacks.clear();
            staged.clear();
            if (textures != nullptr) textures->Seal(Queue, serial);
            lastSerial = serial;
            for (auto& write : pendingWrites) {
                if (write.serial == 0) write.serial = serial;
            }
            for (const auto* surface : dirtySurfaces) noteWrite(surface->guestAddress, surface->guestBytes);
            recordedSinceSubmit = 0;
            ++epoch;
        }
    }

    void complete(std::uint64_t through) {
        while (!completions.empty() && completions.begin()->first <= through) {
            auto entry = std::move(completions.begin()->second);
            completions.erase(completions.begin());
            const auto latency = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - entry.submitted).count();
            completionMsTotal += latency;
            completionMsMax = std::max(completionMsMax, latency);
            ++completedFlushes;
            for (const auto& item : entry.writeBacks) std::memcpy(item.host, item.staged, item.bytes);
            for (const auto& buffer : entry.staged) device->ReleaseBuffer(buffer);
            for (const auto& label : entry.labels) storeLabel(label);
            for (auto& action : entry.actions) action();
        }
        const auto before = pendingWrites.size();
        std::erase_if(pendingWrites, [&](const PendingWrite& write) { return write.serial != 0 && write.serial <= through; });
        if (pendingTotal().fetch_sub(before - pendingWrites.size(), std::memory_order_acq_rel) == before - pendingWrites.size()) {
            pendingLow().store(~0ull, std::memory_order_release);
            pendingHigh().store(0, std::memory_order_release);
        }
        pendingCount.store(pendingWrites.size(), std::memory_order_release);
        checkFault();
    }

    static void storeLabel(const LabelStore& label) {
        if (label.bytes.size() == 8 && label.address % 8 == 0) {
            std::uint64_t value;
            std::memcpy(&value, label.bytes.data(), 8);
            std::atomic_ref<std::uint64_t>(*reinterpret_cast<std::uint64_t*>(label.address)).store(value, std::memory_order_release);
            return;
        }
        for (std::size_t done = 0; done < label.bytes.size(); done += 4) {
            std::uint32_t value;
            std::memcpy(&value, label.bytes.data() + done, 4);
            std::atomic_ref<std::uint32_t>(*reinterpret_cast<std::uint32_t*>(label.address + done)).store(value, std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_release);
    }

    Completion* afterRecorded() {
        reap();
        if (open) flush(false);
        if (completions.empty()) return nullptr;
        auto& [serial, completion] = *completions.rbegin();
        if (!completion.waking) {
            completion.waking = true;
            if (waker == nullptr) {
                waker = std::make_shared<Waker>();
                completer = std::thread([this, signal = waker] { runCompleter(*signal); });
            }
            sync->OnComplete(Queue, serial, [signal = waker] { signal->Notify(); });
        }
        return &completion;
    }

    void runCompleter(Waker& signal) {
        for (;;) {
            {
                std::unique_lock lock(signal.mutex);
                signal.ready.wait(lock, [&] { return signal.pending || signal.stop; });
                if (signal.stop) return;
                signal.pending = false;
            }
            try {
                std::lock_guard lock(mutex);
                reap();
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[metal] completion failed: %s\n", error.what());
            }
        }
    }

    void noteWrite(std::uint64_t address, std::uint64_t bytes, std::uint64_t serial = 0) {
        const auto end = address + bytes;
        if (!pendingWrites.empty() && pendingWrites.back().serial == serial && pendingWrites.back().begin <= address && pendingWrites.back().end >= end) return;
        pendingWrites.push_back({address, end, serial});
        for (auto low = pendingLow().load(std::memory_order_acquire); address < low && !pendingLow().compare_exchange_weak(low, address, std::memory_order_acq_rel);) {}
        for (auto high = pendingHigh().load(std::memory_order_acquire); end > high && !pendingHigh().compare_exchange_weak(high, end, std::memory_order_acq_rel);) {}
        pendingTotal().fetch_add(1, std::memory_order_acq_rel);
        pendingCount.store(pendingWrites.size(), std::memory_order_release);
    }

    static std::mutex& targetsMutex() {
        static std::mutex lock;
        return lock;
    }

    static std::vector<State*>& targets() {
        static std::vector<State*> list;
        return list;
    }

    static std::atomic<std::size_t>& pendingTotal() {
        static std::atomic<std::size_t> total{0};
        return total;
    }

    static std::atomic<std::uint64_t>& pendingLow() {
        static std::atomic<std::uint64_t> low{~0ull};
        return low;
    }

    static std::atomic<std::uint64_t>& pendingHigh() {
        static std::atomic<std::uint64_t> high{0};
        return high;
    }

    static void flushHook(std::uint64_t address, std::size_t bytes) {
        if (pendingTotal().load(std::memory_order_acquire) == 0) return;
        if (address + bytes <= pendingLow().load(std::memory_order_acquire) || address >= pendingHigh().load(std::memory_order_acquire)) return;
        static thread_local std::vector<State*> active;
        active.clear();
        {
            std::lock_guard lock(targetsMutex());
            for (auto* target : targets()) {
                if (target->pendingCount.load(std::memory_order_acquire) == 0) continue;
                target->hookUsers.fetch_add(1, std::memory_order_acq_rel);
                active.push_back(target);
            }
        }
        const auto targetsHeld = active;
        std::exception_ptr failure;
        for (auto* target : targetsHeld) {
            try {
                if (!failure) target->flushFor(address, bytes);
            } catch (...) {
                failure = std::current_exception();
            }
            target->hookUsers.fetch_sub(1, std::memory_order_acq_rel);
        }
        if (failure) std::rethrow_exception(failure);
    }

    void flushFor(std::uint64_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        reap();
        if (!encodingNow && writebackOverlapping(address, bytes, nullptr)) {
            flush();
            ++hookSubmits;
        }
        const auto end = address + bytes;
        bool open = false;
        std::uint64_t wait = 0;
        for (const auto& write : pendingWrites) {
            if (write.end <= address || write.begin >= end) continue;
            if (write.serial == 0) open = true;
            else wait = std::max(wait, write.serial);
        }
        if (open && !encodingNow) {
            flush();
            wait = lastSerial;
            ++hookSubmits;
        }
        if (wait == 0) return;
        if (device->CompletedSerial(Queue) < wait) ++hookWaits;
        drain(wait);
    }

    bool copyWhenIdle(std::uint64_t destination, std::uint64_t source, std::size_t bytes) {
        std::lock_guard lock(mutex);
        reap();
        if (open || encodingNow || !completions.empty() || !pendingWrites.empty() || recordedSinceSubmit != 0) return false;
        GuestMemory::StoreOwnBytes(destination, bytes, [&] { std::memcpy(reinterpret_cast<void*>(destination), reinterpret_cast<const void*>(source), bytes); });
        return true;
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
        const auto address = static_cast<unsigned long long>(record->address);
        if (record->reason == ShaderRecompiler::BdaAbi::FaultReason::UnlinkedCall) std::fprintf(stderr, "[metal] shader stage %u called address 0x%llx at program counter 0x%x, which no linked callee of that call matches\n", record->stage, address, static_cast<unsigned>(record->instruction));
        else if (record->reason == ShaderRecompiler::BdaAbi::FaultReason::WindowLaneSpill) std::fprintf(stderr, "[metal] shader stage %u: a relative write stored %u in lane %llu of vector window slot %llu, which holds a scalar spilled with v_writelane_b32 that a later v_readlane_b32 was replaced with\n", record->stage, record->bytes, address >> 32u, address & 0xffffffffu);
        else std::fprintf(stderr, "[metal] BDA access failed: address=0x%llx instruction=0x%x bytes=%u stage=%u reason=%u\n", address, static_cast<unsigned>(record->instruction), record->bytes, record->stage, static_cast<unsigned>(record->reason));
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
                if (Trace()) {
                    std::fprintf(stderr, "[metal] compute pipeline %016llx%016llx:\n%.*s\n", static_cast<unsigned long long>(identity.high), static_cast<unsigned long long>(identity.low), static_cast<int>(source->size()), source->data());
                    for (const auto& constant : request.constants) std::fprintf(stderr, "[metal] function constant %u = 0x%x\n", constant.index, constant.value);
                }
                ticket = compiler->Compute(request);
                pipelines.emplace(identity, PipelineEntry{std::move(source), ticket});
                pipelinesCreated.fetch_add(1, std::memory_order_relaxed);
                GlobalComputePipelinesCreated().fetch_add(1, std::memory_order_relaxed);
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

    void attachWindow(const PresentationWindow& target) {
        if (target.createSurface == nullptr) throw std::runtime_error("AGC graphics: the presentation window has no Metal layer provider");
        const auto layer = reinterpret_cast<void*>(target.createSurface(target.context, nullptr));
        if (layer == nullptr) throw std::runtime_error("AGC graphics: the video output could not create a Metal layer for its window");
        std::string failure;
        presenter = Metal::Presenter::Create(*device, layer, Metal::PresenterOptions::FromEnvironment(), &failure);
        if (presenter == nullptr) throw std::runtime_error("AGC graphics: Metal presenter could not be created: " + failure);
        window = target.context;
    }

    void stagePresent(std::uint32_t width, std::uint32_t height, std::span<const std::byte> bgra) {
        if (width == 0 || height == 0 || bgra.size() < static_cast<std::size_t>(width) * height * 4u) throw std::runtime_error("AGC graphics: Metal present source is empty or too small");
        const auto slot = presentTarget(width, height);
        [presentTextures[slot] replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 withBytes:bgra.data() bytesPerRow:static_cast<NSUInteger>(width) * 4u];
        presentAfter = 0;
        pendingPresent = slot;
    }

    std::size_t presentTarget(std::uint32_t width, std::uint32_t height) {
        const auto slot = presentSlot;
        presentSlot = (presentSlot + 1u) % PresentSlots;
        if (presentSerials[slot] != 0 && !device->WaitSerial(Queue, presentSerials[slot], WaitTimeoutMs)) throw std::runtime_error("AGC graphics: Metal present slot did not retire in time");
        auto& texture = presentTextures[slot];
        if (texture == nil || texture.width != width || texture.height != height) {
            if (texture != nil) {
                id<MTLTexture> retired = texture;
                texture = nil;
                device->RetireAllocation((__bridge void*)retired, {});
            }
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:width height:height mipmapped:NO];
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            texture = [native newTextureWithDescriptor:descriptor];
            if (texture == nil) throw std::runtime_error("AGC graphics: Metal present texture could not be created");
            texture.label = @"AnyPS5 present source";
            device->AddResidentAllocation((__bridge void*)texture);
        }
        presentWidth = width;
        presentHeight = height;
        return slot;
    }

    id<MTLComputePipelineState> scanout() {
        if (scanoutPipeline != nil) return scanoutPipeline;
        NSError* error = nil;
        id<MTLLibrary> library = [native newLibraryWithSource:[NSString stringWithUTF8String:ScanoutSource] options:nil error:&error];
        id<MTLFunction> function = library != nil ? [library newFunctionWithName:@"aps5_scanout"] : nil;
        if (function == nil) throw std::runtime_error(std::string("AGC graphics: Metal scanout kernel failed to compile: ") + (error.localizedDescription.UTF8String ?: ""));
        scanoutPipeline = [native newComputePipelineStateWithFunction:function error:&error];
        if (scanoutPipeline == nil) throw std::runtime_error(std::string("AGC graphics: Metal scanout pipeline failed to build: ") + (error.localizedDescription.UTF8String ?: ""));
        id<MTLFunction> textureFunction = [library newFunctionWithName:@"aps5_scanout_texture"];
        scanoutTexturePipeline = textureFunction != nil ? [native newComputePipelineStateWithFunction:textureFunction error:&error] : nil;
        if (scanoutTexturePipeline == nil) throw std::runtime_error(std::string("AGC graphics: Metal texture scanout pipeline failed to build: ") + (error.localizedDescription.UTF8String ?: ""));
        MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 2;
        descriptor.maxTextureBindCount = 2;
        scanoutTable = [native newArgumentTableWithDescriptor:descriptor error:&error];
        if (scanoutTable == nil) throw std::runtime_error("AGC graphics: Metal scanout argument table could not be created");
        return scanoutPipeline;
    }

    // A frame that was just rendered into a cached surface is scanned out straight from its texture: no 4K writeback to guest memory and no detiling kernel.
    static bool DirectScanoutFormat(MTLPixelFormat format, std::uint32_t& redLow) {
        switch (format) {
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGB10A2Unorm: redLow = 1u; return true;
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGR10A2Unorm: redLow = 0u; return true;
        default: return false;
        }
    }

    bool stageDirectScanout(const DisplayBuffer& buffer) {
        ++scanoutFrames;
        if (buffer.tilingMode > 1u) return false;
        CachedSurface* found = nullptr;
        std::uint32_t textureRedLow = 0;
        for (auto& [key, surface] : surfaces) {
            if (surface.guestAddress != buffer.address || scanoutFrames - surface.drawnFrame > 2u) continue;
            id<MTLTexture> texture = (__bridge id<MTLTexture>)surface.texture.texture;
            std::uint32_t redLow = 0;
            if (texture == nil || (texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType2DArray) || texture.sampleCount != 1 || texture.width != buffer.width || texture.height != buffer.height || !DirectScanoutFormat(texture.pixelFormat, redLow)) continue;
            if (found != nullptr && found->drawnFrame > surface.drawnFrame) continue;
            found = &surface;
            textureRedLow = redLow;
        }
        if (found == nullptr) {
            if (directDeclined++ < 3u || directDeclined % 1200u == 0u) {
                std::fprintf(stderr, "[metal] direct scanout declined (#%llu): display buffer %llx %ux%u tiling %u format %llx; surfaces:", static_cast<unsigned long long>(directDeclined), static_cast<unsigned long long>(buffer.address), buffer.width, buffer.height, static_cast<unsigned>(buffer.tilingMode), static_cast<unsigned long long>(buffer.pixelFormat));
                std::size_t listed = 0;
                for (auto& [key, surface] : surfaces) {
                    if (surface.guestAddress == 0 || scanoutFrames - surface.drawnFrame > 2u || listed++ >= 6u) continue;
                    id<MTLTexture> texture = (__bridge id<MTLTexture>)surface.texture.texture;
                    std::fprintf(stderr, " [%llx %lux%lu fmt %lu type %lu samples %lu dirty %d]", static_cast<unsigned long long>(surface.guestAddress), static_cast<unsigned long>(texture.width), static_cast<unsigned long>(texture.height), static_cast<unsigned long>(texture.pixelFormat), static_cast<unsigned long>(texture.textureType), static_cast<unsigned long>(texture.sampleCount), surface.dirty ? 1 : 0);
                }
                std::fprintf(stderr, "\n");
            }
            return false;
        }
        id<MTLTexture> texture = (__bridge id<MTLTexture>)found->texture.texture;
        if (texture.textureType == MTLTextureType2DArray) {
            Metal::ViewInfo view;
            view.pixelFormat = found->texture.pixelFormat;
            view.kind = Metal::TextureKind::Tex2D;
            view.changeKind = true;
            texture = (__bridge id<MTLTexture>)viewOf(*found, view).texture;
        }
        found->scanned = true;
        const std::uint32_t swap = textureRedLow != (DisplayRedLow(buffer.pixelFormat) ? 1u : 0u) ? 1u : 0u;
        const auto slot = presentTarget(buffer.width, buffer.height);
        auto compute = computeEncoder();
        [compute barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        [scanoutTable setAddress:uploadBytes(&swap, sizeof(swap)) atIndex:0];
        [scanoutTable setTexture:texture.gpuResourceID atIndex:0];
        [scanoutTable setTexture:presentTextures[slot].gpuResourceID atIndex:1];
        [compute setComputePipelineState:scanoutTexturePipeline];
        [compute setArgumentTable:scanoutTable];
        [compute dispatchThreads:MTLSizeMake(buffer.width, buffer.height, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        ++encodedDispatches;
        ++cbScanouts;
        flush(false);
        reap();
        presentAfter = lastSerial;
        pendingPresent = slot;
        return true;
    }

    void stageScanout(const DisplayBuffer& buffer) {
        const auto bytes = DisplayBufferSize(buffer);
        GuestMemory::CheckRange(reinterpret_cast<const void*>(buffer.address), bytes, buffer.alignment);
        auto state = scanout();
        registry->Refresh();
        begin();
        if (stageDirectScanout(buffer)) return;
        writebackDirty();
        const auto source = resolve(buffer.address, bytes, false, {});
        const ScanoutParams params{buffer.width, buffer.height, buffer.pitchInPixel == 0 ? buffer.width : buffer.pitchInPixel, (buffer.width + 127u) / 128u, buffer.tilingMode == 0 ? 1u : 0u, DisplayRedLow(buffer.pixelFormat) ? 1u : 0u, DisplayTenBit(buffer.pixelFormat) ? 1u : 0u};
        const auto slot = presentTarget(buffer.width, buffer.height);
        auto compute = computeEncoder();
        [compute barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        [scanoutTable setAddress:source atIndex:0];
        [scanoutTable setAddress:uploadBytes(&params, sizeof(params)) atIndex:1];
        [scanoutTable setTexture:presentTextures[slot].gpuResourceID atIndex:0];
        [compute setComputePipelineState:state];
        [compute setArgumentTable:scanoutTable];
        [compute dispatchThreads:MTLSizeMake(buffer.width, buffer.height, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        ++encodedDispatches;
        ++cbScanouts;
        flush();
        reap();
        presentAfter = lastSerial;
        pendingPresent = slot;
    }

    double presentPending() {
        if (!pendingPresent) return 0.0;
        const auto slot = *pendingPresent;
        pendingPresent.reset();
        Metal::PresentSource source;
        source.texture = (__bridge void*)presentTextures[slot];
        source.width = presentWidth;
        source.height = presentHeight;
        const auto scanned = presentAfter;
        presentAfter = 0;
        const auto began = std::chrono::steady_clock::now();
        if (scanned != 0) device->WaitOnQueue(Queue, Queue, scanned);
        const auto result = presenter->Present(source);
        const auto presented = std::chrono::steady_clock::now();
        if (result.serial != 0) presentSerials[slot] = result.serial;
        if (scanned != 0 && !device->WaitSerial(Queue, scanned, WaitTimeoutMs)) throw std::runtime_error("AGC graphics: Metal scanout did not finish in time");
        const auto waited = std::chrono::steady_clock::now();
        presentMs[0] += std::chrono::duration<double, std::milli>(presented - began).count();
        presentMs[1] += std::chrono::duration<double, std::milli>(waited - presented).count();
        presentMs[2] += result.pacedMs;
        presentMs[3] += result.waitMs;
        reportPresents();
        return result.pacedMs + result.waitMs;
    }

    void reportPresents() {
        const auto now = std::chrono::steady_clock::now();
        ++reportFrames;
        const bool first = reportAt == std::chrono::steady_clock::time_point{};
        if (!first && now - reportAt < std::chrono::seconds(10)) return;
        const auto stats = presenter->Stats();
        const auto seconds = first ? 0.0 : std::chrono::duration<double>(now - reportAt).count();
        const auto presented = stats.presented - reportStats.presented;
        const auto drawCount = draws.load(std::memory_order_relaxed);
        const auto dispatchCount = dispatches.load(std::memory_order_relaxed);
        const std::array<std::uint64_t, 3> labels{labelsOnCompletion.load(std::memory_order_relaxed), labelsOnCpu.load(std::memory_order_relaxed), actionsOnCompletion.load(std::memory_order_relaxed)};
        std::fprintf(stderr, "[metal] present: %llu flips in %.1f s (%.1f fps), %llu presented, %llu suspended, %llu without drawable, %llu upscaled; source %ux%u, refresh %.0f Hz, effective %.1f Hz; %llu draws, %llu dispatches; CPU accesses to GPU-written ranges: %llu submits, %llu waits; labels: %llu on completion, %llu on CPU, %llu completion actions; %llu KiB snapshot uploads\n",
                     static_cast<unsigned long long>(reportFrames), seconds, seconds > 0.0 ? static_cast<double>(presented) / seconds : 0.0, static_cast<unsigned long long>(presented),
                     static_cast<unsigned long long>(stats.suspended - reportStats.suspended), static_cast<unsigned long long>(stats.noDrawable - reportStats.noDrawable),
                     static_cast<unsigned long long>(stats.upscaled - reportStats.upscaled), presentWidth, presentHeight, stats.refreshHz, stats.effectiveHz,
                     static_cast<unsigned long long>(drawCount - reportDraws), static_cast<unsigned long long>(dispatchCount - reportDispatches), static_cast<unsigned long long>(hookSubmits - reportSubmits), static_cast<unsigned long long>(hookWaits - reportWaits),
                     static_cast<unsigned long long>(labels[0] - reportLabels[0]), static_cast<unsigned long long>(labels[1] - reportLabels[1]), static_cast<unsigned long long>(labels[2] - reportLabels[2]),
                     static_cast<unsigned long long>((snapshotBytes - reportSnapshotBytes) / 1024));
        if (reportFrames != 0u) std::fprintf(stderr, "[metal] present per flip: %.2f ms Present call (paced %.2f ms, drawable wait %.2f ms), %.2f ms wait for scanout serial\n", presentMs[0] / static_cast<double>(reportFrames), presentMs[2] / static_cast<double>(reportFrames), presentMs[3] / static_cast<double>(reportFrames), presentMs[1] / static_cast<double>(reportFrames));
        std::fprintf(stderr, "[metal] flushes %llu, writeback surfaces %llu, submit-to-completion %.2f ms avg, %.2f ms max over %llu completions\n", static_cast<unsigned long long>(flushes), static_cast<unsigned long long>(writebackSurfaces), completedFlushes != 0u ? completionMsTotal / static_cast<double>(completedFlushes) : 0.0, completionMsMax, static_cast<unsigned long long>(completedFlushes));
        if (std::getenv("APS5_METAL_GPU_TIME") != nullptr) {
            NSProcessInfo* process = [NSProcessInfo processInfo];
            std::fprintf(stderr, "[metal] device state: thermal %ld, low power mode %d, allocated %.0f MB of %.0f MB recommended\n", static_cast<long>(process.thermalState), process.lowPowerModeEnabled ? 1 : 0, static_cast<double>(native.currentAllocatedSize) / 1048576.0, static_cast<double>(native.recommendedMaxWorkingSetSize) / 1048576.0);
        }
        const auto gpu = device->TakeGpuTime();
        if (gpu.enabled) std::fprintf(stderr, "[metal] gpu busy %.1f ms over %llu command buffers in %.1f s (%.1f %% of wall), longest %.2f ms (%s)\n", gpu.busyMs, static_cast<unsigned long long>(gpu.commandBuffers), seconds, seconds > 0.0 ? gpu.busyMs / (seconds * 10.0) : 0.0, gpu.maxMs, gpu.longestTag.c_str());
        flushes = 0;
        writebackSurfaces = 0;
        completedFlushes = 0;
        completionMsTotal = 0.0;
        completionMsMax = 0.0;
        presentMs = {};
        reportSnapshotBytes = snapshotBytes;
        reportLabels = labels;
        reportAt = now;
        reportFrames = 0;
        reportDraws = drawCount;
        reportDispatches = dispatchCount;
        reportSubmits = hookSubmits;
        reportWaits = hookWaits;
        reportStats = stats;
    }

    void settleForPresent() {
        flush();
        if (lastSerial != 0 && device->CompletedSerial(Queue) < lastSerial) drain(lastSerial);
        else reap();
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
        const std::span<const std::uint32_t> words(binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 8u, 8u);
        const bool written = element < binding.imageWritten.size() && binding.imageWritten[element];
        const auto kind = ImageKindOf(binding.binding);
        std::optional<std::uint32_t> storageMip;
        if (kind.storage) {
            storageMip = 0u;
            for (auto previous = element; previous > 0u && std::equal(words.begin(), words.end(), binding.guestDescriptor.begin() + static_cast<std::ptrdiff_t>(previous - 1u) * 8); --previous) ++*storageMip;
        }
        return imageView(words, *binding.imageShape, written, kind.numeric, true, snapshots, uses, storageMip);
    }

    std::uint64_t tableImage(const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t element, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::vector<ImageUse>& uses) {
        const std::span<const std::uint32_t> words(binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 8u, 8u);
        const bool written = element < binding.imageWritten.size() && binding.imageWritten[element];
        return imageView(words, binding.imageShapes.at(element), written, std::nullopt, false, snapshots, uses);
    }

    std::uint64_t imageView(std::span<const std::uint32_t> words, ShaderRecompiler::DescriptorImageShape shape, bool written, std::optional<Numeric> numeric, bool descriptorSwizzle, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::vector<ImageUse>& uses, std::optional<std::uint32_t> storageMip = std::nullopt) {
        const auto viewKind = ViewKindOf(shape);
        if (std::all_of(words.begin(), words.end(), [](std::uint32_t word) { return word == 0u; }) || Graphics::IsNullTextureResource(words)) return nullTexture(viewKind).gpuResourceID._impl;
        auto resource = Graphics::ArrayLayerWindow(Graphics::DecodeTextureResource(words), MaxTextureArrayLayers);
        if (const auto* format = Metal::FindFormat(resource.format); numeric && *numeric != Numeric::Float && format != nullptr && format->transform == Metal::TextureTransform::Repack101111) resource.format = RawPackedFormat;
        const auto [found, info, layers] = surfaceFor(resource);
        if ((viewKind == Metal::TextureKind::Tex3D) != (info.kind == Metal::TextureKind::Tex3D) || (viewKind == Metal::TextureKind::Tex2DMsaa) != (info.kind == Metal::TextureKind::Tex2DMsaa)) throw std::runtime_error("AGC graphics: Metal image dimension does not match the shader image type");
        auto& surface = *found;
        const auto bytes = Graphics::DescribeSurface(resource).guestBytes;
        const auto address = resolve(resource.baseAddress, bytes, written, snapshots, &surface);
        uses.push_back({&surface, resource, address, written});
        Metal::ViewInfo view;
        view.kind = viewKind;
        view.changeKind = true;
        view.pixelFormat = numeric ? NumericView(surface.texture.pixelFormat, *numeric) : surface.texture.pixelFormat;
        view.baseLevel = std::min(resource.baseLevel, info.mipLevels - 1u);
        view.levelCount = std::min(std::max(resource.lastLevel, view.baseLevel), info.mipLevels - 1u) - view.baseLevel + 1u;
        if (storageMip) {
            view.baseLevel = std::min(view.baseLevel + *storageMip, info.mipLevels - 1u);
            view.levelCount = 1;
        }
        if (viewKind == Metal::TextureKind::Tex3D || viewKind == Metal::TextureKind::Tex2DMsaa) {
            view.baseSlice = 0;
            view.sliceCount = 1;
        } else {
            view.baseSlice = std::min(resource.baseArray, layers - 1u);
            view.sliceCount = viewKind == Metal::TextureKind::Tex2D ? 1u : layers - view.baseSlice;
        }
        if (!written && descriptorSwizzle) view.swizzle = Metal::ViewSwizzle(resource.format, {resource.dstSelX, resource.dstSelY, resource.dstSelZ, resource.dstSelW});
        else if (!written) view.swizzle = Metal::ViewSwizzle(resource.format, {4, 5, 6, 7});
        if (Trace()) std::fprintf(stderr, "[metal] image view: format %u surface pixel %u view pixel %u numeric %d kind %u slices %u\n", static_cast<unsigned>(resource.format), static_cast<unsigned>(surface.texture.pixelFormat), static_cast<unsigned>(view.pixelFormat), numeric ? static_cast<int>(*numeric) : -1, static_cast<unsigned>(view.kind), view.sliceCount);
        return viewOf(surface, view).resourceId;
    }

    SurfaceRef surfaceFor(const Graphics::GuestTextureResource& resource) {
        const auto layers = resource.depthOrLastArray + 1u;
        auto info = Metal::DescribeTexture(resource);
        if (info.kind != Metal::TextureKind::Tex3D && info.kind != Metal::TextureKind::Tex2DMsaa) {
            info.kind = Metal::TextureKind::Tex2DArray;
            info.arrayLength = layers;
        }
        const auto* format = Metal::FindFormat(resource.format);
        info.shaderWrite = true;
        info.renderTarget = format != nullptr && !format->blockCompressed;
        info.label = "AnyPS5 guest texture";
        const SurfaceKey key{resource.baseAddress,
                             static_cast<std::uint64_t>(resource.format) | static_cast<std::uint64_t>(info.kind) << 32u | static_cast<std::uint64_t>(resource.tileMode) << 40u,
                             static_cast<std::uint64_t>(info.width) | static_cast<std::uint64_t>(info.height) << 32u,
                             static_cast<std::uint64_t>(layers) | static_cast<std::uint64_t>(info.mipLevels) << 32u,
                             static_cast<std::uint64_t>(resource.pipeBankXor) | static_cast<std::uint64_t>(info.samples) << 32u};
        auto found = surfaces.find(key);
        if (found == surfaces.end()) {
            std::string failure;
            auto created = textureManager().Create(info, &failure);
            if (!created) throw std::runtime_error("AGC graphics: Metal texture could not be created: " + failure);
            found = surfaces.emplace(key, CachedSurface{created, {}}).first;
        }
        return {&found->second, info, layers};
    }

    Metal::TextureHandle viewOf(CachedSurface& surface, const Metal::ViewInfo& view) {
        const ViewKey viewKey{static_cast<std::uint32_t>(view.kind), static_cast<std::uint32_t>(view.pixelFormat), view.baseLevel, view.levelCount, view.baseSlice, view.sliceCount,
                              static_cast<std::uint32_t>(view.swizzle[0]) | static_cast<std::uint32_t>(view.swizzle[1]) << 8u | static_cast<std::uint32_t>(view.swizzle[2]) << 16u | static_cast<std::uint32_t>(view.swizzle[3]) << 24u};
        auto existing = surface.views.find(viewKey);
        if (existing == surface.views.end()) {
            auto created = textureManager().View(surface.texture, view);
            if (!created) throw std::runtime_error("AGC graphics: Metal texture view could not be created");
            existing = surface.views.emplace(viewKey, created).first;
        }
        return existing->second;
    }

    bool needsUpload(const std::vector<ImageUse>& uses) const {
        return std::any_of(uses.begin(), uses.end(), [&](const ImageUse& use) { return !use.surface->dirty && use.surface->validEpoch != epoch; });
    }

    void encodeUploads(id<MTL4ComputeCommandEncoder> compute, const std::vector<ImageUse>& uses) {
        for (const auto& use : uses) {
            auto& surface = *use.surface;
            if (surface.dirty || surface.validEpoch == epoch) continue;
            surface.validEpoch = epoch;
            textures->EncodeUpload((__bridge void*)compute, surface.texture, use.resource, use.address, textures->DefaultUploadPath());
            ++cbUploads;
            cbMovedBytes += Graphics::DescribeSurface(use.resource).guestBytes;
        }
    }

    std::uint64_t sampler(const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t element, bool point, const std::array<std::uint32_t, 4>* override = nullptr) {
        const std::span<const std::uint32_t> words = override != nullptr ? std::span<const std::uint32_t>(*override) : std::span<const std::uint32_t>(binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u, 4u);
        const bool unnormalized = element < binding.samplerUnnormalized.size() && binding.samplerUnnormalized[element];
        const bool integerBorder = element < binding.samplerIntegerBorder.size() && binding.samplerIntegerBorder[element];
        const bool compare = element < binding.samplerDepthCompare.size() && binding.samplerDepthCompare[element];
        const SamplerKey key{words[0], words[1], words[2], words[3], (unnormalized ? 1u : 0u) | (integerBorder ? 2u : 0u) | (point ? 4u : 0u) | (compare ? 8u : 0u)};
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
        if (descriptor.minFilter == MTLSamplerMinMagFilterNearest && descriptor.magFilter == MTLSamplerMinMagFilterNearest) descriptor.maxAnisotropy = 1;
        if (point) {
            descriptor.minFilter = MTLSamplerMinMagFilterNearest;
            descriptor.magFilter = MTLSamplerMinMagFilterNearest;
            if (descriptor.mipFilter == MTLSamplerMipFilterLinear) descriptor.mipFilter = MTLSamplerMipFilterNearest;
            descriptor.maxAnisotropy = 1;
        }
        if (compare) descriptor.compareFunction = static_cast<MTLCompareFunction>(decoded.compareOp);
        descriptor.supportArgumentBuffers = YES;
        descriptor.label = @"AnyPS5 guest sampler";
        id<MTLSamplerState> state = [native newSamplerStateWithDescriptor:descriptor];
        if (state == nil) throw std::runtime_error("AGC graphics: Metal sampler state could not be created");
        if (Trace()) std::fprintf(stderr, "[metal] sampler state: min %d mag %d mip %d point %d unnormalized %d aniso %d lod %.2f..%.2f\n", static_cast<int>(descriptor.minFilter), static_cast<int>(descriptor.magFilter), static_cast<int>(descriptor.mipFilter), point ? 1 : 0, unnormalized ? 1 : 0, static_cast<int>(descriptor.maxAnisotropy), descriptor.lodMinClamp, descriptor.lodMaxClamp);
        samplers.emplace(key, state);
        return state.gpuResourceID._impl;
    }

    void bindStage(const ShaderRecompiler::RecompileResult& shader, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::array<std::uint64_t, ArgumentSlots>& slots, std::vector<ImageUse>& uses) {
        slots[SlotIndex(BufferSlot::PushConstants)] = uploadBytes(shader.pushConstants.data(), shader.pushConstants.size(), ShaderRecompiler::MetalAbi::PushConstantBytes);
        slots[SlotIndex(BufferSlot::FaultBuffer)] = fault.gpuAddress;
        std::vector<std::pair<std::uint32_t, std::uint64_t>> imageEntries;
        std::uint32_t pointSamplers = 0;
        std::optional<std::uint32_t> tableElements;
        std::vector<std::uint32_t> map;
        std::uint32_t otherSamplers = 0;
        std::uint32_t integerSamplers = 0;
        std::uint32_t singleLevelSamplers = 0;
        std::uint32_t mipmappedSamplers = 0;
        for (const auto& binding : shader.bindings) {
            const auto use = SamplerUseOf(binding);
            pointSamplers |= use.integer;
            integerSamplers |= use.integer;
            otherSamplers |= use.other;
            singleLevelSamplers |= use.singleLevel;
            mipmappedSamplers |= use.mipmapped;
        }
        pointSamplers &= ~otherSamplers;
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
                    if (Trace()) std::fprintf(stderr, "[metal] buffer %u.%u: 0x%llx+0x%llx %s through %s\n", binding.binding, element, static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes), written ? "written" : "read", resolvedBy);
                }
                slots[SlotIndex(BufferSlot::GuestBuffers)] = uploadBytes(entries.data(), entries.size() * sizeof(entries[0]));
                break;
            }
            case ShaderRecompiler::DescriptorRole::GuestImages: {
                if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 8u) throw std::runtime_error("AGC graphics: guest image descriptor must contain eight DWORDs per element");
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto index = ShaderRecompiler::MetalAbi::GuestImageIndex(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count), element);
                    if (Trace()) std::fprintf(stderr, "[metal] image binding %u element %u -> table index 0x%x\n", binding.binding, element, index);
                    imageEntries.emplace_back(index, image(binding, element, snapshots, uses));
                }
                break;
            }
            case ShaderRecompiler::DescriptorRole::GuestSamplers: {
                if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) throw std::runtime_error("AGC graphics: guest sampler descriptor must contain four DWORDs per element");
                std::vector<ShaderRecompiler::MetalAbi::ResourceIdEntry> entries(std::max<std::uint32_t>(binding.count, 1u), {0});
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const bool point = element < 32u && ((pointSamplers >> element) & 1u) != 0u;
                    std::optional<std::array<std::uint32_t, 4>> words;
                    if (element < 32u) words = Graphics::SingleLevelSamplerWords(std::span<const std::uint32_t, 4>(binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u, 4u), ((singleLevelSamplers >> element) & 1u) != 0u, ((mipmappedSamplers >> element) & 1u) != 0u);
                    entries[element].resourceId = sampler(binding, element, point, words ? &*words : nullptr);
                    if (Trace()) std::fprintf(stderr, "[metal] sampler %u.%u: point %u, words %08x %08x %08x %08x, integer mask %x, other mask %x\n", binding.binding, element, point ? 1u : 0u, binding.guestDescriptor[element * 4u], binding.guestDescriptor[element * 4u + 1u], binding.guestDescriptor[element * 4u + 2u], binding.guestDescriptor[element * 4u + 3u], integerSamplers, otherSamplers);
                }
                slots[SlotIndex(BufferSlot::GuestSamplers)] = uploadBytes(entries.data(), entries.size() * sizeof(entries[0]));
                break;
            }
            case ShaderRecompiler::DescriptorRole::ImageTable:
            case ShaderRecompiler::DescriptorRole::ImageStorageTable: {
                if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 8u || binding.imageShapes.size() != binding.count) throw std::runtime_error("AGC graphics: image table binding is malformed");
                if (tableElements && *tableElements != binding.count) throw std::runtime_error("AGC graphics: a stage's image tables disagree on their elements");
                tableElements = binding.count;
                std::vector<ShaderRecompiler::MetalAbi::ResourceIdEntry> entries(std::max<std::uint32_t>(binding.count, 1u), {0});
                for (std::uint32_t element = 0; element < binding.count; ++element) entries[element].resourceId = tableImage(binding, element, snapshots, uses);
                const auto role = binding.role == ShaderRecompiler::DescriptorRole::ImageTable ? BufferSlot::ImageTable : BufferSlot::ImageStorageTable;
                slots[SlotIndex(role)] = uploadBytes(entries.data(), entries.size() * sizeof(entries[0]));
                break;
            }
            case ShaderRecompiler::DescriptorRole::SamplerTable: {
                if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u || binding.samplerDepthCompare.size() != binding.count || binding.count % ShaderRecompiler::ImageTableAbi::SamplerEntryElements != 0u) throw std::runtime_error("AGC graphics: sampler table binding is malformed");
                std::vector<ShaderRecompiler::MetalAbi::ResourceIdEntry> entries(std::max<std::uint32_t>(binding.count, 1u), {0});
                for (std::uint32_t element = 0; element < binding.count; ++element) entries[element].resourceId = sampler(binding, element, false);
                slots[SlotIndex(BufferSlot::SamplerTable)] = uploadBytes(entries.data(), entries.size() * sizeof(entries[0]));
                break;
            }
            case ShaderRecompiler::DescriptorRole::ImageTableMap: {
                namespace TableAbi = ShaderRecompiler::ImageTableAbi;
                if (binding.count != 1 || binding.guestDescriptor.size() < TableAbi::HeaderWords || binding.guestDescriptor[TableAbi::HeaderVersion] != TableAbi::Version) throw std::runtime_error("AGC graphics: image table map binding is malformed");
                map = binding.guestDescriptor;
                map[TableAbi::HeaderElementBase] = 0;
                break;
            }
            case ShaderRecompiler::DescriptorRole::BdaPagetable:
                break;
            case ShaderRecompiler::DescriptorRole::FaultBuffer:
                break;
            default:
                throw std::runtime_error("AGC graphics: Metal does not bind descriptor role " + std::to_string(static_cast<int>(binding.role)) + " yet");
            }
        }
        if (!map.empty()) {
            if (map[ShaderRecompiler::ImageTableAbi::HeaderElementCount] != tableElements.value_or(0u)) throw std::runtime_error("AGC graphics: image table map disagrees with its image binding");
            slots[SlotIndex(BufferSlot::ImageTableMap)] = uploadBytes(map.data(), map.size() * sizeof(map[0]));
        } else if (tableElements) {
            throw std::runtime_error("AGC graphics: an image table stage has no map");
        }
        if (!imageEntries.empty()) {
            const auto [lowest, highest] = std::minmax_element(imageEntries.begin(), imageEntries.end());
            const auto first = lowest->first;
            std::vector<ShaderRecompiler::MetalAbi::ResourceIdEntry> entries(highest->first - first + 1u, {0});
            for (const auto& [index, resourceId] : imageEntries) entries[index - first].resourceId = resourceId;
            slots[SlotIndex(BufferSlot::GuestImages)] = uploadBytes(entries.data(), entries.size() * sizeof(entries[0])) - static_cast<std::uint64_t>(first) * sizeof(entries[0]);
        }
        if (shader.bdaAbiVersion != 0) slots[SlotIndex(BufferSlot::BdaPagetable)] = bdaTable(snapshots);
    }

    void dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::span<const Graphics::GuestMemorySnapshot> snapshots) {
        if (shader.backend != ShaderRecompiler::ShaderBackendKind::Msl) throw std::runtime_error("AGC graphics: the Metal backend dispatches only MSL results");
        if (shader.msl.empty()) throw std::runtime_error("AGC graphics: the MSL result has no source");
        if (shader.workgroupMemoryDwords != 0) throw std::runtime_error("AGC graphics: Metal dispatch with device-memory LDS is not wired yet");
        if (x == 0 || y == 0 || z == 0) return;
        const auto threads = shader.hostThreadgroup;
        if (threads[0] == 0 || threads[1] == 0 || threads[2] == 0) throw std::runtime_error("AGC graphics: the MSL result has no threadgroup size");
        if (y > groupLimit[1] || z > groupLimit[2]) throw std::runtime_error("AGC graphics: Metal dispatch exceeds the threadgroup count limit");
        id<MTLComputePipelineState> state = pipeline(shader);
        @autoreleasepool {
            std::lock_guard lock(mutex);
            const EncodingScope encoding;
            registry->Refresh();
            writebackDirty();
            begin();
            std::array<std::uint64_t, ArgumentSlots> slots{};
            std::vector<ImageUse> uses;
            bindStage(shader, snapshots, slots, uses);
            writebackDirty();
            auto compute = computeEncoder();
            if (encodedDispatches != 0) [compute barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
            if (needsUpload(uses)) {
                encodeUploads(compute, uses);
                [compute barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            }
            for (std::uint32_t slot = 0; slot < ArgumentSlots; ++slot) {
                if (slots[slot] != 0) [table setAddress:slots[slot] atIndex:slot];
            }
            [compute setComputePipelineState:state];
            [compute setArgumentTable:table];
            [compute dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(threads[0], threads[1], threads[2])];
            if (Trace()) std::fprintf(stderr, "[metal] dispatch %016llx%016llx: %ux%ux%u groups of %ux%ux%u, %zu snapshots\n", static_cast<unsigned long long>(shader.Identity().high), static_cast<unsigned long long>(shader.Identity().low), x, y, z, threads[0], threads[1], threads[2], snapshots.size());
            if (std::any_of(uses.begin(), uses.end(), [](const ImageUse& use) { return use.written; })) {
                [compute barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
                std::vector<const CachedSurface*> stored;
                for (const auto& use : uses) {
                    if (!use.written || std::find(stored.begin(), stored.end(), use.surface) != stored.end()) continue;
                    stored.push_back(use.surface);
                    textures->EncodeWriteback((__bridge void*)compute, use.surface->texture, use.resource, use.address);
                }
            }
            ++epoch;
            ++encodedDispatches;
            ++cbDispatches;
            ++recordedSinceSubmit;
            dispatches.fetch_add(1, std::memory_order_relaxed);
        }
    }

    Metal::PipelineCache& pipelineCache() {
        if (renderPipelines != nullptr) return *renderPipelines;
        std::string failure;
        renderPipelines = Metal::PipelineCache::Create(*device, *compiler, &failure);
        if (renderPipelines == nullptr) throw std::runtime_error("AGC graphics: Metal pipeline cache could not be created: " + failure);
        return *renderPipelines;
    }

    const StageSource& stageSource(const ShaderRecompiler::RecompileResult& shader) {
        if (shader.backend != ShaderRecompiler::ShaderBackendKind::Msl || shader.msl.empty()) throw std::runtime_error("AGC graphics: Metal draws need MSL stage results");
        const auto identity = shader.Identity();
        if (const auto found = stageSources.find(identity); found != stageSources.end()) return found->second;
        StageSource entry;
        entry.source = std::make_shared<const std::string>(shader.msl.Source());
        const auto hash = shader.msl.ContentHash();
        entry.key = Metal::CompileKeyBuilder(0x4d534c47ull).Add(hash.low).Add(hash.high).Finish();
        const auto declared = DeclaredConstants(*entry.source);
        for (const auto& constant : shader.specialization) {
            if (declared.contains(constant.id)) entry.constants.push_back({constant.id, constant.value, false});
        }
        return stageSources.emplace(identity, std::move(entry)).first->second;
    }

    static Metal::LibrarySource librarySource(const StageSource& stage, const char* name) {
        Metal::LibrarySource library;
        library.key = stage.key;
        library.owner = stage.source;
        library.text = *stage.source;
        library.name = name;
        library.fastMath = true;
        return library;
    }

    void* renderPipeline(const Metal::RenderPipelineRequest& request) {
        const auto key = Metal::PipelineCache::RenderKey(request);
        auto found = drawPipelines.find(key);
        if (found == drawPipelines.end()) {
            found = drawPipelines.emplace(key, pipelineCache().Render(request)).first;
            pipelinesCreated.fetch_add(1, std::memory_order_relaxed);
        }
        const auto& handle = *found->second;
        if (auto* native = handle.Native()) return native;
        pipelineCache().Promote(handle, Metal::CompilePriority::Urgent);
        handle.Wait(PipelineTimeoutMs);
        if (auto* native = handle.Native()) return native;
        if (Verbose()) {
            std::fprintf(stderr, "%.*s\n", static_cast<int>(request.vertexLibrary.text.size()), request.vertexLibrary.text.data());
            if (!request.fragmentFunction.empty()) std::fprintf(stderr, "%.*s\n", static_cast<int>(request.fragmentLibrary.text.size()), request.fragmentLibrary.text.data());
        }
        throw Graphics::UnrenderableDraw("AGC graphics: Metal render pipeline failed to build: " + (handle.Failed() ? handle.Error() : std::string("timed out")));
    }

    DepthSurface& depthSurface(const Graphics::DepthTarget& target, Metal::PixelFormat format) {
        const DepthKey key{target.address, static_cast<std::uint64_t>(target.extent.width) | static_cast<std::uint64_t>(target.extent.height) << 32u, static_cast<std::uint64_t>(format)};
        if (const auto found = depthSurfaces.find(key); found != depthSurfaces.end()) {
            ApplyDepthFills(found->second, target);
            return found->second;
        }
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:static_cast<MTLPixelFormat>(format) width:target.extent.width height:target.extent.height mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        descriptor.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> texture = [native newTextureWithDescriptor:descriptor];
        if (texture == nil) throw std::runtime_error("AGC graphics: Metal depth target could not be created");
        texture.label = @"AnyPS5 depth target";
        device->AddResidentAllocation((__bridge void*)texture);
        return depthSurfaces.emplace(key, DepthSurface{texture, false, false, Graphics::DepthMetadataFillGeneration(), {}, {}, {}, {}}).first->second;
    }

    void openPass(const std::array<ColorAttachment, Metal::MaxColorAttachments>& colors, DepthSurface* depth, const Graphics::DepthTarget* target, bool stencil, std::uint32_t width, std::uint32_t height, std::vector<std::uint64_t> key) {
        begin();
        endRender();
        endCompute();
        ++cbPasses;
        MTL4RenderPassDescriptor* descriptor = [MTL4RenderPassDescriptor new];
        for (std::uint32_t index = 0; index < Metal::MaxColorAttachments; ++index) {
            if (colors[index].texture == nil) continue;
            MTLRenderPassColorAttachmentDescriptor* attachment = descriptor.colorAttachments[index];
            attachment.texture = colors[index].texture;
            attachment.level = colors[index].level;
            attachment.depthPlane = colors[index].depthPlane;
            attachment.loadAction = MTLLoadActionLoad;
            attachment.storeAction = MTLStoreActionStore;
        }
        if (depth != nullptr) {
            const auto load = depth->initialized ? MTLLoadActionLoad : MTLLoadActionClear;
            const auto stencilLoad = depth->stencilInitialized ? MTLLoadActionLoad : MTLLoadActionClear;
            MTLRenderPassDepthAttachmentDescriptor* attachment = descriptor.depthAttachment;
            attachment.texture = depth->texture;
            attachment.loadAction = load;
            attachment.storeAction = MTLStoreActionStore;
            attachment.clearDepth = depth->fillDepth.value_or(target->clearDepth);
            if (stencil) {
                MTLRenderPassStencilAttachmentDescriptor* stencilAttachment = descriptor.stencilAttachment;
                stencilAttachment.texture = depth->texture;
                stencilAttachment.loadAction = stencilLoad;
                stencilAttachment.storeAction = MTLStoreActionStore;
                stencilAttachment.clearStencil = depth->fillStencil.value_or(target->clearStencil);
            }
            if (target->htileAddress != 0u && (depth->htilePattern || !depth->htileWords.empty())) {
                std::vector<std::uint32_t> words = std::move(depth->htileWords);
                if (words.empty()) words.assign(HtileBytes(target->extent) / 4u, *depth->htilePattern);
                for (auto& word : words) word = ExpandedHtileWord(word, target->htileStencil);
                GuestMemory::Write(target->htileAddress, std::as_bytes(std::span(words)), 4);
            }
            depth->htileWords.clear();
            depth->htilePattern.reset();
            depth->initialized = true;
            if (stencil) depth->stencilInitialized = true;
            depth->fillDepth.reset();
            depth->fillStencil.reset();
        }
        descriptor.renderTargetWidth = width;
        descriptor.renderTargetHeight = height;
        if (depth == nullptr && std::all_of(colors.begin(), colors.end(), [](const ColorAttachment& color) { return color.texture == nil; })) descriptor.defaultRasterSampleCount = 1;
        id<MTL4CommandBuffer> commands = (__bridge id<MTL4CommandBuffer>)open.commands;
        render = [commands renderCommandEncoderWithDescriptor:descriptor];
        if (render == nil) throw std::runtime_error("AGC graphics: Metal render encoder could not be created");
        [render barrierAfterQueueStages:MTLStageVertex | MTLStageFragment | MTLStageDispatch | MTLStageBlit beforeStages:MTLStageVertex | MTLStageFragment visibilityOptions:MTL4VisibilityOptionDevice];
        [render setArgumentTable:vertexTable atStages:MTLRenderStageVertex];
        [render setArgumentTable:fragmentTable atStages:MTLRenderStageFragment];
        boundPipeline = nullptr;
        boundDepthStencil = nullptr;
        dynamicValid = false;
        passKey = std::move(key);
        passWidth = width;
        passHeight = height;
    }

    void applyDynamic(const Metal::DynamicState& dynamic) {
        const auto changed = dynamicValid ? Metal::DiffDynamic(boundDynamic, dynamic) : Metal::DynamicAll;
        if (changed == 0) return;
        if ((changed & Metal::DynamicViewport) != 0) {
            const auto& v = dynamic.viewport;
            [render setViewport:MTLViewport{v.originX, v.originY, v.width, v.height, v.znear, v.zfar}];
        }
        if ((changed & Metal::DynamicScissor) != 0) {
            const auto& r = dynamic.scissor;
            [render setScissorRect:MTLScissorRect{r.x, r.y, r.width, r.height}];
        }
        if ((changed & Metal::DynamicDepthBias) != 0) [render setDepthBias:dynamic.depthBias.constant slopeScale:dynamic.depthBias.slope clamp:dynamic.depthBias.clamp];
        if ((changed & Metal::DynamicBlendColor) != 0) [render setBlendColorRed:dynamic.blendColor[0] green:dynamic.blendColor[1] blue:dynamic.blendColor[2] alpha:dynamic.blendColor[3]];
        if ((changed & Metal::DynamicStencilReference) != 0) [render setStencilFrontReferenceValue:dynamic.stencilFrontReference backReferenceValue:dynamic.stencilBackReference];
        if ((changed & Metal::DynamicCull) != 0) [render setCullMode:static_cast<MTLCullMode>(dynamic.cull)];
        if ((changed & Metal::DynamicWinding) != 0) [render setFrontFacingWinding:static_cast<MTLWinding>(dynamic.winding)];
        if ((changed & Metal::DynamicDepthClip) != 0) [render setDepthClipMode:static_cast<MTLDepthClipMode>(dynamic.depthClip)];
        boundDynamic = dynamic;
        dynamicValid = true;
    }

    void draw(const Graphics::State& graphics, const Pm4::DrawParameters& params, std::span<const Graphics::CompiledShader> shaders, std::span<const Graphics::GuestMemorySnapshot> snapshots) {
        if (params.indirect) throw std::runtime_error("AGC graphics: Metal indirect draws are not wired yet");
        if (params.indexCount == 0 || params.instanceCount == 0) return;
        if (params.indexed && params.indexSize != 2u && params.indexSize != 4u) throw std::runtime_error("AGC graphics: only uint16 and uint32 index buffers are supported");
        const Graphics::CompiledShader* vertex = nullptr;
        const Graphics::CompiledShader* fragment = nullptr;
        for (const auto& shader : shaders) {
            if (shader.stage == ShaderRecompiler::ShaderStage::Vertex) vertex = &shader;
            else if (shader.stage == ShaderRecompiler::ShaderStage::Fragment) fragment = &shader;
            else throw std::runtime_error("AGC graphics: Metal draws support only vertex and pixel stages");
        }
        if (vertex == nullptr) throw std::runtime_error("AGC graphics: Metal draw has no vertex stage");
        const auto translated = Metal::TranslateState(graphics, HasFlatInputs(shaders));
        if (translated.rejection != nullptr) throw Graphics::UnrenderableDraw(std::string("AGC graphics: ") + translated.rejection);
        if (translated.cullsEverything) return;
        if (translated.pipeline.sampleCount > 1) throw std::runtime_error("AGC graphics: multisampled Metal draws are not wired yet");
        @autoreleasepool {
            std::lock_guard lock(mutex);
            const EncodingScope encoding;
            registry->Refresh();
            begin();
            Metal::RenderPipelineRequest request;
            request.state = translated.pipeline;
            const auto& vertexSource = stageSource(*vertex->program);
            request.vertexLibrary = librarySource(vertexSource, "AnyPS5 guest vertex");
            request.vertexFunction = std::string(ShaderRecompiler::MslEntryPoint);
            request.vertexConstants = vertexSource.constants;
            if (fragment != nullptr) {
                const auto& fragmentSource = stageSource(*fragment->program);
                request.fragmentLibrary = librarySource(fragmentSource, "AnyPS5 guest pixel");
                request.fragmentFunction = std::string(ShaderRecompiler::MslEntryPoint);
                request.fragmentConstants = fragmentSource.constants;
            }
            request.label = "AnyPS5 guest draw";
            request.priority = Metal::CompilePriority::Urgent;
            std::vector<std::uint64_t> vertexBuffers;
            for (const auto& attribute : vertex->program->vertexAttributes) {
                const auto decoded = Graphics::DecodeVertexFormat(attribute);
                const auto format = NativeVertexFormat(decoded.format);
                if (format == MTLVertexFormatInvalid) throw std::runtime_error("AGC graphics: vertex format " + std::to_string(static_cast<int>(decoded.format)) + " has no Metal vertex format");
                const auto buffer = FirstVertexBuffer + static_cast<std::uint32_t>(vertexBuffers.size());
                if (buffer >= RenderSlots) throw std::runtime_error("AGC graphics: Metal draw has more vertex streams than buffer slots");
                request.vertexAttributes.push_back({attribute.location, static_cast<std::uint32_t>(format), buffer, 0});
                const auto packed = (decoded.bytes + 3u) & ~3u;
                if (Graphics::NullVertexDescriptor(attribute)) {
                    request.vertexBuffers.push_back({buffer, packed, Metal::VertexStep::Constant});
                    vertexBuffers.push_back(uploadBytes(nullptr, 0, 16));
                    continue;
                }
                const auto& fields = attribute.resource.fields;
                const auto stride = (fields[1] >> 16u) & 0x3fffu;
                const auto address = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
                if (stride % 4u != 0u) throw std::runtime_error("AGC graphics: Metal vertex stride " + std::to_string(stride) + " is not a multiple of four");
                const auto step = stride == 0u ? Metal::VertexStep::Constant : attribute.fetchIndex == 0u ? Metal::VertexStep::Vertex : Metal::VertexStep::Instance;
                request.vertexBuffers.push_back({buffer, stride == 0u ? packed : stride, step});
                const auto resolved = resolve(address, Graphics::VertexBufferExtent(attribute), false, snapshots);
                if (resolved % 4u != 0u) throw std::runtime_error("AGC graphics: Metal vertex buffer address is not four-byte aligned");
                vertexBuffers.push_back(resolved);
            }
            void* pipelineState = renderPipeline(request);

            std::array<std::uint64_t, ArgumentSlots> vertexSlots{};
            std::array<std::uint64_t, ArgumentSlots> fragmentSlots{};
            std::vector<ImageUse> uses;
            bindStage(*vertex->program, snapshots, vertexSlots, uses);
            if (fragment != nullptr) bindStage(*fragment->program, snapshots, fragmentSlots, uses);

            std::array<ColorAttachment, Metal::MaxColorAttachments> colors{};
            std::vector<std::pair<CachedSurface*, Graphics::GuestTextureResource>> targets;
            std::vector<std::uint64_t> key;
            std::uint32_t width = graphics.renderExtent.width;
            std::uint32_t height = graphics.renderExtent.height;
            for (const auto& color : graphics.colors) {
                if (color.exportIndex >= request.state.colorCount) continue;
                const auto format = request.state.colors[color.exportIndex].format;
                if (format == Metal::PixelFormat::Invalid) continue;
                const auto resource = Graphics::SurfaceForTarget(color);
                const auto [surface, info, layers] = surfaceFor(resource);
                Metal::TextureHandle handle = surface->texture;
                if (handle.pixelFormat != format) {
                    Metal::ViewInfo view;
                    view.pixelFormat = format;
                    view.kind = info.kind;
                    view.levelCount = info.mipLevels;
                    view.sliceCount = info.kind == Metal::TextureKind::Tex3D ? 1u : layers;
                    handle = viewOf(*surface, view);
                }
                id<MTLTexture> texture = (__bridge id<MTLTexture>)handle.texture;
                const auto level = std::min(color.mip, info.mipLevels - 1u);
                const auto plane = info.kind == Metal::TextureKind::Tex3D ? color.depthSlice : 0u;
                colors[color.exportIndex] = {texture, level, plane};
                width = std::min<std::uint32_t>(width, std::max<std::uint32_t>(static_cast<std::uint32_t>(texture.width) >> level, 1u));
                height = std::min<std::uint32_t>(height, std::max<std::uint32_t>(static_cast<std::uint32_t>(texture.height) >> level, 1u));
                targets.emplace_back(surface, resource);
                key.push_back(reinterpret_cast<std::uintptr_t>(handle.texture));
                key.push_back(static_cast<std::uint64_t>(color.exportIndex) | static_cast<std::uint64_t>(level) << 8u | static_cast<std::uint64_t>(plane) << 16u);
            }
            DepthSurface* depth = nullptr;
            const bool stencil = request.state.stencilFormat != Metal::PixelFormat::Invalid;
            if (graphics.depth && request.state.depthFormat != Metal::PixelFormat::Invalid) {
                depth = &depthSurface(*graphics.depth, request.state.depthFormat);
                width = std::min<std::uint32_t>(width, static_cast<std::uint32_t>(depth->texture.width));
                height = std::min<std::uint32_t>(height, static_cast<std::uint32_t>(depth->texture.height));
                key.push_back(reinterpret_cast<std::uintptr_t>((__bridge void*)depth->texture));
            }
            if (width == 0 || height == 0) return;
            key.push_back(static_cast<std::uint64_t>(width) | static_cast<std::uint64_t>(height) << 32u);

            if (depth != nullptr && graphics.depth && !(render != nil && key == passKey)) ScanHtile(*depth, *graphics.depth);
            const bool reuse = render != nil && key == passKey && (depth == nullptr || (depth->initialized && (!stencil || depth->stencilInitialized)));
            std::vector<ImageUse> loads = uses;
            if (!reuse) {
                for (auto& [surface, resource] : targets) {
                    if (surface->dirty) continue;
                    surface->resource = resource;
                    surface->address = resolve(resource.baseAddress, Graphics::DescribeSurface(resource).guestBytes, true, {}, surface);
                    loads.push_back({surface, resource, surface->address, false});
                }
            }
            if (!reuse || needsUpload(uses)) {
                if (needsUpload(loads)) {
                    auto compute = computeEncoder();
                    [compute barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
                    encodeUploads(compute, loads);
                }
                openPass(colors, depth, graphics.depth ? &*graphics.depth : nullptr, stencil, width, height, std::move(key));
            }

            auto dynamic = translated.dynamic;
            dynamic.scissor.x = std::min(dynamic.scissor.x, passWidth);
            dynamic.scissor.y = std::min(dynamic.scissor.y, passHeight);
            dynamic.scissor.width = std::min(dynamic.scissor.width, passWidth - dynamic.scissor.x);
            dynamic.scissor.height = std::min(dynamic.scissor.height, passHeight - dynamic.scissor.y);
            if (dynamic.scissor.width == 0 || dynamic.scissor.height == 0) return;
            if (pipelineState != boundPipeline) {
                [render setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)pipelineState];
                boundPipeline = pipelineState;
            }
            if (depth != nullptr) {
                void* depthStencil = pipelineCache().DepthStencil(translated.depthStencil);
                if (depthStencil != boundDepthStencil) {
                    [render setDepthStencilState:(__bridge id<MTLDepthStencilState>)depthStencil];
                    boundDepthStencil = depthStencil;
                }
            }
            applyDynamic(dynamic);
            for (std::uint32_t slot = 0; slot < ArgumentSlots; ++slot) {
                if (vertexSlots[slot] != 0) [vertexTable setAddress:vertexSlots[slot] atIndex:slot];
                if (fragmentSlots[slot] != 0) [fragmentTable setAddress:fragmentSlots[slot] atIndex:slot];
            }
            for (std::size_t index = 0; index < vertexBuffers.size(); ++index) [vertexTable setAddress:vertexBuffers[index] atIndex:FirstVertexBuffer + index];
            const auto primitive = static_cast<MTLPrimitiveType>(translated.primitive);
            if (params.indexed) {
                const auto bytes = static_cast<std::uint64_t>(params.indexCount) * params.indexSize;
                const auto indices = resolve(params.indexAddress, bytes, false, snapshots, nullptr, false);
                [render drawIndexedPrimitives:primitive indexCount:params.indexCount indexType:params.indexSize == 2u ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32 indexBuffer:indices indexBufferLength:bytes instanceCount:params.instanceCount baseVertex:static_cast<NSInteger>(params.firstVertex) baseInstance:params.firstInstance];
            } else {
                [render drawPrimitives:primitive vertexStart:params.firstVertex vertexCount:params.indexCount instanceCount:params.instanceCount baseInstance:params.firstInstance];
            }
            for (auto& [surface, resource] : targets) {
                surface->validEpoch = epoch;
                surface->drawnFrame = scanoutFrames;
                if (surface->dirty) continue;
                surface->dirty = true;
                surface->guestAddress = resource.baseAddress;
                surface->guestBytes = Graphics::DescribeSurface(resource).guestBytes;
                dirtySurfaces.push_back(surface);
            }
            ++recordedSinceSubmit;
            ++cbDraws;
            if (gpuTimeEnabled()) {
                char text[320];
                int used = std::snprintf(text, sizeof(text), "last draw: pass %ux%u, %u colors, depth %d, prim %u, %u indices x %u, viewport %.0fx%.0f, scissor %ux%u, vs %08llx ps %08llx, images", passWidth, passHeight, static_cast<unsigned>(targets.size()), depth != nullptr ? 1 : 0, static_cast<unsigned>(translated.primitive), params.indexCount, params.instanceCount, dynamic.viewport.width, dynamic.viewport.height, dynamic.scissor.width, dynamic.scissor.height, static_cast<unsigned long long>(vertex->program->msl.ContentHash().low & 0xffffffffull), static_cast<unsigned long long>(fragment != nullptr ? fragment->program->msl.ContentHash().low & 0xffffffffull : 0ull));
                for (std::size_t index = 0; index < uses.size() && index < 6 && used > 0 && used < static_cast<int>(sizeof(text)); ++index) used += std::snprintf(text + used, sizeof(text) - static_cast<std::size_t>(used), " %ux%u", uses[index].resource.width, uses[index].resource.height);
                cbLastDraw = text;
            }
            draws.fetch_add(1, std::memory_order_relaxed);
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
        descriptor.maxBufferBindCount = RenderSlots;
        s.vertexTable = [s.native newArgumentTableWithDescriptor:descriptor error:&error];
        s.fragmentTable = [s.native newArgumentTableWithDescriptor:descriptor error:&error];
        if (s.vertexTable == nil || s.fragmentTable == nil) throw std::runtime_error("AGC graphics: Metal render argument tables could not be created");
        s.fault = s.device->NewSharedBuffer(ShaderRecompiler::BdaAbi::FaultBufferBytes, "AnyPS5 fault buffer");
        if (!s.fault) throw std::runtime_error("AGC graphics: Metal fault buffer could not be allocated");
        std::memset(s.fault.contents, 0, ShaderRecompiler::BdaAbi::FaultBufferBytes);
        if (window != nullptr && window->context != nullptr) s.attachWindow(*window);
        {
            std::lock_guard lock(State::targetsMutex());
            State::targets().push_back(&s);
        }
        GuestMemory::SetFlushHook(&State::flushHook);
    }
}

VulkanDevice::~VulkanDevice() = default;

auto VulkanDevice::ReplayComputePipeline(const ShaderRecompiler::RecompileResult&, bool) -> bool { return false; }

auto VulkanDevice::WaitForPipelineJournalReplay() -> void {}

auto VulkanDevice::SavePipelineCache() -> void {}

auto VulkanDevice::SubmittedEpoch() const -> std::uint64_t {
    auto& s = *state;
    if (s.sync == nullptr) return 0;
    std::lock_guard lock(s.mutex);
    return s.lastSerial;
}

auto VulkanDevice::GpuCopyPossible(std::uint64_t, std::uint64_t, std::size_t) const -> bool { return false; }

auto VulkanDevice::HostSampleOrder(const Graphics::State&) const -> Graphics::SampleOrder { return {}; }

auto VulkanDevice::RealFramePending() const -> bool { return false; }

auto VulkanDevice::graphicsContext() const -> Graphics::Context { return {}; }

auto VulkanDevice::SealPassWriteOnly() -> void {}

auto VulkanDevice::AcquireRealImage() -> bool { return false; }

auto VulkanDevice::QueueRealFrame() -> void {}

auto VulkanDevice::WaitRecordedFor(std::uint64_t serial, std::uint64_t timeoutNanos) -> bool {
    if (serial == 0 || state->device == nullptr) return true;
    const auto milliseconds = std::min<std::uint64_t>((timeoutNanos + 999999u) / 1000000u, WaitTimeoutMs);
    return state->device->WaitSerial(Queue, serial, static_cast<std::uint32_t>(milliseconds));
}

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
    s.writebackDirty();
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

auto VulkanDevice::WriteLabelOnGpu(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t, std::uint32_t, bool) -> int {
    auto& s = *state;
    if (s.sync == nullptr || bytes.empty() || bytes.size() > Metal::Sync::MaxLabelBytes || address % 4 != 0 || bytes.size() % 4 != 0) return 4;
    std::lock_guard lock(s.mutex);
    if (GuestMemory::CurrentPacket().opcode != ReleaseMemOpcode || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes.size(), true)) return 3;
    auto* completion = s.afterRecorded();
    if (completion == nullptr) {
        s.labelsOnCpu.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    completion->labels.push_back({address, std::vector<std::byte>(bytes.begin(), bytes.end())});
    s.noteWrite(address, bytes.size(), s.completions.rbegin()->first);
    s.labelsOnCompletion.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

auto VulkanDevice::AfterRecordedWork(std::function<void()> action, bool) -> bool {
    auto& s = *state;
    if (s.sync == nullptr) return false;
    std::lock_guard lock(s.mutex);
    auto* completion = s.afterRecorded();
    if (completion == nullptr) return false;
    completion->actions.push_back(std::move(action));
    s.actionsOnCompletion.fetch_add(1, std::memory_order_relaxed);
    return true;
}

auto VulkanDevice::PendingLabel(std::uint64_t, std::size_t, std::uint64_t, Graphics::Recorder::LabelRefusal*) const -> std::optional<Graphics::Recorder::LabelHit> { return {}; }

auto VulkanDevice::OpenWriteOverlaps(std::uint64_t, std::size_t) const -> bool { return false; }

auto VulkanDevice::RecordedWritesSettled(std::uint64_t, std::size_t) const -> bool {
    auto& s = *state;
    if (s.sync == nullptr) return true;
    std::lock_guard lock(s.mutex);
    s.reap();
    return s.completions.empty() && s.recordedSinceSubmit == 0;
}

auto VulkanDevice::SettleCompletions() -> void {
    auto& s = *state;
    if (s.sync == nullptr) return;
    std::lock_guard lock(s.mutex);
    if (!s.completions.empty()) s.drain(s.completions.rbegin()->first);
}

auto VulkanDevice::StoresPendingOver(std::uint64_t, std::size_t) const -> bool { return false; }

auto VulkanDevice::FillBuffer(std::uint64_t, std::size_t, std::span<const std::uint32_t, 4>) -> bool { return false; }

auto VulkanDevice::DumpSamplesOnGpu(std::uint64_t) -> bool { return false; }

auto VulkanDevice::CopyBuffer(std::uint64_t destination, std::uint64_t source, std::size_t bytes, std::size_t cpuMax, std::size_t, std::size_t, std::uint64_t, std::uint32_t, const CopyWriterNote&) -> CopyOutcome {
    CopyOutcome outcome{2, false, 0.0, CopyOutcome::None, false, false, 0.0, 0, 0, 0, false};
    if (bytes == 0) return outcome;
    if (bytes > cpuMax) {
        outcome.reason = CopyOutcome::Size;
        return outcome;
    }
    if (!state->copyWhenIdle(destination, source, bytes)) {
        outcome.reason = CopyOutcome::Reader;
        return outcome;
    }
    outcome.path = 0;
    return outcome;
}

auto VulkanDevice::CopyVerifyCounts() -> CopyVerification { return {}; }

auto VulkanDevice::CopyAliasCounts() -> CopyAliasing { return {}; }

auto VulkanDevice::ComputePipelineCounts() -> ComputePipelineCreation { return {GlobalComputePipelinesCreated().load(std::memory_order_relaxed), 0}; }

auto VulkanDevice::DispatchMoveCounts() -> DispatchMoveCount { return {}; }

auto VulkanDevice::ResolveMemory(std::uint64_t, std::size_t, bool) -> void {}

auto VulkanDevice::Window() const -> void* { return state->window; }

auto VulkanDevice::Resize(std::uint32_t width, std::uint32_t height) -> void {
    if (state->presenter != nullptr && width != 0 && height != 0) state->presenter->SetDrawableSize(width, height);
}

auto VulkanDevice::SelectDynamicRange(const DisplayBuffer*) -> void {}

auto VulkanDevice::Presentable() const -> bool { return state->presenter != nullptr; }

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

auto VulkanDevice::AcquireImage() -> bool { return state->presenter != nullptr; }

auto VulkanDevice::PresentClear(std::uint32_t width, std::uint32_t height, bool opaque) -> bool {
    auto& s = *state;
    if (s.presenter == nullptr) return false;
    std::vector<std::byte> pixels(static_cast<std::size_t>(width) * height * 4u);
    if (opaque) {
        for (std::size_t index = 3; index < pixels.size(); index += 4) pixels[index] = std::byte{0xff};
    }
    std::lock_guard lock(s.mutex);
    @autoreleasepool {
        s.settleForPresent();
        s.stagePresent(width, height, pixels);
    }
    return true;
}

auto VulkanDevice::PresentPixels(std::uint32_t width, std::uint32_t height, std::span<const std::byte> pixels) -> void {
    auto& s = *state;
    if (s.presenter == nullptr) return;
    std::lock_guard lock(s.mutex);
    @autoreleasepool {
        s.settleForPresent();
        s.stagePresent(width, height, pixels);
        s.presentPending();
    }
}

auto VulkanDevice::PresentDisplayBuffer(const DisplayBuffer& buffer) -> bool {
    auto& s = *state;
    if (s.presenter == nullptr) return false;
    std::lock_guard lock(s.mutex);
    @autoreleasepool {
        if (GpuScanout(buffer)) {
            s.stageScanout(buffer);
            return true;
        }
        s.settleForPresent();
        const auto pixels = ReadDisplayBuffer(buffer);
        s.stagePresent(buffer.width, buffer.height, pixels);
    }
    return true;
}

auto VulkanDevice::FinishPresent() -> double {
    auto& s = *state;
    if (s.presenter == nullptr) return 0.0;
    @autoreleasepool {
        return s.presentPending();
    }
}

auto VulkanDevice::QueuePresent() -> void {}

auto VulkanDevice::FlipBatches(std::uint64_t& submissions, std::uint64_t& unsignaled) const -> void {
    submissions = 0;
    unsignaled = 0;
}

auto VulkanDevice::PresentCounts() -> PresentStatistics { return {}; }

auto VulkanDevice::PresyncCounts() -> PresyncStatistics { return {}; }

auto VulkanDevice::PrepareDispatch(const ShaderRecompiler::RecompileResult& shader, std::span<const Graphics::GuestMemorySnapshot>, std::uint64_t, std::uint64_t) -> std::shared_ptr<PreparedDispatch> {
    if (shader.backend == ShaderRecompiler::ShaderBackendKind::Msl && !shader.msl.empty()) static_cast<void>(state->pipeline(shader));
    return {};
}

auto VulkanDevice::PreparePhaseMs(const PreparedDispatch&) -> std::span<const double, 5> {
    static constexpr double zeros[5] = {};
    return std::span<const double, 5>(zeros);
}

auto VulkanDevice::Dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>*, std::optional<std::pair<std::uint64_t, std::size_t>>) -> void {
    state->dispatch(shader, x, y, z, snapshots);
}

auto VulkanDevice::DispatchIndirect(const ShaderRecompiler::RecompileResult& shader, std::uint64_t arguments, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>*, const std::array<std::uint32_t, 3>&) -> IndirectOutcome {
    const auto started = std::chrono::steady_clock::now();
    WaitIdle();
    std::array<std::uint32_t, 3> groups{};
    std::memcpy(groups.data(), reinterpret_cast<const void*>(arguments), sizeof(groups));
    const auto readMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    state->dispatch(shader, groups[0], groups[1], groups[2], snapshots);
    return {1, readMs};
}

auto VulkanDevice::PrepareRecipe(const std::shared_ptr<const Recipe>&, bool) -> std::shared_ptr<RecipeHit> { return {}; }

auto VulkanDevice::DispatchRecipe(const ShaderRecompiler::RecompileResult&, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t, std::uint64_t, const std::shared_ptr<RecipeHit>&, IndirectOutcome&, const std::shared_ptr<PreparedDispatch>&, bool, const std::array<std::uint32_t, 3>&) -> RecipeOutcome { return RecipeOutcome::Rebuild; }

auto VulkanDevice::DispatchRecipes() -> bool { return false; }

auto VulkanDevice::VerifyRecipes() -> bool { return false; }

auto VulkanDevice::TemplateDataRefresh() -> bool { return false; }

auto VulkanDevice::NoteRecipe(RecipeEvent, bool) -> void {}

auto VulkanDevice::NoteRecipe(RecipeEvent, RecipeKind) -> void {}

auto VulkanDevice::NoteDrawRecipeMiss(DrawRecipePrecheck) -> void {}

auto VulkanDevice::Draw(const Graphics::State& graphics, const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> shaders, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>*) -> void {
    state->draw(graphics, draw, shaders, snapshots);
}

auto VulkanDevice::KnownDrawRejection(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> shaders) const -> std::optional<std::string> { return DrawRejection(graphics, shaders); }

auto VulkanDevice::ColorMetadataPass(const Graphics::ColorMetadataPass&) -> void {}

auto VulkanDevice::DrawFromRecipe(const Graphics::State&, const Pm4::DrawParameters&, std::span<const Graphics::CompiledShader>, std::span<const Graphics::GuestMemorySnapshot>, const std::shared_ptr<const DrawRecipe>&) -> RecipeOutcome { return RecipeOutcome::Rebuild; }

auto VulkanDevice::DrawIndirectSupport() const -> IndirectDrawSupport { return {false, false, false}; }

auto VulkanDevice::EnqueueDraw(const Graphics::State& graphics, const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> shaders, std::span<const Graphics::GuestMemorySnapshot> snapshots) -> void {
    state->draw(graphics, draw, shaders, snapshots);
}

auto VulkanDevice::PrebuildGraphicsLibraries(const Graphics::ShaderStages&, std::span<const Graphics::CompiledShader>) -> bool { return false; }

}
