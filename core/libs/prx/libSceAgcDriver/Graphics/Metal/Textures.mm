#include "prx/libSceAgcDriver/Graphics/Metal/Textures.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureJobs.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr std::uint64_t MiB = 1ull << 20u;
constexpr std::uint64_t StagingRegionAlign = 256;
constexpr std::uint64_t StagingPitchAlign = 16;

#define AGC_CHECK_FORMAT(name) static_assert(static_cast<std::uint16_t>(PixelFormat::name) == static_cast<std::uint16_t>(MTLPixelFormat##name), #name)
AGC_CHECK_FORMAT(R8Unorm);
AGC_CHECK_FORMAT(R8Snorm);
AGC_CHECK_FORMAT(R8Uint);
AGC_CHECK_FORMAT(R8Sint);
AGC_CHECK_FORMAT(R16Unorm);
AGC_CHECK_FORMAT(R16Snorm);
AGC_CHECK_FORMAT(R16Uint);
AGC_CHECK_FORMAT(R16Sint);
AGC_CHECK_FORMAT(R16Float);
AGC_CHECK_FORMAT(RG8Unorm);
AGC_CHECK_FORMAT(RG8Snorm);
AGC_CHECK_FORMAT(RG8Uint);
AGC_CHECK_FORMAT(RG8Sint);
AGC_CHECK_FORMAT(B5G6R5Unorm);
AGC_CHECK_FORMAT(ABGR4Unorm);
AGC_CHECK_FORMAT(BGR5A1Unorm);
AGC_CHECK_FORMAT(R32Uint);
AGC_CHECK_FORMAT(R32Sint);
AGC_CHECK_FORMAT(R32Float);
AGC_CHECK_FORMAT(RG16Unorm);
AGC_CHECK_FORMAT(RG16Snorm);
AGC_CHECK_FORMAT(RG16Uint);
AGC_CHECK_FORMAT(RG16Sint);
AGC_CHECK_FORMAT(RG16Float);
AGC_CHECK_FORMAT(RGBA8Unorm);
AGC_CHECK_FORMAT(RGBA8Snorm);
AGC_CHECK_FORMAT(RGBA8Uint);
AGC_CHECK_FORMAT(RGBA8Sint);
AGC_CHECK_FORMAT(RGB10A2Unorm);
AGC_CHECK_FORMAT(RGB10A2Uint);
AGC_CHECK_FORMAT(RG11B10Float);
AGC_CHECK_FORMAT(RGB9E5Float);
AGC_CHECK_FORMAT(RG32Uint);
AGC_CHECK_FORMAT(RG32Sint);
AGC_CHECK_FORMAT(RG32Float);
AGC_CHECK_FORMAT(RGBA16Unorm);
AGC_CHECK_FORMAT(RGBA16Snorm);
AGC_CHECK_FORMAT(RGBA16Uint);
AGC_CHECK_FORMAT(RGBA16Sint);
AGC_CHECK_FORMAT(RGBA16Float);
AGC_CHECK_FORMAT(RGBA32Uint);
AGC_CHECK_FORMAT(RGBA32Sint);
AGC_CHECK_FORMAT(RGBA32Float);
static_assert(static_cast<std::uint16_t>(PixelFormat::R8UnormSrgb) == static_cast<std::uint16_t>(MTLPixelFormatR8Unorm_sRGB));
static_assert(static_cast<std::uint16_t>(PixelFormat::RG8UnormSrgb) == static_cast<std::uint16_t>(MTLPixelFormatRG8Unorm_sRGB));
static_assert(static_cast<std::uint16_t>(PixelFormat::RGBA8UnormSrgb) == static_cast<std::uint16_t>(MTLPixelFormatRGBA8Unorm_sRGB));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC1RGBA) == static_cast<std::uint16_t>(MTLPixelFormatBC1_RGBA));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC1RGBASrgb) == static_cast<std::uint16_t>(MTLPixelFormatBC1_RGBA_sRGB));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC2RGBA) == static_cast<std::uint16_t>(MTLPixelFormatBC2_RGBA));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC2RGBASrgb) == static_cast<std::uint16_t>(MTLPixelFormatBC2_RGBA_sRGB));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC3RGBA) == static_cast<std::uint16_t>(MTLPixelFormatBC3_RGBA));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC3RGBASrgb) == static_cast<std::uint16_t>(MTLPixelFormatBC3_RGBA_sRGB));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC4RUnorm) == static_cast<std::uint16_t>(MTLPixelFormatBC4_RUnorm));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC4RSnorm) == static_cast<std::uint16_t>(MTLPixelFormatBC4_RSnorm));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC5RGUnorm) == static_cast<std::uint16_t>(MTLPixelFormatBC5_RGUnorm));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC5RGSnorm) == static_cast<std::uint16_t>(MTLPixelFormatBC5_RGSnorm));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC6HRGBFloat) == static_cast<std::uint16_t>(MTLPixelFormatBC6H_RGBFloat));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC6HRGBUfloat) == static_cast<std::uint16_t>(MTLPixelFormatBC6H_RGBUfloat));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC7RGBAUnorm) == static_cast<std::uint16_t>(MTLPixelFormatBC7_RGBAUnorm));
static_assert(static_cast<std::uint16_t>(PixelFormat::BC7RGBAUnormSrgb) == static_cast<std::uint16_t>(MTLPixelFormatBC7_RGBAUnorm_sRGB));
static_assert(static_cast<std::uint32_t>(TextureSwizzle::Zero) == static_cast<std::uint32_t>(MTLTextureSwizzleZero));
static_assert(static_cast<std::uint32_t>(TextureSwizzle::One) == static_cast<std::uint32_t>(MTLTextureSwizzleOne));
static_assert(static_cast<std::uint32_t>(TextureSwizzle::Red) == static_cast<std::uint32_t>(MTLTextureSwizzleRed));
static_assert(static_cast<std::uint32_t>(TextureSwizzle::Green) == static_cast<std::uint32_t>(MTLTextureSwizzleGreen));
static_assert(static_cast<std::uint32_t>(TextureSwizzle::Blue) == static_cast<std::uint32_t>(MTLTextureSwizzleBlue));
static_assert(static_cast<std::uint32_t>(TextureSwizzle::Alpha) == static_cast<std::uint32_t>(MTLTextureSwizzleAlpha));

std::uint64_t envMebibytes(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == 0) return fallback;
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || parsed == 0) return fallback;
    return parsed * MiB;
}

MTLTextureType nativeType(TextureKind kind) {
    switch (kind) {
        case TextureKind::Tex1D: return MTLTextureType1D;
        case TextureKind::Tex1DArray: return MTLTextureType1DArray;
        case TextureKind::Tex2D: return MTLTextureType2D;
        case TextureKind::Tex2DArray: return MTLTextureType2DArray;
        case TextureKind::Cube: return MTLTextureTypeCube;
        case TextureKind::CubeArray: return MTLTextureTypeCubeArray;
        case TextureKind::Tex3D: return MTLTextureType3D;
        case TextureKind::Tex2DMsaa: return MTLTextureType2DMultisample;
    }
    return MTLTextureType2D;
}

std::uint32_t sliceCount(const TextureCreateInfo& info) {
    switch (info.kind) {
        case TextureKind::Cube: return 6u;
        case TextureKind::CubeArray: return 6u * info.arrayLength;
        case TextureKind::Tex1DArray:
        case TextureKind::Tex2DArray: return info.arrayLength;
        default: return 1u;
    }
}

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct Object {
    TextureCreateInfo info;
    const FormatInfo* format = nullptr;
    id<MTLTexture> texture = nil;
    std::vector<id<MTLTexture>> writeViews;
    std::vector<id<MTLTexture>> views;
    std::uint64_t bytes = 0;
    std::uint32_t slices = 1;
};

struct Totals {
    std::atomic<std::uint64_t> bytes{0};
    std::atomic<std::uint64_t> textures{0};
    std::atomic<std::uint64_t> views{0};
};

}

struct TextureManager::State {
    Device& device;
    id<MTLDevice> native = nil;
    std::unique_ptr<TextureDetiler> detiler;
    std::mutex mutex;
    std::shared_ptr<Totals> totals = std::make_shared<Totals>();
    std::atomic<std::uint64_t> uploads{0};
    std::atomic<std::uint64_t> writebacks{0};
    std::atomic<std::uint64_t> stagingBytes{0};
    std::uint64_t budget = 0;
    bool bcSupported = false;
    UploadPath defaultPath = UploadPath::Direct;
    std::vector<NativeBuffer> staging;

    explicit State(Device& device) : device(device) {}

    id<MTLTexture> writeView(Object& object, std::uint32_t level) {
        if (object.writeViews.size() <= level) object.writeViews.resize(object.info.mipLevels, nil);
        if (object.writeViews[level] != nil) return object.writeViews[level];
        const auto texelBytes = object.format->texelBytes;
        const auto format = static_cast<MTLPixelFormat>(WriteViewFormat(texelBytes));
        id<MTLTexture> view = nil;
        if (object.info.kind == TextureKind::Tex3D) {
            view = [object.texture newTextureViewWithPixelFormat:format textureType:MTLTextureType3D levels:NSMakeRange(level, 1) slices:NSMakeRange(0, 1)];
        } else if (object.info.kind == TextureKind::Tex1D || object.info.kind == TextureKind::Tex1DArray) {
            view = [object.texture newTextureViewWithPixelFormat:format textureType:MTLTextureType1DArray levels:NSMakeRange(level, 1) slices:NSMakeRange(0, object.slices)];
        } else {
            view = [object.texture newTextureViewWithPixelFormat:format textureType:MTLTextureType2DArray levels:NSMakeRange(level, 1) slices:NSMakeRange(0, object.slices)];
        }
        object.writeViews[level] = view;
        return view;
    }
};

TextureCreateInfo DescribeTexture(const GuestTextureResource& descriptor) {
    TextureCreateInfo info;
    info.guestFormat = descriptor.format;
    info.width = std::max(descriptor.width, 1u);
    info.height = std::max(descriptor.height, 1u);
    info.mipLevels = std::max(descriptor.mipCount, 1u);
    const auto layers = descriptor.depthOrLastArray + 1u;
    switch (descriptor.dimension) {
        case TextureDimension::k1D:
            info.kind = TextureKind::Tex1D;
            info.height = 1;
            break;
        case TextureDimension::k1DArray:
            info.kind = TextureKind::Tex1DArray;
            info.height = 1;
            info.arrayLength = layers;
            break;
        case TextureDimension::k2D:
            info.kind = TextureKind::Tex2D;
            break;
        case TextureDimension::k2DArray:
            info.kind = TextureKind::Tex2DArray;
            info.arrayLength = layers;
            break;
        case TextureDimension::kCube:
            info.kind = layers > 6u ? TextureKind::CubeArray : TextureKind::Cube;
            info.arrayLength = std::max(layers / 6u, 1u);
            break;
        case TextureDimension::k3D:
            info.kind = TextureKind::Tex3D;
            info.depth = layers;
            break;
        case TextureDimension::k2DMsaa:
            info.kind = TextureKind::Tex2DMsaa;
            info.samples = std::max(descriptor.samples, 1u);
            info.mipLevels = 1;
            break;
    }
    return info;
}

std::unique_ptr<TextureManager> TextureManager::Create(Device& device, MetalCompiler& compiler, std::string* failure) {
    auto state = std::make_unique<State>(device);
    state->native = (__bridge id<MTLDevice>)device.NativeDevice();
    state->detiler = TextureDetiler::Create(device, compiler, failure);
    if (state->detiler == nullptr) return nullptr;
    state->bcSupported = state->native.supportsBCTextureCompression;
    state->budget = envMebibytes("APS5_METAL_TEXTURE_MIB", device.Limits().recommendedWorkingSetBytes / 3);
    const char* path = std::getenv("APS5_METAL_UPLOAD");
    if (path != nullptr && std::strcmp(path, "staged") == 0) state->defaultPath = UploadPath::Staged;
    return std::unique_ptr<TextureManager>(new TextureManager(std::move(state)));
}

TextureManager::TextureManager(std::unique_ptr<State> state) : state(std::move(state)) {}

TextureManager::~TextureManager() {
    for (auto& buffer : state->staging) state->device.ReleaseBuffer(buffer);
}

bool TextureManager::SupportsFormat(std::uint32_t guestFormat) const {
    const auto* format = FindFormat(guestFormat);
    if (format == nullptr) return false;
    return !format->blockCompressed || state->bcSupported;
}

UploadPath TextureManager::DefaultUploadPath() const {
    return state->defaultPath;
}

TextureHandle TextureManager::Create(const TextureCreateInfo& requested, std::string* failure) {
    @autoreleasepool {
        auto& s = *state;
        TextureCreateInfo info = requested;
        if (info.mipLevels > 1 && info.kind == TextureKind::Tex1D) info.kind = TextureKind::Tex2D;
        if (info.mipLevels > 1 && info.kind == TextureKind::Tex1DArray) info.kind = TextureKind::Tex2DArray;
        const auto fail = [&](const std::string& reason) {
            if (failure != nullptr) *failure = reason;
            return TextureHandle{};
        };
        const auto* format = FindFormat(info.guestFormat);
        if (format == nullptr) return fail("unsupported guest texture format " + std::to_string(info.guestFormat));
        if (format->blockCompressed && !s.bcSupported) return fail("this GPU does not support BC texture compression");
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
        descriptor.textureType = nativeType(info.kind);
        descriptor.pixelFormat = static_cast<MTLPixelFormat>(format->pixelFormat);
        descriptor.width = info.width;
        descriptor.height = info.height;
        descriptor.depth = info.kind == TextureKind::Tex3D ? info.depth : 1;
        descriptor.mipmapLevelCount = info.mipLevels;
        descriptor.arrayLength = info.kind == TextureKind::Cube ? 1 : (info.kind == TextureKind::Tex1DArray || info.kind == TextureKind::Tex2DArray || info.kind == TextureKind::CubeArray ? info.arrayLength : 1);
        descriptor.sampleCount = info.samples;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
        if (info.shaderWrite) usage |= MTLTextureUsageShaderWrite;
        if (info.renderTarget) usage |= MTLTextureUsageRenderTarget;
        static const bool compressedTargets = std::getenv("APS5_METAL_COMPRESSED_TARGETS") != nullptr;
        if (compressedTargets && info.renderTarget) usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        descriptor.usage = usage;
        const auto sizeAndAlign = [s.native heapTextureSizeAndAlignWithDescriptor:descriptor];
        const std::uint64_t bytes = sizeAndAlign.size;
        const auto before = s.totals->bytes.fetch_add(bytes, std::memory_order_relaxed);
        if (before + bytes > s.budget) {
            s.totals->bytes.fetch_sub(bytes, std::memory_order_relaxed);
            return fail("texture memory budget exhausted");
        }
        id<MTLTexture> texture = [s.native newTextureWithDescriptor:descriptor];
        if (texture == nil) {
            s.totals->bytes.fetch_sub(bytes, std::memory_order_relaxed);
            return fail("newTextureWithDescriptor failed");
        }
        if (info.label != nullptr) texture.label = [NSString stringWithUTF8String:info.label];
        s.device.AddResidentAllocation((__bridge void*)texture);
        auto* object = new Object;
        object->info = info;
        object->format = format;
        object->texture = texture;
        object->bytes = bytes;
        object->slices = sliceCount(info);
        s.totals->textures.fetch_add(1, std::memory_order_relaxed);
        if (format->blockCompressed || s.defaultPath == UploadPath::Staged) {
            s.detiler->Warm(DetileKernel::TiledToLinear, format->guestElementBytes, format->transform, format->texelBytes);
        } else {
            const auto kernel = info.kind == TextureKind::Tex3D ? DetileKernel::TiledToVolume : (info.kind == TextureKind::Tex1D || info.kind == TextureKind::Tex1DArray ? DetileKernel::TiledToArray1D : DetileKernel::TiledToArray2D);
            s.detiler->Warm(kernel, format->guestElementBytes, format->transform, format->texelBytes);
        }
        TextureHandle handle;
        handle.object = object;
        handle.texture = (__bridge void*)texture;
        handle.resourceId = texture.gpuResourceID._impl;
        handle.bytes = bytes;
        handle.pixelFormat = format->pixelFormat;
        return handle;
    }
}

TextureHandle TextureManager::View(const TextureHandle& base, const ViewInfo& info) {
    @autoreleasepool {
        auto& s = *state;
        auto* object = static_cast<Object*>(base.object);
        if (object == nullptr) return {};
        const auto format = info.pixelFormat != PixelFormat::Invalid ? info.pixelFormat : object->format->pixelFormat;
        const auto type = info.changeKind ? nativeType(info.kind) : object->texture.textureType;
        MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(static_cast<MTLTextureSwizzle>(info.swizzle[0]), static_cast<MTLTextureSwizzle>(info.swizzle[1]), static_cast<MTLTextureSwizzle>(info.swizzle[2]), static_cast<MTLTextureSwizzle>(info.swizzle[3]));
        id<MTLTexture> view = [object->texture newTextureViewWithPixelFormat:static_cast<MTLPixelFormat>(format) textureType:type levels:NSMakeRange(info.baseLevel, info.levelCount) slices:NSMakeRange(info.baseSlice, info.sliceCount) swizzle:channels];
        if (view == nil) return {};
        {
            std::lock_guard lock(s.mutex);
            object->views.push_back(view);
        }
        s.totals->views.fetch_add(1, std::memory_order_relaxed);
        TextureHandle handle;
        handle.object = object;
        handle.texture = (__bridge void*)view;
        handle.resourceId = view.gpuResourceID._impl;
        handle.bytes = 0;
        handle.pixelFormat = format;
        handle.view = true;
        return handle;
    }
}

void TextureManager::Release(TextureHandle& handle) {
    auto& s = *state;
    auto* object = static_cast<Object*>(handle.object);
    const bool isView = handle.view;
    handle = {};
    if (object == nullptr || isView) return;
    const auto bytes = object->bytes;
    const auto views = object->views.size();
    void* allocation = (__bridge void*)object->texture;
    s.device.RetireAllocation(allocation, [totals = s.totals, object, bytes, views]() {
        totals->bytes.fetch_sub(bytes, std::memory_order_relaxed);
        totals->textures.fetch_sub(1, std::memory_order_relaxed);
        totals->views.fetch_sub(views, std::memory_order_relaxed);
        delete object;
    });
}

void TextureManager::EncodeUpload(void* computeEncoder, const TextureHandle& handle, const GuestTextureResource& descriptor, std::uint64_t surfaceDeviceAddress, UploadPath path) {
    auto& s = *state;
    auto* object = static_cast<Object*>(handle.object);
    if (object == nullptr) throw std::runtime_error("AGC graphics: texture upload needs a texture");
    const auto plan = PlanSurface(descriptor, surfaceDeviceAddress);
    const auto& format = *object->format;
    id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)computeEncoder;
    const bool volume = object->info.kind == TextureKind::Tex3D;
    const auto texelBytes = format.texelBytes;
    if (path == UploadPath::Direct && !format.blockCompressed) {
        const auto kernel = volume ? DetileKernel::TiledToVolume : (object->info.kind == TextureKind::Tex1D || object->info.kind == TextureKind::Tex1DArray ? DetileKernel::TiledToArray1D : DetileKernel::TiledToArray2D);
        for (const auto& job : plan.jobs) {
            id<MTLTexture> view;
            {
                std::lock_guard lock(s.mutex);
                view = s.writeView(*object, job.level);
            }
            s.detiler->Encode(computeEncoder, kernel, plan.elementBytes, format.transform, texelBytes, job.params, (__bridge void*)view, job.columns, job.rows);
        }
        s.uploads.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::vector<std::uint64_t> offsets(plan.jobs.size());
    std::vector<std::uint32_t> pitches(plan.jobs.size());
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < plan.jobs.size(); ++index) {
        const auto& job = plan.jobs[index];
        pitches[index] = static_cast<std::uint32_t>(alignUp(static_cast<std::uint64_t>(job.columns) * texelBytes, StagingPitchAlign));
        offsets[index] = total;
        total = alignUp(total + static_cast<std::uint64_t>(pitches[index]) * job.rows, StagingRegionAlign);
    }
    NativeBuffer staging = s.device.NewPrivateBuffer(total, "AnyPS5 texture staging");
    if (!staging) throw std::runtime_error("AGC graphics: Metal texture staging buffer could not be allocated");
    {
        std::lock_guard lock(s.mutex);
        s.staging.push_back(staging);
    }
    s.stagingBytes.fetch_add(total, std::memory_order_relaxed);
    for (std::size_t index = 0; index < plan.jobs.size(); ++index) {
        auto params = plan.jobs[index].params;
        params.linear = staging.gpuAddress + offsets[index];
        params.linearPitchBytes = pitches[index];
        s.detiler->Encode(computeEncoder, DetileKernel::TiledToLinear, plan.elementBytes, format.transform, texelBytes, params, nullptr, plan.jobs[index].columns, plan.jobs[index].rows);
    }
    [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)staging.handle;
    const auto blockWidth = format.blockCompressed ? 4u : 1u;
    for (std::size_t index = 0; index < plan.jobs.size(); ++index) {
        const auto& job = plan.jobs[index];
        const auto levelWidth = std::max(object->info.width >> job.level, 1u);
        const auto levelHeight = std::max(object->info.height >> job.level, 1u);
        const auto copyWidth = std::min(job.columns * blockWidth, levelWidth);
        const auto copyHeight = std::min(job.rows * blockWidth, levelHeight);
        const auto size = MTLSizeMake(copyWidth, copyHeight, 1);
        const auto origin = volume ? MTLOriginMake(0, 0, job.destinationDepth) : MTLOriginMake(0, 0, 0);
        const auto slice = volume ? 0u : job.destinationLayer;
        [encoder copyFromBuffer:buffer sourceOffset:offsets[index] sourceBytesPerRow:pitches[index] sourceBytesPerImage:static_cast<NSUInteger>(pitches[index]) * job.rows sourceSize:size toTexture:object->texture destinationSlice:slice destinationLevel:job.level destinationOrigin:origin];
    }
    s.uploads.fetch_add(1, std::memory_order_relaxed);
}

void TextureManager::EncodeWriteback(void* computeEncoder, const TextureHandle& handle, const GuestTextureResource& descriptor, std::uint64_t surfaceDeviceAddress) {
    auto& s = *state;
    auto* object = static_cast<Object*>(handle.object);
    if (object == nullptr) throw std::runtime_error("AGC graphics: texture writeback needs a texture");
    const auto& format = *object->format;
    if ((format.transform != TextureTransform::None && format.transform != TextureTransform::Repack101111) || format.blockCompressed) throw std::runtime_error("AGC graphics: this texture format cannot be written back to guest memory on Metal");
    if (object->info.kind == TextureKind::Tex1D || object->info.kind == TextureKind::Tex1DArray) throw std::runtime_error("AGC graphics: 1D texture writeback is not modeled on Metal");
    const auto plan = PlanSurface(descriptor, surfaceDeviceAddress);
    const bool volume = object->info.kind == TextureKind::Tex3D;
    const auto kernel = volume ? DetileKernel::VolumeToTiled : DetileKernel::Array2DToTiled;
    for (const auto& job : plan.jobs) {
        id<MTLTexture> view;
        {
            std::lock_guard lock(s.mutex);
            view = s.writeView(*object, job.level);
        }
        s.detiler->Encode(computeEncoder, kernel, plan.elementBytes, format.transform, format.texelBytes, job.params, (__bridge void*)view, job.columns, job.rows);
    }
    s.writebacks.fetch_add(1, std::memory_order_relaxed);
}

void TextureManager::EncodeLinearToTiled(void* computeEncoder, const GuestTextureResource& descriptor, std::uint64_t surfaceDeviceAddress, std::uint64_t linearDeviceAddress, std::uint32_t linearPitchBytes) {
    auto& s = *state;
    const auto plan = PlanSurface(descriptor, surfaceDeviceAddress);
    if (plan.volume || plan.jobs.empty()) throw std::runtime_error("AGC graphics: linear data can only be tiled into a single 2D surface on Metal");
    auto job = plan.jobs.front();
    job.params.linear = linearDeviceAddress;
    job.params.linearPitchBytes = linearPitchBytes;
    s.detiler->Encode(computeEncoder, DetileKernel::LinearToTiled, plan.elementBytes, TextureTransform::None, plan.texelBytes, job.params, nullptr, job.columns, job.rows);
    s.writebacks.fetch_add(1, std::memory_order_relaxed);
}

void TextureManager::Seal(QueueKind queue, std::uint64_t serial) {
    auto& s = *state;
    s.detiler->Seal(queue, serial);
    std::vector<NativeBuffer> retired;
    {
        std::lock_guard lock(s.mutex);
        retired.swap(s.staging);
    }
    for (const auto& buffer : retired) {
        s.stagingBytes.fetch_sub(buffer.bytes, std::memory_order_relaxed);
        s.device.ReleaseBuffer(buffer);
    }
}

TextureStats TextureManager::Stats() const {
    const auto& s = *state;
    TextureStats stats;
    stats.textures = s.totals->textures.load(std::memory_order_relaxed);
    stats.views = s.totals->views.load(std::memory_order_relaxed);
    stats.bytes = s.totals->bytes.load(std::memory_order_relaxed);
    stats.budgetBytes = s.budget;
    stats.stagingBytes = s.stagingBytes.load(std::memory_order_relaxed);
    stats.uploads = s.uploads.load(std::memory_order_relaxed);
    stats.writebacks = s.writebacks.load(std::memory_order_relaxed);
    stats.detile = s.detiler->Counters();
    return stats;
}

}
