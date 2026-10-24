#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_TEXTURES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_TEXTURES_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace AgcDriver::Graphics::Metal {

enum class TextureKind : std::uint8_t { Tex1D, Tex1DArray, Tex2D, Tex2DArray, Cube, CubeArray, Tex3D, Tex2DMsaa };

enum class UploadPath : std::uint8_t { Direct, Staged };

struct TextureCreateInfo {
    std::uint32_t guestFormat = 0;
    TextureKind kind = TextureKind::Tex2D;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t depth = 1;
    std::uint32_t arrayLength = 1;
    std::uint32_t mipLevels = 1;
    std::uint32_t samples = 1;
    bool renderTarget = false;
    bool shaderWrite = true;
    const char* label = nullptr;
};

struct TextureHandle {
    void* object = nullptr;
    void* texture = nullptr;
    std::uint64_t resourceId = 0;
    std::uint64_t bytes = 0;
    PixelFormat pixelFormat = PixelFormat::Invalid;
    bool view = false;

    explicit operator bool() const { return object != nullptr; }
};

struct ViewInfo {
    PixelFormat pixelFormat = PixelFormat::Invalid;
    TextureKind kind = TextureKind::Tex2D;
    bool changeKind = false;
    std::uint32_t baseLevel = 0;
    std::uint32_t levelCount = 1;
    std::uint32_t baseSlice = 0;
    std::uint32_t sliceCount = 1;
    std::array<TextureSwizzle, 4> swizzle{TextureSwizzle::Red, TextureSwizzle::Green, TextureSwizzle::Blue, TextureSwizzle::Alpha};
};

struct TextureStats {
    std::uint64_t textures = 0;
    std::uint64_t views = 0;
    std::uint64_t bytes = 0;
    std::uint64_t budgetBytes = 0;
    std::uint64_t stagingBytes = 0;
    std::uint64_t uploads = 0;
    std::uint64_t writebacks = 0;
    DetileCounters detile;
};

class TextureManager {
public:
    static std::unique_ptr<TextureManager> Create(Device& device, MetalCompiler& compiler, std::string* failure);
    ~TextureManager();
    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    bool SupportsFormat(std::uint32_t guestFormat) const;
    TextureHandle Create(const TextureCreateInfo& info, std::string* failure);
    TextureHandle View(const TextureHandle& base, const ViewInfo& info);
    void Release(TextureHandle& handle);

    void EncodeUpload(void* computeEncoder, const TextureHandle& texture, const GuestTextureResource& descriptor, std::uint64_t surfaceDeviceAddress, UploadPath path);
    void EncodeWriteback(void* computeEncoder, const TextureHandle& texture, const GuestTextureResource& descriptor, std::uint64_t surfaceDeviceAddress);
    void Seal(QueueKind queue, std::uint64_t serial);

    UploadPath DefaultUploadPath() const;
    TextureStats Stats() const;

    struct State;

private:
    explicit TextureManager(std::unique_ptr<State> state);

    std::unique_ptr<State> state;
};

TextureCreateInfo DescribeTexture(const GuestTextureResource& descriptor);

}

#endif
