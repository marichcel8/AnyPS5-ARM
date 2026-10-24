#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_TEXTUREFORMAT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_TEXTUREFORMAT_HPP

#include <array>
#include <cstdint>

namespace AgcDriver::Graphics::Metal {

enum class PixelFormat : std::uint16_t {
    Invalid = 0,
    R8Unorm = 10,
    R8UnormSrgb = 11,
    R8Snorm = 12,
    R8Uint = 13,
    R8Sint = 14,
    R16Unorm = 20,
    R16Snorm = 22,
    R16Uint = 23,
    R16Sint = 24,
    R16Float = 25,
    RG8Unorm = 30,
    RG8UnormSrgb = 31,
    RG8Snorm = 32,
    RG8Uint = 33,
    RG8Sint = 34,
    B5G6R5Unorm = 40,
    ABGR4Unorm = 42,
    BGR5A1Unorm = 43,
    R32Uint = 53,
    R32Sint = 54,
    R32Float = 55,
    RG16Unorm = 60,
    RG16Snorm = 62,
    RG16Uint = 63,
    RG16Sint = 64,
    RG16Float = 65,
    RGBA8Unorm = 70,
    RGBA8UnormSrgb = 71,
    RGBA8Snorm = 72,
    RGBA8Uint = 73,
    RGBA8Sint = 74,
    BGRA8Unorm = 80,
    BGRA8UnormSrgb = 81,
    RGB10A2Unorm = 90,
    RGB10A2Uint = 91,
    RG11B10Float = 92,
    RGB9E5Float = 93,
    RG32Uint = 103,
    RG32Sint = 104,
    RG32Float = 105,
    RGBA16Unorm = 110,
    RGBA16Snorm = 112,
    RGBA16Uint = 113,
    RGBA16Sint = 114,
    RGBA16Float = 115,
    RGBA32Uint = 123,
    RGBA32Sint = 124,
    RGBA32Float = 125,
    BC1RGBA = 130,
    BC1RGBASrgb = 131,
    BC2RGBA = 132,
    BC2RGBASrgb = 133,
    BC3RGBA = 134,
    BC3RGBASrgb = 135,
    BC4RUnorm = 140,
    BC4RSnorm = 141,
    BC5RGUnorm = 142,
    BC5RGSnorm = 143,
    BC6HRGBFloat = 150,
    BC6HRGBUfloat = 151,
    BC7RGBAUnorm = 152,
    BC7RGBAUnormSrgb = 153,
    Depth16Unorm = 250,
    Depth32Float = 252,
    Stencil8 = 253,
    Depth32FloatStencil8 = 260
};

enum class TextureTransform : std::uint8_t {
    None,
    Repack101111,
    ExpandSnorm2101010,
    ExpandSint2101010
};

enum class TextureSwizzle : std::uint8_t { Zero = 0, One = 1, Red = 2, Green = 3, Blue = 4, Alpha = 5 };

struct FormatInfo {
    std::uint32_t guestFormat;
    PixelFormat pixelFormat;
    std::uint32_t guestElementBytes;
    std::uint32_t texelBytes;
    bool blockCompressed;
    bool srgb;
    bool uploadable;
    TextureTransform transform;
    std::array<std::uint8_t, 4> layoutPermutation;
};

inline constexpr std::uint32_t MaxGuestFormat = 182;

const FormatInfo* FindFormat(std::uint32_t guestFormat);
bool IsRemappedFormat(std::uint32_t guestFormat);
PixelFormat WriteViewFormat(std::uint32_t texelBytes);
std::array<TextureSwizzle, 4> ViewSwizzle(std::uint32_t guestFormat, const std::array<std::uint8_t, 4>& dstSel);

}

#endif
