#include "prx/libSceAgcDriver/Graphics/Metal/TextureFormat.hpp"
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr std::array<std::uint8_t, 4> Identity{0, 1, 2, 3};
constexpr std::array<std::uint8_t, 4> SwapRedBlue{2, 1, 0, 3};

constexpr FormatInfo entry(std::uint32_t guest, PixelFormat pixel, std::uint32_t bytes, bool srgb = false, TextureTransform transform = TextureTransform::None, std::uint32_t texelBytes = 0, bool uploadable = true, std::array<std::uint8_t, 4> permutation = Identity) {
    return {guest, pixel, bytes, texelBytes != 0 ? texelBytes : bytes, false, srgb, uploadable, transform, permutation};
}

constexpr FormatInfo block(std::uint32_t guest, PixelFormat pixel, std::uint32_t bytes, bool srgb = false) {
    return {guest, pixel, bytes, bytes, true, srgb, true, TextureTransform::None, Identity};
}

constexpr FormatInfo kFormats[] = {
    entry(1, PixelFormat::R8Unorm, 1),
    entry(2, PixelFormat::R8Snorm, 1),
    entry(5, PixelFormat::R8Uint, 1),
    entry(6, PixelFormat::R8Sint, 1),
    entry(7, PixelFormat::R16Unorm, 2),
    entry(8, PixelFormat::R16Snorm, 2),
    entry(11, PixelFormat::R16Uint, 2),
    entry(12, PixelFormat::R16Sint, 2),
    entry(13, PixelFormat::R16Float, 2),
    entry(14, PixelFormat::RG8Unorm, 2),
    entry(15, PixelFormat::RG8Snorm, 2),
    entry(18, PixelFormat::RG8Uint, 2),
    entry(19, PixelFormat::RG8Sint, 2),
    entry(20, PixelFormat::R32Uint, 4),
    entry(21, PixelFormat::R32Sint, 4),
    entry(22, PixelFormat::R32Float, 4),
    entry(23, PixelFormat::RG16Unorm, 4),
    entry(24, PixelFormat::RG16Snorm, 4),
    entry(27, PixelFormat::RG16Uint, 4),
    entry(28, PixelFormat::RG16Sint, 4),
    entry(29, PixelFormat::RG16Float, 4),
    entry(36, PixelFormat::RG11B10Float, 4),
    entry(43, PixelFormat::RG11B10Float, 4, false, TextureTransform::Repack101111),
    entry(50, PixelFormat::RGB10A2Unorm, 4),
    entry(51, PixelFormat::RGBA16Snorm, 4, false, TextureTransform::ExpandSnorm2101010, 8),
    entry(54, PixelFormat::RGB10A2Uint, 4),
    entry(55, PixelFormat::RGBA16Sint, 4, false, TextureTransform::ExpandSint2101010, 8),
    entry(56, PixelFormat::RGBA8Unorm, 4),
    entry(57, PixelFormat::RGBA8Snorm, 4),
    entry(60, PixelFormat::RGBA8Uint, 4),
    entry(61, PixelFormat::RGBA8Sint, 4),
    entry(62, PixelFormat::RG32Uint, 8),
    entry(63, PixelFormat::RG32Sint, 8),
    entry(64, PixelFormat::RG32Float, 8),
    entry(65, PixelFormat::RGBA16Unorm, 8),
    entry(66, PixelFormat::RGBA16Snorm, 8),
    entry(69, PixelFormat::RGBA16Uint, 8),
    entry(70, PixelFormat::RGBA16Sint, 8),
    entry(71, PixelFormat::RGBA16Float, 8),
    entry(72, PixelFormat::RGBA32Uint, 12, false, TextureTransform::None, 16, false),
    entry(73, PixelFormat::RGBA32Sint, 12, false, TextureTransform::None, 16, false),
    entry(74, PixelFormat::RGBA32Float, 12, false, TextureTransform::None, 16, false),
    entry(75, PixelFormat::RGBA32Uint, 16),
    entry(76, PixelFormat::RGBA32Sint, 16),
    entry(77, PixelFormat::RGBA32Float, 16),
    entry(128, PixelFormat::R8UnormSrgb, 1, true),
    entry(129, PixelFormat::RG8UnormSrgb, 2, true),
    entry(130, PixelFormat::RGBA8UnormSrgb, 4, true),
    entry(132, PixelFormat::RGB9E5Float, 4),
    entry(133, PixelFormat::B5G6R5Unorm, 2),
    entry(134, PixelFormat::BGR5A1Unorm, 2),
    entry(136, PixelFormat::ABGR4Unorm, 2, false, TextureTransform::None, 0, true, SwapRedBlue),
    block(169, PixelFormat::BC1RGBA, 8),
    block(170, PixelFormat::BC1RGBASrgb, 8, true),
    block(171, PixelFormat::BC2RGBA, 16),
    block(172, PixelFormat::BC2RGBASrgb, 16, true),
    block(173, PixelFormat::BC3RGBA, 16),
    block(174, PixelFormat::BC3RGBASrgb, 16, true),
    block(175, PixelFormat::BC4RUnorm, 8),
    block(176, PixelFormat::BC4RSnorm, 8),
    block(177, PixelFormat::BC5RGUnorm, 16),
    block(178, PixelFormat::BC5RGSnorm, 16),
    block(179, PixelFormat::BC6HRGBUfloat, 16),
    block(180, PixelFormat::BC6HRGBFloat, 16),
    block(181, PixelFormat::BC7RGBAUnorm, 16),
    block(182, PixelFormat::BC7RGBAUnormSrgb, 16, true),
};

constexpr auto MakeLookup() {
    std::array<const FormatInfo*, MaxGuestFormat + 1> table{};
    for (const auto& item : kFormats) table[item.guestFormat] = &item;
    return table;
}

constexpr auto kLookup = MakeLookup();

std::uint32_t remap(std::uint32_t guestFormat) {
    return guestFormat == 30 || guestFormat == 34 || guestFormat == 43 ? 20u : guestFormat;
}

TextureSwizzle componentFor(std::uint8_t dstSel) {
    switch (dstSel) {
        case 0: return TextureSwizzle::Zero;
        case 1: return TextureSwizzle::One;
        case 4: return TextureSwizzle::Red;
        case 5: return TextureSwizzle::Green;
        case 6: return TextureSwizzle::Blue;
        case 7: return TextureSwizzle::Alpha;
        default: throw std::runtime_error("AGC graphics: guest texture descriptor has an invalid destination channel selector " + std::to_string(dstSel));
    }
}

TextureSwizzle guestChannel(std::uint32_t guestFormat, TextureSwizzle component) {
    if (component < TextureSwizzle::Red) return component;
    const auto index = static_cast<std::size_t>(component) - static_cast<std::size_t>(TextureSwizzle::Red);
    constexpr std::array<TextureSwizzle, 4> bgrb{TextureSwizzle::Blue, TextureSwizzle::Green, TextureSwizzle::Red, TextureSwizzle::Blue};
    constexpr std::array<TextureSwizzle, 4> bgra{TextureSwizzle::Blue, TextureSwizzle::Green, TextureSwizzle::Red, TextureSwizzle::Alpha};
    constexpr std::array<TextureSwizzle, 4> argb{TextureSwizzle::Alpha, TextureSwizzle::Red, TextureSwizzle::Green, TextureSwizzle::Blue};
    switch (guestFormat) {
        case 133: return bgrb[index];
        case 134: return bgra[index];
        case 136: return argb[index];
        case 43: return bgra[index];
        default: return component;
    }
}

}

const FormatInfo* FindFormat(std::uint32_t guestFormat) {
    const auto mapped = guestFormat == 30 || guestFormat == 34 ? 20u : guestFormat;
    return mapped <= MaxGuestFormat ? kLookup[mapped] : nullptr;
}

bool IsRemappedFormat(std::uint32_t guestFormat) {
    return remap(guestFormat) != guestFormat;
}

PixelFormat WriteViewFormat(std::uint32_t texelBytes) {
    switch (texelBytes) {
        case 1: return PixelFormat::R8Uint;
        case 2: return PixelFormat::R16Uint;
        case 4: return PixelFormat::R32Uint;
        case 8: return PixelFormat::RG32Uint;
        case 16: return PixelFormat::RGBA32Uint;
        default: return PixelFormat::Invalid;
    }
}

std::array<TextureSwizzle, 4> ViewSwizzle(std::uint32_t guestFormat, const std::array<std::uint8_t, 4>& dstSel) {
    const auto* format = FindFormat(guestFormat);
    if (format == nullptr) throw std::runtime_error("AGC graphics: unsupported guest texture format " + std::to_string(guestFormat));
    constexpr std::array<TextureSwizzle, 4> identity{TextureSwizzle::Red, TextureSwizzle::Green, TextureSwizzle::Blue, TextureSwizzle::Alpha};
    if (IsRemappedFormat(guestFormat) && format->transform != TextureTransform::Repack101111) return identity;
    std::array<TextureSwizzle, 4> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        auto channel = guestChannel(guestFormat, componentFor(dstSel[index]));
        if (channel >= TextureSwizzle::Red) {
            const auto source = static_cast<std::size_t>(channel) - static_cast<std::size_t>(TextureSwizzle::Red);
            channel = static_cast<TextureSwizzle>(static_cast<std::size_t>(TextureSwizzle::Red) + format->layoutPermutation[source]);
        }
        result[index] = channel;
    }
    return result;
}

}
