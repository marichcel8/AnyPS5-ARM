#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_RENDERSTATE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_RENDERSTATE_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/CompileKey.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureFormat.hpp"
#include <array>
#include <cstdint>
#include <cstring>

namespace AgcDriver::Graphics::Metal {

inline constexpr std::uint32_t MaxColorAttachments = 8;

enum class BlendFactor : std::uint8_t {
    Zero = 0,
    One = 1,
    SourceColor = 2,
    OneMinusSourceColor = 3,
    SourceAlpha = 4,
    OneMinusSourceAlpha = 5,
    DestinationColor = 6,
    OneMinusDestinationColor = 7,
    DestinationAlpha = 8,
    OneMinusDestinationAlpha = 9,
    SourceAlphaSaturated = 10,
    BlendColor = 11,
    OneMinusBlendColor = 12,
    BlendAlpha = 13,
    OneMinusBlendAlpha = 14,
    Source1Color = 15,
    OneMinusSource1Color = 16,
    Source1Alpha = 17,
    OneMinusSource1Alpha = 18
};

enum class BlendOperation : std::uint8_t { Add = 0, Subtract = 1, ReverseSubtract = 2, Min = 3, Max = 4 };
enum class CompareFunction : std::uint8_t { Never = 0, Less = 1, Equal = 2, LessEqual = 3, Greater = 4, NotEqual = 5, GreaterEqual = 6, Always = 7 };
enum class StencilOperation : std::uint8_t { Keep = 0, Zero = 1, Replace = 2, IncrementClamp = 3, DecrementClamp = 4, Invert = 5, IncrementWrap = 6, DecrementWrap = 7 };
enum class CullMode : std::uint8_t { None = 0, Front = 1, Back = 2 };
enum class Winding : std::uint8_t { Clockwise = 0, CounterClockwise = 1 };
enum class DepthClipMode : std::uint8_t { Clip = 0, Clamp = 1 };
enum class FillMode : std::uint8_t { Fill = 0, Lines = 1 };
enum class PrimitiveType : std::uint8_t { Point = 0, Line = 1, LineStrip = 2, Triangle = 3, TriangleStrip = 4 };
enum class TopologyClass : std::uint8_t { Unspecified = 0, Point = 1, Line = 2, Triangle = 3 };
enum class IndexType : std::uint8_t { UInt16 = 0, UInt32 = 1 };

inline constexpr std::uint8_t ColorWriteAlpha = 0x1u;
inline constexpr std::uint8_t ColorWriteBlue = 0x2u;
inline constexpr std::uint8_t ColorWriteGreen = 0x4u;
inline constexpr std::uint8_t ColorWriteRed = 0x8u;
inline constexpr std::uint8_t ColorWriteAll = 0xfu;

constexpr TopologyClass TopologyClassOf(PrimitiveType primitive) {
    switch (primitive) {
    case PrimitiveType::Point: return TopologyClass::Point;
    case PrimitiveType::Line:
    case PrimitiveType::LineStrip: return TopologyClass::Line;
    case PrimitiveType::Triangle:
    case PrimitiveType::TriangleStrip: return TopologyClass::Triangle;
    }
    return TopologyClass::Unspecified;
}

struct ColorAttachmentState {
    PixelFormat format = PixelFormat::Invalid;
    std::uint8_t writeMask = ColorWriteAll;
    bool blend = false;
    BlendFactor sourceRgb = BlendFactor::One;
    BlendFactor destinationRgb = BlendFactor::Zero;
    BlendOperation rgbOperation = BlendOperation::Add;
    BlendFactor sourceAlpha = BlendFactor::One;
    BlendFactor destinationAlpha = BlendFactor::Zero;
    BlendOperation alphaOperation = BlendOperation::Add;

    bool operator==(const ColorAttachmentState&) const = default;

    [[nodiscard]] std::uint64_t Pack() const {
        std::uint64_t word = static_cast<std::uint64_t>(format);
        word |= static_cast<std::uint64_t>(writeMask) << 16u;
        if (!blend) return word;
        word |= 1ull << 20u;
        word |= static_cast<std::uint64_t>(sourceRgb) << 21u;
        word |= static_cast<std::uint64_t>(destinationRgb) << 26u;
        word |= static_cast<std::uint64_t>(rgbOperation) << 31u;
        word |= static_cast<std::uint64_t>(sourceAlpha) << 34u;
        word |= static_cast<std::uint64_t>(destinationAlpha) << 39u;
        word |= static_cast<std::uint64_t>(alphaOperation) << 44u;
        return word;
    }
};

struct RenderPipelineState {
    std::array<ColorAttachmentState, MaxColorAttachments> colors{};
    std::uint8_t colorCount = 0;
    PixelFormat depthFormat = PixelFormat::Invalid;
    PixelFormat stencilFormat = PixelFormat::Invalid;
    std::uint8_t sampleCount = 1;
    TopologyClass topology = TopologyClass::Triangle;
    bool alphaToCoverage = false;
    bool rasterization = true;

    bool operator==(const RenderPipelineState&) const = default;

    void AddTo(CompileKeyBuilder& builder) const {
        builder.Add(colorCount);
        for (std::uint32_t index = 0; index < colorCount; ++index) builder.Add(colors[index].Pack());
        std::uint64_t word = static_cast<std::uint64_t>(depthFormat);
        word |= static_cast<std::uint64_t>(stencilFormat) << 16u;
        word |= static_cast<std::uint64_t>(sampleCount) << 32u;
        word |= static_cast<std::uint64_t>(topology) << 40u;
        word |= static_cast<std::uint64_t>(alphaToCoverage) << 44u;
        word |= static_cast<std::uint64_t>(rasterization) << 45u;
        builder.Add(word);
    }
};

struct StencilFace {
    StencilOperation fail = StencilOperation::Keep;
    StencilOperation depthFail = StencilOperation::Keep;
    StencilOperation pass = StencilOperation::Keep;
    CompareFunction compare = CompareFunction::Always;
    std::uint8_t readMask = 0xffu;
    std::uint8_t writeMask = 0xffu;

    bool operator==(const StencilFace&) const = default;

    [[nodiscard]] std::uint32_t Pack() const {
        return static_cast<std::uint32_t>(fail) | (static_cast<std::uint32_t>(depthFail) << 3u) | (static_cast<std::uint32_t>(pass) << 6u) | (static_cast<std::uint32_t>(compare) << 9u) | (static_cast<std::uint32_t>(readMask) << 12u) | (static_cast<std::uint32_t>(writeMask) << 20u);
    }
};

struct DepthStencilState {
    CompareFunction depthCompare = CompareFunction::Always;
    bool depthWrite = false;
    bool stencil = false;
    StencilFace front;
    StencilFace back;

    bool operator==(const DepthStencilState&) const = default;

    [[nodiscard]] std::uint64_t Pack() const {
        std::uint64_t word = static_cast<std::uint64_t>(depthCompare) | (static_cast<std::uint64_t>(depthWrite) << 3u) | (static_cast<std::uint64_t>(stencil) << 4u) | (1ull << 5u);
        if (!stencil) return word;
        return word | (static_cast<std::uint64_t>(front.Pack()) << 6u) | (static_cast<std::uint64_t>(back.Pack()) << 34u);
    }
};

struct Viewport {
    float originX = 0.0f;
    float originY = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float znear = 0.0f;
    float zfar = 1.0f;

    bool operator==(const Viewport&) const = default;
};

struct ScissorRect {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    bool operator==(const ScissorRect&) const = default;
};

struct DepthBias {
    float constant = 0.0f;
    float slope = 0.0f;
    float clamp = 0.0f;

    bool operator==(const DepthBias&) const = default;
};

struct DynamicState {
    Viewport viewport;
    ScissorRect scissor;
    DepthBias depthBias;
    std::array<float, 4> blendColor{};
    std::uint32_t stencilFrontReference = 0;
    std::uint32_t stencilBackReference = 0;
    CullMode cull = CullMode::None;
    Winding winding = Winding::Clockwise;
    DepthClipMode depthClip = DepthClipMode::Clip;
    FillMode fill = FillMode::Fill;
};

inline constexpr std::uint32_t DynamicViewport = 1u << 0u;
inline constexpr std::uint32_t DynamicScissor = 1u << 1u;
inline constexpr std::uint32_t DynamicDepthBias = 1u << 2u;
inline constexpr std::uint32_t DynamicBlendColor = 1u << 3u;
inline constexpr std::uint32_t DynamicStencilReference = 1u << 4u;
inline constexpr std::uint32_t DynamicCull = 1u << 5u;
inline constexpr std::uint32_t DynamicWinding = 1u << 6u;
inline constexpr std::uint32_t DynamicDepthClip = 1u << 7u;
inline constexpr std::uint32_t DynamicFill = 1u << 8u;
inline constexpr std::uint32_t DynamicAll = 0x1ffu;

inline std::uint32_t DiffDynamic(const DynamicState& current, const DynamicState& next) {
    std::uint32_t changed = 0;
    if (std::memcmp(&current.viewport, &next.viewport, sizeof(Viewport)) != 0) changed |= DynamicViewport;
    if (std::memcmp(&current.scissor, &next.scissor, sizeof(ScissorRect)) != 0) changed |= DynamicScissor;
    if (std::memcmp(&current.depthBias, &next.depthBias, sizeof(DepthBias)) != 0) changed |= DynamicDepthBias;
    if (std::memcmp(current.blendColor.data(), next.blendColor.data(), sizeof(float) * 4u) != 0) changed |= DynamicBlendColor;
    if (current.stencilFrontReference != next.stencilFrontReference || current.stencilBackReference != next.stencilBackReference) changed |= DynamicStencilReference;
    if (current.cull != next.cull) changed |= DynamicCull;
    if (current.winding != next.winding) changed |= DynamicWinding;
    if (current.depthClip != next.depthClip) changed |= DynamicDepthClip;
    if (current.fill != next.fill) changed |= DynamicFill;
    return changed;
}

}

#endif
