#include "prx/libSceAgcDriver/Graphics/Metal/RenderStateTranslate.hpp"
#include <algorithm>

namespace AgcDriver::Graphics::Metal {

namespace {

bool IntegerFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::R8Uint:
    case PixelFormat::R8Sint:
    case PixelFormat::R16Uint:
    case PixelFormat::R16Sint:
    case PixelFormat::RG8Uint:
    case PixelFormat::RG8Sint:
    case PixelFormat::R32Uint:
    case PixelFormat::R32Sint:
    case PixelFormat::RG16Uint:
    case PixelFormat::RG16Sint:
    case PixelFormat::RGBA8Uint:
    case PixelFormat::RGBA8Sint:
    case PixelFormat::RGB10A2Uint:
    case PixelFormat::RG32Uint:
    case PixelFormat::RG32Sint:
    case PixelFormat::RGBA16Uint:
    case PixelFormat::RGBA16Sint:
    case PixelFormat::RGBA32Uint:
    case PixelFormat::RGBA32Sint: return true;
    default: return false;
    }
}

std::uint8_t TranslateWriteMask(VkColorComponentFlags mask) {
    std::uint8_t result = 0;
    if ((mask & VK_COLOR_COMPONENT_R_BIT) != 0) result |= ColorWriteRed;
    if ((mask & VK_COLOR_COMPONENT_G_BIT) != 0) result |= ColorWriteGreen;
    if ((mask & VK_COLOR_COMPONENT_B_BIT) != 0) result |= ColorWriteBlue;
    if ((mask & VK_COLOR_COMPONENT_A_BIT) != 0) result |= ColorWriteAlpha;
    return result;
}

BlendOperation TranslateBlendOperation(VkBlendOp operation) {
    switch (operation) {
    case VK_BLEND_OP_SUBTRACT: return BlendOperation::Subtract;
    case VK_BLEND_OP_REVERSE_SUBTRACT: return BlendOperation::ReverseSubtract;
    case VK_BLEND_OP_MIN: return BlendOperation::Min;
    case VK_BLEND_OP_MAX: return BlendOperation::Max;
    default: return BlendOperation::Add;
    }
}

CompareFunction TranslateCompare(VkCompareOp compare) {
    return static_cast<std::uint32_t>(compare) <= 7u ? static_cast<CompareFunction>(compare) : CompareFunction::Always;
}

StencilOperation TranslateStencilOperation(VkStencilOp operation) {
    return static_cast<std::uint32_t>(operation) <= 7u ? static_cast<StencilOperation>(operation) : StencilOperation::Keep;
}

StencilFace TranslateStencilFace(const VkStencilOpState& face) {
    StencilFace result;
    result.fail = TranslateStencilOperation(face.failOp);
    result.depthFail = TranslateStencilOperation(face.depthFailOp);
    result.pass = TranslateStencilOperation(face.passOp);
    result.compare = TranslateCompare(face.compareOp);
    result.readMask = static_cast<std::uint8_t>(face.compareMask);
    result.writeMask = static_cast<std::uint8_t>(face.writeMask);
    return result;
}

bool TranslatePrimitive(VkPrimitiveTopology topology, PrimitiveType& primitive) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: primitive = PrimitiveType::Point; return true;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: primitive = PrimitiveType::Line; return true;
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: primitive = PrimitiveType::LineStrip; return true;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: primitive = PrimitiveType::Triangle; return true;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: primitive = PrimitiveType::TriangleStrip; return true;
    default: return false;
    }
}

ScissorRect ClampScissor(const VkRect2D& scissor, VkExtent2D extent) {
    const auto clampAxis = [](std::int32_t offset, std::uint32_t size, std::uint32_t limit, std::uint32_t& start, std::uint32_t& length) {
        const auto first = std::clamp<std::int64_t>(offset, 0, limit);
        const auto last = std::clamp<std::int64_t>(static_cast<std::int64_t>(offset) + size, 0, limit);
        start = static_cast<std::uint32_t>(first);
        length = static_cast<std::uint32_t>(std::max<std::int64_t>(last - first, 0));
    };
    ScissorRect result;
    clampAxis(scissor.offset.x, scissor.extent.width, extent.width, result.x, result.width);
    clampAxis(scissor.offset.y, scissor.extent.height, extent.height, result.y, result.height);
    return result;
}

}

PixelFormat ColorTargetFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM: return PixelFormat::R8Unorm;
    case VK_FORMAT_R8_SNORM: return PixelFormat::R8Snorm;
    case VK_FORMAT_R8_UINT: return PixelFormat::R8Uint;
    case VK_FORMAT_R8_SINT: return PixelFormat::R8Sint;
    case VK_FORMAT_R8_SRGB: return PixelFormat::R8UnormSrgb;
    case VK_FORMAT_R16_UNORM: return PixelFormat::R16Unorm;
    case VK_FORMAT_R16_SNORM: return PixelFormat::R16Snorm;
    case VK_FORMAT_R16_UINT: return PixelFormat::R16Uint;
    case VK_FORMAT_R16_SINT: return PixelFormat::R16Sint;
    case VK_FORMAT_R16_SFLOAT: return PixelFormat::R16Float;
    case VK_FORMAT_R8G8_UNORM: return PixelFormat::RG8Unorm;
    case VK_FORMAT_R8G8_SNORM: return PixelFormat::RG8Snorm;
    case VK_FORMAT_R8G8_UINT: return PixelFormat::RG8Uint;
    case VK_FORMAT_R8G8_SINT: return PixelFormat::RG8Sint;
    case VK_FORMAT_R8G8_SRGB: return PixelFormat::RG8UnormSrgb;
    case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_B5G6R5_UNORM_PACK16: return PixelFormat::B5G6R5Unorm;
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
    case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
    case VK_FORMAT_B5G5R5A1_UNORM_PACK16: return PixelFormat::BGR5A1Unorm;
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
    case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return PixelFormat::ABGR4Unorm;
    case VK_FORMAT_R32_UINT: return PixelFormat::R32Uint;
    case VK_FORMAT_R32_SINT: return PixelFormat::R32Sint;
    case VK_FORMAT_R32_SFLOAT: return PixelFormat::R32Float;
    case VK_FORMAT_R16G16_UNORM: return PixelFormat::RG16Unorm;
    case VK_FORMAT_R16G16_SNORM: return PixelFormat::RG16Snorm;
    case VK_FORMAT_R16G16_UINT: return PixelFormat::RG16Uint;
    case VK_FORMAT_R16G16_SINT: return PixelFormat::RG16Sint;
    case VK_FORMAT_R16G16_SFLOAT: return PixelFormat::RG16Float;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return PixelFormat::RGBA8Unorm;
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return PixelFormat::RGBA8UnormSrgb;
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_A8B8G8R8_SNORM_PACK32: return PixelFormat::RGBA8Snorm;
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_A8B8G8R8_UINT_PACK32: return PixelFormat::RGBA8Uint;
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_A8B8G8R8_SINT_PACK32: return PixelFormat::RGBA8Sint;
    case VK_FORMAT_B8G8R8A8_UNORM: return PixelFormat::BGRA8Unorm;
    case VK_FORMAT_B8G8R8A8_SRGB: return PixelFormat::BGRA8UnormSrgb;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return PixelFormat::RGB10A2Unorm;
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: return PixelFormat::RGB10A2Uint;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return PixelFormat::RG11B10Float;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return PixelFormat::RGB9E5Float;
    case VK_FORMAT_R32G32_UINT: return PixelFormat::RG32Uint;
    case VK_FORMAT_R32G32_SINT: return PixelFormat::RG32Sint;
    case VK_FORMAT_R32G32_SFLOAT: return PixelFormat::RG32Float;
    case VK_FORMAT_R16G16B16A16_UNORM: return PixelFormat::RGBA16Unorm;
    case VK_FORMAT_R16G16B16A16_SNORM: return PixelFormat::RGBA16Snorm;
    case VK_FORMAT_R16G16B16A16_UINT: return PixelFormat::RGBA16Uint;
    case VK_FORMAT_R16G16B16A16_SINT: return PixelFormat::RGBA16Sint;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return PixelFormat::RGBA16Float;
    case VK_FORMAT_R32G32B32A32_UINT: return PixelFormat::RGBA32Uint;
    case VK_FORMAT_R32G32B32A32_SINT: return PixelFormat::RGBA32Sint;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return PixelFormat::RGBA32Float;
    default: return PixelFormat::Invalid;
    }
}

PixelFormat DepthTargetFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_D16_UNORM: return PixelFormat::Depth16Unorm;
    case VK_FORMAT_D32_SFLOAT: return PixelFormat::Depth32Float;
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return PixelFormat::Depth32FloatStencil8;
    case VK_FORMAT_S8_UINT: return PixelFormat::Stencil8;
    default: return PixelFormat::Invalid;
    }
}

bool DepthTargetHasStencil(VkFormat format) {
    return format == VK_FORMAT_D16_UNORM_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_S8_UINT;
}

BlendFactor TranslateBlendFactor(VkBlendFactor factor) {
    switch (factor) {
    case VK_BLEND_FACTOR_ZERO: return BlendFactor::Zero;
    case VK_BLEND_FACTOR_ONE: return BlendFactor::One;
    case VK_BLEND_FACTOR_SRC_COLOR: return BlendFactor::SourceColor;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return BlendFactor::OneMinusSourceColor;
    case VK_BLEND_FACTOR_DST_COLOR: return BlendFactor::DestinationColor;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return BlendFactor::OneMinusDestinationColor;
    case VK_BLEND_FACTOR_SRC_ALPHA: return BlendFactor::SourceAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return BlendFactor::OneMinusSourceAlpha;
    case VK_BLEND_FACTOR_DST_ALPHA: return BlendFactor::DestinationAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return BlendFactor::OneMinusDestinationAlpha;
    case VK_BLEND_FACTOR_CONSTANT_COLOR: return BlendFactor::BlendColor;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return BlendFactor::OneMinusBlendColor;
    case VK_BLEND_FACTOR_CONSTANT_ALPHA: return BlendFactor::BlendAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return BlendFactor::OneMinusBlendAlpha;
    case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return BlendFactor::SourceAlphaSaturated;
    case VK_BLEND_FACTOR_SRC1_COLOR: return BlendFactor::Source1Color;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: return BlendFactor::OneMinusSource1Color;
    case VK_BLEND_FACTOR_SRC1_ALPHA: return BlendFactor::Source1Alpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: return BlendFactor::OneMinusSource1Alpha;
    default: return BlendFactor::One;
    }
}

Winding TranslateFrontFace(VkFrontFace face) {
    return face == VK_FRONT_FACE_CLOCKWISE ? Winding::Clockwise : Winding::CounterClockwise;
}

TranslatedState TranslateState(const State& state, bool flatInputs) {
    TranslatedState result;
    auto reject = [&](const char* reason) {
        if (result.rejection == nullptr) result.rejection = reason;
    };
    if (state.stages.path != ShaderPath::Vertex) reject("geometry and tessellation stages need the Metal mesh path");
    if (state.rectList) reject("rect lists need the vertex-stage expansion");
    if (!TranslatePrimitive(state.topology, result.primitive)) reject("topology needs primitive conversion on Metal");
    if (state.depthBoundsTest) reject("depth bounds test is not available on Metal");
    if (state.colorRop != 0xccu) reject("device does not support color logic operations: Metal render pipelines have no logic op");
    if (state.conservativeRasterization != VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT) reject("conservative rasterization requires VK_EXT_conservative_rasterization, which Metal does not provide");
    if (state.polygonMode == VK_POLYGON_MODE_POINT) reject("a point polygon mode needs the fillModeNonSolid feature, which Metal draws do not provide");
    if (flatInputs && state.provokingVertexMode != VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT) reject("last provoking vertex is not available on Metal");
    if (state.customSampleLocations || state.sampleLocationGrid) reject("custom sample locations are not mapped on Metal");
    if (state.blends.size() > MaxColorAttachments) reject("more color targets than Metal attachments");

    auto& pipeline = result.pipeline;
    pipeline.colorCount = static_cast<std::uint8_t>(std::min<std::size_t>(state.blends.size(), MaxColorAttachments));
    for (std::uint32_t index = 0; index < pipeline.colorCount; ++index) pipeline.colors[index].writeMask = 0;
    for (const auto& color : state.colors) {
        if (color.exportIndex >= pipeline.colorCount) continue;
        auto& attachment = pipeline.colors[color.exportIndex];
        const auto& blend = state.blends[color.exportIndex];
        attachment.format = ColorTargetFormat(color.format);
        if (attachment.format == PixelFormat::Invalid) reject("color target format has no Metal render format");
        attachment.writeMask = TranslateWriteMask(blend.colorWriteMask);
        attachment.blend = blend.blendEnable != VK_FALSE && !IntegerFormat(attachment.format);
        if (!attachment.blend) continue;
        attachment.sourceRgb = TranslateBlendFactor(blend.srcColorBlendFactor);
        attachment.destinationRgb = TranslateBlendFactor(blend.dstColorBlendFactor);
        attachment.rgbOperation = TranslateBlendOperation(blend.colorBlendOp);
        attachment.sourceAlpha = TranslateBlendFactor(blend.srcAlphaBlendFactor);
        attachment.destinationAlpha = TranslateBlendFactor(blend.dstAlphaBlendFactor);
        attachment.alphaOperation = TranslateBlendOperation(blend.alphaBlendOp);
    }
    const bool hasStencil = state.depth && DepthTargetHasStencil(state.depth->format);
    if (state.depth) {
        pipeline.depthFormat = DepthTargetFormat(state.depth->format);
        if (pipeline.depthFormat == PixelFormat::Invalid) reject("depth target format has no Metal depth format");
        if (hasStencil) pipeline.stencilFormat = pipeline.depthFormat;
    }
    pipeline.sampleCount = static_cast<std::uint8_t>(std::max<std::uint32_t>(state.samples, 1u));
    pipeline.topology = TopologyClassOf(result.primitive);
    pipeline.alphaToCoverage = state.alphaToCoverage || state.ditheredAlphaToCoverage;

    auto& depthStencil = result.depthStencil;
    if (state.depth && state.depthTest) {
        depthStencil.depthCompare = TranslateCompare(state.depthCompare);
        depthStencil.depthWrite = state.depthWrite;
    }
    if (hasStencil && state.stencilTest) {
        depthStencil.stencil = true;
        depthStencil.front = TranslateStencilFace(state.stencilFront);
        depthStencil.back = TranslateStencilFace(state.stencilBack);
    }

    auto& dynamic = result.dynamic;
    dynamic.viewport = {state.viewport.x, state.viewport.y, state.viewport.width, state.viewport.height, state.viewport.minDepth, state.viewport.maxDepth};
    dynamic.scissor = ClampScissor(state.scissor, state.renderExtent);
    if (dynamic.scissor.width == 0 || dynamic.scissor.height == 0) result.cullsEverything = true;
    if (state.depthBias) dynamic.depthBias = {state.depthBiasConstant, state.depthBiasSlope, state.depthBiasClamp};
    dynamic.blendColor = state.blendConstants;
    if (depthStencil.stencil) {
        dynamic.stencilFrontReference = state.stencilFront.reference & 0xffu;
        dynamic.stencilBackReference = state.stencilBack.reference & 0xffu;
    }
    const bool triangles = pipeline.topology == TopologyClass::Triangle;
    if (triangles && (state.cullMode & VK_CULL_MODE_FRONT_AND_BACK) == VK_CULL_MODE_FRONT_AND_BACK) result.cullsEverything = true;
    else if (triangles && (state.cullMode & VK_CULL_MODE_FRONT_BIT) != 0) dynamic.cull = CullMode::Front;
    else if (triangles && (state.cullMode & VK_CULL_MODE_BACK_BIT) != 0) dynamic.cull = CullMode::Back;
    if (triangles && state.polygonMode == VK_POLYGON_MODE_LINE) dynamic.fill = FillMode::Lines;
    dynamic.winding = TranslateFrontFace(state.frontFace);
    dynamic.depthClip = state.depthClamp ? DepthClipMode::Clamp : DepthClipMode::Clip;
    result.restartOnLists = !state.primitiveRestart;
    return result;
}

}
