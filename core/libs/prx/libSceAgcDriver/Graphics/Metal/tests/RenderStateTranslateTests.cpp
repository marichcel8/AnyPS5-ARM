#include "prx/libSceAgcDriver/Graphics/Metal/RenderStateTranslate.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using namespace AgcDriver::Graphics;
namespace M = AgcDriver::Graphics::Metal;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

State BaseState() {
    State state{};
    state.stages.path = ShaderPath::Vertex;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.renderExtent = {1280, 720};
    state.viewport = {0.0f, 0.0f, 1280.0f, 720.0f, 0.0f, 1.0f};
    state.scissor = {{0, 0}, {1280, 720}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.samples = 1;
    ColorTarget color{};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.exportIndex = 0;
    state.colors.push_back(color);
    state.color = color;
    state.hasColorTarget = true;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    state.blends.push_back(blend);
    return state;
}

void TestBasics() {
    const auto translated = M::TranslateState(BaseState());
    Require(translated.rejection == nullptr, "plain triangle list is accepted");
    Require(translated.primitive == M::PrimitiveType::Triangle && translated.pipeline.topology == M::TopologyClass::Triangle, "triangle list");
    Require(translated.pipeline.colorCount == 1 && translated.pipeline.colors[0].format == M::PixelFormat::RGBA8Unorm, "color format");
    Require(translated.pipeline.colors[0].writeMask == M::ColorWriteAll && !translated.pipeline.colors[0].blend, "write mask and blend");
    Require(translated.depthStencil.depthCompare == M::CompareFunction::Always && !translated.depthStencil.depthWrite, "no depth target means no depth test");
    Require(translated.dynamic.scissor.width == 1280 && translated.dynamic.scissor.height == 720, "scissor");
    Require(translated.dynamic.winding == M::Winding::CounterClockwise, "front face");
}

void TestBlendAndMasks() {
    auto state = BaseState();
    auto& blend = state.blends[0];
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_A_BIT;
    const auto translated = M::TranslateState(state);
    const auto& color = translated.pipeline.colors[0];
    Require(color.blend, "blend enabled");
    Require(color.sourceRgb == M::BlendFactor::SourceAlpha && color.destinationRgb == M::BlendFactor::OneMinusSourceAlpha, "rgb factors");
    Require(color.sourceAlpha == M::BlendFactor::BlendAlpha && color.destinationAlpha == M::BlendFactor::OneMinusSource1Alpha, "alpha factors");
    Require(color.alphaOperation == M::BlendOperation::ReverseSubtract, "alpha operation");
    Require(color.writeMask == (M::ColorWriteRed | M::ColorWriteAlpha), "write mask bits are reversed");
    Require(M::TranslateBlendFactor(VK_BLEND_FACTOR_DST_COLOR) == M::BlendFactor::DestinationColor, "vulkan and metal orders differ for destination color");
    Require(M::TranslateBlendFactor(VK_BLEND_FACTOR_SRC_ALPHA_SATURATE) == M::BlendFactor::SourceAlphaSaturated, "alpha saturate");

    state.colors[0].format = VK_FORMAT_R32_UINT;
    Require(!M::TranslateState(state).pipeline.colors[0].blend, "integer targets never blend");
}

void TestExportGaps() {
    auto state = BaseState();
    state.colors[0].exportIndex = 2;
    state.blends.resize(3);
    state.blends[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    const auto translated = M::TranslateState(state);
    Require(translated.pipeline.colorCount == 3, "attachment count follows the export slots");
    Require(translated.pipeline.colors[0].format == M::PixelFormat::Invalid && translated.pipeline.colors[0].writeMask == 0, "gap attachment is empty");
    Require(translated.pipeline.colors[2].format == M::PixelFormat::RGBA8Unorm, "export index places the target");
}

void TestDepthStencil() {
    auto state = BaseState();
    DepthTarget depth{};
    depth.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    state.depth = depth;
    state.depthTest = true;
    state.depthWrite = true;
    state.depthCompare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    state.stencilTest = true;
    state.stencilFront = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_INCREMENT_AND_WRAP, VK_COMPARE_OP_EQUAL, 0x1ff, 0x0f, 0x181};
    state.stencilBack = state.stencilFront;
    state.stencilBack.reference = 7;
    const auto translated = M::TranslateState(state);
    Require(translated.pipeline.depthFormat == M::PixelFormat::Depth32FloatStencil8 && translated.pipeline.stencilFormat == translated.pipeline.depthFormat, "combined depth stencil format");
    Require(translated.depthStencil.depthCompare == M::CompareFunction::GreaterEqual && translated.depthStencil.depthWrite, "depth test");
    Require(translated.depthStencil.stencil && translated.depthStencil.front.pass == M::StencilOperation::Replace && translated.depthStencil.front.depthFail == M::StencilOperation::IncrementWrap, "stencil ops");
    Require(translated.depthStencil.front.readMask == 0xff && translated.depthStencil.front.writeMask == 0x0f, "stencil masks are 8 bit");
    Require(translated.dynamic.stencilFrontReference == 0x81 && translated.dynamic.stencilBackReference == 7, "stencil references");

    state.depthTest = false;
    Require(!M::TranslateState(state).depthStencil.depthWrite, "depth writes need the depth test");
    state.depth->format = VK_FORMAT_D16_UNORM;
    const auto d16 = M::TranslateState(state);
    Require(d16.pipeline.depthFormat == M::PixelFormat::Depth16Unorm && d16.pipeline.stencilFormat == M::PixelFormat::Invalid && !d16.depthStencil.stencil, "depth only");
}

void TestRasterAndRejections() {
    auto state = BaseState();
    state.cullMode = VK_CULL_MODE_BACK_BIT;
    state.frontFace = VK_FRONT_FACE_CLOCKWISE;
    state.depthClamp = true;
    state.depthBias = true;
    state.depthBiasConstant = 2.0f;
    state.depthBiasSlope = 1.5f;
    auto translated = M::TranslateState(state);
    Require(translated.dynamic.cull == M::CullMode::Back && translated.dynamic.winding == M::Winding::Clockwise, "cull and winding");
    Require(translated.dynamic.depthClip == M::DepthClipMode::Clamp && translated.dynamic.depthBias.constant == 2.0f && translated.dynamic.depthBias.slope == 1.5f, "clamp and bias");

    state.cullMode = VK_CULL_MODE_FRONT_AND_BACK;
    Require(M::TranslateState(state).cullsEverything, "front and back culling draws nothing");
    state = BaseState();
    state.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    state.cullMode = VK_CULL_MODE_FRONT_AND_BACK;
    translated = M::TranslateState(state);
    Require(!translated.cullsEverything && translated.dynamic.cull == M::CullMode::None && translated.pipeline.topology == M::TopologyClass::Line, "lines are never culled");

    state = BaseState();
    state.scissor = {{-100, 600}, {400, 400}};
    translated = M::TranslateState(state);
    Require(translated.dynamic.scissor.x == 0 && translated.dynamic.scissor.width == 300 && translated.dynamic.scissor.y == 600 && translated.dynamic.scissor.height == 120, "scissor clamps to the render extent");
    state.scissor = {{2000, 0}, {10, 10}};
    Require(M::TranslateState(state).cullsEverything, "empty scissor draws nothing");

    state = BaseState();
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    Require(M::TranslateState(state).rejection != nullptr, "fans need conversion");
    state = BaseState();
    state.rectList = true;
    Require(M::TranslateState(state).rejection != nullptr, "rect lists need expansion");
    state = BaseState();
    state.depthBoundsTest = true;
    Require(M::TranslateState(state).rejection != nullptr, "depth bounds are rejected");
    state = BaseState();
    state.provokingVertexMode = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT;
    Require(M::TranslateState(state).rejection != nullptr, "last provoking vertex is rejected");
    Require(M::TranslateState(state, false).rejection == nullptr, "last provoking vertex is ignored without flat inputs");
    state = BaseState();
    state.colors[0].format = VK_FORMAT_R8G8B8_UNORM;
    Require(M::TranslateState(state).rejection != nullptr, "three-component targets are rejected");
}

}

int main() {
    TestBasics();
    TestBlendAndMasks();
    TestExportGaps();
    TestDepthStencil();
    TestRasterAndRejections();
    std::printf("agc_metal_render_state: ok\n");
    return 0;
}
