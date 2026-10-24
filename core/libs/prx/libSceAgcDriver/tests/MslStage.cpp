#include "MslStagePrograms.hpp"
#include "MetalAbi.hpp"
#include "PipelineSpecialization.hpp"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace ShaderRecompiler;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Check(const RecompileRequest& request, std::string_view name, std::initializer_list<std::string_view> markers) {
    const auto result = Recompile(request);
    const auto source = std::string(result.msl.Source());
    Require(result.backend == ShaderBackendKind::Msl, std::string(name) + ": the result is not an MSL artifact");
    Require(result.spirv.empty(), std::string(name) + ": an MSL artifact also carries SPIR-V");
    for (const auto marker : markers) Require(source.find(marker) != std::string::npos, std::string(name) + ": the source lacks " + std::string(marker) + "\n" + source);
    Require(source.find("kernel void") == std::string::npos, std::string(name) + ": a graphics program was emitted as a kernel");
    return source;
}

bool HasConstant(const RecompileResult& result, std::uint32_t id, std::uint32_t value) {
    return std::ranges::any_of(result.specialization, [&](const PipelineSpecializationConstant& constant) { return constant.id == id && constant.value == value; });
}

void CheckVertex() {
    Check(MslStagePrograms::GridVertexRequest(0x10000u), "grid vertex", {"vertex aps5_stage_out aps5_main(", "uint aps5_vertex_id [[vertex_id]]", "float4 position [[position]]", "float4 p0 [[user(locn0)]]", "float4 p1 [[user(locn1)]]", "[[clip_distance]] [1]", "aps5_c60000", "return aps5_out;"});
    const auto request = MslStagePrograms::FetchVertexRequest(0x11000u);
    Check(request, "fetch vertex", {"aps5_stage_in aps5_in [[stage_in]]", "float4 a0f [[attribute(0), function_constant(aps5_vf0)]]", "uint4 a0u [[attribute(0), function_constant(aps5_vu0)]]"});
    const auto result = Recompile(request);
    Require(result.vertexInputs.size() == 1u && result.vertexInputs[0].location == 0u, "fetch vertex: the result does not describe its vertex attribute");
    Require(result.vertexAttributes.size() == 1u, "fetch vertex: the invocation does not describe its vertex attribute");
    const auto first = PipelineSpecialization::VertexBase;
    Require(HasConstant(result, first, 4u) && HasConstant(result, first + 3u, 7u) && HasConstant(result, first + 5u, 0u), "fetch vertex: the vertex class and selector constants are missing");
}

void CheckPixel() {
    const auto interpolated = MslStagePrograms::InterpolatedPixelRequest(0x20000u);
    Check(interpolated, "interpolated pixel", {"fragment aps5_stage_out aps5_main(", "float4 p0 [[user(locn0), center_perspective]]", "float4 c0 [[color(0)]]", "aps5_c8192"});
    const auto result = Recompile(interpolated);
    Require(HasConstant(result, PipelineSpecialization::ExportBase + 3u, 3u), "interpolated pixel: the export mapping constants are missing");
    Require(result.fragmentParameters.size() == 1u, "interpolated pixel: the result does not describe its parameter");
    Check(MslStagePrograms::KillDepthPixelRequest(0x21000u), "kill depth pixel", {"[[user(locn1), flat]]", "float depth [[depth(any)]]", "discard_fragment();", "aps5_out.depth ="});
    Check(MslStagePrograms::ConstantPixelRequest(0x22000u), "constant pixel", {"fragment aps5_stage_out aps5_main(", "aps5_out.c0 ="});
}

void Rejected(const std::string& name, const std::function<void(RecompileRequest&)>& change) {
    auto request = MslStagePrograms::InterpolatedPixelRequest(0x23000u);
    change(request);
    bool rejected = false;
    try {
        static_cast<void>(Recompile(request));
    } catch (const std::exception&) {
        rejected = true;
    }
    Require(rejected, name + " recompiled to MSL");
}

void CheckUnsupported() {
    Rejected("a quad pixel mask", [](RecompileRequest& request) { request.context.pixel->quadPixelMask = 0x7u; });
    Rejected("a primitive-ordered pixel shader", [](RecompileRequest& request) { request.context.pixel->orderedPixelShader = true; });
    Rejected("a wave64 pixel shader", [](RecompileRequest& request) {
        request.context.waveSize = 64;
        request.context.pixel->wave32 = false;
    });
}

}

int main() {
    try {
        CheckVertex();
        CheckPixel();
        CheckUnsupported();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MslStage: %s\n", error.what());
        return 1;
    }
    std::puts("MslStage: vertex and pixel programs recompile to MSL stage functions with their inputs, exports and pipeline constants");
    return 0;
}
