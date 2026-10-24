#include "MslGeometryPrograms.hpp"
#include "MetalAbi.hpp"
#include <cstdio>
#include <exception>
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
    Require(result.backend == ShaderBackendKind::Msl && result.spirv.empty(), std::string(name) + ": the result is not an MSL-only artifact");
    for (const auto marker : markers) Require(source.find(marker) != std::string::npos, std::string(name) + ": the source lacks " + std::string(marker) + "\n" + source);
    if (std::getenv("APS5_PRINT_MSL") != nullptr) std::printf("// %s\n%s\n", std::string(name).c_str(), source.c_str());
    return source;
}

MslGeometryLayout Layout(const std::string& source, std::string_view name, MslGeometryKind kind) {
    const auto layout = ParseMslGeometryLayout(source);
    Require(layout.has_value() && layout->kind == kind, std::string(name) + ": the source carries no geometry layout of its kind");
    return *layout;
}

void CheckTriangles() {
    using namespace MslGeometryPrograms;
    const auto local = Check(Tessellation(ShaderStage::Local, LocalCode, TrianglePatches), "local", {"kernel void aps5_main(", "uint aps5_tid [[thread_position_in_grid]]", "constant aps5_geometry& aps5_geo [[buffer(14)]]", "aps5_geo.local[aps5_tid * aps5_geo.local_words"});
    const auto hull = Check(Tessellation(ShaderStage::TessellationControl, HullCode, TrianglePatches), "hull", {"kernel void aps5_main(", "uint2 aps5_tid [[thread_position_in_grid]]", "aps5_geo.factors[aps5_patch * 4u", "aps5_geo.control[(aps5_patch * 3u + aps5_tid.x) * aps5_geo.control_words + 0u]", "half(", "aps5_push[30]", "aps5_push[31]"});
    const auto domain = Check(Tessellation(ShaderStage::TessellationEvaluation, DomainCode, TrianglePatches), "domain", {"[[patch(triangle, 0)]] vertex aps5_stage_out aps5_main(", "float3 aps5_tess_coord [[position_in_patch]]", "uint aps5_patch_id [[patch_id]]", "aps5_geo.control[(aps5_patch * 3u", "float4 position [[position]]", "return aps5_out;"});
    const auto localLayout = Layout(local, "local", MslGeometryKind::Local);
    const auto hullLayout = Layout(hull, "hull", MslGeometryKind::Hull);
    const auto domainLayout = Layout(domain, "domain", MslGeometryKind::Domain);
    Require(localLayout.localStride != 0u && localLayout.localStride == hullLayout.localStride, "the local and hull programs disagree on the control point record");
    Require(hullLayout.controlStride != 0u && hullLayout.controlStride == domainLayout.controlStride, "the hull and domain programs disagree on the output control point record");
    Require(hullLayout.inputControlPoints == 3u && hullLayout.outputControlPoints == 3u && hullLayout.domain == 1u && hullLayout.partitioning == 2u && hullLayout.outputTopology == 2u, "the hull layout does not describe the patch");
}

void CheckQuads() {
    using namespace MslGeometryPrograms;
    Check(Tessellation(ShaderStage::TessellationControl, QuadHullCode, QuadPatches), "quad hull", {"aps5_geo.factors[aps5_patch * 6u"});
    const auto domain = Check(Tessellation(ShaderStage::TessellationEvaluation, QuadDomainCode, QuadPatches), "quad domain", {"[[patch(quad, 0)]] vertex", "float2 aps5_tess_coord [[position_in_patch]]"});
    Require(Layout(domain, "quad domain", MslGeometryKind::Domain).domain == 2u, "the quad domain layout lost its domain");
}

void CheckMesh() {
    using namespace MslGeometryPrograms;
    const auto small = Check(Mesh(Wave32GeometryCode, SmallSubgroup, 32u), "wave32 mesh", {"[[mesh]] void aps5_main(", "aps5_mesh_t aps5_mesh", "metal::mesh<aps5_mesh_vertex, aps5_mesh_primitive, 12, 4, metal::topology::triangle>", "aps5_mesh.set_primitive_count(", "aps5_mesh.set_vertex(aps5_index, aps5_out)", "aps5_mesh.set_index(aps5_index * 3u", "bool culled [[primitive_culled]]", "threadgroup uint aps5_lds[1024]"});
    const auto layout = Layout(small, "wave32 mesh", MslGeometryKind::Mesh);
    Require(layout.meshVertices == 12u && layout.meshPrimitives == 4u && layout.meshThreads == 64u && layout.meshPrimitiveSize == 3u, "the mesh layout does not describe the subgroup");
    Check(Mesh(FastLaunchCode, FastLaunchSubgroup, 32u), "fast-launch mesh", {"metal::mesh<aps5_mesh_vertex, aps5_mesh_primitive, 3, 1, metal::topology::triangle>", "aps5_prim = "});
}

void CheckParser() {
    Require(!ParseMslGeometryLayout("kernel void aps5_main() {}").has_value(), "a compute source parsed as a geometry layout");
    Require(!ParseMslGeometryLayout("constant uint aps5_layout[13] = {9u, 0u};").has_value(), "a truncated layout parsed");
}

}

int main() {
    try {
        CheckTriangles();
        CheckQuads();
        CheckMesh();
        CheckParser();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MslGeometry: %s\n", error.what());
        return 1;
    }
    std::puts("MslGeometry: local, hull and domain programs recompile to Metal tessellation kernels and a post-tessellation vertex function, mesh programs to Metal mesh functions");
    return 0;
}
