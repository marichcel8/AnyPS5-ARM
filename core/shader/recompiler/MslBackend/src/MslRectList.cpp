#include "MslState.hpp"
#include "MslWrap.hpp"
#include "MetalAbi.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <charconv>
#include <string>
#include <string_view>
#include <vector>

namespace ShaderRecompiler {

namespace {

using namespace MslWrap;

constexpr std::uint32_t MaxRectsPerGroup = 64u;
constexpr std::uint32_t RectOutputBudget = 16384u;
constexpr std::uint32_t DefaultFetchFormat = 31u;

std::uint32_t Location(const Member& member) {
    std::uint32_t location = 0;
    const auto& name = member.name;
    if (name.size() < 2u || name.front() != 'a' || std::from_chars(name.data() + 1, name.data() + name.size(), location).ec != std::errc{}) Unsupported("vertex input " + name + " is not named by its location");
    if (location >= 32u) Unsupported("vertex input " + name + " is past the fetch table");
    return location;
}

std::string Fetched(const Member& member, std::uint32_t location) {
    const auto components = Components(member.type);
    const auto source = "aps5_a" + std::to_string(location) + (components == 1u ? ".x" : components == 2u ? ".xy" : components == 3u ? ".xyz" : "");
    const auto scalar = Scalar(member.type);
    if (scalar == "uint") return source;
    if (scalar != "float" && scalar != "int") Unsupported("vertex input " + member.name + " has type " + member.type);
    return "as_type<" + member.type + ">(" + source + ")";
}

std::string Interpolated(const Member& member, std::string_view suffix) {
    const auto at = [&](std::string_view corner) { return "aps5_rect_v[aps5_b + " + std::string(corner) + "]." + member.name + std::string(suffix); };
    return at("0u") + " * aps5_w.x + (" + at("1u") + " * aps5_w.y + " + at("2u") + " * aps5_w.z)";
}

MslRectList BuildRectList(std::string_view vertex) {
    const auto entry = ParseEntry(vertex, "vertex aps5_stage_out " + std::string(MslEntryPoint) + "(");
    const auto& inputs = entry.inputs;
    const auto& outputs = entry.outputs;
    const auto& parameters = entry.parameters;
    const auto& outputMembers = entry.outputMembers;
    const auto vertexBytes = entry.vertexBytes;
    const auto rects = std::clamp(RectOutputBudget / std::max(vertexBytes * 4u, 1u), 1u, MaxRectsPerGroup);
    const auto threads = Msl::U32(rects * 3u);

    std::string text(vertex.substr(0, inputs.begin != std::string_view::npos ? inputs.begin : outputs.begin));
    text += Msl::VertexFetchText();
    text += "struct aps5_rect_draw { device const uchar* indices; uint index_bytes; uint rects; uint first; int base_vertex; uint base_instance; uint reserved; };\n";
    std::vector<std::uint32_t> locations;
    std::vector<std::pair<const Member*, std::uint32_t>> inputMembers;
    const auto inputList = inputs.begin != std::string_view::npos ? Members(inputs.body) : std::vector<Member>{};
    for (const auto& member : inputList) {
        if (!member.attribute.starts_with("attribute(")) Unsupported("vertex input " + member.name + " has attribute " + member.attribute);
        const auto location = Location(member);
        inputMembers.emplace_back(&member, location);
        if (std::ranges::find(locations, location) != locations.end()) continue;
        locations.push_back(location);
        const auto id = std::to_string(MetalAbi::LocalFetchFormatConstant + location);
        if (vertex.find("aps5_fc" + id + " ") != std::string_view::npos) Unsupported("the source already declares fetch format constant " + id);
        text += "constant uint aps5_fc" + id + " [[function_constant(" + id + ")]];\n";
        text += "constant uint aps5_c" + id + " = is_function_constant_defined(aps5_fc" + id + ") ? aps5_fc" + id + " : " + std::to_string(DefaultFetchFormat) + "u;\n";
    }
    if (inputs.begin != std::string_view::npos) {
        text += "struct aps5_stage_in {\n";
        for (const auto& member : inputList) text += "    " + member.type + " " + member.name + ";\n";
        text += "};\n";
    }
    text += std::string(vertex.substr(outputs.begin, outputs.end - outputs.begin));
    text += OutputStruct("aps5_rect_out", outputMembers);
    text += "struct aps5_rect_primitive {\n    bool culled [[primitive_culled]];\n};\n";
    text += "using aps5_rect_mesh = metal::mesh<aps5_rect_out, aps5_rect_primitive, " + std::to_string(rects * 4u) + ", " + std::to_string(rects * 2u) + ", metal::topology::triangle>;\n";

    std::vector<std::string> helper;
    std::vector<std::string> calls;
    std::vector<std::string> arguments{"aps5_rect_mesh aps5_mesh", "uint aps5_index [[thread_index_in_threadgroup]]", "uint3 aps5_group [[threadgroup_position_in_grid]]", "device const aps5_rect_draw& aps5_rect_draw_params [[buffer(" + std::to_string(MetalAbi::GeometrySlot) + ")]]", "constant aps5_fetch* aps5_fetch_table [[buffer(" + std::to_string(MetalAbi::VertexFetchSlot) + ")]]"};
    for (const auto& parameter : parameters) {
        helper.push_back(parameter.declaration);
        if (parameter.attribute == "stage_in" || parameter.attribute == "vertex_id" || parameter.attribute == "instance_id") {
            calls.push_back(parameter.name);
            continue;
        }
        if (parameter.attribute == "base_instance") {
            calls.push_back("aps5_rect_draw_params.base_instance");
            continue;
        }
        const bool buffer = parameter.attribute.starts_with("buffer(");
        if (!buffer && !parameter.attribute.starts_with("texture(") && !parameter.attribute.starts_with("sampler(")) Unsupported("parameter " + parameter.name + " has attribute " + parameter.attribute);
        if (buffer && (parameter.attribute == "buffer(" + std::to_string(MetalAbi::GeometrySlot) + ")" || parameter.attribute == "buffer(" + std::to_string(MetalAbi::VertexFetchSlot) + ")")) Unsupported("parameter " + parameter.name + " uses a slot the rect-list draw needs");
        calls.push_back(parameter.name);
        arguments.push_back(parameter.declaration + " [[" + parameter.attribute + "]]");
    }
    text += "static inline aps5_stage_out aps5_rect_vertex(" + Join(helper, ", ") + ")\n{\n";
    text += std::string(entry.body);
    text += "}\n";

    text += "[[mesh]] void " + std::string(MslEntryPoint) + "(\n    " + Join(arguments, ",\n    ") + ")\n{\n";
    text += "    threadgroup aps5_rect_out aps5_rect_v[" + threads + "];\n";
    text += "    const uint aps5_first_rect = aps5_group.x * " + Msl::U32(rects) + ";\n";
    text += "    const uint aps5_count = min(" + Msl::U32(rects) + ", aps5_rect_draw_params.rects - min(aps5_first_rect, aps5_rect_draw_params.rects));\n";
    text += "    if (aps5_index < aps5_count * 3u) {\n";
    text += "        const uint aps5_at = aps5_rect_draw_params.first + aps5_first_rect * 3u + aps5_index;\n";
    text += "        const uint aps5_index_value = aps5_rect_draw_params.index_bytes == 0u ? aps5_at : aps5_rect_draw_params.index_bytes == 2u ? uint(reinterpret_cast<device const ushort*>(aps5_rect_draw_params.indices)[aps5_at]) : reinterpret_cast<device const uint*>(aps5_rect_draw_params.indices)[aps5_at];\n";
    text += "        const uint aps5_vertex_id = uint(int(aps5_index_value) + aps5_rect_draw_params.base_vertex);\n";
    text += "        const uint aps5_instance_id = aps5_rect_draw_params.base_instance + aps5_group.y;\n";
    for (const auto location : locations) {
        const auto id = std::to_string(MetalAbi::LocalFetchFormatConstant + location);
        text += "        const uint4 aps5_a" + std::to_string(location) + " = aps5_fetch_attribute(aps5_fetch_table[" + std::to_string(location) + "], aps5_vertex_id, aps5_instance_id, aps5_c" + id + ");\n";
    }
    if (inputs.begin != std::string_view::npos) {
        text += "        aps5_stage_in aps5_in;\n";
        for (const auto& [member, location] : inputMembers) text += "        aps5_in." + member->name + " = " + Fetched(*member, location) + ";\n";
    }
    text += "        const aps5_stage_out aps5_own = aps5_rect_vertex(" + Join(calls, ", ") + ");\n";
    text += "        aps5_rect_out aps5_emit;\n";
    text += CopyOutputs(outputMembers, "aps5_emit", "aps5_own", "        ");
    text += "        aps5_rect_v[aps5_index] = aps5_emit;\n";
    text += "        aps5_mesh.set_vertex(aps5_index, aps5_emit);\n";
    text += "    }\n";
    text += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    text += "    if (aps5_index == 0u) aps5_mesh.set_primitive_count(aps5_count * 2u);\n";
    text += "    if (aps5_index >= aps5_count) return;\n";
    text += "    const uint aps5_b = aps5_index * 3u;\n";
    text += "    const float4 aps5_p0 = aps5_rect_v[aps5_b].position;\n";
    text += "    const float4 aps5_p1 = aps5_rect_v[aps5_b + 1u].position;\n";
    text += "    const float4 aps5_p2 = aps5_rect_v[aps5_b + 2u].position;\n";
    text += "    const bool2 aps5_e0 = aps5_p0.xy == aps5_p1.xy;\n";
    text += "    const bool2 aps5_e1 = aps5_p1.xy == aps5_p2.xy;\n";
    text += "    const bool2 aps5_e2 = aps5_p2.xy == aps5_p0.xy;\n";
    text += "    const bool aps5_corner0 = (aps5_e0.x && aps5_e2.y) || (aps5_e0.y && aps5_e2.x);\n";
    text += "    const bool aps5_corner1 = (aps5_e1.x && aps5_e0.y) || (aps5_e1.y && aps5_e0.x);\n";
    text += "    const bool aps5_corner2 = (aps5_e2.x && aps5_e1.y) || (aps5_e2.y && aps5_e1.x);\n";
    text += "    const float3 aps5_w = float3(aps5_corner0 ? -1.0f : 1.0f, aps5_corner1 ? -1.0f : 1.0f, aps5_corner2 ? -1.0f : 1.0f);\n";
    text += "    const float aps5_cross = (aps5_p1.x - aps5_p0.x) * (aps5_p2.y - aps5_p0.y) - (aps5_p1.y - aps5_p0.y) * (aps5_p2.x - aps5_p0.x);\n";
    text += "    aps5_rect_out aps5_q;\n";
    for (const auto& member : outputMembers) {
        if (Arrayed(member)) {
            text += "    for (uint aps5_k = 0u; aps5_k < " + Msl::U32(member.count) + "; ++aps5_k) aps5_q." + member.name + "[aps5_k] = " + Interpolated(member, "[aps5_k]") + ";\n";
        } else if (Scalar(member.type) != "float") {
            text += "    aps5_q." + member.name + " = aps5_rect_v[aps5_b]." + member.name + ";\n";
        } else {
            text += "    aps5_q." + member.name + " = " + Interpolated(member, "") + ";\n";
        }
    }
    text += "    const bool aps5_valid = uint(aps5_corner0) + uint(aps5_corner1) + uint(aps5_corner2) == 1u && all(isfinite(aps5_p0)) && all(isfinite(aps5_p1)) && all(isfinite(aps5_p2)) && all(isfinite(aps5_q.position)) && aps5_p0.w > 0.0f && aps5_p0.w == aps5_p1.w && aps5_p0.w == aps5_p2.w && aps5_cross != 0.0f;\n";
    text += "    const uint aps5_v3 = aps5_count * 3u + aps5_index;\n";
    text += "    aps5_mesh.set_vertex(aps5_v3, aps5_q);\n";
    text += "    const uint aps5_corner = aps5_corner1 ? 1u : aps5_corner2 ? 2u : 0u;\n";
    text += "    const uint aps5_a = aps5_b + (aps5_corner + 1u) % 3u;\n";
    text += "    const uint aps5_z = aps5_b + (aps5_corner + 2u) % 3u;\n";
    text += "    const bool aps5_turn = aps5_corner == 1u;\n";
    text += "    const uint aps5_i = aps5_index * 6u;\n";
    text += "    aps5_mesh.set_index(aps5_i, uchar(aps5_b));\n";
    text += "    aps5_mesh.set_index(aps5_i + 1u, uchar(aps5_b + 1u));\n";
    text += "    aps5_mesh.set_index(aps5_i + 2u, uchar(aps5_b + 2u));\n";
    text += "    aps5_mesh.set_index(aps5_i + 3u, uchar(aps5_turn ? aps5_z : aps5_a));\n";
    text += "    aps5_mesh.set_index(aps5_i + 4u, uchar(aps5_turn ? aps5_a : aps5_v3));\n";
    text += "    aps5_mesh.set_index(aps5_i + 5u, uchar(aps5_turn ? aps5_v3 : aps5_z));\n";
    text += "    aps5_rect_primitive aps5_r = {};\n";
    text += "    aps5_r.culled = !aps5_valid;\n";
    text += "    aps5_mesh.set_primitive(aps5_index * 2u, aps5_r);\n";
    text += "    aps5_mesh.set_primitive(aps5_index * 2u + 1u, aps5_r);\n";
    text += "}\n";
    return {std::move(text), rects};
}

}

MslRectList BuildMslRectList(std::string_view vertex) {
    try {
        return BuildRectList(vertex);
    } catch (const MslUnsupported& error) {
        throw MslUnsupported("MSL backend: a rect-list mesh function cannot wrap this vertex function: " + std::string(error.what()));
    }
}

}
