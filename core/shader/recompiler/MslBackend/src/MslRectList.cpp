#include "MslState.hpp"
#include "MetalAbi.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <charconv>
#include <string>
#include <string_view>
#include <vector>

namespace ShaderRecompiler {

namespace {

constexpr std::uint32_t MaxRectsPerGroup = 64u;
constexpr std::uint32_t RectOutputBudget = 16384u;
constexpr std::uint32_t DefaultFetchFormat = 31u;

struct Member {
    std::string type;
    std::string name;
    std::string attribute;
    std::uint32_t count = 1u;
};

struct Parameter {
    std::string declaration;
    std::string name;
    std::string attribute;
};

struct Span {
    std::size_t begin = std::string_view::npos;
    std::size_t end = std::string_view::npos;
    std::string_view body;
};

[[noreturn]] void Unsupported(std::string_view reason) {
    throw MslUnsupported("MSL backend: a rect-list mesh function cannot wrap this vertex function: " + std::string(reason));
}

std::string_view Trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\n')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n' || text.back() == ',')) text.remove_suffix(1);
    return text;
}

std::vector<std::string_view> Lines(std::string_view text) {
    std::vector<std::string_view> lines;
    while (!text.empty()) {
        const auto end = text.find('\n');
        const auto line = Trim(text.substr(0, end));
        if (!line.empty()) lines.push_back(line);
        if (end == std::string_view::npos) break;
        text.remove_prefix(end + 1u);
    }
    return lines;
}

Span Struct(std::string_view source, std::string_view name) {
    Span span;
    const auto head = "struct " + std::string(name) + " {\n";
    span.begin = source.find(head);
    if (span.begin == std::string_view::npos) return span;
    const auto close = source.find("\n};\n", span.begin);
    if (close == std::string_view::npos) Unsupported("struct " + std::string(name) + " is not closed");
    span.end = close + 4u;
    span.body = source.substr(span.begin + head.size(), close - span.begin - head.size());
    return span;
}

void Attributed(std::string_view text, std::string_view& before, std::string& attribute, std::string_view& after) {
    const auto open = text.find("[[");
    const auto close = text.find("]]", open == std::string_view::npos ? 0u : open);
    if (open == std::string_view::npos || close == std::string_view::npos) Unsupported("\"" + std::string(text) + "\" has no attribute");
    before = Trim(text.substr(0, open));
    attribute = std::string(text.substr(open + 2u, close - open - 2u));
    after = Trim(text.substr(close + 2u));
}

std::string LastToken(std::string_view declaration) {
    const auto at = declaration.find_last_of(" *&");
    return std::string(at == std::string_view::npos ? declaration : declaration.substr(at + 1u));
}

std::vector<Member> Members(std::string_view body) {
    std::vector<Member> members;
    for (auto line : Lines(body)) {
        if (!line.ends_with(';')) Unsupported("struct member \"" + std::string(line) + "\" is not one declaration");
        line.remove_suffix(1);
        std::string_view before;
        std::string_view after;
        Member member;
        Attributed(line, before, member.attribute, after);
        const auto space = before.find_last_of(' ');
        if (space == std::string_view::npos) Unsupported("struct member \"" + std::string(line) + "\" has no type");
        member.type = std::string(Trim(before.substr(0, space)));
        member.name = std::string(before.substr(space + 1u));
        if (!after.empty()) {
            if (after.size() < 3u || after.front() != '[' || after.back() != ']') Unsupported("struct member \"" + std::string(line) + "\" has an unexpected suffix");
            const auto digits = after.substr(1u, after.size() - 2u);
            if (std::from_chars(digits.data(), digits.data() + digits.size(), member.count).ec != std::errc{}) Unsupported("struct member \"" + std::string(line) + "\" has an unreadable array size");
        }
        members.push_back(std::move(member));
    }
    return members;
}

std::uint32_t Components(std::string_view type) {
    const auto last = type.empty() ? '\0' : type.back();
    return last >= '2' && last <= '4' ? static_cast<std::uint32_t>(last - '0') : 1u;
}

std::string Scalar(std::string_view type) {
    return std::string(Components(type) == 1u ? type : type.substr(0, type.size() - 1u));
}

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

std::uint32_t MemberBytes(const Member& member) {
    const auto scalar = Scalar(member.type);
    if (scalar != "float" && scalar != "uint" && scalar != "int") Unsupported("vertex output " + member.name + " has type " + member.type);
    const auto components = Components(member.type);
    return (components == 3u ? 4u : components) * 4u * member.count;
}

std::string Interpolated(const Member& member, std::string_view suffix) {
    const auto at = [&](std::string_view corner) { return "aps5_rect_v[aps5_b + " + std::string(corner) + "]." + member.name + std::string(suffix); };
    return at("0u") + " * aps5_w.x + (" + at("1u") + " * aps5_w.y + " + at("2u") + " * aps5_w.z)";
}

}

MslRectList BuildMslRectList(std::string_view vertex) {
    const auto signature = "vertex aps5_stage_out " + std::string(MslEntryPoint) + "(";
    const auto at = vertex.find(signature);
    if (at == std::string_view::npos) Unsupported("the source has no vertex entry point");
    const auto open = vertex.find(")\n{\n", at);
    if (open == std::string_view::npos || !vertex.ends_with("}\n")) Unsupported("the vertex entry point is not one function at the end of the source");
    const auto inputs = Struct(vertex, "aps5_stage_in");
    const auto outputs = Struct(vertex, "aps5_stage_out");
    if (outputs.begin == std::string_view::npos) Unsupported("the source has no stage output struct");
    if (inputs.begin != std::string_view::npos && (inputs.begin > outputs.begin || !Trim(vertex.substr(inputs.end, outputs.begin - inputs.end)).empty())) Unsupported("the stage input struct does not precede the output struct");
    if (!Trim(vertex.substr(outputs.end, at - outputs.end)).empty()) Unsupported("the stage output struct does not precede the entry point");
    if (vertex.find("struct aps5_fetch {") != std::string_view::npos || vertex.find("aps5_rect_") != std::string_view::npos) Unsupported("the source already declares rect-list names");

    std::vector<Parameter> parameters;
    for (const auto line : Lines(vertex.substr(at + signature.size(), open - at - signature.size()))) {
        std::string_view before;
        std::string_view after;
        Parameter parameter;
        Attributed(line, before, parameter.attribute, after);
        if (!after.empty()) Unsupported("parameter \"" + std::string(line) + "\" has an unexpected suffix");
        parameter.declaration = std::string(before);
        parameter.name = LastToken(before);
        parameters.push_back(std::move(parameter));
    }

    std::vector<Member> outputMembers;
    std::uint32_t vertexBytes = 0;
    for (auto& member : Members(outputs.body)) {
        if (member.attribute == "render_target_array_index" || member.attribute == "viewport_array_index") Unsupported("the vertex function exports a layer or viewport, which mesh vertices cannot carry");
        if (member.attribute == "point_size") continue;
        if (member.attribute != "position" && member.attribute != "clip_distance" && !member.attribute.starts_with("user(")) Unsupported("vertex output " + member.name + " has attribute " + member.attribute);
        vertexBytes += MemberBytes(member);
        outputMembers.push_back(std::move(member));
    }
    vertexBytes = (vertexBytes + 15u) & ~15u;
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
    text += "struct aps5_rect_out {\n";
    for (const auto& member : outputMembers) text += "    " + member.type + " " + member.name + " [[" + member.attribute + "]]" + (member.count > 1u || member.attribute == "clip_distance" ? " [" + std::to_string(member.count) + "]" : std::string()) + ";\n";
    text += "};\nstruct aps5_rect_primitive {\n    bool culled [[primitive_culled]];\n};\n";
    text += "using aps5_rect_mesh = metal::mesh<aps5_rect_out, aps5_rect_primitive, " + std::to_string(rects * 4u) + ", " + std::to_string(rects * 2u) + ", metal::topology::triangle>;\n";

    std::vector<std::string> helper;
    std::vector<std::string> calls;
    std::vector<std::string> entry{"aps5_rect_mesh aps5_mesh", "uint aps5_index [[thread_index_in_threadgroup]]", "uint3 aps5_group [[threadgroup_position_in_grid]]", "device const aps5_rect_draw& aps5_rect_draw_params [[buffer(" + std::to_string(MetalAbi::GeometrySlot) + ")]]", "constant aps5_fetch* aps5_fetch_table [[buffer(" + std::to_string(MetalAbi::VertexFetchSlot) + ")]]"};
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
        entry.push_back(parameter.declaration + " [[" + parameter.attribute + "]]");
    }
    const auto join = [](const std::vector<std::string>& items, std::string_view separator) {
        std::string joined;
        for (std::size_t index = 0; index < items.size(); ++index) joined += (index == 0u ? "" : std::string(separator)) + items[index];
        return joined;
    };
    text += "static inline aps5_stage_out aps5_rect_vertex(" + join(helper, ", ") + ")\n{\n";
    text += std::string(vertex.substr(open + 4u, vertex.size() - 2u - open - 4u));
    text += "}\n";

    text += "[[mesh]] void " + std::string(MslEntryPoint) + "(\n    " + join(entry, ",\n    ") + ")\n{\n";
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
    text += "        const aps5_stage_out aps5_own = aps5_rect_vertex(" + join(calls, ", ") + ");\n";
    text += "        aps5_rect_out aps5_emit;\n";
    for (const auto& member : outputMembers) {
        if (member.count > 1u || member.attribute == "clip_distance") {
            text += "        for (uint aps5_k = 0u; aps5_k < " + Msl::U32(member.count) + "; ++aps5_k) aps5_emit." + member.name + "[aps5_k] = aps5_own." + member.name + "[aps5_k];\n";
        } else {
            text += "        aps5_emit." + member.name + " = aps5_own." + member.name + ";\n";
        }
    }
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
        if (member.count > 1u || member.attribute == "clip_distance") {
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
