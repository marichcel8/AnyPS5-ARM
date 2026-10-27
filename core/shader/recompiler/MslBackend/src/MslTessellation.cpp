#include "MslState.hpp"
#include "MslWrap.hpp"
#include "MetalAbi.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace ShaderRecompiler {

namespace {

using namespace MslWrap;

constexpr std::uint32_t MaxTrianglesPerGroup = 126u;
constexpr std::uint32_t MeshOutputBudget = 16384u;
constexpr std::uint32_t MaxStrips = 129u;
constexpr std::uint32_t MaxPatchTriangles = 8448u;
constexpr std::uint32_t MaxGroupsPerPatch = 1024u;
constexpr std::uint32_t PayloadBytes = (64u + 2u * (MaxStrips + 1u) + 15u) & ~15u;

constexpr std::string_view CommonText = R"(struct aps5_ts_shape {
    float edge[4];
    uint edge_n[4];
    float inside[2];
    uint inside_n[2];
};
struct aps5_ts_strip {
    uint outer_ring;
    uint outer_side;
    uint inner_ring;
    uint inner_side;
    bool reversed;
    uint a;
    uint b;
};
static inline float aps5_ts_level(float f, bool edge) {
    const uint bits = as_type<uint>(f);
    if (bits == 0x7f800000u) f = 64.0f;
    else if (bits >= 0x7f800000u || bits == 0u) return edge ? 0.0f : (aps5_ts_mode == 3u ? 2.0f : 1.0f);
    if (aps5_ts_mode == 2u) return min(max(f, 1.0f), 63.0f);
    if (aps5_ts_mode == 3u) return min(max(f, 2.0f), 64.0f);
    const float n = ceil(min(max(f, 1.0f), 64.0f));
    return aps5_ts_mode == 1u ? exp2(ceil(log2(n))) : n;
}
static inline float aps5_ts_base(float f) {
    return aps5_ts_mode == 2u ? 2.0f * floor((f - 1.0f) * 0.5f) + 1.0f : 2.0f * floor(f * 0.5f);
}
static inline uint aps5_ts_segments(float f) {
    if (aps5_ts_mode < 2u) return uint(f);
    const float base = aps5_ts_base(f);
    return uint(base) + (f > base ? 2u : 0u);
}
static inline float aps5_ts_half(float f, uint n, uint i) {
    if (aps5_ts_mode < 2u) return float(i) / float(n);
    const float base = aps5_ts_base(f);
    if (!(f > base)) return float(i) / float(n);
    const uint whole = aps5_ts_mode == 2u ? (uint(base) - 1u) / 2u : uint(base) / 2u;
    return i <= whole ? float(i) / f : (float(whole) + (f - base) * 0.5f) / f;
}
static inline float aps5_ts_param(float f, uint n, uint i) {
    return i * 2u <= n ? aps5_ts_half(f, n, i) : 1.0f - aps5_ts_half(f, n, n - i);
}
static inline uint2 aps5_ts_merge(uint t, uint a, uint b) {
    uint lo = t > b ? t - b : 0u;
    uint hi = min(t, a);
    while (lo < hi) {
        const uint mid = (lo + hi) / 2u;
        if ((mid + 1u) * b <= (t - mid) * a) lo = mid + 1u;
        else hi = mid;
    }
    return uint2(lo, t - lo);
}
)";

constexpr std::string_view TriangleText = R"(static inline uint aps5_ts_row(thread const aps5_ts_shape& s, uint ring, uint side) {
    return ring == 0u ? s.edge_n[side == 0u ? 2u : side - 1u] : s.inside_n[0] - 2u * ring;
}
static inline float3 aps5_ts_point(thread const aps5_ts_shape& s, uint ring, uint side, uint index) {
    const uint next = side == 2u ? 0u : side + 1u;
    float3 r = float3(0.0f);
    if (ring == 0u) {
        const uint e = side == 0u ? 2u : side - 1u;
        const uint n = s.edge_n[e];
        r[side] = aps5_ts_param(s.edge[e], n, n - index);
        r[next] = aps5_ts_param(s.edge[e], n, index);
        return r;
    }
    const uint n = s.inside_n[0];
    const uint m = n - 2u * ring;
    if (m == 0u) return float3(1.0f / 3.0f);
    const float pk = aps5_ts_param(s.inside[0], n, ring);
    const float t = index == 0u ? 0.0f : index >= m ? 1.0f : (aps5_ts_param(s.inside[0], n, ring + index) - pk) / (aps5_ts_param(s.inside[0], n, n - ring) - pk);
    const float d = pk * (2.0f / 3.0f);
    float3 a = float3(d);
    float3 b = float3(d);
    a[side] = 1.0f - 2.0f * d;
    b[next] = 1.0f - 2.0f * d;
    return a * (1.0f - t) + b * t;
}
static inline aps5_ts_strip aps5_ts_decode(thread const aps5_ts_shape& s, uint rings, uint strip) {
    aps5_ts_strip r;
    r.reversed = false;
    if (strip < rings * 3u) {
        r.outer_ring = strip / 3u;
        r.outer_side = strip % 3u;
        r.inner_ring = r.outer_ring + 1u;
        r.inner_side = r.outer_side;
        r.a = aps5_ts_row(s, r.outer_ring, r.outer_side);
        r.b = aps5_ts_row(s, r.inner_ring, r.inner_side);
        return r;
    }
    r.outer_ring = rings;
    r.inner_ring = rings;
    r.outer_side = 0u;
    r.inner_side = 2u;
    r.a = aps5_ts_row(s, rings, 0u);
    r.b = 0u;
    return r;
}
)";

constexpr std::string_view QuadText = R"(static inline uint aps5_ts_row(thread const aps5_ts_shape& s, uint ring, uint side) {
    return ring == 0u ? s.edge_n[side == 3u ? 0u : side + 1u] : s.inside_n[side & 1u] - 2u * ring;
}
static inline float3 aps5_ts_point(thread const aps5_ts_shape& s, uint ring, uint side, uint index) {
    if (ring == 0u) {
        const uint e = side == 3u ? 0u : side + 1u;
        const uint n = s.edge_n[e];
        const float f = aps5_ts_param(s.edge[e], n, index);
        const float g = aps5_ts_param(s.edge[e], n, n - index);
        return side == 0u ? float3(f, 0.0f, 0.0f) : side == 1u ? float3(1.0f, f, 0.0f) : side == 2u ? float3(g, 1.0f, 0.0f) : float3(0.0f, g, 0.0f);
    }
    const uint nu = s.inside_n[0];
    const uint nv = s.inside_n[1];
    const uint iu = side == 0u ? ring + index : side == 1u ? nu - ring : side == 2u ? nu - ring - index : ring;
    const uint iv = side == 0u ? ring : side == 1u ? ring + index : side == 2u ? nv - ring : nv - ring - index;
    return float3(aps5_ts_param(s.inside[0], nu, iu), aps5_ts_param(s.inside[1], nv, iv), 0.0f);
}
static inline aps5_ts_strip aps5_ts_decode(thread const aps5_ts_shape& s, uint rings, uint strip) {
    aps5_ts_strip r;
    r.reversed = false;
    if (strip < rings * 4u) {
        r.outer_ring = strip / 4u;
        r.outer_side = strip % 4u;
        r.inner_ring = r.outer_ring + 1u;
        r.inner_side = r.outer_side;
        r.a = aps5_ts_row(s, r.outer_ring, r.outer_side);
        r.b = aps5_ts_row(s, r.inner_ring, r.inner_side);
        return r;
    }
    const bool narrow = s.inside_n[1] <= s.inside_n[0];
    r.outer_ring = rings;
    r.inner_ring = rings;
    r.outer_side = narrow ? 0u : 1u;
    r.inner_side = narrow ? 2u : 3u;
    r.reversed = true;
    r.a = aps5_ts_row(s, rings, r.outer_side);
    r.b = aps5_ts_row(s, rings, r.inner_side);
    return r;
}
)";

constexpr std::string_view ObjectText = R"(    const uint aps5_ts_patch = aps5_ts_group.y * aps5_geo.patch_count + aps5_ts_group.x;
    device const half* aps5_ts_factors = aps5_geo.factors + aps5_ts_patch * aps5_ts_halves;
    aps5_ts_shape aps5_ts_s;
    uint aps5_ts_most = 0u;
    for (uint aps5_ts_e = 0u; aps5_ts_e < 4u; ++aps5_ts_e) {
        aps5_ts_s.edge[aps5_ts_e] = aps5_ts_e < aps5_ts_edges ? aps5_ts_level(float(aps5_ts_factors[aps5_ts_e]), true) : 0.0f;
        if (aps5_ts_e < aps5_ts_edges && aps5_ts_s.edge[aps5_ts_e] == 0.0f) {
            aps5_ts_grid.set_threadgroups_per_grid(uint3(0u, 1u, 1u));
            return;
        }
        aps5_ts_s.edge_n[aps5_ts_e] = aps5_ts_e < aps5_ts_edges ? aps5_ts_segments(aps5_ts_s.edge[aps5_ts_e]) : 0u;
        aps5_ts_most = max(aps5_ts_most, aps5_ts_s.edge_n[aps5_ts_e]);
    }
    for (uint aps5_ts_i = 0u; aps5_ts_i < 2u; ++aps5_ts_i) {
        float aps5_ts_f = aps5_ts_level(float(aps5_ts_factors[aps5_ts_edges + min(aps5_ts_i, aps5_ts_insides - 1u)]), false);
        if (aps5_ts_segments(aps5_ts_f) == 1u && aps5_ts_most > 1u) aps5_ts_f = aps5_ts_level(1.0009765625f, false);
        aps5_ts_s.inside[aps5_ts_i] = aps5_ts_f;
        aps5_ts_s.inside_n[aps5_ts_i] = aps5_ts_segments(aps5_ts_f);
    }
    const uint aps5_ts_narrow = min(aps5_ts_s.inside_n[0], aps5_ts_s.inside_n[1]);
    const uint aps5_ts_rings = aps5_ts_narrow / 2u;
    const uint aps5_ts_strips = min(aps5_ts_rings * aps5_ts_edges + (aps5_ts_narrow & 1u), aps5_ts_max_strips);
    uint aps5_ts_total = 0u;
    for (uint aps5_ts_strip_index = 0u; aps5_ts_strip_index < aps5_ts_strips; ++aps5_ts_strip_index) {
        aps5_ts_payload_out.first[aps5_ts_strip_index] = ushort(aps5_ts_total);
        const aps5_ts_strip aps5_ts_r = aps5_ts_decode(aps5_ts_s, aps5_ts_rings, aps5_ts_strip_index);
        aps5_ts_total += (aps5_ts_r.a + aps5_ts_r.b + aps5_ts_chunk - 1u) / aps5_ts_chunk;
    }
    aps5_ts_payload_out.first[aps5_ts_strips] = ushort(aps5_ts_total);
    aps5_ts_payload_out.shape = aps5_ts_s;
    aps5_ts_payload_out.patch_id = aps5_ts_group.x;
    aps5_ts_payload_out.instance = aps5_geo.base_instance + aps5_ts_group.y;
    aps5_ts_payload_out.rings = aps5_ts_rings;
    aps5_ts_payload_out.strips = aps5_ts_strips;
    aps5_ts_grid.set_threadgroups_per_grid(uint3(aps5_ts_total, 1u, 1u));
}
)";

constexpr std::string_view MeshText = R"(    const aps5_ts_shape aps5_ts_s = aps5_ts_in.shape;
    const uint aps5_ts_g = aps5_ts_group.x;
    uint aps5_ts_lo = 0u;
    uint aps5_ts_hi = aps5_ts_in.strips;
    while (aps5_ts_hi - aps5_ts_lo > 1u) {
        const uint aps5_ts_mid = (aps5_ts_lo + aps5_ts_hi) / 2u;
        if (uint(aps5_ts_in.first[aps5_ts_mid]) <= aps5_ts_g) aps5_ts_lo = aps5_ts_mid;
        else aps5_ts_hi = aps5_ts_mid;
    }
    const aps5_ts_strip aps5_ts_r = aps5_ts_decode(aps5_ts_s, aps5_ts_in.rings, aps5_ts_lo);
    const uint aps5_ts_t0 = (aps5_ts_g - uint(aps5_ts_in.first[aps5_ts_lo])) * aps5_ts_chunk;
    const uint aps5_ts_t1 = min(aps5_ts_t0 + aps5_ts_chunk, aps5_ts_r.a + aps5_ts_r.b);
    const uint2 aps5_ts_m0 = aps5_ts_merge(aps5_ts_t0, aps5_ts_r.a, aps5_ts_r.b);
    const uint2 aps5_ts_m1 = aps5_ts_merge(aps5_ts_t1, aps5_ts_r.a, aps5_ts_r.b);
    const uint aps5_ts_outer = aps5_ts_m1.x - aps5_ts_m0.x + 1u;
    const uint aps5_ts_vertices = aps5_ts_outer + aps5_ts_m1.y - aps5_ts_m0.y + 1u;
    if (aps5_ts_index == 0u) aps5_mesh.set_primitive_count(aps5_ts_t1 - aps5_ts_t0);
    if (aps5_ts_index < aps5_ts_vertices) {
        const uint aps5_ts_inner = aps5_ts_m0.y + aps5_ts_index - aps5_ts_outer;
        const float3 aps5_ts_c = aps5_ts_index < aps5_ts_outer ? aps5_ts_point(aps5_ts_s, aps5_ts_r.outer_ring, aps5_ts_r.outer_side, aps5_ts_m0.x + aps5_ts_index) : aps5_ts_point(aps5_ts_s, aps5_ts_r.inner_ring, aps5_ts_r.inner_side, aps5_ts_r.reversed ? aps5_ts_r.b - aps5_ts_inner : aps5_ts_inner);
)";

constexpr std::string_view TriangleListText = R"(    if (aps5_ts_index < aps5_ts_t1 - aps5_ts_t0) {
        const uint2 aps5_ts_m = aps5_ts_merge(aps5_ts_t0 + aps5_ts_index, aps5_ts_r.a, aps5_ts_r.b);
        const bool aps5_ts_step = aps5_ts_m.x < aps5_ts_r.a && (aps5_ts_m.y >= aps5_ts_r.b || (aps5_ts_m.x + 1u) * aps5_ts_r.b <= (aps5_ts_m.y + 1u) * aps5_ts_r.a);
        const uint aps5_ts_o = aps5_ts_m.x - aps5_ts_m0.x;
        const uint aps5_ts_i = aps5_ts_outer + aps5_ts_m.y - aps5_ts_m0.y;
        const uint aps5_ts_second = aps5_ts_step ? aps5_ts_o + 1u : aps5_ts_i + 1u;
        const uint aps5_ts_at = aps5_ts_index * 3u;
        aps5_mesh.set_index(aps5_ts_at, uchar(aps5_ts_o));
        aps5_mesh.set_index(aps5_ts_at + 1u, uchar(aps5_ts_flip ? aps5_ts_i : aps5_ts_second));
        aps5_mesh.set_index(aps5_ts_at + 2u, uchar(aps5_ts_flip ? aps5_ts_second : aps5_ts_i));
        aps5_ts_primitive aps5_ts_p = {};
        aps5_ts_p.culled = false;
        aps5_mesh.set_primitive(aps5_ts_index, aps5_ts_p);
    }
}
)";

MslTessellationMesh BuildTessellation(std::string_view domain) {
    const auto layout = ParseMslGeometryLayout(domain);
    if (!layout || layout->kind != MslGeometryKind::Domain) Unsupported("the source is not a domain function");
    if ((layout->domain != 1u && layout->domain != 2u) || layout->partitioning > 3u || (layout->outputTopology != 2u && layout->outputTopology != 3u)) Unsupported("the domain, partitioning or output topology is not tessellated");
    const bool quad = layout->domain == 2u;
    const auto entry = ParseEntry(domain, "[[patch(" + std::string(quad ? "quad" : "triangle") + ", 0)]] vertex aps5_stage_out " + std::string(MslEntryPoint) + "(");
    if (entry.inputs.begin != std::string_view::npos) Unsupported("the domain function has a stage input struct");
    const auto room = std::min(MeshOutputBudget / std::max(entry.vertexBytes, 1u), MaxTrianglesPerGroup + 2u);
    const auto chunk = room > 2u ? room - 2u : 1u;
    const auto groups = MaxPatchTriangles / chunk + MaxStrips + 1u;
    if (groups > MaxGroupsPerPatch) Unsupported("the domain function outputs too many bytes per vertex for a mesh threadgroup");

    std::vector<std::string> helper;
    std::vector<std::string> calls;
    std::vector<std::string> arguments{"aps5_ts_mesh aps5_mesh", "const object_data aps5_ts_payload& aps5_ts_in [[payload]]", "uint aps5_ts_index [[thread_index_in_threadgroup]]", "uint3 aps5_ts_group [[threadgroup_position_in_grid]]"};
    std::string geometry;
    for (const auto& parameter : entry.parameters) {
        helper.push_back(parameter.declaration);
        if (parameter.attribute == "patch_id") {
            calls.push_back("aps5_ts_in.patch_id");
        } else if (parameter.attribute == "position_in_patch") {
            calls.push_back(quad ? "aps5_ts_c.xy" : "aps5_ts_c");
        } else if (parameter.attribute == "instance_id") {
            calls.push_back("aps5_ts_in.instance");
        } else if (parameter.attribute == "base_instance") {
            calls.push_back("aps5_geo.base_instance");
        } else if (parameter.attribute.starts_with("buffer(") || parameter.attribute.starts_with("texture(") || parameter.attribute.starts_with("sampler(")) {
            if (parameter.attribute == "buffer(" + std::to_string(MetalAbi::GeometrySlot) + ")") {
                if (parameter.name != "aps5_geo") Unsupported("the geometry parameters are not named aps5_geo");
                geometry = parameter.declaration + " [[" + parameter.attribute + "]]";
            }
            calls.push_back(parameter.name);
            arguments.push_back(parameter.declaration + " [[" + parameter.attribute + "]]");
        } else {
            Unsupported("parameter " + parameter.name + " has attribute " + parameter.attribute);
        }
    }
    if (geometry.empty()) Unsupported("the domain function does not read the geometry parameters");

    std::string text(domain.substr(0, entry.outputs.begin));
    text += "constant uint aps5_ts_mode = " + Msl::U32(layout->partitioning) + ";\n";
    text += "constant uint aps5_ts_edges = " + Msl::U32(quad ? 4u : 3u) + ";\n";
    text += "constant uint aps5_ts_insides = " + Msl::U32(quad ? 2u : 1u) + ";\n";
    text += "constant uint aps5_ts_halves = " + Msl::U32(MetalAbi::TessellationFactorHalves(layout->domain)) + ";\n";
    text += "constant uint aps5_ts_chunk = " + Msl::U32(chunk) + ";\n";
    text += "constant uint aps5_ts_max_strips = " + Msl::U32(MaxStrips) + ";\n";
    text += "constant bool aps5_ts_flip = " + std::string(layout->outputTopology == 3u ? "true" : "false") + ";\n";
    text += CommonText;
    text += quad ? QuadText : TriangleText;
    text += "struct aps5_ts_payload {\n    aps5_ts_shape shape;\n    uint patch_id;\n    uint instance;\n    uint rings;\n    uint strips;\n    ushort first[" + std::to_string(MaxStrips + 1u) + "];\n};\n";
    text += std::string(domain.substr(entry.outputs.begin, entry.outputs.end - entry.outputs.begin));
    text += OutputStruct("aps5_ts_vertex", entry.outputMembers);
    text += "struct aps5_ts_primitive {\n    bool culled [[primitive_culled]];\n};\n";
    text += "using aps5_ts_mesh = metal::mesh<aps5_ts_vertex, aps5_ts_primitive, " + std::to_string(chunk + 2u) + ", " + std::to_string(chunk) + ", metal::topology::triangle>;\n";
    text += "static inline aps5_stage_out aps5_ts_domain(" + Join(helper, ", ") + ")\n{\n";
    text += std::string(entry.body);
    text += "}\n";
    text += "[[object]] void " + std::string(MslTessellationObjectEntryPoint) + "(\n    object_data aps5_ts_payload& aps5_ts_payload_out [[payload]],\n    mesh_grid_properties aps5_ts_grid,\n    uint3 aps5_ts_group [[threadgroup_position_in_grid]],\n    " + geometry + ")\n{\n";
    text += ObjectText;
    text += "[[mesh]] void " + std::string(MslEntryPoint) + "(\n    " + Join(arguments, ",\n    ") + ")\n{\n";
    text += MeshText;
    text += "        const aps5_stage_out aps5_own = aps5_ts_domain(" + Join(calls, ", ") + ");\n";
    text += "        aps5_ts_vertex aps5_emit;\n";
    text += CopyOutputs(entry.outputMembers, "aps5_emit", "aps5_own", "        ");
    text += "        aps5_mesh.set_vertex(aps5_ts_index, aps5_emit);\n    }\n";
    text += TriangleListText;
    return {std::move(text), chunk + 2u, PayloadBytes, groups};
}

}

MslTessellationMesh BuildMslTessellationMesh(std::string_view domain) {
    try {
        return BuildTessellation(domain);
    } catch (const MslUnsupported& error) {
        throw MslUnsupported("MSL backend: a tessellation mesh function cannot wrap this domain function: " + std::string(error.what()));
    }
}

}
