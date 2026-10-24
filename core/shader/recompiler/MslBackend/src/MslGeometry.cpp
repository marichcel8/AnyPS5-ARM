#include "MslState.hpp"
#include "MetalAbi.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <limits>

namespace ShaderRecompiler::Msl {

namespace {

constexpr std::uint32_t NoDistance = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t MaxClipDistances = 8u;
constexpr std::uint32_t DefaultFetchFormat = 31u;

constexpr std::string_view GeometryText = R"(struct aps5_geometry {
    device uint* local;
    device uint* control;
    device uint* patch;
    device half* factors;
    device const uchar* indices;
    device const uint* mesh_arguments;
    uint local_words;
    uint control_words;
    uint patch_words;
    uint patch_count;
    uint instance_count;
    uint base_instance;
    uint first_index;
    int base_vertex;
    uint index_bytes;
    uint reserved[3];
};
)";

constexpr std::string_view FetchText = R"(struct aps5_fetch { device const uchar* data; uint stride; uint records; };
static inline uint4 aps5_fetch_attribute(constant aps5_fetch& f, uint vertex, uint instance, uint format) {
    const bool integer = (format >= 1u && format <= 6u) || (format >= 13u && format <= 18u) || (format >= 32u && format <= 39u) || format == 45u || format == 46u || format == 49u || format == 50u;
    uint4 r = uint4(0u, 0u, 0u, integer ? 1u : 0x3f800000u);
    const uint index = (f.records & 0x80000000u) != 0u ? instance : vertex;
    if (f.data == nullptr || index >= (f.records & 0x7fffffffu)) return r;
    device const uchar* p = f.data + ulong(index) * ulong(f.stride);
    device const ushort* h = reinterpret_cast<device const ushort*>(p);
    device const uint* w = reinterpret_cast<device const uint*>(p);
    if ((format >= 1u && format <= 3u) || format == 45u) {
        const uint n = format == 45u ? 1u : format + 1u;
        for (uint i = 0u; i < n; ++i) r[i] = uint(p[i]);
    } else if ((format >= 4u && format <= 6u) || format == 46u) {
        const uint n = format == 46u ? 1u : format - 2u;
        for (uint i = 0u; i < n; ++i) r[i] = as_type<uint>(int(as_type<char>(p[i])));
    } else if ((format >= 7u && format <= 9u) || format == 47u) {
        const uint n = format == 47u ? 1u : format - 5u;
        for (uint i = 0u; i < n; ++i) r[i] = as_type<uint>(float(p[i]) / 255.0f);
    } else if ((format >= 10u && format <= 12u) || format == 48u) {
        const uint n = format == 48u ? 1u : format - 8u;
        for (uint i = 0u; i < n; ++i) r[i] = as_type<uint>(max(float(as_type<char>(p[i])) / 127.0f, -1.0f));
    } else if ((format >= 13u && format <= 15u) || format == 49u) {
        const uint n = format == 49u ? 1u : format - 11u;
        for (uint i = 0u; i < n; ++i) r[i] = uint(h[i]);
    } else if ((format >= 16u && format <= 18u) || format == 50u) {
        const uint n = format == 50u ? 1u : format - 14u;
        for (uint i = 0u; i < n; ++i) r[i] = as_type<uint>(int(as_type<short>(h[i])));
    } else if ((format >= 19u && format <= 21u) || format == 51u) {
        const uint n = format == 51u ? 1u : format - 17u;
        for (uint i = 0u; i < n; ++i) r[i] = as_type<uint>(float(h[i]) / 65535.0f);
    } else if ((format >= 22u && format <= 24u) || format == 52u) {
        const uint n = format == 52u ? 1u : format - 20u;
        for (uint i = 0u; i < n; ++i) r[i] = as_type<uint>(max(float(as_type<short>(h[i])) / 32767.0f, -1.0f));
    } else if ((format >= 25u && format <= 27u) || format == 53u) {
        const uint n = format == 53u ? 1u : format - 23u;
        for (uint i = 0u; i < n; ++i) r[i] = as_type<uint>(float(as_type<half>(h[i])));
    } else if (format >= 28u && format <= 39u) {
        const uint n = (format - 28u) % 4u + 1u;
        for (uint i = 0u; i < n; ++i) r[i] = w[i];
    } else if (format == 40u) {
        const uint v = w[0];
        for (uint i = 0u; i < 3u; ++i) r[i] = as_type<uint>(max(float(as_type<int>(v << (22u - 10u * i)) >> 22) / 511.0f, -1.0f));
        r[3] = as_type<uint>(max(float(as_type<int>(v) >> 30), -1.0f));
    } else if (format == 41u) {
        const uint v = w[0];
        for (uint i = 0u; i < 3u; ++i) r[i] = as_type<uint>(float(extract_bits(v, 10u * i, 10u)) / 1023.0f);
        r[3] = as_type<uint>(float(v >> 30u) / 3.0f);
    } else if (format == 42u) {
        r = uint4(as_type<uint>(float(p[2]) / 255.0f), as_type<uint>(float(p[1]) / 255.0f), as_type<uint>(float(p[0]) / 255.0f), as_type<uint>(float(p[3]) / 255.0f));
    } else if (format == 54u) {
        const uint v = w[0];
        r.x = as_type<uint>(float(as_type<half>(ushort((v & 0x7ffu) << 4u))));
        r.y = as_type<uint>(float(as_type<half>(ushort(extract_bits(v, 11u, 11u) << 4u))));
        r.z = as_type<uint>(float(as_type<half>(ushort(extract_bits(v, 22u, 10u) << 5u))));
    } else if (format == 55u) {
        const uint v = w[0];
        const float scale = exp2(float(v >> 27u) - 24.0f);
        for (uint i = 0u; i < 3u; ++i) r[i] = as_type<uint>(float(extract_bits(v, 9u * i, 9u)) * scale);
    }
    return r;
}
)";

IrShaderStage StageOf(const State& state) {
    return state.program.Resources().stage;
}

const ShaderVertexInputInfo& VertexInfo(const State& state) {
    if (state.input.vertex == nullptr) Fail("vertex input info is missing for a geometry program");
    return *state.input.vertex;
}

const ShaderTessellationInputInfo& Tess(const State& state) {
    return VertexInfo(state).tess;
}

const ShaderMeshInputInfo& Mesh(const State& state) {
    return VertexInfo(state).mesh;
}

std::uint32_t Immediate(const IrValue& inst, std::size_t index) {
    const IrValue* value = inst.Argument(index)->Resolve();
    if (!value->HasImmediate()) Fail(inst, "needs an immediate operand");
    return value->ImmediateU32();
}

void Global(State& state, std::string_view name, std::string_view text) {
    state.stageGlobals.emplace(std::string(name), std::string(text));
}

std::string Geometry(State& state) {
    Global(state, "aps5_0geometry", GeometryText);
    state.stageArguments.emplace("aps5_geo", "constant aps5_geometry& aps5_geo [[buffer(" + std::to_string(MetalAbi::GeometrySlot) + ")]]");
    return "aps5_geo";
}

std::string Push(State& state, std::uint32_t dword) {
    state.features |= PushData;
    return "aps5_push[" + std::to_string(dword) + "]";
}

std::string Swizzle(std::uint32_t component) {
    static constexpr std::string_view Names[4] = {"x", "y", "z", "w"};
    if (component >= 4u) Fail("component index is out of range");
    return std::string(Names[component]);
}

std::uint32_t OuterFactors(const State& state) {
    return Tess(state).domain == 2u ? 4u : 3u;
}

std::uint32_t Stride(std::uint32_t bytes) {
    return std::max(bytes, 4u);
}

void RequireTessellation(const State& state) {
    const auto& tess = Tess(state);
    if ((tess.domain != 1u && tess.domain != 2u) || (tess.partitioning != 0u && tess.partitioning != 2u && tess.partitioning != 3u) || (tess.outputTopology != 2u && tess.outputTopology != 3u)) {
        throw MslUnsupported("MSL backend: the tessellation domain, partitioning or output topology has no Metal equivalent (isolines and point output are not emitted)");
    }
    if (tess.inputControlPoints == 0u || tess.outputControlPoints == 0u || tess.outputControlPoints > 32u || tess.inputControlPoints > 32u) Fail("tessellation control point counts are out of range");
}

std::string Offset(const IrValue& address, const std::string& value, std::uint32_t stride) {
    const IrValue* resolved = address.Resolve();
    if (resolved->HasImmediate()) return U32(resolved->ImmediateU32() % stride / 4u);
    return "((" + value + ") % " + U32(stride) + ") / 4u";
}

std::string Vertex(const IrValue& address, const std::string& value, std::uint32_t stride, std::uint32_t count) {
    const IrValue* resolved = address.Resolve();
    if (resolved->HasImmediate()) return U32(std::min(resolved->ImmediateU32() / stride, count - 1u));
    return "min((" + value + ") / " + U32(stride) + ", " + U32(count - 1u) + ")";
}

std::string AttributeWord(State& state, const IrValue& inst) {
    const auto kind = static_cast<TessellationAttribute>(Immediate(inst, 0));
    const auto& tess = Tess(state);
    const auto& metadata = state.program.Metadata();
    const auto geo = Geometry(state);
    const auto& raw = *inst.Argument(1);
    const auto address = Arg(state, inst, 1);
    const auto local = Stride(metadata.tessellationLocalStride);
    const auto control = Stride(metadata.tessellationControlStride);
    switch (kind) {
    case TessellationAttribute::LocalOutput:
        return geo + ".local[aps5_tid * " + geo + ".local_words + " + Offset(raw, address, local) + "]";
    case TessellationAttribute::InstanceOutput:
        return geo + ".local[aps5_tid * " + geo + ".local_words + " + geo + ".local_words - 1u]";
    case TessellationAttribute::ControlInput:
        return geo + ".local[(aps5_patch * " + U32(tess.inputControlPoints) + " + " + Vertex(raw, address, local, tess.inputControlPoints) + ") * " + geo + ".local_words + " + Offset(raw, address, local) + "]";
    case TessellationAttribute::InstanceInput:
        return geo + ".local[aps5_patch * " + U32(tess.inputControlPoints) + " * " + geo + ".local_words + " + geo + ".local_words - 1u]";
    case TessellationAttribute::ControlOutput:
        return geo + ".control[(aps5_patch * " + U32(tess.outputControlPoints) + " + aps5_tid.x) * " + geo + ".control_words + " + Offset(raw, address, control) + "]";
    case TessellationAttribute::EvaluationInput:
        return geo + ".control[(aps5_patch * " + U32(tess.outputControlPoints) + " + " + Vertex(raw, address, control, tess.outputControlPoints) + ") * " + geo + ".control_words + " + Offset(raw, address, control) + "]";
    case TessellationAttribute::PatchOutput:
    case TessellationAttribute::PatchInput: {
        const IrValue* resolved = raw.Resolve();
        return geo + ".patch[aps5_patch * " + geo + ".patch_words + " + (resolved->HasImmediate() ? U32(resolved->ImmediateU32() / 4u) : "(" + address + ") / 4u") + "]";
    }
    case TessellationAttribute::Factor: {
        const auto index = Immediate(inst, 1) / 4u;
        const auto outer = OuterFactors(state);
        if (index >= outer + (outer == 4u ? 2u : 1u)) Fail(inst, "writes a tessellation factor past the domain's factors");
        return geo + ".factors[aps5_patch * " + U32(MetalAbi::TessellationFactorHalves(tess.domain)) + " + " + U32(index) + "]";
    }
    }
    Fail(inst, "has an unknown tessellation attribute");
}

void RequireAttributeStage(State& state, const IrValue& inst) {
    const auto kind = static_cast<TessellationAttribute>(Immediate(inst, 0));
    const auto stage = StageOf(state);
    bool valid = false;
    switch (kind) {
    case TessellationAttribute::LocalOutput:
    case TessellationAttribute::InstanceOutput: valid = stage == IrShaderStage::Local; break;
    case TessellationAttribute::ControlInput:
    case TessellationAttribute::InstanceInput:
    case TessellationAttribute::ControlOutput:
    case TessellationAttribute::PatchOutput:
    case TessellationAttribute::Factor: valid = stage == IrShaderStage::TessellationControl; break;
    case TessellationAttribute::EvaluationInput:
    case TessellationAttribute::PatchInput: valid = stage == IrShaderStage::TessellationEvaluation; break;
    }
    if (!valid) Fail(inst, "accesses a tessellation attribute this stage does not own");
}

void GetTessellation(State& state, const IrValue& inst) {
    RequireAttributeStage(state, inst);
    const auto kind = static_cast<TessellationAttribute>(Immediate(inst, 0));
    const auto word = AttributeWord(state, inst);
    const auto value = kind == TessellationAttribute::Factor ? "as_type<uint>(float(" + word + "))" : word;
    Define(state, inst, "(" + Arg(state, inst, 2) + " ? " + value + " : 0u)");
}

void SetTessellation(State& state, const IrValue& inst) {
    RequireAttributeStage(state, inst);
    const auto kind = static_cast<TessellationAttribute>(Immediate(inst, 0));
    const auto word = AttributeWord(state, inst);
    const auto value = Arg(state, inst, 2);
    Open(state, "if (" + Arg(state, inst, 3) + ") {");
    if (kind != TessellationAttribute::Factor) {
        Line(state, word + " = " + value + ";");
        Close(state);
        return;
    }
    const auto outer = Immediate(inst, 1) / 4u < OuterFactors(state);
    const auto factor = Temp(state);
    const auto minimum = Temp(state);
    const auto lowest = Temp(state);
    const auto clamped = Temp(state);
    const auto ceiling = Tess(state).partitioning == 2u ? "63.0f" : "64.0f";
    const auto levels = TessellationLevelPushOffsetBytes / 4u;
    Line(state, "const float " + factor + " = as_type<float>(" + value + ");");
    Line(state, "const float " + minimum + " = as_type<float>(" + Push(state, levels) + ");");
    Line(state, "const float " + lowest + " = " + (outer ? minimum : "max(" + minimum + ", 1.0f)") + ";");
    Line(state, "const float " + clamped + " = min(max(" + factor + ", " + lowest + "), min(as_type<float>(" + Push(state, levels + 1u) + "), " + ceiling + "));");
    Line(state, word + " = half(" + factor + " > " + (outer ? "0.0f" : "1.0f") + " ? " + clamped + " : " + (outer ? factor : lowest) + ");");
    Close(state);
}

std::string MeshArguments(State& state) {
    const auto geo = Geometry(state);
    return "((" + Push(state, MeshDrawPushOffsetBytes / 4u + MeshArgumentAddressDword) + " | " + Push(state, MeshDrawPushOffsetBytes / 4u + MeshArgumentAddressDword + 1u) + ") != 0u && " + geo + ".mesh_arguments != nullptr)";
}

std::uint32_t MeshTopologySize(const State& state) {
    const auto size = Mesh(state).outputPrimitiveSize;
    if (size < 1u || size > 3u) Fail("the mesh program emits primitives of an unsupported size");
    return size;
}

bool MeshNeedsBarrier(const State& state) {
    for (const auto& output : state.program.Info().outputs) {
        if (output.kind == StageOutputKind::Layer || output.kind == StageOutputKind::ViewportIndex || output.kind == StageOutputKind::CullDistance) return true;
    }
    return false;
}

std::uint32_t MeshCullDistances(const State& state) {
    std::uint32_t count = 0u;
    for (const auto& output : state.program.Info().outputs) {
        if (output.kind == StageOutputKind::CullDistance) count = std::max(count, output.index + 1u);
    }
    return count;
}

void DeclareMeshOutputs(State& state) {
    std::uint32_t clips = 0u;
    bool position = false;
    for (const auto& output : state.program.Info().outputs) {
        switch (output.kind) {
        case StageOutputKind::Position: position = true; break;
        case StageOutputKind::Parameter: state.stageOutputs.emplace("p" + std::to_string(output.index), "float4 p" + std::to_string(output.index) + " [[user(locn" + std::to_string(output.location) + ")]]"); break;
        case StageOutputKind::PointSize: state.stageOutputs.emplace("point_size", "float point_size [[point_size]]"); break;
        case StageOutputKind::ClipDistance: clips = std::max(clips, output.index + 1u); break;
        case StageOutputKind::CullDistance:
        case StageOutputKind::Layer:
        case StageOutputKind::ViewportIndex: break;
        default: Fail("a mesh program declares a pixel output");
        }
    }
    state.stageOutputs.emplace("position", "float4 position [[position]]");
    if (position) {
        state.zeroClip = clips;
        ++clips;
    }
    if (clips > MaxClipDistances) Fail("guest clip distances plus the all-zero clip position distance exceed eight clip distances");
    if (clips != 0u) state.stageOutputs.emplace("clip", "float clip [[clip_distance]] [" + std::to_string(clips) + "]");
}

std::string MeshExit(const State& state) {
    const auto& mesh = Mesh(state);
    const auto size = MeshTopologySize(state);
    const auto last = U32(mesh.maxVertices - 1u);
    const auto culls = MeshCullDistances(state);
    std::string text = "{ ";
    if (MeshNeedsBarrier(state)) text += "threadgroup_barrier(mem_flags::mem_threadgroup); ";
    text += "if (aps5_index < " + U32(mesh.maxVertices) + ") aps5_mesh.set_vertex(aps5_index, aps5_out); ";
    text += "if (aps5_index < " + U32(mesh.maxPrimitives) + ") { aps5_mesh_primitive aps5_p = {}; aps5_p.culled = (aps5_prim & 0x80000000u) != 0u; ";
    for (std::uint32_t corner = 0; corner < size; ++corner) {
        const auto index = "aps5_i" + std::to_string(corner);
        text += "const uint " + index + " = min(extract_bits(aps5_prim, " + U32(corner * 10u) + ", 9u), " + last + "); ";
        text += "aps5_mesh.set_index(aps5_index * " + U32(size) + " + " + U32(corner) + ", uchar(" + index + ")); ";
    }
    const auto provoking = "aps5_i" + std::to_string(std::min(mesh.provokingVertex, size - 1u));
    for (const auto& output : state.program.Info().outputs) {
        if (output.kind == StageOutputKind::Layer) text += "aps5_p.layer = aps5_mesh_layer[" + provoking + "]; ";
        if (output.kind == StageOutputKind::ViewportIndex) text += "aps5_p.viewport = aps5_mesh_viewport[" + provoking + "]; ";
    }
    for (std::uint32_t plane = 0; plane < culls; ++plane) {
        std::string outside;
        for (std::uint32_t corner = 0; corner < size; ++corner) outside += std::string(corner == 0u ? "" : " && ") + "aps5_mesh_cull[aps5_i" + std::to_string(corner) + " * " + U32(culls) + " + " + U32(plane) + "] < 0.0f";
        text += "aps5_p.culled = aps5_p.culled || (" + outside + "); ";
    }
    text += "aps5_mesh.set_primitive(aps5_index, aps5_p); } return; }";
    return text;
}

std::string Prologue(const State& state) {
    std::string text;
    for (const auto& line : state.prologue) text += "    " + line + "\n";
    return text;
}

std::string Arguments(const State& state, const std::string& parameters, std::initializer_list<std::string> leading) {
    std::vector<std::string> arguments(leading);
    for (const auto& [name, declaration] : state.stageArguments) arguments.push_back(declaration);
    std::string all;
    for (const auto& argument : arguments) all += "\n    " + argument + ",";
    if (parameters.empty() && !all.empty()) all.pop_back();
    return all + parameters;
}

std::string Globals(const State& state) {
    std::string text;
    for (const auto& [name, declaration] : state.stageGlobals) text += declaration + "\n";
    return text;
}

std::string LayoutConstant(const State& state) {
    const auto stage = StageOf(state);
    const auto& metadata = state.program.Metadata();
    const auto& tess = Tess(state);
    std::array<std::uint32_t, MslGeometryLayoutFields> fields{};
    fields[0] = static_cast<std::uint32_t>(stage == IrShaderStage::Local ? MslGeometryKind::Local : stage == IrShaderStage::TessellationControl ? MslGeometryKind::Hull : stage == IrShaderStage::TessellationEvaluation ? MslGeometryKind::Domain : MslGeometryKind::Mesh);
    if (stage == IrShaderStage::Mesh) {
        const auto& mesh = Mesh(state);
        fields[9] = mesh.maxVertices;
        fields[10] = mesh.maxPrimitives;
        fields[11] = std::max(mesh.threadsNum[0], 1u) * std::max(mesh.threadsNum[1], 1u) * std::max(mesh.threadsNum[2], 1u);
        fields[12] = mesh.outputPrimitiveSize;
    } else {
        fields[1] = metadata.tessellationLocalStride;
        fields[2] = metadata.tessellationControlStride;
        fields[3] = metadata.tessellationPatchStride;
        fields[4] = tess.inputControlPoints;
        fields[5] = tess.outputControlPoints;
        fields[6] = tess.domain;
        fields[7] = tess.partitioning;
        fields[8] = tess.outputTopology;
    }
    std::string text = "constant uint aps5_layout[" + std::to_string(MslGeometryLayoutFields) + "] = {";
    for (std::size_t index = 0; index < fields.size(); ++index) text += (index == 0u ? "" : ", ") + std::to_string(fields[index]) + "u";
    return text + "};\n";
}

bool HasBarrier(const State& state) {
    for (const auto* block : state.program.BlockOrder()) {
        for (const auto* inst : block->Instructions()) {
            if (inst->Opcode() == IrOpcode::Barrier) return true;
        }
    }
    return false;
}

std::size_t Count(const std::string& text, const std::string& needle) {
    std::size_t count = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) ++count;
    return count;
}

}

bool IsGeometry(IrShaderStage stage) {
    return stage == IrShaderStage::Local || stage == IrShaderStage::TessellationControl || stage == IrShaderStage::TessellationEvaluation || stage == IrShaderStage::Mesh;
}

std::string StageConstant(State& state, std::uint32_t id, std::uint32_t fallback) {
    state.constants.emplace(id, fallback);
    return "aps5_c" + std::to_string(id);
}

void PrepareGeometry(State& state) {
    const auto stage = StageOf(state);
    static_cast<void>(VertexInfo(state));
    if (stage == IrShaderStage::Mesh) {
        const auto& mesh = Mesh(state);
        if (mesh.maxVertices == 0u || mesh.maxPrimitives == 0u) Fail("the mesh program has no output vertices or primitives");
        if (state.target.mesh && (mesh.maxVertices > state.target.mesh->maxOutputVertices || mesh.maxPrimitives > state.target.mesh->maxOutputPrimitives)) throw MslUnsupported("MSL backend: the mesh program outputs more vertices or primitives than a Metal mesh threadgroup holds");
        if (mesh.maxVertices > 256u) throw MslUnsupported("MSL backend: Metal mesh functions index at most 256 vertices");
        static_cast<void>(MeshTopologySize(state));
        state.features |= LocalIndex;
        DeclareMeshOutputs(state);
        state.exit = MeshExit(state);
        return;
    }
    RequireTessellation(state);
    if (stage == IrShaderStage::TessellationEvaluation) {
        DeclareVertexOutputs(state);
        Geometry(state);
        state.stageArguments.emplace("aps5_patch_id", "uint aps5_patch_id [[patch_id]]");
        state.stageArguments.emplace("aps5_tess_coord", std::string(Tess(state).domain == 2u ? "float2" : "float3") + " aps5_tess_coord [[position_in_patch]]");
        state.stageArguments.emplace("aps5_instance", "uint aps5_instance [[instance_id]]");
        state.stageArguments.emplace("aps5_base_instance", "uint aps5_base_instance [[base_instance]]");
        state.prologue.push_back("const uint aps5_patch = (aps5_instance - aps5_base_instance) * aps5_geo.patch_count + aps5_patch_id;");
        state.exit = "return aps5_out;";
        return;
    }
    const auto geo = Geometry(state);
    const bool guard = !HasBarrier(state);
    if (stage == IrShaderStage::Local) {
        const auto inputs = U32(Tess(state).inputControlPoints);
        state.prologue.push_back("const uint aps5_span = " + geo + ".patch_count * " + inputs + ";");
        if (guard) state.prologue.push_back("if (aps5_tid >= aps5_span * " + geo + ".instance_count) return;");
        state.prologue.push_back("const uint aps5_instance = aps5_tid / max(aps5_span, 1u);");
        state.prologue.push_back("const uint aps5_at = " + geo + ".first_index + aps5_tid - aps5_instance * aps5_span;");
        state.prologue.push_back("const uint aps5_vertex_id = " + geo + ".index_bytes == 0u ? aps5_at : uint(int(" + geo + ".index_bytes == 2u ? uint(reinterpret_cast<device const ushort*>(" + geo + ".indices)[aps5_at]) : reinterpret_cast<device const uint*>(" + geo + ".indices)[aps5_at]) + " + geo + ".base_vertex);");
        state.prologue.push_back("const uint aps5_instance_index = " + geo + ".base_instance + aps5_instance;");
    } else {
        state.prologue.push_back("const uint aps5_patch = aps5_tid.y;");
        if (guard) state.prologue.push_back("if (aps5_patch >= " + geo + ".patch_count * " + geo + ".instance_count || aps5_tid.x >= " + U32(Tess(state).outputControlPoints) + ") return;");
    }
    state.exit = "return;";
}

std::string GeometryBuiltin(State& state, StageInputKind kind, std::uint32_t component) {
    switch (StageOf(state)) {
    case IrShaderStage::Local:
        if (kind == StageInputKind::VertexIndex) return "aps5_vertex_id";
        if (kind == StageInputKind::InstanceIndex) return "aps5_instance_index";
        if (kind == StageInputKind::BaseInstance) return Geometry(state) + ".base_instance";
        break;
    case IrShaderStage::TessellationControl:
        if (kind == StageInputKind::InvocationId) return "aps5_tid.x";
        if (kind == StageInputKind::PrimitiveId) return "(aps5_patch % max(" + Geometry(state) + ".patch_count, 1u))";
        if (kind == StageInputKind::BaseInstance) return Geometry(state) + ".base_instance";
        break;
    case IrShaderStage::TessellationEvaluation:
        if (kind == StageInputKind::TessCoord) return "as_type<uint>(aps5_tess_coord." + Swizzle(component) + ")";
        if (kind == StageInputKind::PrimitiveId) return "aps5_patch_id";
        break;
    default: break;
    }
    throw MslUnsupported("MSL backend: stage input " + std::to_string(static_cast<std::uint32_t>(kind)) + " is not emitted for this geometry stage");
}

std::string LocalAttribute(State& state, std::uint32_t location, std::uint32_t component) {
    if (location >= static_cast<std::uint32_t>(ShaderVertexInputInfo::MaxResources)) Fail("a local program reads an attribute past the vertex fetch table");
    if (!state.localFetches.contains(location)) {
        const auto format = StageConstant(state, MetalAbi::LocalFetchFormatConstant + location, DefaultFetchFormat);
        state.localFetches.emplace(location, 0u);
        Global(state, "aps5_1fetch", FetchText);
        state.stageArguments.emplace("aps5_fetch_table", "constant aps5_fetch* aps5_fetch_table [[buffer(" + std::to_string(MetalAbi::VertexFetchSlot) + ")]]");
        state.prologue.push_back("const uint4 aps5_a" + std::to_string(location) + " = aps5_fetch_attribute(aps5_fetch_table[" + std::to_string(location) + "], aps5_vertex_id, aps5_instance_index, " + format + ");");
    }
    return "aps5_a" + std::to_string(location) + "." + Swizzle(component & 3u);
}

bool MeshAuxExport(State& state, const PositionExportComponent& output, const std::string& raw) {
    if (StageOf(state) != IrShaderStage::Mesh) return false;
    const auto& mesh = Mesh(state);
    if (output.layer) {
        Line(state, "if (aps5_index < " + U32(mesh.maxVertices) + ") aps5_mesh_layer[aps5_index] = " + raw + " & 0x7ffu;");
        return true;
    }
    if (output.viewport) {
        Line(state, "if (aps5_index < " + U32(mesh.maxVertices) + ") aps5_mesh_viewport[aps5_index] = extract_bits(" + raw + ", 16u, 4u);");
        return true;
    }
    if (output.cullDistance != NoDistance) {
        Line(state, "if (aps5_index < " + U32(mesh.maxVertices) + ") aps5_mesh_cull[aps5_index * " + U32(MeshCullDistances(state)) + " + " + U32(output.cullDistance) + "] = as_type<float>(" + raw + ");");
        return true;
    }
    return false;
}

void MeshPrimitive(State& state, const std::string& value) {
    if (StageOf(state) != IrShaderStage::Mesh) return;
    Line(state, "aps5_prim = " + value + ";");
}

bool EmitGeometry(State& state, const IrValue& inst) {
    const auto stage = StageOf(state);
    switch (inst.Opcode()) {
    case IrOpcode::TessellationBase:
        Fail(inst, "must be lowered before MSL emission");
    case IrOpcode::GetTessellationAttribute:
        GetTessellation(state, inst);
        return true;
    case IrOpcode::SetTessellationAttribute:
        SetTessellation(state, inst);
        return true;
    case IrOpcode::LaneId:
        if (stage != IrShaderStage::TessellationControl) return false;
        Define(state, inst, "aps5_tid.x");
        return true;
    case IrOpcode::MeshDrawParameter: {
        const auto index = Immediate(inst, 0);
        if (stage != IrShaderStage::Mesh || index >= PushData::MeshDrawDwordCount) Fail(inst, "is an invalid mesh draw parameter");
        Define(state, inst, Push(state, MeshDrawPushOffsetBytes / 4u + index));
        return true;
    }
    case IrOpcode::MeshArgument: {
        const auto index = Immediate(inst, 0);
        if (stage != IrShaderStage::Mesh || index >= MeshArgumentBytes / 4u) Fail(inst, "is an invalid mesh argument");
        Define(state, inst, "(" + MeshArguments(state) + " ? aps5_geo.mesh_arguments[" + U32(index) + "] : 0u)");
        return true;
    }
    case IrOpcode::MeshRestartStart: {
        if (stage != IrShaderStage::Mesh) Fail(inst, "reads the mesh restart table outside a mesh program");
        const auto present = MeshArguments(state);
        const auto position = "min(" + Arg(state, inst, 0) + ", aps5_geo.mesh_arguments[" + U32(MeshRestartLengthDword) + "] - 1u)";
        Define(state, inst, "((" + Arg(state, inst, 1) + " && " + present + ") ? aps5_geo.mesh_arguments[" + U32(MeshRestartTableDword) + " + " + position + "] : 0u)");
        return true;
    }
    case IrOpcode::MeshAllocate:
        if (stage != IrShaderStage::Mesh) Fail(inst, "allocates mesh outputs outside a mesh program");
        Line(state, "if (aps5_index == 0u) aps5_mesh.set_primitive_count(min(extract_bits(" + Arg(state, inst, 0) + ", 12u, 11u), " + U32(Mesh(state).maxPrimitives) + "));");
        return true;
    case IrOpcode::Barrier:
        if (stage == IrShaderStage::TessellationEvaluation) return true;
        if (stage != IrShaderStage::TessellationControl) return false;
        Line(state, "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);");
        return true;
    default:
        return false;
    }
}

void CheckGeometry(const State& state) {
    const auto stage = StageOf(state);
    if (stage == IrShaderStage::Mesh) {
        if (MeshNeedsBarrier(state) && (Count(state.body, state.exit) > 1u || (!state.body.empty() && Count(state.body, state.exit) == 1u && !state.body.ends_with("    " + state.exit + "\n")))) {
            throw MslUnsupported("MSL backend: a mesh program that exports layers, viewports or cull distances must end in one place, because its primitives read other vertices' values after a barrier");
        }
        if (state.program.WaveSize() == 64u && (state.features & (Lane | Wave | Permute)) != 0u) throw MslUnsupported("MSL backend: a wave64 mesh program uses SIMD-group lane operations, which need two guest lanes per thread in the mesh epilogue");
        return;
    }
    if ((state.features & (Lds | WorkgroupId | LocalId | GlobalId | LocalIndex)) != 0u) throw MslUnsupported("MSL backend: a tessellation program uses threadgroup memory or compute builtins");
    if ((state.features & (Lane | Wave | Permute)) != 0u) throw MslUnsupported("MSL backend: a tessellation program uses SIMD-group lane operations, whose lanes do not match the guest wave");
}

std::string GeometryProgram(State& state, const std::string& prelude, const std::string& declarations, const std::string& parameters) {
    const auto stage = StageOf(state);
    std::string text = prelude + Globals(state) + LayoutConstant(state);
    const auto body = [&](const std::string& head) {
        text += head + "{\n";
        text += declarations;
        text += Prologue(state);
        if (stage == IrShaderStage::Mesh) {
            const auto& mesh = Mesh(state);
            text += "    aps5_mesh_vertex aps5_out = {};\n    uint aps5_prim = 0x80000000u;\n";
            for (const auto& output : state.program.Info().outputs) {
                if (output.kind == StageOutputKind::Layer) text += "    threadgroup uint aps5_mesh_layer[" + std::to_string(mesh.maxVertices) + "];\n";
                if (output.kind == StageOutputKind::ViewportIndex) text += "    threadgroup uint aps5_mesh_viewport[" + std::to_string(mesh.maxVertices) + "];\n";
            }
            if (const auto culls = MeshCullDistances(state); culls != 0u) text += "    threadgroup float aps5_mesh_cull[" + std::to_string(mesh.maxVertices * culls) + "];\n";
        } else if (stage == IrShaderStage::TessellationEvaluation) {
            text += "    aps5_stage_out aps5_out = {};\n";
        }
        text += state.body;
        if (!state.body.ends_with(state.exit + "\n")) text += "    " + state.exit + "\n";
        text += "}\n";
    };
    switch (stage) {
    case IrShaderStage::Local:
        body("kernel void " + std::string(MslEntryPoint) + "(" + Arguments(state, parameters, {"uint aps5_tid [[thread_position_in_grid]]"}) + ")\n");
        break;
    case IrShaderStage::TessellationControl:
        body("kernel void " + std::string(MslEntryPoint) + "(" + Arguments(state, parameters, {"uint2 aps5_tid [[thread_position_in_grid]]"}) + ")\n");
        break;
    case IrShaderStage::TessellationEvaluation: {
        text += "struct aps5_stage_out {\n";
        for (const auto& [name, declaration] : state.stageOutputs) text += "    " + declaration + ";\n";
        text += "};\n";
        const auto domain = Tess(state).domain == 2u ? "quad" : "triangle";
        body("[[patch(" + std::string(domain) + ", 0)]] vertex aps5_stage_out " + std::string(MslEntryPoint) + "(" + Arguments(state, parameters, {}) + ")\n");
        break;
    }
    case IrShaderStage::Mesh: {
        const auto& mesh = Mesh(state);
        text += "struct aps5_mesh_vertex {\n";
        for (const auto& [name, declaration] : state.stageOutputs) text += "    " + declaration + ";\n";
        text += "};\nstruct aps5_mesh_primitive {\n    bool culled [[primitive_culled]];\n";
        for (const auto& output : state.program.Info().outputs) {
            if (output.kind == StageOutputKind::Layer) text += "    uint layer [[render_target_array_index]];\n";
            if (output.kind == StageOutputKind::ViewportIndex) text += "    uint viewport [[viewport_array_index]];\n";
        }
        const auto size = MeshTopologySize(state);
        const auto topology = size == 3u ? "triangle" : size == 2u ? "line" : "point";
        text += "};\nusing aps5_mesh_t = metal::mesh<aps5_mesh_vertex, aps5_mesh_primitive, " + std::to_string(mesh.maxVertices) + ", " + std::to_string(mesh.maxPrimitives) + ", metal::topology::" + topology + ">;\n";
        body("[[mesh]] void " + std::string(MslEntryPoint) + "(" + Arguments(state, parameters, {"aps5_mesh_t aps5_mesh"}) + ")\n");
        break;
    }
    default:
        Fail("a non-geometry program reached the geometry emitter");
    }
    return text;
}

}

namespace ShaderRecompiler {

std::optional<MslGeometryLayout> ParseMslGeometryLayout(std::string_view source) {
    constexpr std::string_view Marker = "constant uint aps5_layout[13] = {";
    const auto at = source.find(Marker);
    if (at == std::string_view::npos) return std::nullopt;
    std::array<std::uint32_t, MslGeometryLayoutFields> fields{};
    auto cursor = source.data() + at + Marker.size();
    const auto end = source.data() + source.size();
    for (auto& field : fields) {
        while (cursor < end && (*cursor == ' ' || *cursor == ',')) ++cursor;
        const auto parsed = std::from_chars(cursor, end, field);
        if (parsed.ec != std::errc{}) return std::nullopt;
        cursor = parsed.ptr;
        if (cursor < end && *cursor == 'u') ++cursor;
    }
    if (fields[0] < 1u || fields[0] > 4u) return std::nullopt;
    MslGeometryLayout layout;
    layout.kind = static_cast<MslGeometryKind>(fields[0]);
    layout.localStride = fields[1];
    layout.controlStride = fields[2];
    layout.patchStride = fields[3];
    layout.inputControlPoints = fields[4];
    layout.outputControlPoints = fields[5];
    layout.domain = fields[6];
    layout.partitioning = fields[7];
    layout.outputTopology = fields[8];
    layout.meshVertices = fields[9];
    layout.meshPrimitives = fields[10];
    layout.meshThreads = fields[11];
    layout.meshPrimitiveSize = fields[12];
    return layout;
}

}
