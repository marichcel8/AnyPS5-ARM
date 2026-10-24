#include "MslState.hpp"
#include "MetalAbi.hpp"
#include "PipelineSpecialization.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <limits>

namespace ShaderRecompiler::Msl {

namespace {

constexpr std::uint32_t PixelParameterLimit = 32u;
constexpr std::uint32_t MaxClipDistances = 8u;
constexpr std::uint32_t NoDistance = std::numeric_limits<std::uint32_t>::max();

IrShaderStage StageOf(const State& state) {
    return state.program.Resources().stage;
}

const ShaderPixelInputInfo& PixelInfo(const State& state) {
    if (state.input.pixel == nullptr) Fail("pixel input info is missing");
    return *state.input.pixel;
}

const ShaderVertexInputInfo& VertexInfo(const State& state) {
    if (state.input.vertex == nullptr) Fail("vertex input info is missing");
    return *state.input.vertex;
}

std::string Swizzle(std::uint32_t component) {
    static constexpr std::string_view Names[4] = {"x", "y", "z", "w"};
    if (component >= 4u) Fail("component index is out of range");
    return std::string(Names[component]);
}

std::string F32(float value) {
    return "as_type<float>(" + U32(std::bit_cast<std::uint32_t>(value)) + ")";
}

std::string Constant(State& state, std::uint32_t id, std::uint32_t fallback) {
    state.constants.emplace(id, fallback);
    return "aps5_c" + std::to_string(id);
}

std::uint32_t Immediate(const IrValue& inst, std::size_t index) {
    const IrValue* value = inst.Argument(index)->Resolve();
    if (!value->HasImmediate()) Fail(inst, "needs an immediate operand");
    return value->ImmediateU32();
}

void Input(State& state, const std::string& name, const std::string& declaration) {
    const auto [found, inserted] = state.stageInputs.emplace(name, declaration);
    if (!inserted && found->second != declaration) Fail("stage input " + name + " is declared twice with different interpolation");
}

void Output(State& state, const std::string& name, const std::string& declaration) {
    state.stageOutputs.emplace(name, declaration);
}

void Argument(State& state, const std::string& name, const std::string& declaration) {
    state.stageArguments.emplace(name, declaration);
}

std::string VertexClassType(IrTextureNumericClass numeric, std::uint32_t components) {
    const std::string scalar = numeric == IrTextureNumericClass::Uint ? "uint" : numeric == IrTextureNumericClass::Sint ? "int" : "float";
    return components == 1u ? scalar : scalar + std::to_string(components);
}

const StageInput* VertexParameter(const State& state, std::uint32_t location) {
    for (const auto& input : state.program.Info().inputs) {
        if (input.kind == StageInputKind::Parameter && input.location == location) return &input;
    }
    return nullptr;
}

std::string VertexComponent(const std::string& member, std::uint32_t components, std::uint32_t component, bool floating) {
    if (component >= components) return "0u";
    const auto value = components == 1u ? member : member + "." + Swizzle(component);
    return floating ? "as_type<uint>(" + value + ")" : "uint(" + value + ")";
}

std::string VertexAttribute(State& state, std::uint32_t location, std::uint32_t component) {
    const auto* input = VertexParameter(state, location);
    if (input == nullptr) return "0u";
    const auto& vertex = VertexInfo(state);
    const auto components = std::clamp(input->componentCount, 1u, 4u);
    const auto name = "a" + std::to_string(location);
    if (vertex.fetchEmbedded) {
        const auto kind = Constant(state, PipelineSpecialization::VertexBase + location * PipelineSpecialization::VertexWords + 5u, 0u);
        const auto guard = [&](std::string_view suffix, std::string_view test) {
            const auto flag = "aps5_v" + std::string(suffix) + std::to_string(location);
            state.stageGlobals.emplace(flag, "constant bool " + flag + " = " + kind + " " + std::string(test) + ";");
            return flag;
        };
        const auto floating = guard("f", "== 0u");
        const auto sint = guard("i", "== 1u");
        const auto uint = guard("u", "== 2u");
        Input(state, name + "f", VertexClassType(IrTextureNumericClass::Float, components) + " " + name + "f [[attribute(" + std::to_string(location) + "), function_constant(" + floating + ")]]");
        Input(state, name + "i", VertexClassType(IrTextureNumericClass::Sint, components) + " " + name + "i [[attribute(" + std::to_string(location) + "), function_constant(" + sint + ")]]");
        Input(state, name + "u", VertexClassType(IrTextureNumericClass::Uint, components) + " " + name + "u [[attribute(" + std::to_string(location) + "), function_constant(" + uint + ")]]");
        return "(" + floating + " ? " + VertexComponent("aps5_in." + name + "f", components, component, true) + " : " + sint + " ? " + VertexComponent("aps5_in." + name + "i", components, component, true) + " : " + VertexComponent("aps5_in." + name + "u", components, component, false) + ")";
    }
    auto numeric = IrTextureNumericClass::Float;
    if (location < static_cast<std::uint32_t>(ShaderVertexInputInfo::MaxResources) && location < static_cast<std::uint32_t>(vertex.resourcesNum)) {
        numeric = VertexInputNumericClass(vertex.resources[location].Format());
        if (numeric == IrTextureNumericClass::Unsupported) numeric = IrTextureNumericClass::Float;
    }
    Input(state, name, VertexClassType(numeric, components) + " " + name + " [[attribute(" + std::to_string(location) + ")]]");
    return VertexComponent("aps5_in." + name, components, component, numeric != IrTextureNumericClass::Uint);
}

std::uint32_t PixelSlot(const State& state, std::uint32_t attribute) {
    const auto& pixel = PixelInfo(state);
    if (attribute >= PixelParameterLimit || attribute >= pixel.inputNum) Fail("pixel interpolator index is out of range");
    return pixel.InputSlot(attribute);
}

bool PixelPerVertex(const State& state, std::uint32_t attribute) {
    for (const auto& input : state.program.Info().inputs) {
        if (input.kind == StageInputKind::Parameter && input.location == attribute) return input.perVertex;
    }
    return false;
}

std::string PixelAttribute(State& state, std::uint32_t attribute, std::uint32_t component) {
    const auto& pixel = PixelInfo(state);
    if (pixel.InputIsDefault(attribute)) return U32(pixel.InputDefaultBits(attribute, component & 3u));
    if (PixelPerVertex(state, attribute)) throw MslUnsupported("MSL backend: pixel input " + std::to_string(attribute) + " is read per vertex, which Metal fragment functions cannot do (barycentric emulation is section S7)");
    if (pixel.InputIsCustom(attribute)) throw MslUnsupported("MSL backend: pixel input " + std::to_string(attribute) + " uses custom interpolation, which needs per-vertex values (section S7)");
    const auto slot = PixelSlot(state, attribute);
    const auto& metadata = state.program.Metadata();
    std::string qualifier = "center_perspective";
    if (pixel.InputIsFlat(attribute)) {
        qualifier = "flat";
    } else if (pixel.InputIsLinear(attribute, metadata.pixelLinearInputs, metadata.pixelPerspectiveInputs)) {
        qualifier = "center_no_perspective";
    }
    const auto name = "p" + std::to_string(slot);
    Input(state, name, "float4 " + name + " [[user(locn" + std::to_string(slot) + "), " + qualifier + "]]");
    return "as_type<uint>(aps5_in." + name + "." + Swizzle(component & 3u) + ")";
}

std::string Attribute(State& state, const IrValue& inst) {
    const auto attribute = Immediate(inst, 0);
    const auto component = Immediate(inst, 1);
    if (StageOf(state) == IrShaderStage::Pixel) return PixelAttribute(state, attribute, component);
    if (inst.Flags<std::uint32_t>() != 1u) return VertexAttribute(state, attribute, component & 3u);
    if (!VertexInfo(state).fetchEmbedded || component >= 4u) Fail(inst, "is an invalid prepared vertex attribute");
    const auto first = PipelineSpecialization::VertexBase + attribute * PipelineSpecialization::VertexWords;
    const auto selector = Constant(state, first + component, component + 4u);
    const auto one = Constant(state, first + 4u, 0x3f800000u);
    std::string value = "(" + selector + " == 1u ? " + one + " : 0u)";
    for (std::uint32_t channel = 0; channel < 4u; ++channel) value = "(" + selector + " == " + U32(channel + 4u) + " ? " + VertexAttribute(state, attribute, channel) + " : " + value + ")";
    return value;
}

std::string InterpolationParameter(State& state, const IrValue& inst) {
    const auto attribute = Immediate(inst, 0);
    const auto component = Immediate(inst, 1);
    const auto mode = Immediate(inst, 2);
    if (PixelInfo(state).InputIsDefault(attribute)) return mode == 2u ? PixelAttribute(state, attribute, component) : std::string("0u");
    return PixelAttribute(state, attribute, component);
}

std::string Coverage(State& state, const std::string& mask) {
    return "aps5_out.mask = " + mask + ";";
}

void DeclareCoverage(State& state) {
    Output(state, "mask", "uint mask [[sample_mask]]");
}

void AlphaCoverage(State& state, const std::string& color) {
    const auto& pixel = PixelInfo(state);
    if (!pixel.psDitheredAlphaToCoverage) {
        Line(state, Coverage(state, "(" + color + ".w < 0.5f || isnan(" + color + ".w)) ? 0u : 0xffffffffu"));
        return;
    }
    static constexpr std::array<std::uint32_t, 64> Masks{{
    0x00000000u, 0x00000010u, 0x00000004u, 0x10000004u, 0x10000002u, 0x10100002u, 0x04100002u, 0x04101002u,
    0x04101001u, 0x04101011u, 0x04041011u, 0x14041011u, 0x12041011u, 0x12141011u, 0x12140411u, 0x12141411u,
    0x12141411u, 0x12141431u, 0x12141415u, 0x32141415u, 0x32121415u, 0x32321415u, 0x16321415u, 0x16323415u,
    0x15323415u, 0x15323435u, 0x15163435u, 0x35163435u, 0x35163235u, 0x35363235u, 0x35361635u, 0x35363635u,
    0x35363635u, 0x35363675u, 0x3536363du, 0x7536363du, 0x75363637u, 0x75763637u, 0x3d763637u, 0x3d767637u,
    0x3d757637u, 0x3d757677u, 0x3d3d7677u, 0x7d3d7677u, 0x773d7677u, 0x777d7677u, 0x777d3e77u, 0x777d7e77u,
    0x777d7e77u, 0x777d7ef7u, 0x777d7e7fu, 0xf77d7e7fu, 0xf7777e7fu, 0xf7f77e7fu, 0x7ff77e7fu, 0x7ff7fe7fu,
    0x7ff7fd7fu, 0x7ff7fdffu, 0x7f7ffdffu, 0xff7ffdffu, 0xff7ff7ffu, 0xfffff7ffu, 0xffff7fffu, 0xffffffffu,
    }};
    std::string table = "constant uint aps5_dither[64] = {";
    for (std::size_t index = 0; index < Masks.size(); ++index) table += (index == 0u ? "" : ", ") + U32(Masks[index]);
    state.stageGlobals.emplace("aps5_dither", table + "};");
    Argument(state, "aps5_frag_coord", "float4 aps5_frag_coord [[position]]");
    const auto index = Temp(state);
    const auto shift = Temp(state);
    Line(state, "const uint " + index + " = uint(clamp(" + color + ".w > 0.0f ? " + color + ".w : 0.0f, 0.0f, 63.0f / 64.0f) * 64.0f);");
    Line(state, "const uint " + shift + " = (((uint(aps5_frag_coord.x) & 1u) << 1u) | (uint(aps5_frag_coord.y) & 1u)) << 3u;");
    Line(state, Coverage(state, "(aps5_dither[" + index + "] >> " + shift + ") & 255u"));
}

std::uint32_t MrtMode(const State& state, const ExportInfo& exp) {
    if (StageOf(state) != IrShaderStage::Pixel || exp.kind != ExportTargetKind::Mrt) return 0u;
    const auto& pixel = PixelInfo(state);
    return exp.index < std::size(pixel.targetOutputMode) ? pixel.targetOutputMode[exp.index] : 0u;
}

std::string ExportVector(State& state, const std::string& data, const ExportInfo& exp, bool uintOutput) {
    const auto mode = MrtMode(state, exp);
    if (exp.compr && !uintOutput) {
        std::array<std::string, 4> f32{"0.0f", "0.0f", "0.0f", "1.0f"};
        for (std::uint32_t pair = 0; pair < 2u; ++pair) {
            if ((exp.en & (3u << (pair * 2u))) == 0u) continue;
            const auto packed = data + "." + Swizzle(pair);
            const auto unpacked = Temp(state);
            const auto expression = mode == 5u ? "unpack_unorm2x16_to_float(" + packed + ")" : mode == 6u ? "unpack_snorm2x16_to_float(" + packed + ")" : "float2(as_type<half2>(" + packed + "))";
            Line(state, "const float2 " + unpacked + " = " + expression + ";");
            for (std::uint32_t lane = 0; lane < 2u; ++lane) {
                const auto component = pair * 2u + lane;
                if (((exp.en >> component) & 1u) != 0u) f32[component] = unpacked + "." + Swizzle(lane);
            }
        }
        return "float4(" + f32[0] + ", " + f32[1] + ", " + f32[2] + ", " + f32[3] + ")";
    }
    std::array<std::string, 4> raw{"0u", "0u", "0u", uintOutput ? "1u" : "0x3f800000u"};
    const auto exported = exp.en & (mode == 1u ? 0x1u : mode == 2u || mode == 3u ? 0x3u : 0xfu);
    if (exp.compr) {
        for (std::uint32_t pair = 0; pair < 2u; ++pair) {
            if ((exp.en & (3u << (pair * 2u))) == 0u) continue;
            for (std::uint32_t lane = 0; lane < 2u; ++lane) {
                const auto component = pair * 2u + lane;
                if (((exp.en >> component) & 1u) != 0u) raw[component] = "extract_bits(" + data + "." + Swizzle(pair) + ", " + U32(lane * 16u) + ", 16u)";
            }
        }
    } else {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            if (((exported >> component) & 1u) != 0u) raw[mode == 3u && component == 1u ? 3u : component] = data + "." + Swizzle(component);
        }
    }
    const auto vector = "uint4(" + raw[0] + ", " + raw[1] + ", " + raw[2] + ", " + raw[3] + ")";
    return uintOutput ? vector : "as_type<float4>(" + vector + ")";
}

bool HasOutput(const State& state, StageOutputKind kind, std::uint32_t index) {
    return std::ranges::any_of(state.program.Info().outputs, [&](const StageOutput& output) { return output.kind == kind && output.index == index; });
}

void AuxPosition(State& state, const std::string& data, const ExportInfo& exp) {
    const auto& vertex = VertexInfo(state);
    for (std::uint32_t component = 0; component < 4u; ++component) {
        if ((exp.en & (1u << component)) == 0u) continue;
        const auto output = DecodePositionExportComponent(vertex.paClVsOutCntl, exp.index, component);
        const auto raw = data + "." + Swizzle(component);
        if (output.layer) Line(state, "aps5_out.layer = " + raw + " & 0x7ffu;");
        if (output.viewport) Line(state, "aps5_out.viewport = extract_bits(" + raw + ", 16u, 4u);");
        if (output.layer || output.viewport) continue;
        if (output.pointSize) {
            Line(state, "aps5_out.point_size = as_type<float>(" + raw + ");");
            continue;
        }
        if (output.cullDistance != NoDistance) throw MslUnsupported("MSL backend: cull distances have no Metal equivalent (section S7)");
        if (output.clipDistance != NoDistance) Line(state, "aps5_out.clip[" + std::to_string(output.clipDistance) + "] = as_type<float>(" + raw + ");");
    }
}

void Position(State& state, const std::string& value) {
    const auto& vertex = VertexInfo(state);
    const auto position = Temp(state);
    Line(state, "float4 " + position + " = " + value + ";");
    if (vertex.clipSpace.enabled) {
        const auto& transform = vertex.clipSpace;
        for (std::uint32_t axis = 0; axis < 2u; ++axis) {
            const auto coordinate = position + "." + Swizzle(axis);
            Line(state, coordinate + " = (" + coordinate + " * " + F32(transform.scale[axis]) + " + " + F32(transform.offset[axis]) + ") / " + F32(transform.halfExtent[axis]) + " - 1.0f;");
        }
    }
    if (state.zeroClip != NoDistance) Line(state, "aps5_out.clip[" + std::to_string(state.zeroClip) + "] = all((as_type<uint4>(" + position + ") & 0x7fffffffu) == 0u) ? -1.0f : 0.0f;");
    const auto flip = Constant(state, MetalAbi::PositionFlipYConstant, 1u);
    const auto depth = Constant(state, MetalAbi::DepthNegativeOneToOneConstant, 0u);
    Line(state, "if (" + flip + " != 0u) " + position + ".y = -" + position + ".y;");
    Line(state, "if (" + depth + " != 0u) " + position + ".z = (" + position + ".z + " + position + ".w) * 0.5f;");
    Line(state, "aps5_out.position = " + position + ";");
}

void Export(State& state, const IrValue& inst) {
    const auto& exports = state.program.Metadata().exportInfo;
    const auto index = inst.Flags<ExportFlags>().index;
    if (index >= exports.size()) Fail(inst, "has no export info");
    const auto& exp = exports[index];
    const auto exec = Arg(state, inst, 1);
    const bool pixel = StageOf(state) == IrShaderStage::Pixel;
    if (pixel && exp.vm && state.validMask) Line(state, "aps5_valid = " + exec + ";");
    if (exp.kind == ExportTargetKind::Null || exp.en == 0u || exp.kind == ExportTargetKind::Primitive) return;
    const auto data = Arg(state, inst, 0);
    Open(state, "if (" + exec + ") {");
    if (exp.kind == ExportTargetKind::Position && exp.index != 0u) {
        AuxPosition(state, data, exp);
    } else if (exp.kind == ExportTargetKind::MrtZ) {
        if ((exp.en & 1u) != 0u && HasOutput(state, StageOutputKind::Depth, 0u)) Line(state, "aps5_out.depth = as_type<float>(" + data + ".x);");
        if ((exp.en & 2u) != 0u && HasOutput(state, StageOutputKind::StencilRef, 0u)) Line(state, "aps5_out.stencil = " + data + ".y;");
        if ((exp.en & 4u) != 0u && HasOutput(state, StageOutputKind::SampleMask, 0u)) Line(state, Coverage(state, data + ".z"));
    } else if (exp.kind == ExportTargetKind::Position) {
        Position(state, ExportVector(state, data, exp, false));
    } else if (exp.kind == ExportTargetKind::Parameter && HasOutput(state, StageOutputKind::Parameter, exp.index)) {
        Line(state, "aps5_out.p" + std::to_string(exp.index) + " = " + ExportVector(state, data, exp, false) + ";");
    } else if (exp.kind == ExportTargetKind::Mrt && HasOutput(state, StageOutputKind::Mrt, exp.index)) {
        if (exp.index >= 8u) Fail(inst, "exports past the eight color targets");
        const bool uintOutput = MrtMode(state, exp) == 7u;
        const auto value = Temp(state);
        Line(state, std::string(uintOutput ? "const uint4 " : "const float4 ") + value + " = " + ExportVector(state, data, exp, uintOutput) + ";");
        std::string mapped = uintOutput ? "uint4(" : "float4(";
        for (std::uint32_t component = 0; component < 4u; ++component) mapped += (component == 0u ? "" : ", ") + value + "[" + Constant(state, PipelineSpecialization::ExportBase + exp.index * 4u + component, component) + "]";
        mapped += ")";
        const auto target = "aps5_out.c" + std::to_string(exp.index);
        Line(state, target + " = " + mapped + ";");
        if (exp.index == 0u && !uintOutput && PixelInfo(state).psAlphaToCoverage) AlphaCoverage(state, target);
    }
    Close(state);
}

std::string Location(const StageOutput& output) {
    return std::to_string(output.location);
}

void DeclareVertexOutputs(State& state) {
    std::uint32_t clips = 0u;
    bool position = false;
    for (const auto& output : state.program.Info().outputs) {
        switch (output.kind) {
        case StageOutputKind::Position: position = true; break;
        case StageOutputKind::Parameter: Output(state, "p" + std::to_string(output.index), "float4 p" + std::to_string(output.index) + " [[user(locn" + Location(output) + ")]]"); break;
        case StageOutputKind::PointSize: Output(state, "point_size", "float point_size [[point_size]]"); break;
        case StageOutputKind::ClipDistance: clips = std::max(clips, output.index + 1u); break;
        case StageOutputKind::CullDistance: throw MslUnsupported("MSL backend: cull distances have no Metal equivalent (section S7)");
        case StageOutputKind::Layer: Output(state, "layer", "uint layer [[render_target_array_index]]"); break;
        case StageOutputKind::ViewportIndex: Output(state, "viewport", "uint viewport [[viewport_array_index]]"); break;
        default: Fail("a vertex program declares a pixel output");
        }
    }
    Output(state, "position", "float4 position [[position]]");
    if (position) {
        state.zeroClip = clips;
        ++clips;
    }
    if (clips > MaxClipDistances) Fail("guest clip distances plus the all-zero clip position distance exceed eight clip distances");
    if (clips != 0u) Output(state, "clip", "float clip [[clip_distance]] [" + std::to_string(clips) + "]");
}

void DeclarePixelOutputs(State& state) {
    const auto& pixel = PixelInfo(state);
    for (const auto& output : state.program.Info().outputs) {
        switch (output.kind) {
        case StageOutputKind::Mrt: {
            const bool uintOutput = output.index < std::size(pixel.targetOutputMode) && pixel.targetOutputMode[output.index] == 7u;
            const auto name = "c" + std::to_string(output.index);
            Output(state, name, std::string(uintOutput ? "uint4 " : "float4 ") + name + " [[color(" + Location(output) + ")]]");
            break;
        }
        case StageOutputKind::Depth: {
            const auto mode = pixel.psConservativeZExport == ConservativeZExport::LessThanZ ? "less" : pixel.psConservativeZExport == ConservativeZExport::GreaterThanZ ? "greater" : "any";
            Output(state, "depth", std::string("float depth [[depth(") + mode + ")]]");
            break;
        }
        case StageOutputKind::StencilRef: Output(state, "stencil", "uint stencil [[stencil]]"); break;
        case StageOutputKind::SampleMask: DeclareCoverage(state); break;
        default: Fail("a pixel program declares a vertex output");
        }
    }
    if (pixel.psAlphaToCoverage) {
        if (pixel.psSampleMaskExportEnable) throw MslUnsupported("MSL backend: alpha-to-coverage with a sample-mask export is unsupported");
        DeclareCoverage(state);
    }
}

}

void PrepareStage(State& state) {
    const auto stage = StageOf(state);
    if (stage == IrShaderStage::Vertex) {
        static_cast<void>(VertexInfo(state));
        DeclareVertexOutputs(state);
    } else {
        const auto& pixel = PixelInfo(state);
        if (pixel.QuadMasked()) throw MslUnsupported("MSL backend: PA_SC_AA_MASK quad pixel or sample masks are not emitted");
        if (pixel.psOrderedPixelShader) throw MslUnsupported("MSL backend: primitive-ordered pixel shaders need raster order groups, which are not emitted");
        if (state.program.Metadata().barycentricEmulation) throw MslUnsupported("MSL backend: barycentric emulation is section S7");
        DeclarePixelOutputs(state);
        for (const auto* block : state.program.BlockOrder()) {
            for (const auto* inst : block->Instructions()) {
                if (inst->Opcode() != IrOpcode::SetAttribute) continue;
                const auto index = inst->Flags<ExportFlags>().index;
                if (index < state.program.Metadata().exportInfo.size() && state.program.Metadata().exportInfo[index].vm) state.validMask = true;
            }
        }
    }
    state.exit = state.validMask ? "if (!aps5_valid) discard_fragment(); return aps5_out;" : "return aps5_out;";
}

std::string StageBuiltin(State& state, StageInputKind kind, std::uint32_t component) {
    const auto stage = StageOf(state);
    const auto argument = [&](const char* name, const char* declaration) {
        Argument(state, name, declaration);
        return std::string(name);
    };
    if (stage == IrShaderStage::Vertex) {
        switch (kind) {
        case StageInputKind::VertexIndex: return argument("aps5_vertex_id", "uint aps5_vertex_id [[vertex_id]]");
        case StageInputKind::InstanceIndex: return argument("aps5_instance_id", "uint aps5_instance_id [[instance_id]]");
        case StageInputKind::BaseInstance: return argument("aps5_base_instance", "uint aps5_base_instance [[base_instance]]");
        default: break;
        }
    } else if (stage == IrShaderStage::Pixel) {
        switch (kind) {
        case StageInputKind::FragCoord: return "as_type<uint>(" + argument("aps5_frag_coord", "float4 aps5_frag_coord [[position]]") + "." + Swizzle(component) + ")";
        case StageInputKind::FrontFacing: return "(" + argument("aps5_front", "bool aps5_front [[front_facing]]") + " ? 0x3f800000u : 0xbf800000u)";
        case StageInputKind::SampleId: return argument("aps5_sample", "uint aps5_sample [[sample_id]]");
        case StageInputKind::PrimitiveId: return argument("aps5_primitive", "uint aps5_primitive [[primitive_id]]");
        case StageInputKind::Layer: return argument("aps5_layer", "uint aps5_layer [[render_target_array_index]]");
        case StageInputKind::HelperInvocation: return "(simd_is_helper_thread() ? 1u : 0u)";
        case StageInputKind::BaryCoordSmooth: return "as_type<uint>(" + argument("aps5_bary", "float3 aps5_bary [[barycentric_coord, center_perspective]]") + "." + Swizzle(component + 1u) + ")";
        case StageInputKind::BaryCoordNoPerspective: return "as_type<uint>(" + argument("aps5_bary_linear", "float3 aps5_bary_linear [[barycentric_coord, center_no_perspective]]") + "." + Swizzle(component + 1u) + ")";
        case StageInputKind::RasterPrimitive: return Constant(state, PipelineSpecialization::RasterPrimitive, static_cast<std::uint32_t>(RasterPrimitive::Triangle));
        case StageInputKind::LayerExported: return Constant(state, PipelineSpecialization::LayerExported, 0u);
        default: break;
        }
    }
    throw MslUnsupported("MSL backend: stage input " + std::to_string(static_cast<std::uint32_t>(kind)) + " is not emitted for this stage");
}

bool EmitStage(State& state, const IrValue& inst) {
    const auto stage = StageOf(state);
    switch (inst.Opcode()) {
    case IrOpcode::GetAttribute:
        if (stage != IrShaderStage::Vertex && stage != IrShaderStage::Pixel) Fail(inst, "reads an attribute outside a vertex or pixel program");
        Define(state, inst, Attribute(state, inst));
        return true;
    case IrOpcode::GetInterpolationParameter:
        if (stage != IrShaderStage::Pixel) Fail(inst, "reads an interpolation parameter outside a pixel program");
        Define(state, inst, InterpolationParameter(state, inst));
        return true;
    case IrOpcode::InterpolateHostP1:
        DefineZero(state, inst);
        return true;
    case IrOpcode::InterpolateHostP2:
        return true;
    case IrOpcode::SetAttribute:
        if (stage != IrShaderStage::Vertex && stage != IrShaderStage::Pixel) Fail(inst, "exports outside a vertex or pixel program");
        Export(state, inst);
        return true;
    case IrOpcode::GetInterpolationParameterF16:
    case IrOpcode::FPInterpolateF32:
    case IrOpcode::FPInterpolateF16:
        throw MslUnsupported("MSL backend: 16-bit interpolation reads per-vertex values, which Metal fragment functions cannot do (section S7)");
    case IrOpcode::Discard:
        Line(state, "discard_fragment();");
        return true;
    case IrOpcode::EmitVertex:
        throw MslUnsupported("MSL backend: geometry emission is section S7");
    default:
        return false;
    }
}

std::string StageHeader(const State& state, const std::string& parameters) {
    const bool pixel = StageOf(state) == IrShaderStage::Pixel;
    std::string text;
    for (const auto& [name, declaration] : state.stageGlobals) text += declaration + "\n";
    const bool input = !state.stageInputs.empty();
    if (input) {
        text += "struct aps5_stage_in {\n";
        for (const auto& [name, declaration] : state.stageInputs) text += "    " + declaration + ";\n";
        text += "};\n";
    }
    text += "struct aps5_stage_out {\n";
    for (const auto& [name, declaration] : state.stageOutputs) text += "    " + declaration + ";\n";
    text += "};\n";
    std::vector<std::string> arguments;
    if (input) arguments.push_back("aps5_stage_in aps5_in [[stage_in]]");
    for (const auto& [name, declaration] : state.stageArguments) arguments.push_back(declaration);
    std::string all;
    for (const auto& argument : arguments) all += "\n    " + argument + ",";
    if (parameters.empty() && !all.empty()) all.pop_back();
    all += parameters;
    if (pixel && PixelInfo(state).psEarlyZ && !PixelInfo(state).psAlphaToCoverage && !state.stageOutputs.contains("depth")) text += "[[early_fragment_tests]] ";
    text += std::string(pixel ? "fragment" : "vertex") + " aps5_stage_out " + std::string(MslEntryPoint) + "(" + all + ")\n{\n";
    text += "    aps5_stage_out aps5_out = {};\n";
    if (state.stageOutputs.contains("mask")) text += "    aps5_out.mask = 0xffffffffu;\n";
    if (state.validMask) text += "    bool aps5_valid = true;\n";
    return text;
}

}
