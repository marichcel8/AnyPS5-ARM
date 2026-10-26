#include "MslState.hpp"
#include "BdaAbi.hpp"
#include "MetalAbi.hpp"
#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>

namespace ShaderRecompiler {

namespace Msl {

void Fail(const IrValue& inst, std::string_view reason) {
    throw MslUnsupported("MSL backend: " + std::string(IrOpcodeName(inst.Opcode())) + " (value " + std::to_string(inst.Id()) + ") " + std::string(reason));
}

void Fail(std::string_view reason) {
    throw MslUnsupported("MSL backend: " + std::string(reason));
}

std::uint32_t& CurrentHalf() {
    static thread_local std::uint32_t half = 0u;
    return half;
}

std::string HalfSuffix() {
    return CurrentHalf() == 0u ? std::string() : "_" + std::to_string(CurrentHalf());
}

std::string Name(const IrValue& value) {
    return "v" + std::to_string(value.Id()) + HalfSuffix();
}

std::string U32(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), value < 10u ? "%uu" : "0x%xu", value);
    return text;
}

std::string TypeName(IrType type) {
    switch (type) {
    case IrType::Bool: return "bool";
    case IrType::U8:
    case IrType::U16:
    case IrType::U32:
    case IrType::S32:
    case IrType::F16: return "uint";
    case IrType::F32: return "float";
    case IrType::U64:
    case IrType::S64: return "ulong";
    case IrType::Vec2U32: return "uint2";
    case IrType::Vec3U32: return "uint3";
    case IrType::Vec4U32: return "uint4";
    case IrType::Vec2F32: return "float2";
    case IrType::Vec3F32: return "float3";
    case IrType::Vec4F32: return "float4";
    default: return {};
    }
}

namespace {

std::string FloatLiteral(std::uint32_t bits) {
    const auto value = std::bit_cast<float>(bits);
    if (!std::isfinite(value) || bits == 0x80000000u) return "as_type<float>(" + U32(bits) + ")";
    char text[48];
    std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(value));
    std::string literal(text);
    if (literal.find_first_of(".e") == std::string::npos) literal += ".0";
    return literal + "f";
}

}

std::string Literal(const IrValue& value) {
    switch (value.Type()) {
    case IrType::Bool: return value.ImmediateBool() ? "true" : "false";
    case IrType::F32: return FloatLiteral(value.ImmediateU32());
    case IrType::U64:
    case IrType::S64: {
        char text[32];
        std::snprintf(text, sizeof(text), "0x%llxul", static_cast<unsigned long long>(value.ImmediateU64()));
        return text;
    }
    case IrType::U8: return U32(value.ImmediateU8());
    case IrType::U16: return U32(value.ImmediateU16());
    case IrType::F16: return U32(value.ImmediateF16Bits());
    case IrType::U32:
    case IrType::S32: return U32(value.ImmediateU32());
    default: Fail(value, "is a constant of a type MSL does not hold");
    }
}

std::string Value(State& state, const IrValue* value) {
    if (value == nullptr) Fail("an instruction has a null argument");
    const IrValue* resolved = value->Resolve();
    if (resolved->HasImmediate() && resolved->ArgumentCount() == 0u) return Literal(*resolved);
    if (resolved->Opcode() == IrOpcode::UndefU1) return "false";
    if (resolved->Opcode() == IrOpcode::UndefU8 || resolved->Opcode() == IrOpcode::UndefU16 || resolved->Opcode() == IrOpcode::UndefU32) return "0u";
    if (resolved->Opcode() == IrOpcode::UndefU64) return "0ul";
    if (!state.hoisted.contains(resolved) && !state.defined.contains(resolved)) Fail(*resolved, "is used before it is defined");
    return Name(*resolved);
}

std::string Arg(State& state, const IrValue& inst, std::size_t index) {
    return Value(state, inst.Argument(index));
}

std::string Temp(State& state) {
    return "t" + std::to_string(state.temps++);
}

void Line(State& state, std::string_view text) {
    state.body.append(state.depth * 4u, ' ');
    state.body += text;
    state.body += '\n';
}

void Open(State& state, std::string_view text) {
    Line(state, text);
    ++state.depth;
}

void Close(State& state, std::string_view text) {
    --state.depth;
    Line(state, text);
}

void Define(State& state, const IrValue& inst, std::string_view expression) {
    const auto type = TypeName(inst.Type());
    if (type.empty()) Fail(inst, "has a result type MSL does not hold");
    if (state.hoisted.contains(&inst)) {
        Line(state, Name(inst) + " = " + std::string(expression) + ";");
    } else {
        Line(state, "const " + type + " " + Name(inst) + " = " + std::string(expression) + ";");
    }
    state.defined.insert(&inst);
}

void DefineZero(State& state, const IrValue& inst) {
    const auto type = TypeName(inst.Type());
    Define(state, inst, type == "bool" ? std::string("false") : type + "(0)");
}

const MemoryInfo& Memory(const State& state, const IrValue& inst) {
    const auto& infos = state.program.Resources().memoryInfo;
    const auto index = inst.Flags<MemoryFlags>().index;
    if (index >= infos.size()) Fail(inst, "has no memory info");
    return infos[index];
}

namespace {

bool IsResourceType(IrType type) {
    switch (type) {
    case IrType::SrtResource:
    case IrType::BufferResource:
    case IrType::AddressResource:
    case IrType::ImageResource:
    case IrType::SamplerResource:
    case IrType::ImageAddress: return true;
    default: return false;
    }
}

bool IsLeaf(const IrValue* value) {
    if (value->HasImmediate() && value->ArgumentCount() == 0u) return true;
    switch (value->Type()) {
    case IrType::ScalarReg:
    case IrType::VectorReg:
    case IrType::Label:
    case IrType::Void: return true;
    default: break;
    }
    switch (value->Opcode()) {
    case IrOpcode::UndefU1:
    case IrOpcode::UndefU8:
    case IrOpcode::UndefU16:
    case IrOpcode::UndefU32:
    case IrOpcode::UndefU64: return true;
    default: return false;
    }
}

class Hoister {
public:
    explicit Hoister(State& state) : state(state) {}

    void Run() {
        for (const auto* block : state.program.BlockOrder()) {
            for (const auto* inst : block->Instructions()) {
                if (inst->IsPhi()) {
                    hoist(inst);
                    for (std::size_t index = 0; index < inst->ArgumentCount(); ++index) use(inst->Argument(index), inst->PhiBlock(index));
                    continue;
                }
                for (std::size_t index = 0; index < inst->ArgumentCount(); ++index) use(inst->Argument(index), block);
            }
        }
        const auto& blocks = state.program.BlockOrder();
        const auto& infos = state.program.Metadata().blockInfo;
        for (std::size_t index = 0; index < blocks.size() && index < infos.size(); ++index) {
            if (infos[index].condition != nullptr) use(infos[index].condition, blocks[index]);
        }
    }

private:
    void hoist(const IrValue* value) {
        if (state.hoisted.insert(value).second) state.hoistOrder.push_back(value);
    }

    void use(const IrValue* argument, const IrBlock* at) {
        if (argument == nullptr) return;
        const IrValue* value = argument->Resolve();
        if (IsLeaf(value)) return;
        if (IsResourceType(value->Type())) {
            for (std::size_t index = 0; index < value->ArgumentCount(); ++index) use(value->Argument(index), at);
            return;
        }
        if (value->Parent() != at) hoist(value);
    }

    State& state;
};

std::uint32_t WaveLdsScope(const IrProgram& program, const ShaderWorkgroupInputInfo* workgroup) {
    if (workgroup == nullptr) return 0u;
    bool writes = false;
    for (const auto* block : program.BlockOrder()) {
        for (const auto* inst : block->Instructions()) {
            const auto access = SharedAccessOf(inst->Opcode());
            writes |= access != SharedAccess::None && access != SharedAccess::Read;
        }
    }
    if (!writes) return 0u;
    const auto threads = std::max(workgroup->threadsNum[0], 1u) * std::max(workgroup->threadsNum[1], 1u) * std::max(workgroup->threadsNum[2], 1u);
    return threads <= program.WaveSize() ? 1u : 2u;
}

bool HasSharedAtomic64(const IrProgram& program) {
    for (const auto* block : program.BlockOrder()) {
        for (const auto* inst : block->Instructions()) {
            const auto name = IrOpcodeName(inst->Opcode());
            if (name.starts_with("SharedAtomic") && name.ends_with("64")) return true;
        }
    }
    return false;
}

bool HasBarrier(const IrProgram& program) {
    for (const auto* block : program.BlockOrder()) {
        for (const auto* inst : block->Instructions()) {
            if (inst->Opcode() == IrOpcode::Barrier) return true;
        }
    }
    return false;
}

std::string Parameters(const State& state) {
    std::vector<std::string> parameters;
    const auto has = [&](std::uint32_t feature) { return (state.features & feature) != 0u; };
    const auto slot = [](MetalAbi::BufferSlot slot) { return std::to_string(MetalAbi::SlotIndex(slot)); };
    if (has(ShaderData)) parameters.push_back("device const uint* aps5_shader_data [[buffer(" + slot(MetalAbi::BufferSlot::ShaderData) + ")]]");
    if (has(PushData)) parameters.push_back("constant uint* aps5_push [[buffer(" + slot(MetalAbi::BufferSlot::PushConstants) + ")]]");
    if (has(Bda)) parameters.push_back("device const aps5_bda_header* aps5_bda_table [[buffer(" + slot(MetalAbi::BufferSlot::BdaPagetable) + ")]]");
    if (has(Fault)) parameters.push_back("device atomic_uint* aps5_fault_buffer [[buffer(" + slot(MetalAbi::BufferSlot::FaultBuffer) + ")]]");
    if (has(Gds)) parameters.push_back("device atomic_uint* aps5_gds [[buffer(" + slot(MetalAbi::BufferSlot::Gds) + ")]]");
    if (has(FlattenedSrt)) parameters.push_back("device const uint* aps5_srt [[buffer(" + slot(MetalAbi::BufferSlot::FlattenedSrt) + ")]]");
    if (has(GuestBuffers)) parameters.push_back("device const aps5_guest_buffer* aps5_buffers [[buffer(" + slot(MetalAbi::BufferSlot::GuestBuffers) + ")]]");
    if (has(Images)) parameters.push_back("device const ulong* aps5_images [[buffer(" + slot(MetalAbi::BufferSlot::GuestImages) + ")]]");
    if (has(Samplers)) parameters.push_back("device const ulong* aps5_samplers [[buffer(" + slot(MetalAbi::BufferSlot::GuestSamplers) + ")]]");
    if (has(Tables)) {
        parameters.push_back("device const ulong* aps5_image_table [[buffer(" + slot(MetalAbi::BufferSlot::ImageTable) + ")]]");
        parameters.push_back("device const ulong* aps5_storage_table [[buffer(" + slot(MetalAbi::BufferSlot::ImageStorageTable) + ")]]");
        parameters.push_back("device const ulong* aps5_sampler_table [[buffer(" + slot(MetalAbi::BufferSlot::SamplerTable) + ")]]");
        parameters.push_back("device const uint* aps5_table_map [[buffer(" + slot(MetalAbi::BufferSlot::ImageTableMap) + ")]]");
    }
    if (has(WorkgroupId)) parameters.push_back("uint3 aps5_group [[threadgroup_position_in_grid]]");
    if (has(LocalId)) parameters.push_back("uint3 aps5_local [[thread_position_in_threadgroup]]");
    if (has(GlobalId)) parameters.push_back("uint3 aps5_global [[thread_position_in_grid]]");
    if (has(LocalIndex)) parameters.push_back("uint aps5_index [[thread_index_in_threadgroup]]");
    if (has(Lane) && state.program.Resources().stage != IrShaderStage::Vertex) parameters.push_back("uint aps5_lane [[thread_index_in_simdgroup]]");
    std::string text;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        text += index == 0u ? "\n    " : ",\n    ";
        text += parameters[index];
    }
    return text;
}

}

void EmitBlockBody(State& state, const IrBlock& block) {
    bool ldsWritten = true;
    bool ldsRead = false;
    for (const IrValue* inst : block.Instructions()) {
        if (inst->IsPhi()) continue;
        if (state.waveLds != 0u) {
            const auto access = SharedAccessOf(inst->Opcode());
            if (inst->Opcode() == IrOpcode::Barrier) {
                ldsWritten = false;
                ldsRead = false;
            } else if (access != SharedAccess::None) {
                const bool writes = access != SharedAccess::Read;
                if (ldsWritten || (writes && ldsRead)) {
                    Line(state, state.waveLds == 1u ? "threadgroup_barrier(mem_flags::mem_threadgroup);" : "simdgroup_barrier(mem_flags::mem_threadgroup);");
                    ldsWritten = false;
                    ldsRead = false;
                }
                ldsWritten |= writes;
                ldsRead |= access != SharedAccess::Write;
            }
        }
        if (state.lanes == 1u) {
            EmitInstruction(state, *inst);
            continue;
        }
        if (IsCrossLane(inst->Opcode())) {
            EmitWave(state, *inst);
            continue;
        }
        ForEachHalf(state, [&](std::uint32_t half) {
            if (half == 0u || inst->Opcode() != IrOpcode::Barrier) EmitInstruction(state, *inst);
        });
    }
}

void EmitInstruction(State& state, const IrValue& inst) {
    switch (inst.Opcode()) {
    case IrOpcode::Identity:
    case IrOpcode::Phi:
    case IrOpcode::Void:
    case IrOpcode::Reference:
    case IrOpcode::ReferenceU32:
    case IrOpcode::ControlNop:
    case IrOpcode::Waitcnt:
    case IrOpcode::Sendmsg:
    case IrOpcode::TtraceData:
    case IrOpcode::InstPrefetch:
    case IrOpcode::GetSrtResource:
    case IrOpcode::GetBufferResource:
    case IrOpcode::GetAddressResource:
    case IrOpcode::GetScratchResource:
    case IrOpcode::GetImageResource:
    case IrOpcode::GetSamplerResource:
    case IrOpcode::MakeImageAddress:
    case IrOpcode::UndefU1:
    case IrOpcode::UndefU8:
    case IrOpcode::UndefU16:
    case IrOpcode::UndefU32:
    case IrOpcode::UndefU64:
        return;
    case IrOpcode::Barrier:
        if (IsGeometry(state.program.Resources().stage) && EmitGeometry(state, inst)) return;
        Line(state, "threadgroup_barrier(mem_flags::mem_threadgroup);");
        return;
    case IrOpcode::UnlinkedCall:
        state.features |= Fault;
        Line(state, "aps5_fault(aps5_c, " + U32(static_cast<std::uint32_t>(BdaAbi::FaultReason::UnlinkedCall)) + ", (ulong(" + Arg(state, inst, 1) + ") << 32u) | ulong(" + Arg(state, inst, 0) + "), 0u, " + Arg(state, inst, 2) + ");");
        if (state.stopOnUnlinkedCall) Line(state, "return;");
        return;
    default:
        break;
    }
    if (EmitStage(state, inst) || EmitWave(state, inst) || EmitAlu(state, inst) || EmitMemory(state, inst)) return;
    Fail(inst, "has no MSL emitter");
}

}

std::string MslEmitter::Emit(const IrProgram& program, const ShaderStageInputInfo& inputInfo, const CompiledBindingLayout& bindings, const MslTarget& target) const {
    using namespace Msl;
    const auto stage = program.Resources().stage;
    const bool compute = stage == IrShaderStage::Compute;
    const bool geometry = IsGeometry(stage);
    if (!compute && !geometry && stage != IrShaderStage::Vertex && stage != IrShaderStage::Pixel) throw MslUnsupported("MSL backend: only compute, vertex, pixel, tessellation and mesh programs are emitted");
    if (target.simdgroupSize != 32u || (program.WaveSize() != 32u && program.WaveSize() != 64u)) throw MslUnsupported("MSL backend: only wave32 and wave64 programs on 32-wide SIMD-groups are emitted");
    if (program.Info().sharedMemoryBytes != 0u) throw MslUnsupported("MSL backend: LDS in device memory is not emitted (section S9)");
    State state{program, inputInfo, bindings.layout, target};
    const bool mesh = stage == IrShaderStage::Mesh;
    if (mesh && inputInfo.vertex == nullptr) throw MslUnsupported("MSL backend: the mesh program has no mesh input info");
    const ShaderWorkgroupInputInfo* workgroup = mesh ? static_cast<const ShaderWorkgroupInputInfo*>(&inputInfo.vertex->mesh) : inputInfo.compute;
    state.lanes = program.WaveSize() == 64u && !geometry && (compute || mesh) ? 2u : 1u;
    if (state.lanes == 2u && workgroup == nullptr) throw MslUnsupported("MSL backend: a wave64 program needs its workgroup size");
    state.ldsDwords = workgroup != nullptr ? workgroup->ldsSizeDwords : 0u;
    state.scratchDwords = program.Info().scratchDwords;
    state.windowDwords = program.Metadata().vectorWindowSize;
    state.waveLds = WaveLdsScope(program, workgroup);
    state.lds64 = HasSharedAtomic64(program);
    state.stopAllowed = compute && state.waveLds == 0u && !HasBarrier(program) && state.lanes == 1u;
    state.stopOnUnlinkedCall = compute && state.waveLds == 0u && !HasBarrier(program);
    if (geometry) {
        PrepareGeometry(state);
    } else if (!compute) {
        PrepareStage(state);
    }
    Hoister{state}.Run();
    AnalyzePrecision(state);
    EmitFlow(state);
    if (geometry) CheckGeometry(state);
    if (stage == IrShaderStage::Vertex && (state.features & Permute) != 0u) throw MslUnsupported("MSL backend: vertex functions cannot run cross-lane permutes");
    if (!compute && !geometry && (state.features & (Lds | WorkgroupId | LocalId | GlobalId | LocalIndex)) != 0u) throw MslUnsupported("MSL backend: a vertex or pixel program uses threadgroup memory or compute builtins");

    std::string declarations;
    const auto declare = [&](std::string_view text) {
        declarations += "    ";
        declarations += text;
        declarations += '\n';
    };
    if ((state.features & (Bda | Fault | GuestBuffers)) != 0u) {
        const auto pointer = [&](std::uint32_t feature, std::string_view name) { return (state.features & feature) != 0u ? std::string(name) : std::string("nullptr"); };
        declare("aps5_ctx aps5_c = {" + pointer(Bda, "aps5_bda_table") + ", " + pointer(Fault, "aps5_fault_buffer") + ", " + pointer(GuestBuffers, "aps5_buffers") + ", " + U32(static_cast<std::uint32_t>(program.Resources().stage)) + ", false};");
    }
    if (stage == IrShaderStage::Vertex && (state.features & Lane) != 0u) declare("const uint aps5_lane = 0u;");
    if ((state.features & Lds) != 0u) declare("threadgroup uint aps5_lds[" + std::to_string(std::max(state.ldsDwords, 1u)) + "];");
    if ((state.features & Lds64) != 0u) declare("threadgroup atomic_uint aps5_lds_lock;");
    ForEachHalf(state, [&](std::uint32_t) {
        if ((state.features & Scratch) != 0u) declare("uint aps5_scratch" + HalfSuffix() + "[" + std::to_string(std::max(state.scratchDwords, 1u)) + "];");
        if ((state.features & VectorWindow) != 0u) declare("uint aps5_window" + HalfSuffix() + "[" + std::to_string(std::max(state.windowDwords, 1u)) + "] = {};");
        for (const auto* value : state.hoistOrder) {
            const auto type = TypeName(value->Type());
            if (type.empty()) Fail(*value, "is carried across blocks with a type MSL does not hold");
            declare(type + " " + Name(*value) + " = " + (type == "bool" ? std::string("false") : type + "(0)") + ";");
        }
    });

    std::string text = Prelude(state);
    if (geometry) return GeometryProgram(state, text, declarations, Parameters(state));
    if (!compute) {
        text += StageHeader(state, Parameters(state));
        text += declarations;
        if (stage == IrShaderStage::Pixel && (state.target.experiment & MslExperimentConstantOutput) != 0u) {
            text += "    return aps5_out;\n}\n";
            return text;
        }
        text += state.body;
        if (!state.body.ends_with(state.exit + "\n")) text += "    " + state.exit + "\n";
        text += "}\n";
        return text;
    }
    if (workgroup != nullptr) {
        const auto threads = std::max(workgroup->threadsNum[0], 1u) * std::max(workgroup->threadsNum[1], 1u) * std::max(workgroup->threadsNum[2], 1u);
        text += "[[max_total_threads_per_threadgroup(" + std::to_string(state.lanes == 2u ? (threads + 63u) / 64u * 32u : threads) + ")]] ";
    }
    text += "kernel void " + std::string(MslEntryPoint) + "(" + Parameters(state) + ")\n{\n";
    text += declarations;
    if ((state.features & Lds64) != 0u) text += "    if (aps5_index == 0u) atomic_store_explicit(&aps5_lds_lock, 0u, memory_order_relaxed);\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    text += state.body;
    text += "}\n";
    return text;
}

SpirvTarget MslFrontendTarget(const MslTarget& target) {
    static constexpr std::array<std::uint32_t, 3> Capabilities{spv::CapabilityFloat16, spv::CapabilityRoundingModeRTE, spv::CapabilityDenormPreserve};
    SpirvTarget frontend{};
    frontend.vulkanVersion = 0u;
    frontend.spirvVersion = 0x00010600u;
    frontend.subgroupSize = target.simdgroupSize;
    frontend.bdaAbiVersion = target.bdaAbiVersion;
    frontend.supportedCapabilities = Capabilities;
    frontend.fragmentShaderBarycentricEnabled = target.fragmentBarycentric;
    frontend.maxWorkgroupSize = target.maxThreadgroupSize;
    frontend.maxWorkgroupInvocations = target.maxThreadsPerThreadgroup;
    frontend.maxWorkgroupSharedMemoryBytes = target.maxThreadgroupMemoryBytes;
    frontend.depthRangeRemap = true;
    frontend.mesh = target.mesh;
    frontend.tessellation = target.tessellation;
    frontend.subgroupStages = target.subgroupStages;
    frontend.relaxedFloat = target.relaxedFloat;
    frontend.vertexSubgroups = target.vertexSimdgroups;
    frontend.float16DenormPreserve = target.float16DenormPreserve;
    frontend.spannedWaves = target.spannedWaves;
    return frontend;
}

}
