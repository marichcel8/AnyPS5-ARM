#include "MslState.hpp"
#include <algorithm>

namespace ShaderRecompiler::Msl {

namespace {

std::string Cast(const IrValue& inst, std::string expression, IrType from, IrType to) {
    const auto source = TypeName(from);
    const auto target = TypeName(to);
    if (source == target || source.empty() || target.empty()) return expression;
    const auto sameSize = [&](std::string_view left, std::string_view right) {
        return (source == left && target == right) || (source == right && target == left);
    };
    if (sameSize("uint", "float") || sameSize("ulong", "uint2") || sameSize("uint2", "float2") || sameSize("uint4", "float4")) return "as_type<" + target + ">(" + expression + ")";
    Fail(inst, "mixes " + source + " and " + target + " values");
}

void Out(State& state, const IrValue& inst, std::string_view expression) {
    auto produced = IrOpcodeType(inst.Opcode());
    if (produced == IrType::Opaque) produced = inst.Type();
    Define(state, inst, Cast(inst, std::string(expression), produced, inst.Type()));
}

std::string ShaderDataWord(State& state, std::uint32_t index) {
    if (state.layout.UsesPushData()) {
        state.features |= PushData;
        return "aps5_push[" + std::to_string(index) + "]";
    }
    state.features |= ShaderData;
    return "aps5_shader_data[" + std::to_string(index) + "]";
}

std::string Component(std::uint32_t index) {
    static constexpr std::string_view Names[4] = {"x", "y", "z", "w"};
    if (index >= 4u) Fail("component index is out of range");
    return std::string(Names[index]);
}

std::string Immediate(const IrValue& inst, std::size_t index) {
    const IrValue* value = inst.Argument(index)->Resolve();
    if (!value->HasImmediate()) Fail(inst, "needs an immediate operand");
    return std::to_string(value->ImmediateU32());
}

std::uint32_t ImmediateValue(const IrValue& inst, std::size_t index) {
    const IrValue* value = inst.Argument(index)->Resolve();
    if (!value->HasImmediate()) Fail(inst, "needs an immediate operand");
    return value->ImmediateU32();
}

std::string Nan(std::string_view value) {
    return "aps5_nan(as_type<uint>(" + std::string(value) + "))";
}

std::string GuestIndex() {
    return "(aps5_index + (aps5_index & ~31u) + " + U32(CurrentHalf() * 32u) + ")";
}

std::string Builtin(State& state, const IrValue& inst) {
    const auto kind = static_cast<StageInputKind>(ImmediateValue(inst, 0));
    const auto component = ImmediateValue(inst, 1);
    switch (kind) {
    case StageInputKind::WorkgroupId:
        state.features |= WorkgroupId;
        return "aps5_group." + Component(component);
    case StageInputKind::LocalInvocationId:
    case StageInputKind::GlobalInvocationId: {
        if (state.lanes == 1u) {
            state.features |= kind == StageInputKind::LocalInvocationId ? LocalId : GlobalId;
            return std::string(kind == StageInputKind::LocalInvocationId ? "aps5_local." : "aps5_global.") + Component(component);
        }
        const auto* workgroup = state.input.compute;
        if (workgroup == nullptr || component >= 3u) Fail(inst, "needs the workgroup size");
        std::uint32_t divisor = 1u;
        for (std::uint32_t axis = 0; axis < component; ++axis) divisor *= std::max(workgroup->threadsNum[axis], 1u);
        const auto size = std::max(workgroup->threadsNum[component], 1u);
        state.features |= LocalIndex;
        const auto local = "((" + GuestIndex() + " / " + U32(divisor) + ") % " + U32(size) + ")";
        if (kind == StageInputKind::LocalInvocationId) return local;
        state.features |= WorkgroupId;
        return "(" + local + " + aps5_group." + Component(component) + " * " + U32(size) + ")";
    }
    case StageInputKind::LocalInvocationIndex:
        state.features |= LocalIndex;
        return state.lanes == 1u ? std::string("aps5_index") : GuestIndex();
    case StageInputKind::DispatchThreadLimit:
        return ShaderDataWord(state, state.layout.DispatchThreadLimitDword() + component);
    default:
        return StageBuiltin(state, kind, component);
    }
}

std::string UserData(State& state, const IrValue& inst) {
    const auto reg = inst.Argument(0)->Register().index;
    const auto& registers = state.layout.userDataRegisters;
    const auto found = std::lower_bound(registers.begin(), registers.end(), reg);
    if (found == registers.end() || *found != reg) return "0u";
    return ShaderDataWord(state, static_cast<std::uint32_t>(found - registers.begin()));
}

std::string Bits(State& state, const IrValue& inst, std::size_t index) {
    const IrValue* value = inst.Argument(index)->Resolve();
    if (value->HasImmediate() && value->ArgumentCount() == 0u && value->Type() == IrType::F32) return U32(value->ImmediateU32());
    return "as_type<uint>(" + Typed(state, inst, index) + ")";
}

bool EmitPrecise(State& state, const IrValue& inst) {
    const auto level = Precision(state, inst);
    if (level == PrecisionFast) return false;
    const auto helper = [&](std::string_view name, std::size_t count) {
        state.features |= level == PrecisionStrict ? StrictFloat : IeeeFloat;
        std::string text = std::string(level == PrecisionStrict ? "aps5_s_" : "aps5_i_") + std::string(name) + "(";
        for (std::size_t index = 0; index < count; ++index) text += (index == 0u ? "" : ", ") + Bits(state, inst, index);
        return text + ")";
    };
    const auto value = [&](std::string_view name, std::size_t count) { Out(state, inst, "as_type<float>(" + helper(name, count) + ")"); };
    const auto test = [&](std::string_view name) { Out(state, inst, helper(name, 2)); };
    switch (inst.Opcode()) {
    case IrOpcode::FPAdd32: value("add", 2); return true;
    case IrOpcode::FPSub32: value("sub", 2); return true;
    case IrOpcode::FPMul32: value("mul", 2); return true;
    case IrOpcode::FPMad32: value("mad", 3); return true;
    case IrOpcode::FPFma32: value("fma", 3); return true;
    case IrOpcode::FPMin32: value("min", 2); return true;
    case IrOpcode::FPMax32: value("max", 2); return true;
    case IrOpcode::FPMinTri32: value("min3", 3); return true;
    case IrOpcode::FPMaxTri32: value("max3", 3); return true;
    case IrOpcode::FPMedTri32: value("med3", 3); return true;
    case IrOpcode::FPRecip32:
    case IrOpcode::FPRecipIFlag32: value("recip", 1); return true;
    case IrOpcode::FPRecipSqrt32: value("rsqrt", 1); return true;
    case IrOpcode::FPSqrt: value("sqrt", 1); return true;
    case IrOpcode::FPSin: value("sin", 1); return true;
    case IrOpcode::FPCos: value("cos", 1); return true;
    case IrOpcode::FPExp2:
    case IrOpcode::FPLog2:
        if ((state.target.experiment & MslExperimentNoPow) != 0u) {
            Out(state, inst, Typed(state, inst, 0));
            return true;
        }
        value(inst.Opcode() == IrOpcode::FPExp2 ? "exp2" : "log2", 1);
        return true;
    case IrOpcode::FPRoundEven32: value("rint", 1); return true;
    case IrOpcode::FPFloor32: value("floor", 1); return true;
    case IrOpcode::FPCeil32: value("ceil", 1); return true;
    case IrOpcode::FPTrunc32: value("trunc", 1); return true;
    case IrOpcode::FPFract32: value("fract", 1); return true;
    case IrOpcode::FPOrdEqual32: test("eq"); return true;
    case IrOpcode::FPOrdNotEqual32: test("one"); return true;
    case IrOpcode::FPOrdLessThan32: test("lt"); return true;
    case IrOpcode::FPOrdGreaterThan32: test("gt"); return true;
    case IrOpcode::FPOrdLessThanEqual32: test("le"); return true;
    case IrOpcode::FPOrdGreaterThanEqual32: test("ge"); return true;
    case IrOpcode::FPUnordEqual32: test("ueq"); return true;
    case IrOpcode::FPUnordNotEqual32: test("une"); return true;
    case IrOpcode::FPUnordLessThan32: test("ult"); return true;
    case IrOpcode::FPUnordGreaterThan32: test("ugt"); return true;
    case IrOpcode::FPUnordLessThanEqual32: test("ule"); return true;
    case IrOpcode::FPUnordGreaterThanEqual32: test("uge"); return true;
    case IrOpcode::Select:
    case IrOpcode::SelectF32:
        if (inst.Type() != IrType::F32) return false;
        Out(state, inst, "as_type<float>(" + Typed(state, inst, 0) + " ? " + Bits(state, inst, 1) + " : " + Bits(state, inst, 2) + ")");
        return true;
    default: return false;
    }
}

bool EmitFloat64(State& state, const IrValue& inst) {
    const auto call = [&](std::string_view function, std::size_t count) {
        state.features |= Float64;
        std::string text = std::string(function) + "(";
        for (std::size_t index = 0; index < count; ++index) text += (index == 0u ? "" : ", ") + Typed(state, inst, index);
        Out(state, inst, text + ")");
    };
    switch (inst.Opcode()) {
    case IrOpcode::FPAdd64: call("aps5_f64_add", 2); return true;
    case IrOpcode::FPMul64: call("aps5_f64_mul", 2); return true;
    case IrOpcode::FPFma64: call("aps5_f64_fma", 3); return true;
    case IrOpcode::FPFmaScale64:
        state.features |= Float64;
        Out(state, inst, "aps5_f64_fma_scale(" + Typed(state, inst, 0) + ", " + Typed(state, inst, 1) + ", " + Typed(state, inst, 2) + ", int(" + Typed(state, inst, 3) + "))");
        return true;
    case IrOpcode::FPMin64: call("aps5_f64_min", 2); return true;
    case IrOpcode::FPMax64: call("aps5_f64_max", 2); return true;
    case IrOpcode::FPSaturate64: call("aps5_f64_saturate", 1); return true;
    case IrOpcode::FPLdexp64: call("aps5_f64_ldexp", 2); return true;
    case IrOpcode::FPRoundEven64: call("aps5_f64_round_even", 1); return true;
    case IrOpcode::FPFloor64: call("aps5_f64_floor", 1); return true;
    case IrOpcode::FPCeil64: call("aps5_f64_ceil", 1); return true;
    case IrOpcode::FPTrunc64: call("aps5_f64_trunc", 1); return true;
    case IrOpcode::FPFract64: call("aps5_f64_fract", 1); return true;
    case IrOpcode::FPFrexpMant64: call("aps5_f64_frexp_mant", 1); return true;
    case IrOpcode::FPFrexpExp64: call("aps5_f64_frexp_exp", 1); return true;
    case IrOpcode::FPRcp64: call("aps5_f64_rcp", 1); return true;
    case IrOpcode::FPRsq64: call("aps5_f64_rsq", 1); return true;
    case IrOpcode::FPSqrt64: call("aps5_f64_sqrt", 1); return true;
    case IrOpcode::FPTrigPreop64: call("aps5_f64_trig_preop", 2); return true;
    case IrOpcode::ConvertF32F64:
        state.features |= Float64;
        Out(state, inst, "as_type<float>(aps5_f64_to_f32(" + Typed(state, inst, 0) + "))");
        return true;
    case IrOpcode::ConvertF64F32:
        state.features |= Float64;
        Out(state, inst, "aps5_f64_from_f32(" + Bits(state, inst, 0) + ")");
        return true;
    case IrOpcode::ConvertF64S32: call("aps5_f64_from_i32", 1); return true;
    case IrOpcode::ConvertF64U32: call("aps5_f64_from_u32", 1); return true;
    case IrOpcode::ConvertS32F64: call("aps5_f64_to_i32", 1); return true;
    case IrOpcode::ConvertU32F64: call("aps5_f64_to_u32", 1); return true;
    case IrOpcode::FPDot2F32F16:
        state.features |= Float64 | Dot2;
        call("aps5_dot2_f32_f16", 3);
        return true;
    default: return false;
    }
}

}

std::string Typed(State& state, const IrValue& inst, std::size_t index) {
    const IrValue* value = inst.Argument(index)->Resolve();
    auto expected = index < IrOpcodeOperandCount(inst.Opcode()) ? IrOpcodeArgumentType(inst.Opcode(), index) : value->Type();
    if (expected == IrType::Opaque) expected = value->Type();
    return Cast(inst, Value(state, value), value->Type(), expected);
}

bool EmitAlu(State& state, const IrValue& inst) {
    const auto a = [&](std::size_t index) { return Typed(state, inst, index); };
    const auto binary = [&](std::string_view op) { Out(state, inst, a(0) + " " + std::string(op) + " " + a(1)); };
    const auto call = [&](std::string_view function, std::size_t count) {
        std::string text(function);
        text += "(";
        for (std::size_t index = 0; index < count; ++index) text += (index == 0u ? "" : ", ") + a(index);
        Out(state, inst, text + ")");
    };
    const auto signedCall = [&](std::string_view function, std::size_t count) {
        std::string text = "uint(" + std::string(function) + "(";
        for (std::size_t index = 0; index < count; ++index) text += (index == 0u ? "int(" : ", int(") + a(index) + ")";
        Out(state, inst, text + "))");
    };
    const auto signedCompare = [&](std::string_view op, std::string_view type) {
        Out(state, inst, std::string(type) + "(" + a(0) + ") " + std::string(op) + " " + std::string(type) + "(" + a(1) + ")");
    };
    const auto unordered = [&](std::string_view op) {
        Out(state, inst, "(" + Nan(a(0)) + " || " + Nan(a(1)) + " || " + a(0) + " " + std::string(op) + " " + a(1) + ")");
    };
    if (EmitPrecise(state, inst) || EmitFloat64(state, inst)) return true;
    switch (inst.Opcode()) {
    case IrOpcode::IAdd32:
    case IrOpcode::IAdd64: binary("+"); return true;
    case IrOpcode::ISub32:
    case IrOpcode::ISub64: binary("-"); return true;
    case IrOpcode::IMul32:
    case IrOpcode::IMul64: binary("*"); return true;
    case IrOpcode::UDiv32: binary("/"); return true;
    case IrOpcode::SMulHi: Out(state, inst, "uint(mulhi(int(" + a(0) + "), int(" + a(1) + ")))"); return true;
    case IrOpcode::UMulHi: call("mulhi", 2); return true;
    case IrOpcode::IAbs32: Out(state, inst, "(int(" + a(0) + ") < 0 ? 0u - " + a(0) + " : " + a(0) + ")"); return true;
    case IrOpcode::IAddCarry32: Out(state, inst, "uint2(" + a(0) + " + " + a(1) + ", " + a(0) + " + " + a(1) + " < " + a(0) + " ? 1u : 0u)"); return true;
    case IrOpcode::ShiftLeftLogical32: Out(state, inst, a(0) + " << (" + a(1) + " & 31u)"); return true;
    case IrOpcode::ShiftRightLogical32: Out(state, inst, a(0) + " >> (" + a(1) + " & 31u)"); return true;
    case IrOpcode::ShiftRightArithmetic32: Out(state, inst, "uint(int(" + a(0) + ") >> (" + a(1) + " & 31u))"); return true;
    case IrOpcode::ShiftLeftLogical64: Out(state, inst, a(0) + " << (" + a(1) + " & 63u)"); return true;
    case IrOpcode::ShiftRightLogical64: Out(state, inst, a(0) + " >> (" + a(1) + " & 63u)"); return true;
    case IrOpcode::ShiftRightArithmetic64: Out(state, inst, "ulong(long(" + a(0) + ") >> (" + a(1) + " & 63u))"); return true;
    case IrOpcode::BitwiseAnd32:
    case IrOpcode::BitwiseAnd64: binary("&"); return true;
    case IrOpcode::BitwiseOr32: binary("|"); return true;
    case IrOpcode::BitwiseXor32: binary("^"); return true;
    case IrOpcode::BitwiseNot32: Out(state, inst, "~" + a(0)); return true;
    case IrOpcode::BitReverse32: call("reverse_bits", 1); return true;
    case IrOpcode::BitCount32: call("popcount", 1); return true;
    case IrOpcode::BitCount64: Out(state, inst, "uint(popcount(" + a(0) + "))"); return true;
    case IrOpcode::FindUMsb32: call("aps5_msb", 1); return true;
    case IrOpcode::FindUMsb64: Out(state, inst, "(uint(" + a(0) + " >> 32u) != 0u ? aps5_msb(uint(" + a(0) + " >> 32u)) + 32u : aps5_msb(uint(" + a(0) + ")))"); return true;
    case IrOpcode::FindILsb32: call("aps5_lsb", 1); return true;
    case IrOpcode::SMin32: signedCall("min", 2); return true;
    case IrOpcode::UMin32: call("min", 2); return true;
    case IrOpcode::SMax32: signedCall("max", 2); return true;
    case IrOpcode::UMax32: call("max", 2); return true;
    case IrOpcode::SMinTri32: Out(state, inst, "uint(min(min(int(" + a(0) + "), int(" + a(1) + ")), int(" + a(2) + ")))"); return true;
    case IrOpcode::UMinTri32: Out(state, inst, "min(min(" + a(0) + ", " + a(1) + "), " + a(2) + ")"); return true;
    case IrOpcode::SMaxTri32: Out(state, inst, "uint(max(max(int(" + a(0) + "), int(" + a(1) + ")), int(" + a(2) + ")))"); return true;
    case IrOpcode::UMaxTri32: Out(state, inst, "max(max(" + a(0) + ", " + a(1) + "), " + a(2) + ")"); return true;
    case IrOpcode::SMedTri32: signedCall("aps5_smed3", 3); return true;
    case IrOpcode::UMedTri32: call("aps5_med3", 3); return true;
    case IrOpcode::SLessThan32: signedCompare("<", "int"); return true;
    case IrOpcode::SLessThan64: signedCompare("<", "long"); return true;
    case IrOpcode::SLessThanEqual32: signedCompare("<=", "int"); return true;
    case IrOpcode::SGreaterThan32: signedCompare(">", "int"); return true;
    case IrOpcode::SGreaterThanEqual32: signedCompare(">=", "int"); return true;
    case IrOpcode::ULessThan32:
    case IrOpcode::ULessThan64: binary("<"); return true;
    case IrOpcode::ULessThanEqual32: binary("<="); return true;
    case IrOpcode::UGreaterThan32:
    case IrOpcode::UGreaterThan64: binary(">"); return true;
    case IrOpcode::UGreaterThanEqual32: binary(">="); return true;
    case IrOpcode::IEqual32:
    case IrOpcode::IEqual64: binary("=="); return true;
    case IrOpcode::INotEqual32:
    case IrOpcode::INotEqual64: binary("!="); return true;
    case IrOpcode::LogicalOr: binary("||"); return true;
    case IrOpcode::LogicalAnd: binary("&&"); return true;
    case IrOpcode::LogicalXor: binary("!="); return true;
    case IrOpcode::LogicalNot: Out(state, inst, "!" + a(0)); return true;
    case IrOpcode::Select:
    case IrOpcode::SelectU1:
    case IrOpcode::SelectU32:
    case IrOpcode::SelectF32: Out(state, inst, a(0) + " ? " + a(1) + " : " + a(2)); return true;
    case IrOpcode::BitFieldInsert: call("insert_bits", 4); return true;
    case IrOpcode::BitFieldUExtract: call("extract_bits", 3); return true;
    case IrOpcode::BitFieldSExtract: Out(state, inst, "uint(extract_bits(int(" + a(0) + "), " + a(1) + ", " + a(2) + "))"); return true;
    case IrOpcode::WqmU64: Out(state, inst, "(ulong(aps5_wqm(uint(" + a(0) + " >> 32u))) << 32u) | ulong(aps5_wqm(uint(" + a(0) + ")))"); return true;
    case IrOpcode::CompositeConstructU64: Out(state, inst, "(ulong(" + a(1) + ") << 32u) | ulong(" + a(0) + ")"); return true;
    case IrOpcode::CompositeConstructU32x2: call("uint2", 2); return true;
    case IrOpcode::CompositeConstructU32x3: call("uint3", 3); return true;
    case IrOpcode::CompositeConstructU32x4: call("uint4", 4); return true;
    case IrOpcode::CompositeConstructF32x2: call("float2", 2); return true;
    case IrOpcode::CompositeExtractU64: Out(state, inst, ImmediateValue(inst, 1) == 0u ? "uint(" + a(0) + ")" : "uint(" + a(0) + " >> 32u)"); return true;
    case IrOpcode::CompositeExtractU32x2:
    case IrOpcode::CompositeExtractU32x3:
    case IrOpcode::CompositeExtractU32x4: Out(state, inst, a(0) + "." + Component(ImmediateValue(inst, 1))); return true;
    case IrOpcode::BitCastU16F16:
    case IrOpcode::BitCastF16U16: Out(state, inst, a(0)); return true;
    case IrOpcode::BitCastU32F32: Out(state, inst, "as_type<uint>(" + a(0) + ")"); return true;
    case IrOpcode::BitCastF32U32: Out(state, inst, "as_type<float>(" + a(0) + ")"); return true;
    case IrOpcode::ConvertU16U32:
    case IrOpcode::ConvertU32U16: Out(state, inst, a(0) + " & 0xffffu"); return true;
    case IrOpcode::ConvertU8U32:
    case IrOpcode::ConvertU32U8: Out(state, inst, a(0) + " & 0xffu"); return true;
    case IrOpcode::ConvertF32F16: state.features |= Half; call("aps5_f16_to_f32", 1); return true;
    case IrOpcode::ConvertF16F32: state.features |= Half; Out(state, inst, "aps5_f32_to_f16(" + Bits(state, inst, 0) + ")"); return true;
    case IrOpcode::ConvertS32F32: Out(state, inst, "uint(int(" + a(0) + "))"); return true;
    case IrOpcode::ConvertU32F32: Out(state, inst, "uint(" + a(0) + ")"); return true;
    case IrOpcode::ConvertF32S32: Out(state, inst, "float(int(" + a(0) + "))"); return true;
    case IrOpcode::ConvertF32U32: Out(state, inst, "float(" + a(0) + ")"); return true;
    case IrOpcode::PackHalf2x16:
        state.features |= Half;
        Out(state, inst, "aps5_f32_to_f16(as_type<uint>(" + a(0) + ".x)) | (aps5_f32_to_f16(as_type<uint>(" + a(0) + ".y)) << 16u)");
        return true;
    case IrOpcode::PackFloat2x16Rtz:
        state.features |= Half;
        Out(state, inst, "aps5_f32_to_f16_rtz(" + Bits(state, inst, 0) + ") | (aps5_f32_to_f16_rtz(" + Bits(state, inst, 1) + ") << 16u)");
        return true;
    case IrOpcode::FPAdd16x2: state.features |= Half; Out(state, inst, "aps5_half2(" + a(0) + ", " + a(1) + ", false)"); return true;
    case IrOpcode::FPMul16x2: state.features |= Half; Out(state, inst, "aps5_half2(" + a(0) + ", " + a(1) + ", true)"); return true;
    case IrOpcode::FPAbs32: Out(state, inst, "as_type<float>(as_type<uint>(" + a(0) + ") & 0x7fffffffu)"); return true;
    case IrOpcode::FPNeg32: Out(state, inst, "as_type<float>(as_type<uint>(" + a(0) + ") ^ 0x80000000u)"); return true;
    case IrOpcode::FPSaturate32: Out(state, inst, "aps5_saturate(" + Bits(state, inst, 0) + ")"); return true;
    case IrOpcode::FPOrdEqual32: binary("=="); return true;
    case IrOpcode::FPOrdLessThan32: binary("<"); return true;
    case IrOpcode::FPOrdGreaterThan32: binary(">"); return true;
    case IrOpcode::FPOrdLessThanEqual32: binary("<="); return true;
    case IrOpcode::FPOrdGreaterThanEqual32: binary(">="); return true;
    case IrOpcode::FPOrdNotEqual32: Out(state, inst, "(!" + Nan(a(0)) + " && !" + Nan(a(1)) + " && " + a(0) + " != " + a(1) + ")"); return true;
    case IrOpcode::FPUnordEqual32: unordered("=="); return true;
    case IrOpcode::FPUnordNotEqual32: unordered("!="); return true;
    case IrOpcode::FPUnordLessThan32: unordered("<"); return true;
    case IrOpcode::FPUnordGreaterThan32: unordered(">"); return true;
    case IrOpcode::FPUnordLessThanEqual32: unordered("<="); return true;
    case IrOpcode::FPUnordGreaterThanEqual32: unordered(">="); return true;
    case IrOpcode::FPIsNan32: Out(state, inst, "aps5_nan(" + Bits(state, inst, 0) + ")"); return true;
    case IrOpcode::FPCmpClass32: state.features |= FloatClass; Out(state, inst, "aps5_class(" + Bits(state, inst, 0) + ", " + a(1) + ")"); return true;
    case IrOpcode::FPAdd32: binary("+"); return true;
    case IrOpcode::FPSub32: binary("-"); return true;
    case IrOpcode::FPMul32: binary("*"); return true;
    case IrOpcode::FPMad32: Out(state, inst, a(0) + " * " + a(1) + " + " + a(2)); return true;
    case IrOpcode::FPFma32: call("fma", 3); return true;
    case IrOpcode::FPMin32: call("fmin", 2); return true;
    case IrOpcode::FPMax32: call("fmax", 2); return true;
    case IrOpcode::FPMinTri32: Out(state, inst, "fmin(fmin(" + a(0) + ", " + a(1) + "), " + a(2) + ")"); return true;
    case IrOpcode::FPMaxTri32: Out(state, inst, "fmax(fmax(" + a(0) + ", " + a(1) + "), " + a(2) + ")"); return true;
    case IrOpcode::FPMedTri32: call("aps5_med3f", 3); return true;
    case IrOpcode::FPRecip32:
    case IrOpcode::FPRecipIFlag32: Out(state, inst, "1.0f / " + a(0)); return true;
    case IrOpcode::FPRecipSqrt32: call("rsqrt", 1); return true;
    case IrOpcode::FPSqrt: call("sqrt", 1); return true;
    case IrOpcode::FPSin: Out(state, inst, "sinpi(2.0f * " + a(0) + ")"); return true;
    case IrOpcode::FPCos: Out(state, inst, "cospi(2.0f * " + a(0) + ")"); return true;
    case IrOpcode::FPExp2:
    case IrOpcode::FPLog2:
        if ((state.target.experiment & MslExperimentNoPow) != 0u) {
            Out(state, inst, a(0));
            return true;
        }
        call(inst.Opcode() == IrOpcode::FPExp2 ? "exp2" : "log2", 1);
        return true;
    case IrOpcode::FPRoundEven32: call("rint", 1); return true;
    case IrOpcode::FPFloor32: call("floor", 1); return true;
    case IrOpcode::FPCeil32: call("ceil", 1); return true;
    case IrOpcode::FPTrunc32: call("trunc", 1); return true;
    case IrOpcode::FPFract32: call("aps5_fract", 1); return true;
    case IrOpcode::GetUserData: Out(state, inst, UserData(state, inst)); return true;
    case IrOpcode::GetShaderBase: Out(state, inst, "0ul"); return true;
    case IrOpcode::GetBuiltin: Out(state, inst, Builtin(state, inst)); return true;
    case IrOpcode::ShaderClock:
    case IrOpcode::RealtimeClock:
        throw MslUnsupported("MSL backend: shader clocks are not emitted");
    default:
        break;
    }
    if (inst.Opcode() == IrOpcode::FPInterpolateF32 || inst.Opcode() == IrOpcode::FPInterpolateF16) throw MslUnsupported("MSL backend: " + std::string(IrOpcodeName(inst.Opcode())) + " is not emitted (graphics stages are section S6)");
    return false;
}

}
