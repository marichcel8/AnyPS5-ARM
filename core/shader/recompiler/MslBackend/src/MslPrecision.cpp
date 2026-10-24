#include "MslState.hpp"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace ShaderRecompiler::Msl {

namespace {

constexpr int Unbounded = 129;

struct Facts {
    bool nan = true;
    bool infinite = true;
    bool negative = true;
    int high = Unbounded;

    friend bool operator==(const Facts&, const Facts&) = default;
};

constexpr Facts Unknown{};
constexpr Facts Nothing{false, false, false, -127};

Facts Bound(Facts facts) {
    if (facts.high >= Unbounded) {
        facts.infinite = true;
        facts.high = Unbounded;
    }
    return facts;
}

Facts FromBits(std::uint32_t bits) {
    const auto exponent = static_cast<int>((bits >> 23u) & 0xffu);
    const bool negative = (bits >> 31u) != 0u;
    if (exponent == 0xff) return {(bits & 0x007fffffu) != 0u, (bits & 0x007fffffu) == 0u, negative, Unbounded};
    return {false, false, negative, exponent == 0 ? -126 : exponent - 126};
}

Facts Join(Facts left, Facts right) {
    return {left.nan || right.nan, left.infinite || right.infinite, left.negative || right.negative, std::max(left.high, right.high)};
}

Facts Add(Facts left, Facts right, bool subtract) {
    return Bound({left.nan || right.nan || (left.infinite && right.infinite), left.infinite || right.infinite, subtract || left.negative || right.negative, std::max(left.high, right.high) + 1});
}

Facts Multiply(Facts left, Facts right) {
    const bool infinite = left.infinite || right.infinite;
    return Bound({left.nan || right.nan || infinite, infinite, left.negative || right.negative, std::min(left.high + right.high, Unbounded)});
}

Facts Choose(Facts left, Facts right) {
    return {left.nan && right.nan, left.infinite || right.infinite, left.negative || right.negative, std::max(left.high, right.high)};
}

bool Special(Facts facts) {
    return facts.nan || facts.infinite;
}

bool Tracked(const IrValue& value) {
    return value.Type() == IrType::F32 || value.Opcode() == IrOpcode::BitCastU32F32;
}

bool Samples(IrOpcode opcode) {
    return opcode == IrOpcode::ImageSampleExplicitLod || opcode == IrOpcode::ImageSampleImplicitLod || opcode == IrOpcode::ImageSampleRaw || opcode == IrOpcode::ImageGatherRaw || opcode == IrOpcode::ImageQueryLod;
}

bool PassesBits(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::BitwiseAnd32:
    case IrOpcode::BitwiseAnd64:
    case IrOpcode::BitwiseOr32:
    case IrOpcode::BitwiseXor32:
    case IrOpcode::BitwiseNot32:
    case IrOpcode::ShiftLeftLogical32:
    case IrOpcode::ShiftRightLogical32:
    case IrOpcode::ShiftRightArithmetic32:
    case IrOpcode::ShiftLeftLogical64:
    case IrOpcode::ShiftRightLogical64:
    case IrOpcode::ShiftRightArithmetic64:
    case IrOpcode::IAdd32:
    case IrOpcode::IAdd64:
    case IrOpcode::ISub32:
    case IrOpcode::ISub64:
    case IrOpcode::IMul32:
    case IrOpcode::IMul64:
    case IrOpcode::BitFieldInsert:
    case IrOpcode::BitFieldUExtract:
    case IrOpcode::BitFieldSExtract:
    case IrOpcode::Select:
    case IrOpcode::SelectU32:
    case IrOpcode::Phi:
    case IrOpcode::MakeImageAddress:
    case IrOpcode::CompositeConstructU64:
    case IrOpcode::CompositeConstructU32x2:
    case IrOpcode::CompositeConstructU32x3:
    case IrOpcode::CompositeConstructU32x4:
    case IrOpcode::CompositeExtractU64:
    case IrOpcode::CompositeExtractU32x2:
    case IrOpcode::CompositeExtractU32x3:
    case IrOpcode::CompositeExtractU32x4:
    case IrOpcode::ConvertU16U32:
    case IrOpcode::ConvertU32U16:
    case IrOpcode::ConvertU8U32:
    case IrOpcode::ConvertU32U8:
    case IrOpcode::ReadLane:
    case IrOpcode::ReadFirstLane:
    case IrOpcode::WriteLane: return true;
    default: return false;
    }
}

bool Atomic(IrOpcode opcode) {
    return BufferAccessOf(opcode) == BufferAccess::Atomic || SharedAccessOf(opcode) == SharedAccess::Atomic || AddressOpcodeInfoOf(opcode).access == AddressAccess::Atomic;
}

class Analysis {
public:
    explicit Analysis(State& state) : state(state) {}

    void Run() {
        Forward();
        for (const auto* block : state.program.BlockOrder()) {
            for (const auto* inst : block->Instructions()) Sink(*inst);
        }
    }

private:
    Facts Of(const IrValue* argument) const {
        if (argument == nullptr) return Unknown;
        const IrValue* value = argument->Resolve();
        if (value->HasImmediate() && value->ArgumentCount() == 0u) {
            switch (value->Type()) {
            case IrType::F32:
            case IrType::U32:
            case IrType::S32: return FromBits(value->ImmediateU32());
            default: return Unknown;
            }
        }
        if (value->Opcode() == IrOpcode::UndefU32) return FromBits(0u);
        const auto found = facts.find(value);
        if (found != facts.end()) return found->second;
        return Tracked(*value) ? Nothing : Unknown;
    }

    Facts Transfer(const IrValue& inst) const {
        const auto a = [&](std::size_t index) { return Of(inst.Argument(index)); };
        switch (inst.Opcode()) {
        case IrOpcode::BitCastU32F32:
        case IrOpcode::BitCastF32U32: return a(0);
        case IrOpcode::ConvertF32U32: return {false, false, false, 33};
        case IrOpcode::ConvertF32S32: return {false, false, true, 32};
        case IrOpcode::FPAdd32: return Add(a(0), a(1), false);
        case IrOpcode::FPSub32: return Add(a(0), a(1), true);
        case IrOpcode::FPMul32: return Multiply(a(0), a(1));
        case IrOpcode::FPMad32:
        case IrOpcode::FPFma32: return Add(Multiply(a(0), a(1)), a(2), false);
        case IrOpcode::FPMin32:
        case IrOpcode::FPMax32: return Choose(a(0), a(1));
        case IrOpcode::FPMinTri32:
        case IrOpcode::FPMaxTri32:
        case IrOpcode::FPMedTri32: return Choose(Choose(a(0), a(1)), a(2));
        case IrOpcode::FPSaturate32: return {false, false, false, 1};
        case IrOpcode::FPAbs32: return {a(0).nan, a(0).infinite, false, a(0).high};
        case IrOpcode::FPNeg32: return {a(0).nan, a(0).infinite, true, a(0).high};
        case IrOpcode::FPRecip32:
        case IrOpcode::FPRecipIFlag32: return {a(0).nan, true, a(0).negative, Unbounded};
        case IrOpcode::FPRecipSqrt32: return {a(0).nan || a(0).negative, true, false, Unbounded};
        case IrOpcode::FPSqrt: return Bound({a(0).nan || a(0).negative, a(0).infinite, false, a(0).high / 2 + 1});
        case IrOpcode::FPSin:
        case IrOpcode::FPCos: return {a(0).nan || a(0).infinite, false, true, 1};
        case IrOpcode::FPExp2: {
            const bool overflow = a(0).infinite || a(0).high > 7;
            return {a(0).nan, overflow, false, overflow ? Unbounded : (a(0).high <= 0 ? 1 : 1 << a(0).high)};
        }
        case IrOpcode::FPLog2: return {a(0).nan || a(0).negative, true, true, 8};
        case IrOpcode::FPRoundEven32:
        case IrOpcode::FPFloor32:
        case IrOpcode::FPCeil32:
        case IrOpcode::FPTrunc32: return Bound({a(0).nan, a(0).infinite, a(0).negative, std::max(a(0).high, 1) + 1});
        case IrOpcode::FPFract32: return {a(0).nan || a(0).infinite, false, false, 1};
        case IrOpcode::Select:
        case IrOpcode::SelectF32: return Join(a(1), a(2));
        case IrOpcode::Phi: {
            Facts joined = Nothing;
            for (std::size_t index = 0; index < inst.ArgumentCount(); ++index) joined = Join(joined, a(index));
            return joined;
        }
        default: return Unknown;
        }
    }

    void Forward() {
        bool changed = true;
        for (std::uint32_t pass = 0; changed && pass < 16u; ++pass) {
            changed = false;
            for (const auto* block : state.program.BlockOrder()) {
                for (const auto* inst : block->Instructions()) {
                    if (!Tracked(*inst)) continue;
                    auto next = Transfer(*inst);
                    const auto [found, inserted] = facts.try_emplace(inst, next);
                    if (inserted || found->second == next) continue;
                    found->second = pass >= 3u ? Unknown : Join(found->second, next);
                    changed = true;
                }
            }
        }
    }

    void Mark(const IrValue* argument, std::uint8_t level) {
        if (argument == nullptr) return;
        const IrValue* value = argument->Resolve();
        if (value->HasImmediate() && value->ArgumentCount() == 0u) return;
        if (!Tracked(*value)) return;
        if (level == PrecisionIeee && !Special(Of(value))) return;
        auto& current = state.precision[value];
        if (current >= level) return;
        current = level;
        for (std::size_t index = 0; index < value->ArgumentCount(); ++index) Mark(value->Argument(index), level);
    }

    void Observe(const IrValue& inst, std::uint8_t level, std::initializer_list<std::size_t> operands) {
        bool special = level == PrecisionStrict;
        for (const auto index : operands) {
            const IrValue* operand = inst.Argument(index);
            if (operand != nullptr && operand->Resolve()->Opcode() == IrOpcode::CompositeConstructF32x2) {
                for (std::size_t part = 0; part < operand->Resolve()->ArgumentCount(); ++part) special |= Special(Of(operand->Resolve()->Argument(part)));
            } else {
                special |= Special(Of(operand));
            }
        }
        if (!special) return;
        auto& current = state.precision[&inst];
        current = std::max(current, level);
        for (const auto index : operands) {
            const IrValue* operand = inst.Argument(index);
            if (operand != nullptr && operand->Resolve()->Opcode() == IrOpcode::CompositeConstructF32x2) {
                for (std::size_t part = 0; part < operand->Resolve()->ArgumentCount(); ++part) Mark(operand->Resolve()->Argument(part), level);
            } else {
                Mark(operand, level);
            }
        }
    }

    bool Inspected(const IrValue& bits) const {
        std::vector<const IrValue*> pending(bits.Uses().begin(), bits.Uses().end());
        std::vector<const IrValue*> seen;
        while (!pending.empty()) {
            const IrValue* user = pending.back();
            pending.pop_back();
            if (std::find(seen.begin(), seen.end(), user) != seen.end()) continue;
            seen.push_back(user);
            const auto opcode = user->Opcode();
            if (user->IsIdentity() || PassesBits(opcode)) {
                pending.insert(pending.end(), user->Uses().begin(), user->Uses().end());
                continue;
            }
            if (opcode == IrOpcode::BitCastF32U32 || Samples(opcode)) continue;
            if (user->MayHaveSideEffects() && !Atomic(opcode)) continue;
            return true;
        }
        return false;
    }

    void Sink(const IrValue& inst) {
        switch (inst.Opcode()) {
        case IrOpcode::FPOrdEqual32:
        case IrOpcode::FPUnordEqual32:
        case IrOpcode::FPOrdNotEqual32:
        case IrOpcode::FPUnordNotEqual32:
        case IrOpcode::FPOrdLessThan32:
        case IrOpcode::FPUnordLessThan32:
        case IrOpcode::FPOrdGreaterThan32:
        case IrOpcode::FPUnordGreaterThan32:
        case IrOpcode::FPOrdLessThanEqual32:
        case IrOpcode::FPUnordLessThanEqual32:
        case IrOpcode::FPOrdGreaterThanEqual32:
        case IrOpcode::FPUnordGreaterThanEqual32:
        case IrOpcode::FPMin32:
        case IrOpcode::FPMax32:
        case IrOpcode::PackFloat2x16Rtz: Observe(inst, PrecisionIeee, {0, 1}); return;
        case IrOpcode::FPMinTri32:
        case IrOpcode::FPMaxTri32:
        case IrOpcode::FPMedTri32: Observe(inst, PrecisionIeee, {0, 1, 2}); return;
        case IrOpcode::FPIsNan32:
        case IrOpcode::FPSaturate32:
        case IrOpcode::FPFract32:
        case IrOpcode::ConvertS32F32:
        case IrOpcode::ConvertU32F32:
        case IrOpcode::ConvertF16F32:
        case IrOpcode::PackHalf2x16: Observe(inst, PrecisionIeee, {0}); return;
        case IrOpcode::FPCmpClass32: {
            const IrValue* mask = inst.Argument(1)->Resolve();
            const bool onlySpecial = mask->HasImmediate() && (mask->ImmediateU32() & ~0x207u) == 0u;
            Observe(inst, onlySpecial ? PrecisionIeee : PrecisionStrict, {0});
            return;
        }
        case IrOpcode::BitCastU32F32:
            if (Inspected(inst)) Observe(inst, PrecisionStrict, {0});
            return;
        default: return;
        }
    }

    State& state;
    std::unordered_map<const IrValue*, Facts> facts;
};

}

void AnalyzePrecision(State& state) {
    if (!state.target.relaxedFloat) return;
    Analysis{state}.Run();
}

std::uint8_t Precision(const State& state, const IrValue& inst) {
    if (!state.target.relaxedFloat) return PrecisionStrict;
    const auto found = state.precision.find(&inst);
    return found == state.precision.end() ? PrecisionFast : found->second;
}

}
