#include "MslState.hpp"
#include <algorithm>
#include <cstdint>
#include <optional>

namespace ShaderRecompiler::Msl {

namespace {

constexpr const char* HalfLane = "aps5_lane";
constexpr const char* Exchange = "aps5_index, aps5_wave, aps5_phase";

struct Move {
    std::string lane;
    std::string valid;
    std::optional<std::uint32_t> xorMask;
    std::optional<std::uint32_t> up;
    std::optional<std::uint32_t> down;
};

std::string Ballot32(State& state, const std::string& predicate) {
    state.features |= Wave;
    return "aps5_ballot32(" + predicate + ")";
}

std::string SimdLane(State& state) {
    state.features |= Lane;
    return HalfLane;
}

std::string Xor(State& state, std::uint32_t mask) {
    return "(" + SimdLane(state) + " ^ " + U32(mask) + ")";
}

std::string Row(State& state) {
    return "(" + SimdLane(state) + " & ~15u)";
}

std::string RowLane(State& state) {
    return "(" + SimdLane(state) + " & 15u)";
}

std::string QuadPerm(State& state, std::uint32_t control) {
    return "((" + SimdLane(state) + " & ~3u) | ((" + U32(control & 0xffu) + " >> ((" + SimdLane(state) + " & 3u) * 2u)) & 3u))";
}

Move QuadMove(State& state, std::uint32_t control) {
    control &= 0xffu;
    for (std::uint32_t mask = 0; mask < 4u; ++mask) {
        bool matches = true;
        for (std::uint32_t lane = 0; lane < 4u; ++lane) matches &= ((control >> (lane * 2u)) & 3u) == (lane ^ mask);
        if (!matches) continue;
        if (mask == 0u) return {SimdLane(state), {}, std::nullopt, std::nullopt, std::nullopt};
        return {Xor(state, mask), {}, mask, std::nullopt, std::nullopt};
    }
    return {QuadPerm(state, control), {}, std::nullopt, std::nullopt, std::nullopt};
}

Move DppMove(State& state, std::uint32_t control) {
    const auto lane = SimdLane(state);
    if ((control & DppMoveFlags::Lanes8) != 0u) {
        const auto selects = control & ~DppMoveFlags::Lanes8;
        return {"((" + lane + " & ~7u) | ((" + U32(selects) + " >> ((" + lane + " & 7u) * 3u)) & 7u))", {}, std::nullopt, std::nullopt, std::nullopt};
    }
    if (control <= 0xffu) return QuadMove(state, control);
    const auto amount = control & 0xfu;
    if (control >= 0x101u && control <= 0x10fu) return {"(" + lane + " + " + U32(amount) + ")", RowLane(state) + " < " + U32(16u - amount), std::nullopt, std::nullopt, amount};
    if (control >= 0x111u && control <= 0x11fu) return {"(" + lane + " - " + U32(amount) + ")", RowLane(state) + " >= " + U32(amount), std::nullopt, amount, std::nullopt};
    if (control >= 0x121u && control <= 0x12fu) return {"(" + Row(state) + " | ((" + lane + " - " + U32(amount) + ") & 15u))", {}, std::nullopt, std::nullopt, std::nullopt};
    if (control == 0x140u) return {Xor(state, 15u), {}, 15u, std::nullopt, std::nullopt};
    if (control == 0x141u) return {Xor(state, 7u), {}, 7u, std::nullopt, std::nullopt};
    if (control >= 0x150u && control <= 0x15fu) return {"(" + Row(state) + " | " + U32(amount) + ")", {}, std::nullopt, std::nullopt, std::nullopt};
    if (control >= 0x160u && control <= 0x16fu) return {Xor(state, amount), {}, amount, std::nullopt, std::nullopt};
    Fail("DPP control " + std::to_string(control) + " is not emitted");
}

Move SwizzleMove(State& state, std::uint32_t control) {
    const auto lane = SimdLane(state);
    if ((control & 0xc000u) == 0xc000u) {
        const std::uint32_t mask = control & 0x1fu;
        const std::uint32_t rotate = (control >> 5u) & 0x1fu;
        const std::uint32_t delta = (control & 0x400u) != 0u ? ((32u - rotate) & 0x1fu) : rotate;
        return {"((" + lane + " & " + U32(mask) + ") | (((" + lane + " + " + U32(delta) + ") & 31u) & " + U32(~mask & 31u) + "))", {}, std::nullopt, std::nullopt, std::nullopt};
    }
    if ((control & 0x8000u) != 0u) return QuadMove(state, (control & 0xffu));
    const std::uint32_t andMask = control & 0x1fu;
    const std::uint32_t orMask = (control >> 5u) & 0x1fu;
    const std::uint32_t xorMask = (control >> 10u) & 0x1fu;
    if (andMask == 0x1fu && orMask == 0u) return {Xor(state, xorMask), {}, xorMask, std::nullopt, std::nullopt};
    return {"((((" + lane + " & " + U32(andMask) + ") | " + U32(orMask) + ") ^ " + U32(xorMask) + ") & 31u)", {}, std::nullopt, std::nullopt, std::nullopt};
}

std::string Fetch(const std::string& value, const Move& move) {
    if (move.xorMask) return "simd_shuffle_xor(" + value + ", ushort(" + U32(*move.xorMask) + "))";
    if (move.up) return "simd_shuffle_up(" + value + ", ushort(" + U32(*move.up) + "))";
    if (move.down) return "simd_shuffle_down(" + value + ", ushort(" + U32(*move.down) + "))";
    return "simd_shuffle(" + value + ", ushort(" + move.lane + " & 31u))";
}

std::string Join(const std::vector<std::string>& conditions) {
    std::string text;
    for (const auto& condition : conditions) text += (text.empty() ? "" : " && ") + std::string("(") + condition + ")";
    return text;
}

std::string SourceActive(State& state, const std::string& exec, const std::string& lane) {
    const auto mask = Temp(state);
    Line(state, "const uint " + mask + " = " + Ballot32(state, exec) + ";");
    return "((" + mask + " >> (" + lane + " & 31u)) & 1u) != 0u";
}

std::string Fetched(State& state, const std::string& value, const Move& move) {
    const auto name = Temp(state);
    Line(state, "const uint " + name + " = " + Fetch(value, move) + ";");
    return name;
}

bool EmitDpp(State& state, const IrValue& inst) {
    const auto flags = inst.Flags<DppMoveFlags>();
    const auto move = DppMove(state, flags.control);
    const bool lanes8 = (flags.control & DppMoveFlags::Lanes8) != 0u;
    if (inst.Opcode() == IrOpcode::DppMoveU32) {
        const auto moved = Fetched(state, Arg(state, inst, 0), move);
        std::vector<std::string> conditions;
        if (!move.valid.empty()) conditions.push_back(move.valid);
        if (!flags.fetchInactive) conditions.push_back(SourceActive(state, Arg(state, inst, 1), move.lane));
        Define(state, inst, conditions.empty() ? moved : Join(conditions) + " ? " + moved + " : 0u");
        return true;
    }
    std::vector<std::string> conditions{Arg(state, inst, 2)};
    const auto lane = WaveLane(state);
    if (flags.rowMask != 0xfu) conditions.push_back("((" + U32(flags.rowMask) + " >> ((" + lane + " >> 4u) & 3u)) & 1u) != 0u");
    if (flags.bankMask != 0xfu) conditions.push_back("((" + U32(flags.bankMask) + " >> ((" + lane + " >> 2u) & 3u)) & 1u) != 0u");
    if (!flags.boundControl) {
        if (!move.valid.empty()) conditions.push_back(move.valid);
        if (!flags.fetchInactive && !lanes8) conditions.push_back(SourceActive(state, Arg(state, inst, 2), move.lane));
    }
    Define(state, inst, Join(conditions) + " ? " + Arg(state, inst, 0) + " : " + Arg(state, inst, 1));
    return true;
}

bool EmitMaskedRead(State& state, const IrValue& inst, const Move& move) {
    const auto fetched = Fetched(state, Arg(state, inst, 0), move);
    Define(state, inst, SourceActive(state, Arg(state, inst, 2), move.lane) + " ? " + fetched + " : 0u");
    return true;
}

bool EmitPermlane(State& state, const IrValue& inst) {
    const auto flags = inst.Flags<PermlaneFlags>();
    const auto lane = SimdLane(state);
    const auto selected = Temp(state);
    const auto target = Temp(state);
    Line(state, "const uint " + selected + " = (" + RowLane(state) + " >= 8u) ? " + Arg(state, inst, 2) + " : " + Arg(state, inst, 1) + ";");
    Line(state, "const uint " + target + " = " + (flags.x16 ? "(" + Row(state) + " ^ 16u)" : Row(state)) + " | ((" + selected + " >> ((" + lane + " & 7u) << 2u)) & 15u);");
    const auto shuffled = Fetched(state, Arg(state, inst, 0), Move{target, {}, std::nullopt, std::nullopt, std::nullopt});
    if (flags.fetchInactive) {
        Define(state, inst, shuffled);
        return true;
    }
    Define(state, inst, SourceActive(state, Arg(state, inst, 3), target) + " ? " + shuffled + " : " + (flags.boundControl ? std::string("0u") : Arg(state, inst, 4)));
    return true;
}

bool EmitBallot(State& state, const IrValue& inst) {
    if (!state.spanned) {
        Define(state, inst, "uint4(" + Ballot32(state, Arg(state, inst, 0)) + ", 0u, 0u, 0u)");
        return true;
    }
    state.features |= Wave | WaveExchange | LocalIndex;
    Define(state, inst, "uint4(aps5_ballot64(" + Arg(state, inst, 0) + ", " + Exchange + "), 0u, 0u)");
    return true;
}

bool EmitReadLane(State& state, const IrValue& inst) {
    if (!state.spanned) {
        Define(state, inst, "simd_shuffle(" + Arg(state, inst, 0) + ", ushort(" + Arg(state, inst, 1) + " & 31u))");
        return true;
    }
    state.features |= WaveExchange | LocalIndex;
    Define(state, inst, "aps5_read64(" + Arg(state, inst, 0) + ", " + Arg(state, inst, 1) + " & 63u, " + Exchange + ")");
    return true;
}

bool EmitReadFirstLane(State& state, const IrValue& inst) {
    const IrValue* active = inst.Argument(1)->Resolve();
    const bool everyone = active->HasImmediate() && active->ImmediateBool();
    if (!state.spanned) {
        if (everyone) {
            Define(state, inst, "simd_broadcast_first(" + Arg(state, inst, 0) + ")");
            return true;
        }
        const auto mask = Temp(state);
        Line(state, "const uint " + mask + " = " + Ballot32(state, Arg(state, inst, 1)) + ";");
        Define(state, inst, mask + " != 0u ? simd_shuffle(" + Arg(state, inst, 0) + ", ushort(ctz(" + mask + "))) : simd_broadcast_first(" + Arg(state, inst, 0) + ")");
        return true;
    }
    state.features |= Wave | WaveExchange | LocalIndex;
    if (everyone) Define(state, inst, "aps5_read64(" + Arg(state, inst, 0) + ", 0u, " + Exchange + ")");
    else Define(state, inst, "aps5_readfirst64(" + Arg(state, inst, 0) + ", " + Arg(state, inst, 1) + ", " + Exchange + ")");
    return true;
}

}

std::string WaveLane(State& state) {
    if (!state.spanned) return SimdLane(state);
    state.features |= LocalIndex;
    return "(aps5_index & 63u)";
}

std::string WaveAny(State& state, const std::string& predicate) {
    if (!state.spanned) return "simd_any(" + predicate + ")";
    state.features |= Wave | WaveExchange | LocalIndex;
    return "aps5_any64(" + predicate + ", " + Exchange + ")";
}

bool EmitWave(State& state, const IrValue& inst) {
    switch (inst.Opcode()) {
    case IrOpcode::LaneId:
        Define(state, inst, WaveLane(state));
        return true;
    case IrOpcode::WriteLane:
        Define(state, inst, "(" + Arg(state, inst, 1) + " & " + U32(state.spanned ? 63u : 31u) + ") == " + WaveLane(state) + " ? " + Arg(state, inst, 0) + " : " + Arg(state, inst, 2));
        return true;
    case IrOpcode::Ballot: return EmitBallot(state, inst);
    case IrOpcode::ReadLane: return EmitReadLane(state, inst);
    case IrOpcode::ReadFirstLane: return EmitReadFirstLane(state, inst);
    case IrOpcode::DppMoveU32:
    case IrOpcode::DppUpdateU32: return EmitDpp(state, inst);
    case IrOpcode::Permlane16U32: return EmitPermlane(state, inst);
    case IrOpcode::SwizzleU32: {
        const IrValue* control = inst.Argument(1)->Resolve();
        return EmitMaskedRead(state, inst, SwizzleMove(state, control->HasImmediate() ? control->ImmediateU32() : 0u));
    }
    case IrOpcode::BpermuteU32: {
        const auto index = Temp(state);
        Line(state, "const uint " + index + " = (" + Arg(state, inst, 1) + " >> 2u) & 31u;");
        return EmitMaskedRead(state, inst, Move{index, {}, std::nullopt, std::nullopt, std::nullopt});
    }
    case IrOpcode::PermuteU32:
        state.features |= Lane | Wave | Permute;
        Define(state, inst, "aps5_permute(" + Arg(state, inst, 0) + ", " + Arg(state, inst, 1) + ", " + Arg(state, inst, 2) + ", " + HalfLane + ")");
        return true;
    default:
        return false;
    }
}

}
