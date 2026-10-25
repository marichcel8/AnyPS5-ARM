#include "LifterImpl.hpp"

#include <cstdlib>
#include <unordered_map>

namespace Translator {

namespace {

constexpr std::uint8_t AllFlags = (1u << Cf) | (1u << Pf) | (1u << Af) | (1u << Zf) | (1u << Sf) | (1u << Of);

struct FlagUse {
    std::uint8_t reads = 0;
    std::uint8_t writes = 0;
};

std::uint8_t ConditionFlags(Condition condition) {
    switch (condition) {
    case Condition::O: case Condition::No: return 1u << Of;
    case Condition::B: case Condition::Ae: return 1u << Cf;
    case Condition::E: case Condition::Ne: return 1u << Zf;
    case Condition::Be: case Condition::A: return (1u << Cf) | (1u << Zf);
    case Condition::S: case Condition::Ns: return 1u << Sf;
    case Condition::P: case Condition::Np: return 1u << Pf;
    case Condition::L: case Condition::Ge: return (1u << Sf) | (1u << Of);
    case Condition::Le: case Condition::G: return (1u << Zf) | (1u << Sf) | (1u << Of);
    case Condition::LoopE: case Condition::LoopNe: return 1u << Zf;
    default: return 0;
    }
}

bool IsShift(unsigned id) {
    switch (id) {
    case X86_INS_SHL: case X86_INS_SAL: case X86_INS_SHR: case X86_INS_SAR: case X86_INS_ROL: case X86_INS_ROR:
    case X86_INS_RCL: case X86_INS_RCR: case X86_INS_SHLD: case X86_INS_SHRD:
        return true;
    default:
        return false;
    }
}

FlagUse UseOf(const Instruction& instruction) {
    const auto id = instruction.id;
    if (instruction.size == 0 || instruction.mnemonic.empty() || instruction.mnemonic.front() == 'f') return {AllFlags, 0};
    if (id == X86_INS_CALL) return {0, AllFlags};
    static const std::uint64_t tested[] = {X86_EFLAGS_TEST_CF, X86_EFLAGS_TEST_PF, X86_EFLAGS_TEST_AF, X86_EFLAGS_TEST_ZF, X86_EFLAGS_TEST_SF, X86_EFLAGS_TEST_OF};
    static const std::uint64_t prior[] = {X86_EFLAGS_PRIOR_CF, X86_EFLAGS_PRIOR_PF, X86_EFLAGS_PRIOR_AF, X86_EFLAGS_PRIOR_ZF, X86_EFLAGS_PRIOR_SF, X86_EFLAGS_PRIOR_OF};
    static const std::uint64_t written[] = {
        X86_EFLAGS_MODIFY_CF | X86_EFLAGS_RESET_CF | X86_EFLAGS_SET_CF | X86_EFLAGS_UNDEFINED_CF,
        X86_EFLAGS_MODIFY_PF | X86_EFLAGS_RESET_PF | X86_EFLAGS_SET_PF | X86_EFLAGS_UNDEFINED_PF,
        X86_EFLAGS_MODIFY_AF | X86_EFLAGS_RESET_AF | X86_EFLAGS_SET_AF | X86_EFLAGS_UNDEFINED_AF,
        X86_EFLAGS_MODIFY_ZF | X86_EFLAGS_RESET_ZF | X86_EFLAGS_SET_ZF | X86_EFLAGS_UNDEFINED_ZF,
        X86_EFLAGS_MODIFY_SF | X86_EFLAGS_RESET_SF | X86_EFLAGS_SET_SF | X86_EFLAGS_UNDEFINED_SF,
        X86_EFLAGS_MODIFY_OF | X86_EFLAGS_RESET_OF | X86_EFLAGS_SET_OF | X86_EFLAGS_UNDEFINED_OF,
    };
    const std::uint64_t eflags = instruction.x86.eflags;
    FlagUse use;
    for (unsigned flag = 0; flag < 6; ++flag) {
        if (eflags & (tested[flag] | prior[flag])) use.reads |= static_cast<std::uint8_t>(1u << flag);
        if (eflags & written[flag]) use.writes |= static_cast<std::uint8_t>(1u << flag);
    }
    use.reads |= ConditionFlags(ConditionOf(id));
    switch (id) {
    case X86_INS_ADC: case X86_INS_SBB: case X86_INS_CMC: case X86_INS_RCL: case X86_INS_RCR: case X86_INS_ADCX:
        use.reads |= 1u << Cf;
        break;
    case X86_INS_ADOX:
        use.reads |= 1u << Of;
        break;
    case X86_INS_PUSHF: case X86_INS_PUSHFQ: case X86_INS_LAHF:
        use.reads = AllFlags;
        break;
    default:
        break;
    }
    if (IsShift(id)) {
        const auto& count = instruction.Op(instruction.x86.op_count - 1u);
        const auto mask = instruction.Bits(0) == 64 ? 63u : 31u;
        const bool constantCount = instruction.x86.op_count == 1 || (count.type == X86_OP_IMM && (static_cast<std::uint64_t>(count.imm) & mask) != 0);
        if (!constantCount || id == X86_INS_RCL || id == X86_INS_RCR) {
            use.reads |= use.writes;
            use.writes = 0;
        }
    }
    if (instruction.Prefixed(X86_PREFIX_REP) || instruction.Prefixed(X86_PREFIX_REPNE)) {
        use.reads |= use.writes;
        use.writes = 0;
    }
    return use;
}

}

std::map<std::uint64_t, std::uint8_t> Lifter::Impl::DeadFlags(std::uint64_t entry, const std::set<std::uint64_t>& addresses) {
    std::map<std::uint64_t, std::uint8_t> dead;
    static const bool keepAll = std::getenv("APS5_TRANSLATOR_ALL_FLAGS") != nullptr;
    if (keepAll) return dead;
    const std::vector<std::uint64_t> order(addresses.begin(), addresses.end());
    std::unordered_map<std::uint64_t, std::uint32_t> index;
    index.reserve(order.size());
    for (std::uint32_t i = 0; i < order.size(); ++i) index.emplace(order[i], i);
    std::vector<FlagUse> uses(order.size());
    std::vector<std::uint8_t> exitLive(order.size(), 0);
    std::vector<std::vector<std::uint32_t>> successors(order.size());
    auto internal = [&](std::uint64_t target) -> std::optional<std::uint32_t> {
        if (target != entry && lifted.count(target) != 0) return std::nullopt;
        if (const auto found = index.find(target); found != index.end()) return found->second;
        return std::nullopt;
    };
    for (std::uint32_t i = 0; i < order.size(); ++i) {
        const auto& instruction = Decode(order[i]);
        uses[i] = UseOf(instruction);
        if (instruction.size == 0) continue;
        const auto id = instruction.id;
        const auto target = instruction.DirectTarget();
        if (id == X86_INS_RET || id == X86_INS_UD2 || id == X86_INS_INT3 || id == X86_INS_HLT) continue;
        if (id == X86_INS_JMP) {
            if (target) {
                if (const auto to = internal(*target)) successors[i].push_back(*to);
                continue;
            }
            const auto table = jumpTables.find(order[i]);
            if (table == jumpTables.end()) {
                exitLive[i] = AllFlags;
                continue;
            }
            for (const auto to : table->second.targets) {
                if (const auto at = index.find(to); at != index.end()) successors[i].push_back(at->second);
                else exitLive[i] = AllFlags;
            }
            if (!table->second.bounded) exitLive[i] = AllFlags;
            continue;
        }
        if (IsConditionalJump(id) && target) {
            if (const auto to = internal(*target)) successors[i].push_back(*to);
        }
        if (const auto next = index.find(instruction.Next()); next != index.end()) successors[i].push_back(next->second);
        else exitLive[i] = AllFlags;
    }
    std::vector<std::uint8_t> liveIn(order.size(), 0), liveOut(order.size(), 0);
    for (bool changed = true; changed;) {
        changed = false;
        for (auto i = order.size(); i-- > 0;) {
            std::uint8_t out = exitLive[i];
            for (const auto to : successors[i]) out |= liveIn[to];
            const auto in = static_cast<std::uint8_t>(uses[i].reads | (out & ~uses[i].writes));
            liveOut[i] = out;
            if (in != liveIn[i]) {
                liveIn[i] = in;
                changed = true;
            }
        }
    }
    for (std::uint32_t i = 0; i < order.size(); ++i) {
        const auto mask = static_cast<std::uint8_t>(uses[i].writes & ~liveOut[i]);
        if (mask != 0) dead.emplace(order[i], mask);
    }
    return dead;
}

}
