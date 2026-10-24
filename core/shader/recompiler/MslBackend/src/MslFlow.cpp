#include "MslState.hpp"
#include "ControlFlow/ControlFlowGraph.hpp"
#include <unordered_map>

namespace ShaderRecompiler::Msl {

namespace {

struct Loop {
    const IrBlock* header = nullptr;
    const IrBlock* merge = nullptr;
    const IrBlock* cont = nullptr;
    std::string flag;
    bool wrapped = false;
};

class Flow {
public:
    explicit Flow(State& state) : state(state) {
        const auto& blocks = state.program.BlockOrder();
        const auto& infos = state.program.Metadata().blockInfo;
        if (blocks.size() != infos.size()) Fail("control flow block metadata is inconsistent");
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            if (blocks[index] == nullptr) Fail("control flow contains a null block");
            byBlock.emplace(blocks[index], &infos[index]);
            byId.emplace(infos[index].id, blocks[index]);
        }
    }

    void Run() {
        const auto& blocks = state.program.BlockOrder();
        if (blocks.empty()) Fail("control flow has no blocks");
        if (Sequence(blocks.front(), nullptr) != nullptr) Fail("control flow ends outside its function");
    }

private:
    const BlockInfo& Info(const IrBlock* block) const {
        const auto found = byBlock.find(block);
        if (found == byBlock.end()) Fail("control flow block has no metadata");
        return *found->second;
    }

    const IrBlock* Target(std::uint32_t id) const {
        const auto found = byId.find(id);
        if (found == byId.end()) Fail("control flow target block is missing");
        return found->second;
    }

    const IrBlock* Sequence(const IrBlock* block, const IrBlock* stop) {
        while (block != nullptr && block != stop) {
            if (const auto jumped = JumpOut(block, stop); jumped != block) return jumped;
            block = Info(block).terminator.loopHeader ? EmitLoop(block) : EmitBlock(block, stop);
        }
        return block;
    }

    const IrBlock* JumpOut(const IrBlock* target, const IrBlock* stop) {
        if (target == stop || loops.empty()) return target;
        auto& loop = loops.back();
        if (target == loop.merge) {
            if (loop.wrapped) Line(state, loop.flag + " = true;");
            Line(state, "break;");
            return nullptr;
        }
        if (target == loop.cont && loop.wrapped) {
            Line(state, "break;");
            return nullptr;
        }
        if (target == loop.header) {
            if (loop.wrapped) Fail("control flow re-enters a loop header from inside its body");
            Line(state, "continue;");
            return nullptr;
        }
        for (std::size_t index = 0; index + 1u < loops.size(); ++index) {
            const auto& outer = loops[index];
            if (target == outer.merge || target == outer.cont || target == outer.header) Fail("control flow leaves more than one loop at once");
        }
        return target;
    }

    const IrBlock* Goto(const IrBlock* from, const IrBlock* target, const IrBlock* stop) {
        CopyPhis(from, target);
        return JumpOut(target, stop);
    }

    void CopyPhis(const IrBlock* from, const IrBlock* target) {
        std::vector<std::pair<const IrValue*, const IrValue*>> copies;
        for (const IrValue* inst : target->Instructions()) {
            if (!inst->IsPhi()) continue;
            for (std::size_t index = 0; index < inst->ArgumentCount(); ++index) {
                if (inst->PhiBlock(index) != from) continue;
                copies.emplace_back(inst, inst->Argument(index));
                break;
            }
        }
        if (copies.empty()) return;
        bool overlapping = false;
        for (const auto& copy : copies) {
            const IrValue* source = copy.second->Resolve();
            overlapping |= source->IsPhi() && source->Parent() == target;
        }
        if (!overlapping) {
            for (const auto& copy : copies) Line(state, Name(*copy.first) + " = " + Value(state, copy.second) + ";");
            return;
        }
        std::vector<std::string> temps;
        for (const auto& copy : copies) {
            temps.push_back(Temp(state));
            Line(state, "const " + TypeName(copy.first->Type()) + " " + temps.back() + " = " + Value(state, copy.second) + ";");
        }
        for (std::size_t index = 0; index < copies.size(); ++index) Line(state, Name(*copies[index].first) + " = " + temps[index] + ";");
    }

    std::string Condition(const BlockInfo& info) {
        if (info.condition == nullptr) Fail("a conditional branch has no condition");
        const auto value = Value(state, info.condition);
        switch (info.terminator.condition) {
        case BranchCondition::ExecNonZero:
        case BranchCondition::VccNonZero: return WaveAny(state, value);
        case BranchCondition::ExecZero:
        case BranchCondition::VccZero: return "!" + WaveAny(state, "!" + value);
        default: return value;
        }
    }

    void Body(const IrBlock* block) {
        if (!emitted.insert(block).second) Fail("control flow emits block " + std::to_string(block->Id()) + " twice");
        EmitBlockBody(state, *block);
    }

    const IrBlock* Arm(const IrBlock* from, const IrBlock* target, const IrBlock* stop) {
        return Sequence(Goto(from, target, stop), stop);
    }

    const IrBlock* Terminate(const IrBlock* block, const BlockInfo& info, const IrBlock* stop) {
        const auto& term = info.terminator;
        switch (term.kind) {
        case TerminatorKind::Branch:
            return Goto(block, Target(term.trueBlock), stop);
        case TerminatorKind::ConditionalBranch: {
            const auto* whenTrue = Target(term.trueBlock);
            const auto* whenFalse = Target(term.falseBlock);
            if (whenTrue == whenFalse) return Goto(block, whenTrue, stop);
            const IrBlock* merge = !term.loopHeader && term.mergeBlock != InvalidControlFlowId ? Target(term.mergeBlock) : nullptr;
            const auto* until = merge != nullptr ? merge : stop;
            Open(state, "if (" + Condition(info) + ") {");
            const auto* trueEnd = Arm(block, whenTrue, until);
            Close(state, "} else {");
            ++state.depth;
            const auto* falseEnd = Arm(block, whenFalse, until);
            Close(state);
            if (merge == nullptr) return (trueEnd == nullptr && falseEnd == nullptr) ? nullptr : stop;
            if (trueEnd == nullptr && falseEnd == nullptr) return nullptr;
            return JumpOut(merge, stop);
        }
        default:
            Line(state, "return;");
            return nullptr;
        }
    }

    const IrBlock* EmitBlock(const IrBlock* block, const IrBlock* stop) {
        Body(block);
        return Terminate(block, Info(block), stop);
    }

    const IrBlock* EmitLoop(const IrBlock* header) {
        const auto& info = Info(header);
        Loop loop;
        loop.header = header;
        loop.merge = Target(info.terminator.mergeBlock);
        loop.cont = Target(info.terminator.continueBlock);
        loop.wrapped = loop.cont != header;
        Open(state, "while (true) {");
        if (loop.wrapped) {
            loop.flag = Temp(state);
            Line(state, "bool " + loop.flag + " = false;");
            Open(state, "do {");
            loops.push_back(loop);
            Body(header);
            Sequence(Terminate(header, info, loop.cont), loop.cont);
            Close(state, "} while (false);");
            Line(state, "if (" + loop.flag + ") break;");
            loops.back().wrapped = false;
            Sequence(loop.cont, nullptr);
        } else {
            loops.push_back(loop);
            Body(header);
            Sequence(Terminate(header, info, nullptr), nullptr);
        }
        loops.pop_back();
        Close(state);
        return loop.merge;
    }

    State& state;
    std::unordered_map<const IrBlock*, const BlockInfo*> byBlock;
    std::unordered_map<std::uint32_t, const IrBlock*> byId;
    std::unordered_set<const IrBlock*> emitted;
    std::vector<Loop> loops;
};

}

void EmitFlow(State& state) {
    Flow{state}.Run();
}

}
