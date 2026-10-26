#include "Profile.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <set>

namespace Translator {

void Instrument(llvm::Function& function, std::uint64_t offset) {
    auto& context = function.getContext();
    auto& module = *function.getParent();
    auto* i64 = llvm::Type::getInt64Ty(context);
    auto* type = llvm::StructType::get(context, {i64, i64});
    auto* row = new llvm::GlobalVariable(module, type, false, llvm::GlobalValue::InternalLinkage,
                                         llvm::ConstantStruct::get(type, {llvm::ConstantInt::get(i64, offset), llvm::ConstantInt::get(i64, 0)}), "p_" + function.getName().drop_front(2));
    row->setSection("__DATA,__aps5_prof");
    row->setAlignment(llvm::Align(8));
    llvm::appendToUsed(module, {row});
    llvm::SmallVector<std::pair<const llvm::BasicBlock*, const llvm::BasicBlock*>> backEdges;
    llvm::FindFunctionBackedges(function, backEdges);
    std::set<llvm::Instruction*> sites{function.getEntryBlock().getTerminator()};
    for (const auto& [from, to] : backEdges) sites.insert(&*const_cast<llvm::BasicBlock*>(to)->getFirstInsertionPt());
    for (auto* site : sites) {
        llvm::IRBuilder<> builder(site);
        auto* count = builder.CreateConstInBoundsGEP2_32(type, row, 0, 1);
        builder.CreateStore(builder.CreateAdd(builder.CreateLoad(i64, count), builder.getInt64(1)), count);
    }
}

std::map<std::uint64_t, std::uint64_t> ReadProfile(const std::string& path) {
    std::map<std::uint64_t, std::uint64_t> counts;
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) return counts;
    llvm::SmallVector<llvm::StringRef> lines;
    (*buffer)->getBuffer().split(lines, '\n', -1, false);
    for (const auto line : lines) {
        const auto [offsetText, countText] = line.split(' ');
        std::uint64_t offset = 0, count = 0;
        if (offsetText.getAsInteger(16, offset) || countText.trim().getAsInteger(10, count)) continue;
        counts[offset] += count;
    }
    return counts;
}

}
