#include "LifterImpl.hpp"

#include <cmath>

namespace Translator {

namespace {

constexpr std::uint16_t C0 = 1u << 8, C1 = 1u << 9, C2 = 1u << 10, C3 = 1u << 14;

bool StartsWith(const std::string& text, const char* prefix) {
    return text.rfind(prefix, 0) == 0;
}

bool EndsWith(const std::string& text, char last) {
    return !text.empty() && text.back() == last;
}

}

struct X87 {
    Lifter::Impl& lifter;
    llvm::IRBuilder<>& builder;
    const Instruction& instruction;

    llvm::Type* F64() { return builder.getDoubleTy(); }
    llvm::Value* Top() { return builder.CreateLoad(builder.getInt8Ty(), lifter.Field(StateLayout::X87Top)); }
    void SetTop(llvm::Value* top) { builder.CreateStore(builder.CreateAnd(top, builder.getInt8(7)), lifter.Field(StateLayout::X87Top)); }
    llvm::Value* Slot(unsigned index) {
        auto* physical = builder.CreateAnd(builder.CreateAdd(Top(), builder.getInt8(static_cast<std::uint8_t>(index))), builder.getInt8(7));
        auto* offset = builder.CreateAdd(builder.getInt64(StateLayout::X87), builder.CreateMul(builder.CreateZExt(physical, builder.getInt64Ty()), builder.getInt64(8)));
        return builder.CreateInBoundsGEP(builder.getInt8Ty(), lifter.state, offset);
    }
    llvm::Value* St(unsigned index) { return builder.CreateLoad(F64(), Slot(index)); }
    void SetSt(unsigned index, llvm::Value* value) { builder.CreateStore(value, Slot(index)); }
    void Push(llvm::Value* value) {
        SetTop(builder.CreateSub(Top(), builder.getInt8(1)));
        SetSt(0, value);
    }
    void Pop() { SetTop(builder.CreateAdd(Top(), builder.getInt8(1))); }
    llvm::Value* Control() { return builder.CreateLoad(builder.getInt16Ty(), lifter.Field(StateLayout::X87Control)); }
    llvm::Value* Status() { return builder.CreateLoad(builder.getInt16Ty(), lifter.Field(StateLayout::X87Status)); }
    void SetStatus(llvm::Value* value) { builder.CreateStore(value, lifter.Field(StateLayout::X87Status)); }

    std::vector<unsigned> Stack() const {
        std::vector<unsigned> registers;
        for (unsigned i = 0; i < instruction.x86.op_count; ++i) {
            const auto& op = instruction.Op(i);
            if (op.type == X86_OP_REG && op.reg >= X86_REG_ST0 && op.reg <= X86_REG_ST7) registers.push_back(op.reg - X86_REG_ST0);
        }
        return registers;
    }
    const cs_x86_op* MemoryOperand() const {
        for (unsigned i = 0; i < instruction.x86.op_count; ++i) {
            if (instruction.Op(i).type == X86_OP_MEM) return &instruction.Op(i);
        }
        return nullptr;
    }
    llvm::Value* Address(const cs_x86_op& op) { return lifter.Address(instruction, op.mem); }

    llvm::FunctionCallee Runtime(const char* name, llvm::Type* result, std::vector<llvm::Type*> parameters) {
        return lifter.module->getOrInsertFunction(name, llvm::FunctionType::get(result, parameters, false));
    }

    llvm::Value* LoadFloat(const cs_x86_op& op) {
        auto* address = Address(op);
        switch (op.size) {
        case 4: return builder.CreateFPExt(lifter.Load(builder.getFloatTy(), address), F64());
        case 8: return lifter.Load(F64(), address);
        case 10: return builder.CreateCall(Runtime("aps5_x87_load80", F64(), {builder.getPtrTy()}), {lifter.Pointer(address)});
        default: throw Unsupported{instruction.mnemonic};
        }
    }
    void StoreFloat(const cs_x86_op& op, llvm::Value* value) {
        auto* address = Address(op);
        switch (op.size) {
        case 4: lifter.Store(builder.CreateFPTrunc(value, builder.getFloatTy()), address); return;
        case 8: lifter.Store(value, address); return;
        case 10: builder.CreateCall(Runtime("aps5_x87_store80", builder.getVoidTy(), {builder.getPtrTy(), F64()}), {lifter.Pointer(address), value}); return;
        default: throw Unsupported{instruction.mnemonic};
        }
    }
    llvm::Value* LoadInteger(const cs_x86_op& op) {
        if (op.size != 2 && op.size != 4 && op.size != 8) throw Unsupported{instruction.mnemonic};
        return builder.CreateSIToFP(lifter.Load(lifter.Int(op.size * 8u), Address(op)), F64());
    }
    llvm::Value* Round(llvm::Value* value) {
        auto* mode = builder.CreateAnd(builder.CreateLShr(Control(), 10), builder.getInt16(3));
        auto* nearest = builder.CreateUnaryIntrinsic(llvm::Intrinsic::roundeven, value);
        auto* down = builder.CreateUnaryIntrinsic(llvm::Intrinsic::floor, value);
        auto* up = builder.CreateUnaryIntrinsic(llvm::Intrinsic::ceil, value);
        auto* zero = builder.CreateUnaryIntrinsic(llvm::Intrinsic::trunc, value);
        auto* result = builder.CreateSelect(builder.CreateICmpEQ(mode, builder.getInt16(1)), down, nearest);
        result = builder.CreateSelect(builder.CreateICmpEQ(mode, builder.getInt16(2)), up, result);
        return builder.CreateSelect(builder.CreateICmpEQ(mode, builder.getInt16(3)), zero, result);
    }
    void StoreInteger(const cs_x86_op& op, llvm::Value* value, bool truncate) {
        if (op.size != 2 && op.size != 4 && op.size != 8) throw Unsupported{instruction.mnemonic};
        auto* rounded = truncate ? builder.CreateUnaryIntrinsic(llvm::Intrinsic::trunc, value) : Round(value);
        auto* type = lifter.Int(op.size * 8u);
        auto* minimum = llvm::ConstantFP::get(F64(), -std::ldexp(1.0, static_cast<int>(op.size * 8u - 1)));
        auto* inRange = builder.CreateAnd(builder.CreateFCmpOGE(rounded, minimum), builder.CreateFCmpOLT(rounded, builder.CreateFNeg(minimum)));
        auto* converted = builder.CreateFPToSI(builder.CreateSelect(inRange, rounded, llvm::ConstantFP::get(F64(), 0.0)), type);
        auto* indefinite = llvm::ConstantInt::get(type, llvm::APInt::getSignedMinValue(op.size * 8u));
        lifter.Store(builder.CreateSelect(inRange, converted, indefinite), Address(op));
    }

    llvm::Value* Arithmetic(char operation, llvm::Value* a, llvm::Value* b) {
        switch (operation) {
        case '+': return builder.CreateFAdd(a, b);
        case '-': return builder.CreateFSub(a, b);
        case '*': return builder.CreateFMul(a, b);
        default: return builder.CreateFDiv(a, b);
        }
    }

    void Binary(char operation, bool reverse, bool integer, bool pop) {
        llvm::Value* source = nullptr;
        unsigned destination = 0;
        if (const auto* memory = MemoryOperand()) {
            source = integer ? LoadInteger(*memory) : LoadFloat(*memory);
        } else {
            const auto registers = Stack();
            if (registers.size() >= 2) {
                destination = registers[0];
                source = St(registers[1]);
            } else if (registers.size() == 1) {
                destination = pop ? registers[0] : 0;
                source = St(pop ? 0 : registers[0]);
            } else {
                destination = 1;
                source = St(0);
            }
        }
        auto* target = St(destination);
        SetSt(destination, reverse ? Arithmetic(operation, source, target) : Arithmetic(operation, target, source));
        if (pop) Pop();
    }

    void Compare(llvm::Value* a, llvm::Value* b) {
        auto* unordered = builder.CreateFCmpUNO(a, b);
        auto* less = builder.CreateOr(builder.CreateFCmpOLT(a, b), unordered);
        auto* equal = builder.CreateOr(builder.CreateFCmpOEQ(a, b), unordered);
        auto* bits = builder.CreateOr(builder.CreateSelect(less, builder.getInt16(C0), builder.getInt16(0)),
                                      builder.CreateOr(builder.CreateSelect(unordered, builder.getInt16(C2), builder.getInt16(0)),
                                                       builder.CreateSelect(equal, builder.getInt16(C3), builder.getInt16(0))));
        SetStatus(builder.CreateOr(builder.CreateAnd(Status(), builder.getInt16(static_cast<std::uint16_t>(~(C0 | C1 | C2 | C3)))), bits));
    }

    void CompareFlags(llvm::Value* a, llvm::Value* b) {
        auto* unordered = builder.CreateFCmpUNO(a, b);
        lifter.SetFlag(Zf, builder.CreateOr(builder.CreateFCmpOEQ(a, b), unordered));
        lifter.SetFlag(Pf, unordered);
        lifter.SetFlag(Cf, builder.CreateOr(builder.CreateFCmpOLT(a, b), unordered));
        lifter.SetFlag(Of, builder.getFalse());
        lifter.SetFlag(Sf, builder.getFalse());
        lifter.SetFlag(Af, builder.getFalse());
    }

    void Examine() {
        auto* value = St(0);
        auto* bits = builder.CreateBitCast(value, builder.getInt64Ty());
        auto* exponent = builder.CreateAnd(builder.CreateLShr(bits, 52), builder.getInt64(0x7ff));
        auto* mantissa = builder.CreateAnd(bits, builder.getInt64((1ull << 52) - 1));
        auto* sign = builder.CreateICmpSLT(bits, builder.getInt64(0));
        auto* zeroExponent = builder.CreateICmpEQ(exponent, builder.getInt64(0));
        auto* maxExponent = builder.CreateICmpEQ(exponent, builder.getInt64(0x7ff));
        auto* zeroMantissa = builder.CreateICmpEQ(mantissa, builder.getInt64(0));
        llvm::Value* kind = builder.getInt16(C2);
        kind = builder.CreateSelect(builder.CreateAnd(zeroExponent, zeroMantissa), builder.getInt16(C3), kind);
        kind = builder.CreateSelect(builder.CreateAnd(maxExponent, zeroMantissa), builder.getInt16(C2 | C0), kind);
        kind = builder.CreateSelect(builder.CreateAnd(maxExponent, builder.CreateNot(zeroMantissa)), builder.getInt16(C0), kind);
        kind = builder.CreateOr(kind, builder.CreateSelect(sign, builder.getInt16(C1), builder.getInt16(0)));
        SetStatus(builder.CreateOr(builder.CreateAnd(Status(), builder.getInt16(static_cast<std::uint16_t>(~(C0 | C1 | C2 | C3)))), kind));
    }

    llvm::Value* Unary(llvm::Intrinsic::ID id, llvm::Value* value) { return builder.CreateUnaryIntrinsic(id, value); }

    bool Emit() {
        const auto& m = instruction.mnemonic;
        const auto registers = Stack();
        const auto* memory = MemoryOperand();
        if (m == "wait" || m == "fwait" || m == "fnop" || m == "ffree" || m == "ffreep") return true;
        if (m == "fninit" || m == "finit") {
            builder.CreateStore(builder.getInt16(0x037f), lifter.Field(StateLayout::X87Control));
            SetStatus(builder.getInt16(0));
            SetTop(builder.getInt8(0));
            return true;
        }
        if (m == "fnclex" || m == "fclex") {
            SetStatus(builder.CreateAnd(Status(), builder.getInt16(0x7f00)));
            return true;
        }
        if ((m == "fnstcw" || m == "fstcw") && memory != nullptr) {
            lifter.Store(Control(), Address(*memory));
            return true;
        }
        if (m == "fldcw" && memory != nullptr) {
            builder.CreateStore(lifter.Load(builder.getInt16Ty(), Address(*memory)), lifter.Field(StateLayout::X87Control));
            return true;
        }
        if (m == "fnstsw" || m == "fstsw") {
            auto* top = builder.CreateShl(builder.CreateZExt(Top(), builder.getInt16Ty()), 11);
            auto* word = builder.CreateOr(builder.CreateAnd(Status(), builder.getInt16(static_cast<std::uint16_t>(~0x3800u))), top);
            if (memory != nullptr) lifter.Store(word, Address(*memory));
            else lifter.WriteRegister(X86_REG_AX, word);
            return true;
        }
        if ((m == "fnstenv" || m == "fstenv") && memory != nullptr) {
            auto* base = Address(*memory);
            auto* top = builder.CreateShl(builder.CreateZExt(Top(), builder.getInt16Ty()), 11);
            auto* status = builder.CreateOr(builder.CreateAnd(Status(), builder.getInt16(static_cast<std::uint16_t>(~0x3800u))), top);
            lifter.Store(builder.CreateZExt(Control(), builder.getInt32Ty()), base);
            lifter.Store(builder.CreateZExt(status, builder.getInt32Ty()), builder.CreateAdd(base, builder.getInt64(4)));
            lifter.Store(builder.getInt32(0xffff), builder.CreateAdd(base, builder.getInt64(8)));
            for (unsigned offset = 12; offset < 28; offset += 4) lifter.Store(builder.getInt32(0), builder.CreateAdd(base, builder.getInt64(offset)));
            builder.CreateStore(builder.CreateOr(Control(), builder.getInt16(0x3f)), lifter.Field(StateLayout::X87Control));
            return true;
        }
        if (m == "fldenv" && memory != nullptr) {
            auto* base = Address(*memory);
            auto* status = lifter.Load(builder.getInt16Ty(), builder.CreateAdd(base, builder.getInt64(4)));
            builder.CreateStore(lifter.Load(builder.getInt16Ty(), base), lifter.Field(StateLayout::X87Control));
            SetStatus(builder.CreateAnd(status, builder.getInt16(static_cast<std::uint16_t>(~0x3800u))));
            SetTop(builder.CreateTrunc(builder.CreateLShr(status, 11), builder.getInt8Ty()));
            return true;
        }
        if (m == "fincstp") {
            SetTop(builder.CreateAdd(Top(), builder.getInt8(1)));
            return true;
        }
        if (m == "fdecstp") {
            SetTop(builder.CreateSub(Top(), builder.getInt8(1)));
            return true;
        }
        if (m == "fld") {
            Push(memory != nullptr ? LoadFloat(*memory) : St(registers.empty() ? 0 : registers.back()));
            return true;
        }
        if (m == "fild" && memory != nullptr) {
            Push(LoadInteger(*memory));
            return true;
        }
        if (m == "fbld" && memory != nullptr) {
            Push(builder.CreateCall(Runtime("aps5_x87_loadbcd", F64(), {builder.getPtrTy()}), {lifter.Pointer(Address(*memory))}));
            return true;
        }
        static const std::pair<const char*, double> constants[] = {
            {"fld1", 1.0}, {"fldz", 0.0}, {"fldpi", 3.14159265358979323846}, {"fldl2e", 1.44269504088896340736}, {"fldl2t", 3.32192809488736234787},
            {"fldlg2", 0.30102999566398119521}, {"fldln2", 0.69314718055994530942}};
        for (const auto& [name, value] : constants) {
            if (m == name) {
                Push(llvm::ConstantFP::get(F64(), value));
                return true;
            }
        }
        if (m == "fst" || m == "fstp") {
            auto* value = St(0);
            if (memory != nullptr) StoreFloat(*memory, value);
            else SetSt(registers.empty() ? 1 : registers.front(), value);
            if (m == "fstp") Pop();
            return true;
        }
        if ((m == "fist" || m == "fistp" || m == "fisttp") && memory != nullptr) {
            StoreInteger(*memory, St(0), m == "fisttp");
            if (m != "fist") Pop();
            return true;
        }
        if (m == "fbstp" && memory != nullptr) {
            builder.CreateCall(Runtime("aps5_x87_storebcd", builder.getVoidTy(), {builder.getPtrTy(), F64()}), {lifter.Pointer(Address(*memory)), Round(St(0))});
            Pop();
            return true;
        }
        if (m == "fxch") {
            const unsigned other = registers.empty() ? 1 : registers.back() == 0 && registers.size() > 1 ? registers.front() : registers.back();
            auto* a = St(0);
            auto* b = St(other);
            SetSt(0, b);
            SetSt(other, a);
            return true;
        }
        static const std::pair<const char*, char> operations[] = {{"add", '+'}, {"sub", '-'}, {"mul", '*'}, {"div", '/'}};
        for (const auto& [name, operation] : operations) {
            const std::string base = std::string("f") + name;
            const std::string integer = std::string("fi") + name;
            const bool plain = m == base || m == base + "p";
            const bool reverse = m == base + "r" || m == base + "rp";
            const bool fromInteger = m == integer || m == integer + "r";
            if (plain || reverse || fromInteger) {
                Binary(operation, reverse || (fromInteger && EndsWith(m, 'r')), fromInteger, !fromInteger && EndsWith(m, 'p'));
                return true;
            }
        }
        if (m == "fchs") return SetSt(0, builder.CreateFNeg(St(0))), true;
        if (m == "fabs") return SetSt(0, Unary(llvm::Intrinsic::fabs, St(0))), true;
        if (m == "fsqrt") return SetSt(0, Unary(llvm::Intrinsic::sqrt, St(0))), true;
        if (m == "frndint") return SetSt(0, Round(St(0))), true;
        if (m == "fsin") return SetSt(0, Unary(llvm::Intrinsic::sin, St(0))), true;
        if (m == "fcos") return SetSt(0, Unary(llvm::Intrinsic::cos, St(0))), true;
        if (m == "f2xm1") return SetSt(0, builder.CreateFSub(Unary(llvm::Intrinsic::exp2, St(0)), llvm::ConstantFP::get(F64(), 1.0))), true;
        if (m == "fsincos") {
            auto* value = St(0);
            SetSt(0, Unary(llvm::Intrinsic::sin, value));
            Push(Unary(llvm::Intrinsic::cos, value));
            return true;
        }
        if (m == "fptan") {
            auto* value = St(0);
            SetSt(0, builder.CreateFDiv(Unary(llvm::Intrinsic::sin, value), Unary(llvm::Intrinsic::cos, value)));
            Push(llvm::ConstantFP::get(F64(), 1.0));
            return true;
        }
        if (m == "fpatan" || m == "fyl2x" || m == "fyl2xp1") {
            auto* x = St(0);
            auto* y = St(1);
            llvm::Value* result = nullptr;
            if (m == "fpatan") result = builder.CreateCall(Runtime("atan2", F64(), {F64(), F64()}), {y, x});
            else result = builder.CreateFMul(y, Unary(llvm::Intrinsic::log2, m == "fyl2x" ? x : builder.CreateFAdd(x, llvm::ConstantFP::get(F64(), 1.0))));
            SetSt(1, result);
            Pop();
            return true;
        }
        if (m == "fscale") {
            auto* exponent = builder.CreateFPToSI(Unary(llvm::Intrinsic::trunc, St(1)), builder.getInt32Ty());
            SetSt(0, builder.CreateIntrinsic(llvm::Intrinsic::ldexp, {F64(), builder.getInt32Ty()}, {St(0), exponent}));
            return true;
        }
        if (m == "fprem" || m == "fprem1") {
            auto* a = St(0);
            auto* b = St(1);
            if (m == "fprem") SetSt(0, builder.CreateFRem(a, b));
            else SetSt(0, builder.CreateFSub(a, builder.CreateFMul(b, Unary(llvm::Intrinsic::roundeven, builder.CreateFDiv(a, b)))));
            SetStatus(builder.CreateAnd(Status(), builder.getInt16(static_cast<std::uint16_t>(~C2))));
            return true;
        }
        if (m == "fxtract") {
            auto* value = St(0);
            auto* split = builder.CreateIntrinsic(llvm::Intrinsic::frexp, {F64(), builder.getInt32Ty()}, {value});
            auto* exponent = builder.CreateSIToFP(builder.CreateSub(builder.CreateExtractValue(split, 1), builder.getInt32(1)), F64());
            SetSt(0, exponent);
            Push(builder.CreateFMul(builder.CreateExtractValue(split, 0), llvm::ConstantFP::get(F64(), 2.0)));
            return true;
        }
        if (m == "fxam") {
            Examine();
            return true;
        }
        if (m == "ftst") {
            Compare(St(0), llvm::ConstantFP::get(F64(), 0.0));
            return true;
        }
        if (m == "fcomi" || m == "fucomi" || m == "fcomip" || m == "fucomip" || m == "fcompi" || m == "fucompi") {
            CompareFlags(St(0), St(registers.size() >= 2 ? registers[1] : registers.empty() ? 1 : registers.back()));
            if (m.size() > 6 || m == "fcompi" || m == "fcomip") Pop();
            return true;
        }
        if (m == "fcom" || m == "fcomp" || m == "fucom" || m == "fucomp" || m == "fcompp" || m == "fucompp" || m == "ficom" || m == "ficomp") {
            llvm::Value* other = nullptr;
            if (memory != nullptr) other = StartsWith(m, "fi") ? LoadInteger(*memory) : LoadFloat(*memory);
            else other = St(registers.size() >= 2 ? registers[1] : registers.empty() ? 1 : registers.back());
            Compare(St(0), other);
            if (EndsWith(m, 'p')) Pop();
            if (m == "fcompp" || m == "fucompp") Pop();
            return true;
        }
        static const std::pair<const char*, Condition> moves[] = {{"fcmovb", Condition::B},   {"fcmove", Condition::E},   {"fcmovbe", Condition::Be}, {"fcmovu", Condition::P},
                                                                  {"fcmovnb", Condition::Ae}, {"fcmovne", Condition::Ne}, {"fcmovnbe", Condition::A}, {"fcmovnu", Condition::Np}};
        for (const auto& [name, condition] : moves) {
            if (m == name) {
                const unsigned source = registers.size() >= 2 ? registers[1] : registers.empty() ? 1 : registers.back();
                SetSt(0, builder.CreateSelect(lifter.Test(condition), St(source), St(0)));
                return true;
            }
        }
        return false;
    }
};

bool Lifter::Impl::EmitX87(const Instruction& instruction) {
    const auto& m = instruction.mnemonic;
    if (m.empty() || (m.front() != 'f' && m != "wait")) return false;
    X87 x87{*this, builder, instruction};
    return x87.Emit();
}

}
