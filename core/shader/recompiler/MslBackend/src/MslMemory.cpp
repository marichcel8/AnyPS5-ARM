#include "MslState.hpp"
#include "BdaAbi.hpp"
#include "FlatApertures.hpp"
#include <algorithm>
#include <functional>

namespace ShaderRecompiler::Msl {

namespace {

struct Atomic {
    std::uint32_t op = 0;
    std::string a;
    std::string b = "0u";
};

std::string Pc(const IrValue& inst) {
    return U32(inst.Flags<MemoryFlags>().pc);
}

std::string Active(State& state, const IrValue& inst) {
    const IrValue* active = inst.Argument(inst.ArgumentCount() - 1u)->Resolve();
    if (active->HasImmediate() && active->Type() == IrType::Bool && active->ImmediateBool()) return {};
    return Typed(state, inst, inst.ArgumentCount() - 1u);
}

void OpenActive(State& state, const std::string& active) {
    Open(state, active.empty() ? std::string("{") : "if (" + active + ") {");
}

void StopCheck(State& state) {
    if (state.stopAllowed) Line(state, "if (aps5_c.stop) return;");
}

std::string Component(const std::string& value, std::uint32_t components, std::uint32_t component) {
    static constexpr std::string_view Names[4] = {"x", "y", "z", "w"};
    return components == 1u ? value : value + "." + std::string(Names[component]);
}

std::string ZeroOf(const IrValue& inst) {
    const auto type = TypeName(inst.Type());
    return type == "bool" ? std::string("false") : type + "(0)";
}

std::string Result(State& state, const IrValue& inst) {
    const auto name = Temp(state);
    Line(state, TypeName(inst.Type()) + " " + name + " = " + ZeroOf(inst) + ";");
    return name;
}

std::uint32_t BufferElement(State& state, const IrValue& inst, const MemoryInfo& mem) {
    for (const auto& descriptor : state.layout.descriptors) {
        if (descriptor.kind != DescriptorBindingKind::Buffers) continue;
        const auto found = std::find(descriptor.resources.begin(), descriptor.resources.end(), mem.resource);
        if (found == descriptor.resources.end()) break;
        return static_cast<std::uint32_t>(found - descriptor.resources.begin());
    }
    Fail(inst, "reads a buffer the descriptor layout does not bind");
}

std::string VSharp(State& state, const IrValue& inst, const MemoryInfo& mem) {
    const IrValue* handle = inst.Argument(0)->Resolve();
    if (handle->Opcode() != IrOpcode::GetBufferResource || handle->ArgumentCount() != 4u) Fail(inst, "requires a runtime four-dword V#");
    if (mem.formatted) throw MslUnsupported("MSL backend: formatted buffer access is not emitted (section S3)");
    std::string words;
    for (std::size_t index = 0; index < 4u; ++index) words += Arg(state, *handle, index) + ", ";
    std::string element = "0u";
    if (mem.kind == ResourceKind::Buffer) {
        element = Typed(state, inst, 1u) + " + (extract_bits(" + Arg(state, *handle, 3u) + ", 23u, 1u) != 0u ? " + WaveLane(state) + " : 0u)";
    }
    state.features |= Descriptors | Fault;
    const auto name = Temp(state);
    if (mem.gpuDescriptor) {
        state.features |= Bda;
        Line(state, "const aps5_vsharp " + name + " = aps5_vsharp_gpu(aps5_c, " + words + element + ", " + Pc(inst) + ");");
    } else {
        state.features |= GuestBuffers;
        Line(state, "const aps5_vsharp " + name + " = aps5_vsharp_make(" + words + element + ");");
    }
    return name;
}

std::string BufferLoad(State& state, const IrValue& inst, const MemoryInfo& mem, const std::string& resource, const std::string& byte, std::uint32_t bytes) {
    if (mem.gpuDescriptor) return "aps5_vsharp_load(aps5_c, " + resource + ", " + byte + ", " + U32(bytes) + ", " + Pc(inst) + ")";
    return "aps5_buffer_load(aps5_c, " + U32(BufferElement(state, inst, mem)) + ", " + resource + ", " + byte + ", " + U32(bytes) + ")";
}

void LoadBuffer(State& state, const IrValue& inst, std::uint32_t components) {
    const auto& mem = Memory(state, inst);
    if (mem.planningOnly) {
        DefineZero(state, inst);
        return;
    }
    if (mem.kind != ResourceKind::Buffer) Fail(inst, "must access a buffer resource");
    const auto result = Result(state, inst);
    OpenActive(state, Active(state, inst));
    const auto resource = VSharp(state, inst, mem);
    const auto offset = Typed(state, inst, 2u);
    const auto soffset = Typed(state, inst, 3u);
    const auto bytes = mem.dataBits / 8u;
    const auto byte = Temp(state);
    Line(state, "uint " + byte + " = 0u;");
    for (std::uint32_t component = 0; component < components; ++component) {
        Open(state, "if (aps5_vsharp_byte(" + resource + ", " + offset + " + " + U32(mem.offset + component * 4u) + ", " + soffset + ", " + U32(bytes) + ", " + byte + ")) {");
        Line(state, Component(result, components, component) + " = " + BufferLoad(state, inst, mem, resource, byte, bytes) + ";");
        Close(state);
    }
    Close(state);
    if (mem.gpuDescriptor) StopCheck(state);
    Define(state, inst, result);
}

void StoreBuffer(State& state, const IrValue& inst, std::uint32_t components) {
    const auto& mem = Memory(state, inst);
    if (mem.kind != ResourceKind::Buffer) Fail(inst, "must access a buffer resource");
    OpenActive(state, Active(state, inst));
    const auto resource = VSharp(state, inst, mem);
    const auto offset = Typed(state, inst, 2u);
    const auto soffset = Typed(state, inst, 3u);
    const auto data = Typed(state, inst, inst.ArgumentCount() - 2u);
    const auto bytes = mem.dataBits / 8u;
    const auto byte = Temp(state);
    Line(state, "uint " + byte + " = 0u;");
    for (std::uint32_t component = 0; component < components; ++component) {
        const auto value = Component(data, components, component);
        Open(state, "if (aps5_vsharp_byte(" + resource + ", " + offset + " + " + U32(mem.offset + component * 4u) + ", " + soffset + ", " + U32(bytes) + ", " + byte + ")) {");
        if (mem.gpuDescriptor) {
            state.features |= BdaWrites;
            Line(state, "aps5_vsharp_store(aps5_c, " + resource + ", " + byte + ", " + U32(bytes) + ", " + value + ", " + Pc(inst) + ");");
        } else {
            Line(state, "aps5_buffer_store(aps5_c, " + U32(BufferElement(state, inst, mem)) + ", " + resource + ", " + byte + ", " + U32(bytes) + ", " + value + ");");
        }
        Close(state);
    }
    Close(state);
    if (mem.gpuDescriptor) StopCheck(state);
}

void ReadConstBuffer(State& state, const IrValue& inst) {
    const auto& mem = Memory(state, inst);
    if (mem.planningOnly) {
        DefineZero(state, inst);
        return;
    }
    if (mem.kind != ResourceKind::ScalarBuffer) Fail(inst, "must read a scalar buffer resource");
    const auto result = Result(state, inst);
    Open(state, "{");
    const auto resource = VSharp(state, inst, mem);
    const auto offset = Typed(state, inst, 1u);
    const auto address = Temp(state);
    const auto byte = Temp(state);
    const auto size = Temp(state);
    Line(state, "const uint " + address + " = " + offset + " + " + U32(mem.offset) + ";");
    Line(state, "const uint " + byte + " = " + address + " & ~3u;");
    Line(state, "const ulong " + size + " = " + resource + ".stride == 0u ? ulong(" + resource + ".records) : ulong(" + resource + ".stride) * ulong(" + resource + ".records);");
    Open(state, "if (" + address + " >= " + offset + " && " + resource + ".valid && ulong(" + byte + ") + 4ul <= " + size + ") {");
    Line(state, result + " = " + BufferLoad(state, inst, mem, resource, byte, 4u) + ";");
    Close(state);
    Close(state);
    if (mem.gpuDescriptor) StopCheck(state);
    Define(state, inst, result);
}

Atomic BufferAtomicOf(State& state, const IrValue& inst) {
    const auto count = inst.ArgumentCount();
    const auto data = [&] { return Typed(state, inst, count - 2u); };
    switch (inst.Opcode()) {
    case IrOpcode::BufferAtomicSwap32: return {0u, data()};
    case IrOpcode::BufferAtomicCmpSwap32: return {1u, Typed(state, inst, count - 3u), Typed(state, inst, count - 2u)};
    case IrOpcode::BufferAtomicIAdd32: return {2u, data()};
    case IrOpcode::BufferAtomicISub32: return {3u, data()};
    case IrOpcode::BufferAtomicSMin32: return {4u, data()};
    case IrOpcode::BufferAtomicUMin32: return {5u, data()};
    case IrOpcode::BufferAtomicSMax32: return {6u, data()};
    case IrOpcode::BufferAtomicUMax32: return {7u, data()};
    case IrOpcode::BufferAtomicAnd32: return {8u, data()};
    case IrOpcode::BufferAtomicOr32: return {9u, data()};
    case IrOpcode::BufferAtomicXor32: return {10u, data()};
    case IrOpcode::BufferAtomicInc32: return {11u, data()};
    case IrOpcode::BufferAtomicDec32: return {12u, data()};
    case IrOpcode::BufferAtomicUSubSat32: return {13u, data()};
    case IrOpcode::BufferAtomicFMin32: return {15u, data()};
    case IrOpcode::BufferAtomicFMax32: return {16u, data()};
    case IrOpcode::BufferAtomicFCmpSwap32: return {20u, Typed(state, inst, count - 3u), Typed(state, inst, count - 2u)};
    default: throw MslUnsupported("MSL backend: " + std::string(IrOpcodeName(inst.Opcode())) + " is not emitted (64-bit atomics are section S5)");
    }
}

bool ZeroIdentity(std::uint32_t op) {
    return op == 2u || op == 3u || op == 9u || op == 10u;
}

void BufferAtomic(State& state, const IrValue& inst) {
    const auto& mem = Memory(state, inst);
    if (mem.kind != ResourceKind::Buffer) Fail(inst, "must access a buffer resource");
    const auto atomic = BufferAtomicOf(state, inst);
    const auto result = Result(state, inst);
    auto active = Active(state, inst);
    if (!inst.HasUses() && ZeroIdentity(atomic.op)) active = (active.empty() ? std::string() : active + " && ") + atomic.a + " != 0u";
    OpenActive(state, active);
    const auto resource = VSharp(state, inst, mem);
    const auto byte = Temp(state);
    const auto pointer = Temp(state);
    Line(state, "uint " + byte + " = 0u;");
    Open(state, "if (aps5_vsharp_byte(" + resource + ", " + Typed(state, inst, 2u) + " + " + U32(mem.offset) + ", " + Typed(state, inst, 3u) + ", 4u, " + byte + ")) {");
    if (mem.gpuDescriptor) {
        state.features |= BdaWrites;
        Line(state, "device atomic_uint* " + pointer + " = aps5_vsharp_atomic(aps5_c, " + resource + ", " + byte + ", " + Pc(inst) + ");");
    } else {
        Line(state, "device atomic_uint* " + pointer + " = aps5_buffer_atomic(aps5_c, " + U32(BufferElement(state, inst, mem)) + ", " + resource + ", " + byte + ", " + Pc(inst) + ");");
    }
    Line(state, "if (" + pointer + " != nullptr) " + result + " = aps5_atomic(" + pointer + ", " + U32(atomic.op) + ", " + atomic.a + ", " + atomic.b + ");");
    Close(state);
    Close(state);
    StopCheck(state);
    Define(state, inst, result);
}

struct GuestAddress {
    std::string address;
    std::string ok;
};

GuestAddress AddressOf(State& state, const IrValue& inst, const MemoryInfo& mem) {
    state.features |= Bda | Fault;
    const auto address = Temp(state);
    const auto ok = Temp(state);
    auto low = Typed(state, inst, 1u);
    if (mem.kind == ResourceKind::ScalarAddress) low = "(" + low + " & ~3u)";
    Line(state, "ulong " + address + " = 0ul;");
    if (mem.addressIsFull) {
        Line(state, address + " = (ulong(" + Typed(state, inst, 2u) + ") << 32u) | ulong(" + low + ");");
        Line(state, "bool " + ok + " = true;");
    } else {
        const IrValue* handle = inst.Argument(0)->Resolve();
        if (handle->Opcode() != IrOpcode::GetAddressResource || handle->ArgumentCount() != 2u) Fail(inst, "has no address base pair");
        Line(state, "bool " + ok + " = aps5_bda_add(aps5_c, (ulong(" + Arg(state, *handle, 1u) + ") << 32u) | ulong(" + Arg(state, *handle, 0u) + "), ulong(" + low + "), false, " + Pc(inst) + ", " + address + ");");
    }
    auto immediate = static_cast<std::int32_t>(mem.offset);
    if (mem.kind == ResourceKind::ScalarAddress) immediate = static_cast<std::int32_t>(static_cast<std::uint32_t>(immediate) & ~3u);
    if (immediate != 0) {
        const auto magnitude = immediate < 0 ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(immediate)) : static_cast<std::uint64_t>(immediate);
        Line(state, ok + " = " + ok + " && aps5_bda_add(aps5_c, " + address + ", " + std::to_string(magnitude) + "ul, " + (immediate < 0 ? "true" : "false") + ", " + Pc(inst) + ", " + address + ");");
    }
    return {address, ok};
}

bool RoutesApertures(const State& state, const MemoryInfo& mem) {
    return mem.kind == ResourceKind::Flat && state.program.Resources().stage == IrShaderStage::Compute;
}

struct Local {
    std::string read;
    std::string write;
};

Local LdsArray(State& state) {
    state.features |= Lds;
    const auto arguments = "(aps5_lds, " + U32(state.ldsDwords) + ", ";
    return {"aps5_read_lds" + arguments, "aps5_write_lds" + arguments};
}

Local ScratchArray(State& state) {
    state.features |= Scratch;
    const auto arguments = "(aps5_scratch, " + U32(state.scratchDwords) + ", ";
    return {"aps5_read_scratch" + arguments, "aps5_write_scratch" + arguments};
}

std::string ScratchByte(State& state, const IrValue& inst, const MemoryInfo& mem) {
    const auto immediate = static_cast<std::int32_t>(mem.offset);
    const auto sum = Temp(state);
    Line(state, "const ulong " + sum + " = ((ulong(" + Typed(state, inst, 2u) + ") << 32u) | ulong(" + Typed(state, inst, 1u) + ")) + ulong(long(" + std::to_string(immediate) + "));");
    return "(uint(" + sum + " >> 32u) == 0u ? uint(" + sum + ") : 0xffffffffu)";
}

void Route(State& state, const GuestAddress& guest, const std::function<void(const Local& space, const std::string& low)>& local, const std::function<void()>& global) {
    const auto high = "uint(" + guest.address + " >> 32u)";
    Open(state, "if (" + high + " == " + U32(SharedApertureHigh) + ") {");
    local(LdsArray(state), "uint(" + guest.address + ")");
    Close(state, "} else if (" + high + " == " + U32(PrivateApertureHigh) + ") {");
    ++state.depth;
    local(ScratchArray(state), "uint(" + guest.address + ")");
    Close(state, "} else {");
    ++state.depth;
    global();
    Close(state);
}

void LoadAddress(State& state, const IrValue& inst, std::uint32_t components, std::uint32_t bytes) {
    const auto& mem = Memory(state, inst);
    if (mem.planningOnly) {
        DefineZero(state, inst);
        return;
    }
    const auto result = Result(state, inst);
    OpenActive(state, Active(state, inst));
    if (mem.kind == ResourceKind::Scratch) {
        if (components != 1u) Fail(inst, "must read a physical address resource");
        const auto byte = ScratchByte(state, inst, mem);
        Line(state, result + " = " + ScratchArray(state).read + (bytes == 4u ? "(" + byte + ") & ~3u" : byte) + ", " + U32(bytes) + ");");
        Close(state);
        Define(state, inst, result);
        return;
    }
    if (mem.kind != ResourceKind::Flat && mem.kind != ResourceKind::Global && mem.kind != ResourceKind::ScalarAddress) Fail(inst, "must read a scratch or physical address resource");
    const auto guest = AddressOf(state, inst, mem);
    Open(state, "if (" + guest.ok + ") {");
    const auto global = [&] {
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto target = Component(result, components, component);
            if (component == 0u) {
                Line(state, target + " = aps5_bda_load(aps5_c, " + guest.address + ", " + U32(bytes) + ", " + Pc(inst) + ");");
            } else {
                const auto next = Temp(state);
                Line(state, "ulong " + next + " = 0ul;");
                Line(state, "if (aps5_bda_add(aps5_c, " + guest.address + ", " + std::to_string(component * 4u) + "ul, false, " + Pc(inst) + ", " + next + ")) " + target + " = aps5_bda_load(aps5_c, " + next + ", 4u, " + Pc(inst) + ");");
            }
        }
    };
    if (RoutesApertures(state, mem)) {
        Route(state, guest, [&](const Local& space, const std::string& low) {
            for (std::uint32_t component = 0; component < components; ++component) {
                Line(state, Component(result, components, component) + " = " + space.read + low + " + " + U32(component * 4u) + ", " + U32(bytes) + ");");
            }
        }, global);
    } else {
        global();
    }
    Close(state);
    Close(state);
    StopCheck(state);
    Define(state, inst, result);
}

void StoreAddress(State& state, const IrValue& inst, std::uint32_t components, std::uint32_t bytes) {
    const auto& mem = Memory(state, inst);
    const auto data = Typed(state, inst, inst.ArgumentCount() - 2u);
    OpenActive(state, Active(state, inst));
    if (mem.kind == ResourceKind::Scratch) {
        if (components != 1u) Fail(inst, "must write a physical address resource");
        const auto byte = ScratchByte(state, inst, mem);
        Line(state, ScratchArray(state).write + (bytes == 4u ? "(" + byte + ") & ~3u" : byte) + ", " + U32(bytes) + ", " + data + ");");
        Close(state);
        return;
    }
    if (mem.kind != ResourceKind::Flat && mem.kind != ResourceKind::Global) Fail(inst, "must write a scratch or physical address resource");
    state.features |= BdaWrites;
    const auto guest = AddressOf(state, inst, mem);
    Open(state, "if (" + guest.ok + ") {");
    const auto global = [&] {
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto value = Component(data, components, component);
            if (component == 0u) {
                Line(state, "aps5_bda_store(aps5_c, " + guest.address + ", " + U32(bytes) + ", " + value + ", " + Pc(inst) + ");");
            } else {
                const auto next = Temp(state);
                Line(state, "ulong " + next + " = 0ul;");
                Line(state, "if (aps5_bda_add(aps5_c, " + guest.address + ", " + std::to_string(component * 4u) + "ul, false, " + Pc(inst) + ", " + next + ")) aps5_bda_store(aps5_c, " + next + ", 4u, " + value + ", " + Pc(inst) + ");");
            }
        }
    };
    if (RoutesApertures(state, mem)) {
        Route(state, guest, [&](const Local& space, const std::string& low) {
            for (std::uint32_t component = 0; component < components; ++component) {
                Line(state, space.write + low + " + " + U32(component * 4u) + ", " + U32(bytes) + ", " + Component(data, components, component) + ");");
            }
        }, global);
    } else {
        global();
    }
    Close(state);
    Close(state);
    StopCheck(state);
}

Atomic AddressAtomicOf(State& state, const IrValue& inst) {
    const auto value = [&] { return Typed(state, inst, 3u); };
    switch (inst.Opcode()) {
    case IrOpcode::AddressAtomicSwap32: return {0u, value()};
    case IrOpcode::AddressAtomicCmpSwap32: return {1u, value(), Typed(state, inst, 4u)};
    case IrOpcode::AddressAtomicIAdd32: return {2u, value()};
    case IrOpcode::AddressAtomicISub32: return {3u, value()};
    case IrOpcode::AddressAtomicSMin32: return {4u, value()};
    case IrOpcode::AddressAtomicUMin32: return {5u, value()};
    case IrOpcode::AddressAtomicSMax32: return {6u, value()};
    case IrOpcode::AddressAtomicUMax32: return {7u, value()};
    case IrOpcode::AddressAtomicAnd32: return {8u, value()};
    case IrOpcode::AddressAtomicOr32: return {9u, value()};
    case IrOpcode::AddressAtomicXor32: return {10u, value()};
    case IrOpcode::AddressAtomicInc32: return {11u, value()};
    case IrOpcode::AddressAtomicDec32: return {12u, value()};
    case IrOpcode::AddressAtomicUSubSat32: return {13u, value()};
    case IrOpcode::AddressAtomicFMin32: return {15u, value()};
    case IrOpcode::AddressAtomicFMax32: return {16u, value()};
    case IrOpcode::AddressAtomicFCmpSwap32: return {20u, value(), Typed(state, inst, 4u)};
    default: throw MslUnsupported("MSL backend: " + std::string(IrOpcodeName(inst.Opcode())) + " is not emitted (64-bit atomics are section S5)");
    }
}

void AddressAtomic(State& state, const IrValue& inst) {
    const auto& mem = Memory(state, inst);
    if (mem.kind != ResourceKind::Flat && mem.kind != ResourceKind::Global) Fail(inst, "must access a physical address resource");
    const auto atomic = AddressAtomicOf(state, inst);
    const auto result = Result(state, inst);
    state.features |= BdaWrites;
    OpenActive(state, Active(state, inst));
    const auto guest = AddressOf(state, inst, mem);
    Open(state, "if (" + guest.ok + ") {");
    const auto global = [&] {
        const auto pointer = Temp(state);
        Line(state, "device atomic_uint* " + pointer + " = aps5_bda_atomic(aps5_c, " + guest.address + ", " + U32(BdaAbi::Write) + ", " + Pc(inst) + ");");
        Line(state, "if (" + pointer + " != nullptr) " + result + " = aps5_atomic(" + pointer + ", " + U32(atomic.op) + ", " + atomic.a + ", " + atomic.b + ");");
    };
    if (RoutesApertures(state, mem)) {
        const auto low = "uint(" + guest.address + ")";
        const auto unaligned = "if ((" + low + " & 3u) != 0u) aps5_fault(aps5_c, " + U32(static_cast<std::uint32_t>(BdaAbi::FaultReason::Unaligned)) + ", " + guest.address + ", 4u, " + Pc(inst) + ");";
        const auto high = "uint(" + guest.address + " >> 32u)";
        Open(state, "if (" + high + " == " + U32(SharedApertureHigh) + ") {");
        LdsArray(state);
        Line(state, unaligned);
        Line(state, "else if ((" + low + " >> 2u) < " + U32(state.ldsDwords) + ") " + result + " = aps5_atomic_lds(reinterpret_cast<threadgroup atomic_uint*>(&aps5_lds[" + low + " >> 2u]), " + U32(atomic.op) + ", " + atomic.a + ", " + atomic.b + ");");
        Close(state, "} else if (" + high + " == " + U32(PrivateApertureHigh) + ") {");
        ++state.depth;
        ScratchArray(state);
        Line(state, unaligned);
        Open(state, "else if ((" + low + " >> 2u) < " + U32(state.scratchDwords) + ") {");
        Line(state, result + " = aps5_scratch[" + low + " >> 2u];");
        Line(state, "aps5_scratch[" + low + " >> 2u] = aps5_update(" + U32(atomic.op) + ", " + result + ", " + atomic.a + ", " + atomic.b + ");");
        Close(state);
        Close(state, "} else {");
        ++state.depth;
        global();
        Close(state);
    } else {
        global();
    }
    Close(state);
    Close(state);
    StopCheck(state);
    Define(state, inst, result);
}

const MemoryInfo& SharedMemory(State& state, const IrValue& inst) {
    const auto& mem = Memory(state, inst);
    if (mem.kind == ResourceKind::Gds) throw MslUnsupported("MSL backend: GDS access is not emitted (section S9)");
    if (mem.kind != ResourceKind::Lds) Fail(inst, "must access LDS");
    return mem;
}

std::string SharedByte(State& state, const IrValue& inst, const MemoryInfo& mem) {
    const auto address = Typed(state, inst, 0u);
    return mem.offset == 0u ? address : "(" + address + " + " + U32(mem.offset) + ")";
}

void LoadShared(State& state, const IrValue& inst, std::uint32_t components, std::uint32_t bytes) {
    const auto& mem = SharedMemory(state, inst);
    const auto active = Active(state, inst);
    const auto byte = SharedByte(state, inst, mem);
    const auto lds = LdsArray(state);
    if (components == 1u) {
        const auto read = lds.read + (bytes == 4u ? byte + " & ~3u" : byte) + ", " + U32(bytes) + ")";
        Define(state, inst, active.empty() ? read : active + " ? " + read + " : 0u");
        return;
    }
    const auto base = Temp(state);
    Line(state, "const uint " + base + " = " + byte + " & ~3u;");
    std::string values;
    for (std::uint32_t component = 0; component < components; ++component) values += std::string(component == 0u ? "" : ", ") + lds.read + base + " + " + U32(component * 4u) + ", 4u)";
    const auto read = TypeName(inst.Type()) + "(" + values + ")";
    Define(state, inst, active.empty() ? read : active + " ? " + read + " : " + ZeroOf(inst));
}

void WriteShared(State& state, const IrValue& inst, std::uint32_t components, std::uint32_t bytes) {
    const auto& mem = SharedMemory(state, inst);
    const auto active = Active(state, inst);
    const auto byte = SharedByte(state, inst, mem);
    const auto lds = LdsArray(state);
    OpenActive(state, active);
    if (components == 1u) {
        Line(state, lds.write + (bytes == 4u ? byte + " & ~3u" : byte) + ", " + U32(bytes) + ", " + Typed(state, inst, 1u) + ");");
    } else {
        const auto base = Temp(state);
        Line(state, "const uint " + base + " = " + byte + " & ~3u;");
        for (std::uint32_t component = 0; component < components; ++component) {
            Line(state, lds.write + base + " + " + U32(component * 4u) + ", 4u, " + Typed(state, inst, component + 1u) + ");");
        }
    }
    Close(state);
}

Atomic SharedAtomicOf(State& state, const IrValue& inst, const MemoryInfo& mem) {
    const auto data = [&] { return Typed(state, inst, 1u); };
    const auto flush = mem.flushDenormals ? std::string("1u") : std::string("0u");
    switch (inst.Opcode()) {
    case IrOpcode::SharedAtomicSwap32: return {0u, data()};
    case IrOpcode::SharedAtomicIAdd32: return {2u, data()};
    case IrOpcode::SharedAtomicISub32: return {3u, data()};
    case IrOpcode::SharedAtomicSMin32: return {4u, data()};
    case IrOpcode::SharedAtomicUMin32: return {5u, data()};
    case IrOpcode::SharedAtomicSMax32: return {6u, data()};
    case IrOpcode::SharedAtomicUMax32: return {7u, data()};
    case IrOpcode::SharedAtomicAnd32: return {8u, data()};
    case IrOpcode::SharedAtomicOr32: return {9u, data()};
    case IrOpcode::SharedAtomicXor32: return {10u, data()};
    case IrOpcode::SharedAtomicInc32: return {11u, data()};
    case IrOpcode::SharedAtomicDec32: return {12u, data()};
    case IrOpcode::SharedAtomicRsub32: return {14u, data()};
    case IrOpcode::SharedAtomicFMin32: return {15u, data(), flush};
    case IrOpcode::SharedAtomicFMax32: return {16u, data(), flush};
    case IrOpcode::SharedAtomicFAdd32: return {17u, data(), U32((mem.flushDenormals ? 1u : 0u) | (mem.flushDenormalResults ? 2u : 0u))};
    case IrOpcode::SharedAtomicMskor32: return {18u, Typed(state, inst, 1u), Typed(state, inst, 2u)};
    case IrOpcode::SharedAtomicWrap32: return {19u, Typed(state, inst, 1u), Typed(state, inst, 2u)};
    case IrOpcode::SharedAtomicCmpst32: return {1u, Typed(state, inst, 2u), Typed(state, inst, 1u)};
    case IrOpcode::SharedAtomicCmpstF32: return {mem.flushDenormals ? 21u : 20u, Typed(state, inst, 2u), Typed(state, inst, 1u)};
    default: throw MslUnsupported("MSL backend: " + std::string(IrOpcodeName(inst.Opcode())) + " is not emitted (64-bit atomics are section S5)");
    }
}

void SharedAtomic(State& state, const IrValue& inst) {
    const auto& mem = SharedMemory(state, inst);
    const auto atomic = SharedAtomicOf(state, inst, mem);
    const auto active = Active(state, inst);
    const auto index = Temp(state);
    LdsArray(state);
    Line(state, "const uint " + index + " = " + SharedByte(state, inst, mem) + " >> 2u;");
    const auto apply = "aps5_atomic_lds(reinterpret_cast<threadgroup atomic_uint*>(&aps5_lds[" + index + "]), " + U32(atomic.op) + ", " + atomic.a + ", " + atomic.b + ")";
    const auto condition = (active.empty() ? std::string() : active + " && ") + index + " < " + U32(state.ldsDwords);
    if (!inst.HasUses()) {
        Line(state, "if (" + condition + ") " + apply + ";");
        DefineZero(state, inst);
        return;
    }
    Define(state, inst, "(" + condition + ") ? " + apply + " : 0u");
}

void ReadConst(State& state, const IrValue& inst) {
    state.features |= FlattenedSrt;
    const auto& resources = state.program.Resources();
    const auto& guarded = resources.guardedSrtSlots;
    if (!guarded.empty()) {
        const IrValue* slot = inst.Argument(1)->Resolve();
        if (!slot->HasImmediate()) Fail(inst, "reads the flattened SRT at a runtime slot while some slots are guarded");
        const auto found = std::lower_bound(guarded.begin(), guarded.end(), slot->ImmediateU32());
        if (found != guarded.end() && *found == slot->ImmediateU32()) {
            state.features |= Fault;
            const auto flag = Temp(state);
            const auto record = Temp(state);
            const auto guard = static_cast<std::uint32_t>(found - guarded.begin());
            const auto records = resources.srtGuardOffset + static_cast<std::uint32_t>(guarded.size());
            Line(state, "const uint " + flag + " = aps5_srt[" + U32(resources.srtGuardOffset + guard) + "];");
            Open(state, "if (" + flag + " != 0u) {");
            Line(state, "const uint " + record + " = " + U32(records) + " + (" + flag + " - 1u) * 3u;");
            Line(state, "aps5_fault(aps5_c, " + U32(static_cast<std::uint32_t>(BdaAbi::FaultReason::Unmapped)) + ", (ulong(aps5_srt[" + record + " + 2u]) << 32u) | ulong(aps5_srt[" + record + " + 1u]), 4u, aps5_srt[" + record + "]);");
            if (state.stopAllowed) Line(state, "return;");
            Close(state);
        }
    }
    Define(state, inst, "aps5_srt[" + Typed(state, inst, 1u) + "]");
}

void LoadVectorWindow(State& state, const IrValue& inst) {
    state.features |= VectorWindow;
    const auto index = Typed(state, inst, 0u);
    const IrValue* value = inst.Argument(0)->Resolve();
    if (value->HasImmediate()) {
        Define(state, inst, value->ImmediateU32() < state.windowDwords ? "aps5_window[" + index + "]" : std::string("0u"));
        return;
    }
    Define(state, inst, index + " < " + U32(state.windowDwords) + " ? aps5_window[" + index + "] : 0u");
}

void StoreVectorWindow(State& state, const IrValue& inst) {
    state.features |= VectorWindow;
    const auto index = Typed(state, inst, 0u);
    const auto data = Typed(state, inst, 1u);
    const IrValue* value = inst.Argument(0)->Resolve();
    if (value->HasImmediate()) {
        if (value->ImmediateU32() < state.windowDwords) Line(state, "aps5_window[" + index + "] = " + data + ";");
        return;
    }
    Line(state, "if (" + index + " < " + U32(state.windowDwords) + ") aps5_window[" + index + "] = " + data + ";");
}

}

bool EmitMemory(State& state, const IrValue& inst) {
    switch (inst.Opcode()) {
    case IrOpcode::ReadConst: ReadConst(state, inst); return true;
    case IrOpcode::ReadConstBuffer: ReadConstBuffer(state, inst); return true;
    case IrOpcode::LoadBufferU8: LoadBuffer(state, inst, 1u); return true;
    case IrOpcode::LoadBufferU16: LoadBuffer(state, inst, 1u); return true;
    case IrOpcode::LoadBufferU32: LoadBuffer(state, inst, 1u); return true;
    case IrOpcode::LoadBufferU32x2: LoadBuffer(state, inst, 2u); return true;
    case IrOpcode::LoadBufferU32x3: LoadBuffer(state, inst, 3u); return true;
    case IrOpcode::LoadBufferU32x4: LoadBuffer(state, inst, 4u); return true;
    case IrOpcode::StoreBufferU8:
    case IrOpcode::StoreBufferU16:
    case IrOpcode::StoreBufferU32: StoreBuffer(state, inst, 1u); return true;
    case IrOpcode::StoreBufferU32x2: StoreBuffer(state, inst, 2u); return true;
    case IrOpcode::StoreBufferU32x3: StoreBuffer(state, inst, 3u); return true;
    case IrOpcode::StoreBufferU32x4: StoreBuffer(state, inst, 4u); return true;
    case IrOpcode::BufferAtomicSwap32:
    case IrOpcode::BufferAtomicCmpSwap32:
    case IrOpcode::BufferAtomicIAdd32:
    case IrOpcode::BufferAtomicISub32:
    case IrOpcode::BufferAtomicSMin32:
    case IrOpcode::BufferAtomicUMin32:
    case IrOpcode::BufferAtomicSMax32:
    case IrOpcode::BufferAtomicUMax32:
    case IrOpcode::BufferAtomicAnd32:
    case IrOpcode::BufferAtomicOr32:
    case IrOpcode::BufferAtomicXor32:
    case IrOpcode::BufferAtomicInc32:
    case IrOpcode::BufferAtomicDec32:
    case IrOpcode::BufferAtomicUSubSat32:
    case IrOpcode::BufferAtomicFMin32:
    case IrOpcode::BufferAtomicFMax32:
    case IrOpcode::BufferAtomicFCmpSwap32: BufferAtomic(state, inst); return true;
    case IrOpcode::LoadAddressU8: LoadAddress(state, inst, 1u, 1u); return true;
    case IrOpcode::LoadAddressU16: LoadAddress(state, inst, 1u, 2u); return true;
    case IrOpcode::LoadAddressU32: LoadAddress(state, inst, 1u, 4u); return true;
    case IrOpcode::LoadAddressU32x2: LoadAddress(state, inst, 2u, 4u); return true;
    case IrOpcode::LoadAddressU32x3: LoadAddress(state, inst, 3u, 4u); return true;
    case IrOpcode::LoadAddressU32x4: LoadAddress(state, inst, 4u, 4u); return true;
    case IrOpcode::StoreAddressU8: StoreAddress(state, inst, 1u, 1u); return true;
    case IrOpcode::StoreAddressU16: StoreAddress(state, inst, 1u, 2u); return true;
    case IrOpcode::StoreAddressU32: StoreAddress(state, inst, 1u, 4u); return true;
    case IrOpcode::StoreAddressU32x2: StoreAddress(state, inst, 2u, 4u); return true;
    case IrOpcode::StoreAddressU32x3: StoreAddress(state, inst, 3u, 4u); return true;
    case IrOpcode::StoreAddressU32x4: StoreAddress(state, inst, 4u, 4u); return true;
    case IrOpcode::AddressAtomicSwap32:
    case IrOpcode::AddressAtomicCmpSwap32:
    case IrOpcode::AddressAtomicIAdd32:
    case IrOpcode::AddressAtomicISub32:
    case IrOpcode::AddressAtomicSMin32:
    case IrOpcode::AddressAtomicUMin32:
    case IrOpcode::AddressAtomicSMax32:
    case IrOpcode::AddressAtomicUMax32:
    case IrOpcode::AddressAtomicAnd32:
    case IrOpcode::AddressAtomicOr32:
    case IrOpcode::AddressAtomicXor32:
    case IrOpcode::AddressAtomicInc32:
    case IrOpcode::AddressAtomicDec32:
    case IrOpcode::AddressAtomicUSubSat32:
    case IrOpcode::AddressAtomicFMin32:
    case IrOpcode::AddressAtomicFMax32:
    case IrOpcode::AddressAtomicFCmpSwap32: AddressAtomic(state, inst); return true;
    case IrOpcode::LoadSharedU8: LoadShared(state, inst, 1u, 1u); return true;
    case IrOpcode::LoadSharedU16: LoadShared(state, inst, 1u, 2u); return true;
    case IrOpcode::LoadSharedU32: LoadShared(state, inst, 1u, 4u); return true;
    case IrOpcode::LoadSharedU32x2: LoadShared(state, inst, 2u, 4u); return true;
    case IrOpcode::LoadSharedU32x3: LoadShared(state, inst, 3u, 4u); return true;
    case IrOpcode::LoadSharedU32x4: LoadShared(state, inst, 4u, 4u); return true;
    case IrOpcode::WriteSharedU8: WriteShared(state, inst, 1u, 1u); return true;
    case IrOpcode::WriteSharedU16: WriteShared(state, inst, 1u, 2u); return true;
    case IrOpcode::WriteSharedU32: WriteShared(state, inst, 1u, 4u); return true;
    case IrOpcode::WriteSharedU32x2: WriteShared(state, inst, 2u, 4u); return true;
    case IrOpcode::WriteSharedU32x3: WriteShared(state, inst, 3u, 4u); return true;
    case IrOpcode::WriteSharedU32x4: WriteShared(state, inst, 4u, 4u); return true;
    case IrOpcode::SharedAtomicSwap32:
    case IrOpcode::SharedAtomicIAdd32:
    case IrOpcode::SharedAtomicISub32:
    case IrOpcode::SharedAtomicSMin32:
    case IrOpcode::SharedAtomicUMin32:
    case IrOpcode::SharedAtomicSMax32:
    case IrOpcode::SharedAtomicUMax32:
    case IrOpcode::SharedAtomicAnd32:
    case IrOpcode::SharedAtomicOr32:
    case IrOpcode::SharedAtomicXor32:
    case IrOpcode::SharedAtomicInc32:
    case IrOpcode::SharedAtomicDec32:
    case IrOpcode::SharedAtomicRsub32:
    case IrOpcode::SharedAtomicFMin32:
    case IrOpcode::SharedAtomicFMax32:
    case IrOpcode::SharedAtomicFAdd32:
    case IrOpcode::SharedAtomicMskor32:
    case IrOpcode::SharedAtomicWrap32:
    case IrOpcode::SharedAtomicCmpst32:
    case IrOpcode::SharedAtomicCmpstF32: SharedAtomic(state, inst); return true;
    case IrOpcode::LoadVectorWindow: LoadVectorWindow(state, inst); return true;
    case IrOpcode::StoreVectorWindow: StoreVectorWindow(state, inst); return true;
    default: break;
    }
    const auto name = IrOpcodeName(inst.Opcode());
    if (name.starts_with("Image") || name.starts_with("LoadImage") || name.starts_with("StoreImage")) throw MslUnsupported("MSL backend: " + std::string(name) + " is not emitted (images are section S3)");
    if (name.starts_with("BufferAtomic") || name.starts_with("AddressAtomic") || name.starts_with("SharedAtomic")) throw MslUnsupported("MSL backend: " + std::string(name) + " is not emitted (64-bit atomics are section S5)");
    if (name.starts_with("Data") || name.starts_with("Gds")) throw MslUnsupported("MSL backend: " + std::string(name) + " is not emitted (GDS is section S9)");
    return false;
}

}
