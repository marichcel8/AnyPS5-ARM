#include "MslState.hpp"
#include "BdaAbi.hpp"
#include "FlatApertures.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
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
    if ((state.target.experiment & MslExperimentNoUniformLoad) != 0u && state.program.Resources().stage == IrShaderStage::Pixel && bytes == 4u) return "0x3f800000u";
    if (mem.gpuDescriptor) return "aps5_vsharp_load(aps5_c, " + resource + ", " + byte + ", " + U32(bytes) + ", " + Pc(inst) + ")";
    return "aps5_buffer_load(aps5_c, " + U32(BufferElement(state, inst, mem)) + ", " + resource + ", " + byte + ", " + U32(bytes) + ")";
}

std::string Uint3(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return "uint3(" + U32(x) + ", " + U32(y) + ", " + U32(z) + ")";
}

std::string FormatInfo(const SpirvBufferFormatInfo& info) {
    std::uint32_t layout = info.componentCount | (info.packedBitfield ? 8u : 0u) | (static_cast<std::uint32_t>(info.type) << 4u);
    std::uint32_t bytes = 0;
    std::uint32_t bits = 0;
    for (std::uint32_t component = 0; component < info.componentCount; ++component) {
        layout |= info.componentBits[component] << (8u + component * 6u);
        bytes |= GetFormatComponentByteOffset(info, component) << (component * 8u);
        bits |= info.componentBitOffset[component] << (component * 8u);
    }
    return Uint3(layout, bytes, bits);
}

struct FormatArgs {
    std::string info;
    std::string sel;
};

FormatArgs FormatOf(State& state, const IrValue& inst, const MemoryInfo& mem, const std::string& resource) {
    state.features |= Formatted;
    if (mem.typed) {
        const auto info = GetFormatInfo(DecodeTBufferFormat(mem.dataFormat, mem.numberFormat));
        if (info.type == SpirvFormatComponentType::Unknown) Fail(inst, "typed buffer instruction has an invalid format");
        std::uint32_t sel = 0;
        for (std::uint32_t output = 0; output < 4u; ++output) sel |= (output < info.componentCount ? 4u + output : 0u) << (output * 3u);
        return {FormatInfo(info), U32(sel)};
    }
    const IrValue* word = inst.Argument(0)->Resolve()->Argument(3)->Resolve();
    if (mem.gpuDescriptor && word->HasImmediate()) {
        const auto format = static_cast<IrBufferFormat>((word->ImmediateU32() >> 12u) & 0x7fu);
        if (format == IrBufferFormat::Invalid) Fail(inst, "formatted buffer instruction has an invalid constant format");
        return {FormatInfo(GetFormatInfo(format)), resource + ".word3"};
    }
    return {"aps5_fmt_info(extract_bits(" + resource + ".word3, 12u, 7u))", resource + ".word3"};
}

std::string FormatTarget(State& state, const IrValue& inst, const MemoryInfo& mem) {
    if (mem.gpuDescriptor) return "_gpu(aps5_c, 0u, ";
    return "(aps5_c, " + U32(BufferElement(state, inst, mem)) + ", ";
}

void LoadFormatted(State& state, const IrValue& inst, const MemoryInfo& mem, std::uint32_t components, const std::string& result) {
    const auto resource = VSharp(state, inst, mem);
    const auto format = FormatOf(state, inst, mem, resource);
    const auto offset = Typed(state, inst, 2u) + " + " + U32(mem.offset);
    const bool check = mem.gpuDescriptor && !mem.typed;
    static constexpr std::string_view Swizzles[4] = {".x", ".xy", ".xyz", ""};
    Line(state, result + " = aps5_fmt_load" + FormatTarget(state, inst, mem) + resource + ", " + offset + ", " + Typed(state, inst, 3u) + ", " + format.info + ", " + format.sel + ", " + U32(components) + ", " + (mem.d16 ? "true" : "false") + ", " + (check ? "true" : "false") + ", " + Pc(inst) + ")" + std::string(Swizzles[components - 1u]) + ";");
}

void StoreFormatted(State& state, const IrValue& inst, const MemoryInfo& mem, std::uint32_t components) {
    const auto resource = VSharp(state, inst, mem);
    const auto format = FormatOf(state, inst, mem, resource);
    if (mem.gpuDescriptor) state.features |= BdaWrites;
    const auto offset = Typed(state, inst, 2u) + " + " + U32(mem.offset);
    static constexpr std::string_view Padding[4] = {", 0u, 0u, 0u", ", 0u, 0u", ", 0u", ""};
    const auto data = "uint4(" + Typed(state, inst, inst.ArgumentCount() - 2u) + std::string(Padding[components - 1u]) + ")";
    Line(state, "aps5_fmt_store" + FormatTarget(state, inst, mem) + resource + ", " + offset + ", " + Typed(state, inst, 3u) + ", " + format.info + ", " + U32(components) + ", " + data + ", " + (mem.d16 ? "true" : "false") + ", " + Pc(inst) + ");");
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
    if (mem.formatted) {
        LoadFormatted(state, inst, mem, components, result);
        Close(state);
        if (mem.gpuDescriptor) StopCheck(state);
        Define(state, inst, result);
        return;
    }
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
    if (mem.formatted) {
        StoreFormatted(state, inst, mem, components);
        Close(state);
        if (mem.gpuDescriptor) StopCheck(state);
        return;
    }
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
    const auto arguments = "(aps5_scratch" + HalfSuffix() + ", " + U32(state.scratchDwords) + ", ";
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
        Line(state, result + " = aps5_scratch" + HalfSuffix() + "[" + low + " >> 2u];");
        Line(state, "aps5_scratch" + HalfSuffix() + "[" + low + " >> 2u] = aps5_update(" + U32(atomic.op) + ", " + result + ", " + atomic.a + ", " + atomic.b + ");");
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
    if (mem.kind == ResourceKind::Gds) throw MslUnsupported("MSL backend: GDS loads, stores and atomics are not emitted, only append, consume and ordered count");
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

constexpr std::uint32_t NoAtomic64 = ~0u;

std::uint32_t Atomic64Op(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::BufferAtomicIAdd64:
    case IrOpcode::AddressAtomicIAdd64:
    case IrOpcode::SharedAtomicIAdd64: return 2u;
    case IrOpcode::BufferAtomicISub64:
    case IrOpcode::AddressAtomicISub64:
    case IrOpcode::SharedAtomicISub64: return 3u;
    case IrOpcode::BufferAtomicUMin64:
    case IrOpcode::AddressAtomicUMin64: return 5u;
    case IrOpcode::BufferAtomicUMax64:
    case IrOpcode::AddressAtomicUMax64: return 7u;
    case IrOpcode::BufferAtomicAnd64:
    case IrOpcode::AddressAtomicAnd64:
    case IrOpcode::SharedAtomicAnd64: return 8u;
    case IrOpcode::BufferAtomicOr64:
    case IrOpcode::AddressAtomicOr64:
    case IrOpcode::SharedAtomicOr64: return 9u;
    case IrOpcode::BufferAtomicXor64:
    case IrOpcode::AddressAtomicXor64:
    case IrOpcode::SharedAtomicXor64: return 10u;
    default: return NoAtomic64;
    }
}

std::uint32_t RequireAtomic64(State& state, const IrValue& inst) {
    const auto op = Atomic64Op(inst.Opcode());
    if (op == NoAtomic64) throw MslUnsupported("MSL backend: " + std::string(IrOpcodeName(inst.Opcode())) + " is not emitted (Metal has no 64-bit exchange, compare-and-swap, signed or floating-point atomic, and no 64-bit atomic in threadgroup memory)");
    state.features |= Atomic64;
    return op;
}

void BufferAtomic64(State& state, const IrValue& inst) {
    const auto& mem = Memory(state, inst);
    if (mem.kind != ResourceKind::Buffer) Fail(inst, "must access a buffer resource");
    const auto op = RequireAtomic64(state, inst);
    const auto data = Typed(state, inst, inst.ArgumentCount() - 2u);
    const auto result = Result(state, inst);
    OpenActive(state, Active(state, inst));
    const auto resource = VSharp(state, inst, mem);
    const auto byte = Temp(state);
    const auto pointer = Temp(state);
    Line(state, "uint " + byte + " = 0u;");
    Open(state, "if (aps5_vsharp_byte(" + resource + ", " + Typed(state, inst, 2u) + " + " + U32(mem.offset) + ", " + Typed(state, inst, 3u) + ", 8u, " + byte + ")) {");
    if (mem.gpuDescriptor) {
        state.features |= BdaWrites;
        Line(state, "device atomic_uint* " + pointer + " = aps5_vsharp_atomic8(aps5_c, " + resource + ", " + byte + ", " + Pc(inst) + ");");
    } else {
        Line(state, "device atomic_uint* " + pointer + " = aps5_buffer_atomic8(aps5_c, " + U32(BufferElement(state, inst, mem)) + ", " + resource + ", " + byte + ", " + Pc(inst) + ");");
    }
    Line(state, "if (" + pointer + " != nullptr) " + result + " = aps5_atomic64(" + pointer + ", " + U32(op) + ", " + data + ", " + (inst.HasUses() ? "true" : "false") + ");");
    Close(state);
    Close(state);
    StopCheck(state);
    Define(state, inst, result);
}

void AddressAtomic64(State& state, const IrValue& inst) {
    const auto& mem = Memory(state, inst);
    if (mem.kind != ResourceKind::Flat && mem.kind != ResourceKind::Global) Fail(inst, "must access a physical address resource");
    const auto op = RequireAtomic64(state, inst);
    const auto data = Typed(state, inst, 3u);
    const auto wanted = std::string(inst.HasUses() ? "true" : "false");
    const auto result = Result(state, inst);
    state.features |= BdaWrites;
    OpenActive(state, Active(state, inst));
    const auto guest = AddressOf(state, inst, mem);
    Open(state, "if (" + guest.ok + ") {");
    const auto global = [&] {
        const auto pointer = Temp(state);
        Line(state, "device atomic_uint* " + pointer + " = aps5_bda_atomic8(aps5_c, " + guest.address + ", " + U32(BdaAbi::Write) + ", " + Pc(inst) + ");");
        Line(state, "if (" + pointer + " != nullptr) " + result + " = aps5_atomic64(" + pointer + ", " + U32(op) + ", " + data + ", " + wanted + ");");
    };
    if (RoutesApertures(state, mem) && op != 5u && op != 7u) {
        const auto low = "uint(" + guest.address + ")";
        const auto unaligned = "if ((" + low + " & 3u) != 0u) aps5_fault(aps5_c, " + U32(static_cast<std::uint32_t>(BdaAbi::FaultReason::Unaligned)) + ", " + guest.address + ", 8u, " + Pc(inst) + ");";
        const auto high = "uint(" + guest.address + " >> 32u)";
        Open(state, "if (" + high + " == " + U32(SharedApertureHigh) + ") {");
        LdsArray(state);
        Line(state, unaligned);
        Line(state, "else if (((" + low + " >> 2u) + 1u) < " + U32(state.ldsDwords) + ") " + result + " = aps5_atomic64_lds(reinterpret_cast<threadgroup atomic_uint*>(&aps5_lds[" + low + " >> 2u]), " + U32(op) + ", " + data + ", " + wanted + ");");
        Close(state, "} else if (" + high + " == " + U32(PrivateApertureHigh) + ") {");
        ++state.depth;
        ScratchArray(state);
        Line(state, unaligned);
        Open(state, "else if (((" + low + " >> 2u) + 1u) < " + U32(state.scratchDwords) + ") {");
        Line(state, result + " = (ulong(aps5_scratch[(" + low + " >> 2u) + 1u]) << 32u) | ulong(aps5_scratch[" + low + " >> 2u]);");
        const auto next = Temp(state);
        Line(state, "const ulong " + next + " = aps5_update64(" + U32(op) + ", " + result + ", " + data + ", 0ul);");
        Line(state, "aps5_scratch[" + low + " >> 2u] = uint(" + next + ");");
        Line(state, "aps5_scratch[(" + low + " >> 2u) + 1u] = uint(" + next + " >> 32u);");
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

std::uint32_t SharedAtomic64Code(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::SharedAtomicSwap64: return 0u;
    case IrOpcode::SharedAtomicCmpst64: return 1u;
    case IrOpcode::SharedAtomicIAdd64: return 2u;
    case IrOpcode::SharedAtomicISub64: return 3u;
    case IrOpcode::SharedAtomicSMin64: return 4u;
    case IrOpcode::SharedAtomicUMin64: return 5u;
    case IrOpcode::SharedAtomicSMax64: return 6u;
    case IrOpcode::SharedAtomicUMax64: return 7u;
    case IrOpcode::SharedAtomicAnd64: return 8u;
    case IrOpcode::SharedAtomicOr64: return 9u;
    case IrOpcode::SharedAtomicXor64: return 10u;
    case IrOpcode::SharedAtomicInc64: return 11u;
    case IrOpcode::SharedAtomicDec64: return 12u;
    case IrOpcode::SharedAtomicRsub64: return 14u;
    case IrOpcode::SharedAtomicFMin64: return 15u;
    case IrOpcode::SharedAtomicFMax64: return 16u;
    case IrOpcode::SharedAtomicMskor64: return 18u;
    case IrOpcode::SharedAtomicCmpstF64: return 20u;
    default: throw MslUnsupported("MSL backend: " + std::string(IrOpcodeName(opcode)) + " is not emitted");
    }
}

void SharedAtomic64(State& state, const IrValue& inst) {
    const auto& mem = SharedMemory(state, inst);
    const auto opcode = inst.Opcode();
    const auto op = SharedAtomic64Code(opcode);
    const bool floating = opcode == IrOpcode::SharedAtomicFMin64 || opcode == IrOpcode::SharedAtomicFMax64 || opcode == IrOpcode::SharedAtomicCmpstF64;
    if (floating && mem.flushDenormals) throw MslUnsupported("MSL backend: 64-bit floating-point LDS atomics with denormal flushing are not emitted");
    const bool twoOperands = opcode == IrOpcode::SharedAtomicCmpst64 || opcode == IrOpcode::SharedAtomicCmpstF64 || opcode == IrOpcode::SharedAtomicMskor64;
    const auto a = Typed(state, inst, twoOperands ? 2u : 1u);
    const auto b = twoOperands ? Typed(state, inst, 1u) : std::string("0ul");
    const auto active = Active(state, inst);
    const auto index = Temp(state);
    LdsArray(state);
    state.features |= Lds64 | Atomic64 | Float64 | Lane | LocalIndex;
    Line(state, "const uint " + index + " = " + SharedByte(state, inst, mem) + " >> 2u;");
    const auto condition = (active.empty() ? std::string() : "(" + active + ") && ") + "(" + index + " + 1u) < " + U32(state.ldsDwords);
    Define(state, inst, "aps5_lds64(&aps5_lds_lock, reinterpret_cast<threadgroup atomic_uint*>(&aps5_lds[" + index + "]), aps5_lane, " + condition + ", " + U32(op) + ", " + a + ", " + b + ")");
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
        Define(state, inst, value->ImmediateU32() < state.windowDwords ? "aps5_window" + HalfSuffix() + "[" + index + "]" : std::string("0u"));
        return;
    }
    Define(state, inst, index + " < " + U32(state.windowDwords) + " ? aps5_window" + HalfSuffix() + "[" + index + "] : 0u");
}

void StoreVectorWindow(State& state, const IrValue& inst) {
    state.features |= VectorWindow;
    const auto index = Typed(state, inst, 0u);
    const auto data = Typed(state, inst, 1u);
    const IrValue* value = inst.Argument(0)->Resolve();
    if (value->HasImmediate()) {
        if (value->ImmediateU32() < state.windowDwords) Line(state, "aps5_window" + HalfSuffix() + "[" + index + "] = " + data + ";");
        return;
    }
    Line(state, "if (" + index + " < " + U32(state.windowDwords) + ") aps5_window" + HalfSuffix() + "[" + index + "] = " + data + ";");
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
    case IrOpcode::BufferAtomicSwap64:
    case IrOpcode::BufferAtomicCmpSwap64:
    case IrOpcode::BufferAtomicIAdd64:
    case IrOpcode::BufferAtomicISub64:
    case IrOpcode::BufferAtomicSMin64:
    case IrOpcode::BufferAtomicUMin64:
    case IrOpcode::BufferAtomicSMax64:
    case IrOpcode::BufferAtomicUMax64:
    case IrOpcode::BufferAtomicAnd64:
    case IrOpcode::BufferAtomicOr64:
    case IrOpcode::BufferAtomicXor64:
    case IrOpcode::BufferAtomicFCmpSwap64:
    case IrOpcode::BufferAtomicFMin64:
    case IrOpcode::BufferAtomicFMax64:
    case IrOpcode::BufferAtomicInc64:
    case IrOpcode::BufferAtomicDec64: BufferAtomic64(state, inst); return true;
    case IrOpcode::AddressAtomicSwap64:
    case IrOpcode::AddressAtomicCmpSwap64:
    case IrOpcode::AddressAtomicIAdd64:
    case IrOpcode::AddressAtomicISub64:
    case IrOpcode::AddressAtomicSMin64:
    case IrOpcode::AddressAtomicUMin64:
    case IrOpcode::AddressAtomicSMax64:
    case IrOpcode::AddressAtomicUMax64:
    case IrOpcode::AddressAtomicAnd64:
    case IrOpcode::AddressAtomicOr64:
    case IrOpcode::AddressAtomicXor64:
    case IrOpcode::AddressAtomicFCmpSwap64:
    case IrOpcode::AddressAtomicFMin64:
    case IrOpcode::AddressAtomicFMax64:
    case IrOpcode::AddressAtomicInc64:
    case IrOpcode::AddressAtomicDec64: AddressAtomic64(state, inst); return true;
    case IrOpcode::SharedAtomicSwap64:
    case IrOpcode::SharedAtomicIAdd64:
    case IrOpcode::SharedAtomicISub64:
    case IrOpcode::SharedAtomicRsub64:
    case IrOpcode::SharedAtomicInc64:
    case IrOpcode::SharedAtomicDec64:
    case IrOpcode::SharedAtomicSMin64:
    case IrOpcode::SharedAtomicUMin64:
    case IrOpcode::SharedAtomicSMax64:
    case IrOpcode::SharedAtomicUMax64:
    case IrOpcode::SharedAtomicAnd64:
    case IrOpcode::SharedAtomicOr64:
    case IrOpcode::SharedAtomicXor64:
    case IrOpcode::SharedAtomicFMin64:
    case IrOpcode::SharedAtomicFMax64:
    case IrOpcode::SharedAtomicCmpst64: SharedAtomic64(state, inst); return true;
    case IrOpcode::LoadVectorWindow: LoadVectorWindow(state, inst); return true;
    case IrOpcode::StoreVectorWindow: StoreVectorWindow(state, inst); return true;
    default: break;
    }
    if (EmitImage(state, inst)) return true;
    const auto name = IrOpcodeName(inst.Opcode());
    if (name.starts_with("Image") || name.starts_with("LoadImage") || name.starts_with("StoreImage")) throw MslUnsupported("MSL backend: " + std::string(name) + " is not emitted (images are section S3)");
    if (name.starts_with("BufferAtomic") || name.starts_with("AddressAtomic") || name.starts_with("SharedAtomic")) throw MslUnsupported("MSL backend: " + std::string(name) + " is not emitted (64-bit atomics are section S5)");
    if (name.starts_with("Data") || name.starts_with("Gds")) throw MslUnsupported("MSL backend: " + std::string(name) + " is not emitted");
    return false;
}

bool IsAppendConsume(IrOpcode opcode) {
    return opcode == IrOpcode::DataAppend || opcode == IrOpcode::DataConsume || opcode == IrOpcode::DataOrderedCount || opcode == IrOpcode::DataOrderedSwap;
}

void EmitAppendConsume(State& state, const IrValue& inst) {
    constexpr std::uint32_t GdsDwords = 0x10000u / 4u;
    const auto& mem = Memory(state, inst);
    const bool gds = mem.kind == ResourceKind::Gds;
    if (!gds && mem.kind != ResourceKind::Lds) Fail(inst, "must access LDS or GDS");
    const auto opcode = inst.Opcode();
    const bool ordered = opcode == IrOpcode::DataOrderedCount || opcode == IrOpcode::DataOrderedSwap;
    const bool swap = opcode == IrOpcode::DataOrderedSwap;
    const bool append = opcode != IrOpcode::DataConsume;
    const bool m0Address = gds && !ordered;
    const auto index = Temp(state);
    std::string bounds;
    if (m0Address) {
        const auto m0 = Typed(state, inst, 0u);
        Line(state, "const uint " + index + " = (((" + m0 + " >> 16u) + " + U32(mem.offset) + ") >> 2u);");
        bounds = "(" + m0 + " & 0xffffu) != 0u && ";
    } else {
        Line(state, "const uint " + index + " = " + U32(mem.offset >> 2u) + ";");
    }
    if (gds) state.features |= Gds;
    else LdsArray(state);
    bounds += index + " < " + U32(gds ? GdsDwords : state.ldsDwords);
    state.features |= Lane | Wave;
    const auto low = Temp(state);
    const auto high = Temp(state);
    const auto count = Temp(state);
    const auto first = Temp(state);
    const auto source = Temp(state);
    Line(state, "const uint " + low + " = aps5_ballot32(" + HalfArg(state, inst, 1u, 0u) + ");");
    Line(state, "const uint " + high + " = " + (state.lanes == 2u ? "aps5_ballot32(" + HalfArg(state, inst, 1u, 1u) + ")" : std::string("0u")) + ";");
    Line(state, "const uint " + count + " = popcount(" + low + ") + popcount(" + high + ");");
    Line(state, "const uint " + first + " = " + low + " != 0u ? ctz(" + low + ") : (" + high + " != 0u ? 32u + ctz(" + high + ") : 0u);");
    Line(state, "const uint " + source + " = " + first + " & 31u;");
    std::string amount = count;
    if (ordered) {
        const auto fromLow = Temp(state);
        Line(state, "const uint " + fromLow + " = simd_shuffle(" + HalfArg(state, inst, 4u, 0u) + ", ushort(" + source + "));");
        if (state.lanes == 2u) {
            const auto fromHigh = Temp(state);
            Line(state, "const uint " + fromHigh + " = simd_shuffle(" + HalfArg(state, inst, 4u, 1u) + ", ushort(" + source + "));");
            amount = Temp(state);
            Line(state, "const uint " + amount + " = " + first + " >= 32u ? " + fromHigh + " : " + fromLow + ";");
        } else {
            amount = fromLow;
        }
    }
    const auto old = Temp(state);
    Line(state, "uint " + old + " = 0u;");
    Open(state, "if (aps5_lane == " + source + " && " + count + " != 0u && " + bounds + ") {");
    if (gds) {
        const char* function = swap ? "atomic_exchange_explicit" : append ? "atomic_fetch_add_explicit" : "atomic_fetch_sub_explicit";
        Line(state, old + " = " + function + "(&aps5_gds[" + index + "], " + amount + ", memory_order_relaxed);");
    } else {
        Line(state, old + " = aps5_atomic_lds(reinterpret_cast<threadgroup atomic_uint*>(&aps5_lds[" + index + "]), " + U32(swap ? 0u : append ? 2u : 3u) + ", " + amount + ", 0u);");
    }
    Close(state);
    const auto result = Temp(state);
    Line(state, "const uint " + result + " = simd_shuffle(" + old + ", ushort(" + source + "));");
    if (state.lanes == 2u) DefineHalves(state, inst, result, result);
    else Define(state, inst, result);
}

}
