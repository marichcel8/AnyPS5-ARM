#include "MslState.hpp"
#include "MetalAbi.hpp"
#include "PipelineSpecialization.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "RdnaDecoder/RdnaImageOpDecoder.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <optional>

namespace ShaderRecompiler::Msl {

namespace {

constexpr std::uint32_t NoComponent = 0xffffffffu;

struct Shape {
    std::uint32_t coordinates;
    std::uint32_t spatial;
    bool arrayed;
    bool multisampled;
    std::string_view texture;
};

Shape ShapeOf(RdnaImageDimension dimension) {
    switch (dimension) {
    case RdnaImageDimension::Dim1D: return {1u, 1u, false, false, "texture2d"};
    case RdnaImageDimension::Dim1DArray: return {2u, 1u, true, false, "texture2d_array"};
    case RdnaImageDimension::Dim2D: return {2u, 2u, false, false, "texture2d"};
    case RdnaImageDimension::Dim2DArray: return {3u, 2u, true, false, "texture2d_array"};
    case RdnaImageDimension::Dim2DMsaa: return {2u, 2u, false, true, "texture2d_ms"};
    case RdnaImageDimension::Dim2DMsaaArray: return {3u, 2u, true, true, "texture2d_ms_array"};
    case RdnaImageDimension::Dim3D: return {3u, 3u, false, false, "texture3d"};
    case RdnaImageDimension::Unknown: break;
    }
    Fail("an image has an unknown dimension");
}

struct SampleLayout {
    std::uint32_t offset = NoComponent;
    std::uint32_t dref = NoComponent;
    std::uint32_t bias = NoComponent;
    std::uint32_t coord = 0;
    std::uint32_t lod = NoComponent;
    std::uint32_t clamp = NoComponent;
    std::uint32_t gradX = NoComponent;
    std::uint32_t gradY = NoComponent;
};

struct Access {
    State& state;
    const IrValue& inst;
    const MemoryInfo& mem;
    const ImageResource& image;
    const IrValue& address;
    std::uint32_t element;
};

bool HasFlag(const MemoryInfo& mem, std::uint32_t flag) {
    return (mem.imageSampleFlags & flag) != 0u;
}

SampleLayout LayoutOf(const MemoryInfo& mem) {
    SampleLayout layout;
    std::uint32_t cursor = 0;
    const auto shape = ShapeOf(mem.imageDimension);
    if (HasFlag(mem, RdnaImageSampleFlagOffset)) layout.offset = cursor++;
    if (HasFlag(mem, RdnaImageSampleFlagBias)) layout.bias = cursor++;
    if (HasFlag(mem, RdnaImageSampleFlagCompare)) layout.dref = cursor++;
    if (HasFlag(mem, RdnaImageSampleFlagDerivative)) {
        layout.gradX = cursor;
        cursor += shape.spatial;
        layout.gradY = cursor;
        cursor += shape.spatial;
    }
    layout.coord = cursor;
    cursor += shape.coordinates;
    if (HasFlag(mem, RdnaImageSampleFlagLod)) layout.lod = cursor++;
    if (HasFlag(mem, RdnaImageSampleFlagLodClamp)) layout.clamp = cursor++;
    return layout;
}

std::uint32_t EffectiveDmask(const MemoryInfo& mem) {
    return mem.dmask != 0u ? mem.dmask : 1u;
}

std::uint32_t DmaskComponentIndex(std::uint32_t dmask, std::uint32_t component) {
    std::uint32_t index = 0;
    for (std::uint32_t bit = 0; bit < component; ++bit) index += (dmask >> bit) & 1u;
    return index;
}

std::uint32_t DmaskComponent(const IrValue& inst, std::uint32_t dmask, std::uint32_t index) {
    for (std::uint32_t component = 0; component < 4u; ++component) {
        if (((dmask >> component) & 1u) != 0u && index-- == 0u) return component;
    }
    Fail(inst, "has a dmask with fewer components than requested");
}

std::string Swizzle(std::uint32_t index) {
    static constexpr std::string_view Names[4] = {"x", "y", "z", "w"};
    return std::string(Names[index]);
}

std::string Constant(State& state, std::uint32_t id, std::uint32_t fallback) {
    state.constants.emplace(id, fallback);
    return "aps5_c" + std::to_string(id);
}

std::uint32_t ElementOf(const State& state, DescriptorBindingKind kind, std::uint32_t resource) {
    for (const auto& descriptor : state.layout.descriptors) {
        if (descriptor.kind != kind) continue;
        const auto found = std::find(descriptor.resources.begin(), descriptor.resources.end(), resource);
        if (found == descriptor.resources.end()) break;
        return static_cast<std::uint32_t>(found - descriptor.resources.begin());
    }
    Fail("an image or sampler is absent from the descriptor layout");
}

bool PlainMode(const ImageResource& mode) {
    return mode.conversionFormat == IrBufferFormat::Invalid && mode.packedFormat == IrBufferFormat::Invalid && !mode.depthBits && !mode.packed;
}

std::string ScalarName(const ImageResource& image) {
    if (image.depthCompare) return "float";
    switch (image.numericClass) {
    case IrTextureNumericClass::Float: return "float";
    case IrTextureNumericClass::Uint: return "uint";
    case IrTextureNumericClass::Sint: return "int";
    case IrTextureNumericClass::Unsupported: break;
    }
    Fail("an image has an unsupported numeric class");
}

std::string TextureType(const ImageResource& image, const std::string& scalar = {}) {
    const auto shape = ShapeOf(image.dimension);
    std::string name(shape.texture);
    if (image.depthCompare) {
        if (image.dimension == RdnaImageDimension::Dim1D || image.dimension == RdnaImageDimension::Dim1DArray || image.dimension == RdnaImageDimension::Dim3D) throw MslUnsupported("MSL backend: depth comparison on 1D or 3D images has no Metal depth texture type");
        name = shape.multisampled ? (shape.arrayed ? "depth2d_ms_array" : "depth2d_ms") : (shape.arrayed ? "depth2d_array" : "depth2d");
    }
    const char* access = "sample";
    if (image.resourceClass == ImageResourceClass::Storage) access = image.atomic ? "read_write" : "write";
    else if (shape.multisampled) access = "read";
    return name + "<" + (scalar.empty() ? ScalarName(image) : scalar) + ", access::" + access + ">";
}

std::string Wrapper(State& state, const std::string& type) {
    const auto found = std::find(state.imageTypes.begin(), state.imageTypes.end(), type);
    if (found != state.imageTypes.end()) return "aps5_img" + std::to_string(found - state.imageTypes.begin());
    state.imageTypes.push_back(type);
    return "aps5_img" + std::to_string(state.imageTypes.size() - 1u);
}

std::string TextureRef(State& state, const ImageResource& image, std::uint32_t element, const std::string& scalar = {}) {
    const auto kind = DescriptorBindingForImage(image);
    state.features |= Images;
    const auto type = TextureType(image, scalar);
    return "((device const " + Wrapper(state, type) + "*)aps5_images)[" + U32(MetalAbi::GuestImageIndex(static_cast<std::uint32_t>(kind), element)) + "].t";
}

std::string SamplerRef(State& state, std::uint32_t sampler, bool point) {
    state.features |= Samplers;
    const auto element = ElementOf(state, DescriptorBindingKind::Samplers, sampler) + (point ? 1u : 0u);
    return "((device const aps5_sampler*)aps5_samplers)[" + U32(element) + "].s";
}

std::string AddressU32(const Access& access, std::uint32_t component) {
    const auto layout = GetRdnaImageAddressComponentLayout(access.mem.imageSampleFlags, component);
    const auto packed = layout.bitOffset / 32u;
    if (packed >= access.address.ArgumentCount()) Fail(access.inst, "has an image address component outside of the address operands");
    auto value = Typed(access.state, access.address, packed);
    if (layout.bitWidth == 16u) {
        if ((layout.bitOffset & 31u) != 0u) value = "(" + value + " >> 16u)";
        value = "(" + value + " & 0xffffu)";
    }
    return value;
}

std::string AddressF32(const Access& access, std::uint32_t component) {
    const auto value = AddressU32(access, component);
    if (GetRdnaImageAddressComponentLayout(access.mem.imageSampleFlags, component).bitWidth == 16u) {
        access.state.features |= Half;
        return "aps5_f16_to_f32(" + value + ")";
    }
    return "as_type<float>(" + value + ")";
}

std::string CubeLayer(const std::string& value) {
    return "float(uint(" + value + ") - ((uint(" + value + ") >> 3u) << 1u))";
}

std::string CoordF32(const Access& access, std::uint32_t first, std::uint32_t components, std::uint32_t encoded, bool gradient = false) {
    const bool cube = access.image.cube;
    const auto present = std::min(encoded, components);
    const bool cubeFace = cube && access.mem.imageDimension == RdnaImageDimension::Dim2D;
    if (first == NoComponent || (encoded < components && ((cube && !cubeFace) || encoded == 0u)) || access.mem.imageAddressComponents < first + present) Fail(access.inst, "has an image address with too few coordinate components");
    const auto component = [&](std::uint32_t index) { return index < present ? AddressF32(access, first + index) : std::string("0.0f"); };
    auto x = component(0u);
    if (components == 1u) return x;
    auto y = component(1u);
    if (cube && !gradient) {
        x = "(" + x + " - 1.0f)";
        y = "(" + y + " - 1.0f)";
    }
    if (components == 3u) {
        auto z = component(2u);
        if (cube) z = CubeLayer(z);
        return "float3(" + x + ", " + y + ", " + z + ")";
    }
    return "float2(" + x + ", " + y + ")";
}

std::string CoordU32(const Access& access) {
    const auto components = ShapeOf(access.image.dimension).coordinates;
    const auto encoded = std::min(components, ShapeOf(access.mem.imageDimension).coordinates);
    if (access.mem.imageAddressComponents < encoded) Fail(access.inst, "has an image address with too few coordinate components");
    const auto component = [&](std::uint32_t index) { return index < encoded ? AddressU32(access, index) : std::string("0u"); };
    if (components == 1u) return component(0u);
    if (components == 3u) return "uint3(" + component(0u) + ", " + component(1u) + ", " + component(2u) + ")";
    return "uint2(" + component(0u) + ", " + component(1u) + ")";
}

std::string LodU32(const Access& access) {
    if (!access.mem.imageHasMip) return "0u";
    const auto component = ShapeOf(access.mem.imageDimension).coordinates;
    if (access.mem.imageAddressComponents <= component) Fail(access.inst, "has an image address without the mip level component");
    return AddressU32(access, component);
}

std::string SampleIndex(const Access& access) {
    const auto shape = ShapeOf(access.image.dimension);
    if (!shape.multisampled) return "0u";
    const auto address = ShapeOf(access.mem.imageDimension);
    if (!address.multisampled || access.mem.imageAddressComponents <= address.coordinates) Fail(access.inst, "has no sample index in the image address");
    return AddressU32(access, address.coordinates);
}

std::string SpatialPart(const std::string& coord, const Shape& shape) {
    if (shape.coordinates == 1u) return coord;
    if (shape.arrayed) return shape.spatial == 1u ? coord + ".x" : coord + ".xy";
    return coord;
}

std::string HostPlane(const std::string& spatial, const Shape& shape, bool floating) {
    if (shape.spatial != 1u) return spatial;
    return floating ? "float2(" + spatial + ", 0.5f)" : "uint2(" + spatial + ", 0u)";
}

std::string LayerPart(const std::string& coord, const Shape& shape, bool rounded) {
    const auto layer = coord + (shape.spatial == 1u ? ".y" : ".z");
    return rounded ? "uint(max(floor(" + layer + " + 0.5f), 0.0f))" : "uint(" + layer + ")";
}

std::string Bits(const std::string& value, IrTextureNumericClass numericClass) {
    return numericClass == IrTextureNumericClass::Uint ? value : "as_type<uint>(" + value + ")";
}

std::string ResultVector(Access& access, const std::string& value, IrTextureNumericClass numericClass, bool dref, bool gather) {
    State& state = access.state;
    const auto& mem = access.mem;
    const auto valueClass = dref ? IrTextureNumericClass::Float : numericClass;
    const bool integer = valueClass == IrTextureNumericClass::Uint || valueClass == IrTextureNumericClass::Sint;
    const auto source = Temp(state);
    Line(state, "const auto " + source + " = " + value + ";");
    const auto scalar = [&](std::uint32_t index) { return dref ? source : source + "." + Swizzle(index); };
    if (mem.dataBits == 16u) {
        if (mem.dataDwords > 4u) Fail(access.inst, "has more than four packed image result dwords");
        const auto dmask = EffectiveDmask(mem);
        std::array<std::string, 4> packed{"0u", "0u", "0u", "0u"};
        for (std::uint32_t word = 0; word < mem.dataDwords; ++word) {
            const auto lowIndex = word * 2u;
            const auto highIndex = lowIndex + 1u;
            const auto low = scalar(dref ? 0u : gather ? lowIndex : DmaskComponent(access.inst, dmask, lowIndex));
            const auto high = highIndex < mem.componentCount ? scalar(dref ? 0u : gather ? highIndex : DmaskComponent(access.inst, dmask, highIndex)) : std::string(integer ? "0u" : "0.0f");
            if (integer) {
                packed[word] = "((" + Bits(low, valueClass) + " & 0xffffu) | ((" + Bits(high, valueClass) + " & 0xffffu) << 16u))";
            } else {
                state.features |= Half;
                packed[word] = "(aps5_f32_to_f16(" + low + ") | (aps5_f32_to_f16(" + high + ") << 16u))";
            }
        }
        return "uint4(" + packed[0] + ", " + packed[1] + ", " + packed[2] + ", " + packed[3] + ")";
    }
    std::array<std::string, 4> components;
    for (std::uint32_t index = 0; index < 4u; ++index) {
        if (dref) {
            components[index] = Bits(source, valueClass);
        } else if (access.image.depthUnorm16) {
            components[index] = "uint(" + scalar(index) + " * 65535.0f + 0.5f)";
        } else {
            components[index] = Bits(scalar(index), valueClass);
        }
    }
    return "uint4(" + components[0] + ", " + components[1] + ", " + components[2] + ", " + components[3] + ")";
}

std::optional<std::string> ConstantOffset(const Access& access, const SampleLayout& layout, const Shape& shape) {
    const auto component = GetRdnaImageAddressComponentLayout(access.mem.imageSampleFlags, layout.offset);
    const auto argument = component.bitOffset / 32u;
    if (component.bitWidth != 32u || argument >= access.address.ArgumentCount()) return std::nullopt;
    const auto* value = access.address.Argument(argument)->Resolve();
    if (!value->HasImmediate()) return std::nullopt;
    const auto bits = value->ImmediateU32();
    std::string text = shape.spatial == 1u ? "" : (shape.spatial == 2u ? "int2(" : "int3(");
    for (std::uint32_t index = 0; index < shape.spatial; ++index) {
        const auto field = (bits >> (index * 8u)) & 0x3fu;
        text += (index == 0u ? "" : ", ") + std::to_string(static_cast<std::int32_t>(field ^ 0x20u) - 0x20);
    }
    return shape.spatial == 1u ? text : text + ")";
}

std::string PackedOffset(const Access& access, const SampleLayout& layout, const Shape& shape) {
    if (layout.offset == NoComponent || access.mem.imageAddressComponents <= layout.offset) Fail(access.inst, "has no texel offset in the image address");
    const auto packed = "int(" + AddressU32(access, layout.offset) + ")";
    std::string text = shape.spatial == 1u ? "" : (shape.spatial == 2u ? "int2(" : "int3(");
    for (std::uint32_t index = 0; index < shape.spatial; ++index) text += std::string(index == 0u ? "" : ", ") + "extract_bits(" + packed + ", " + U32(index * 8u) + ", 6u)";
    return shape.spatial == 1u ? text : text + ")";
}

void EmitQueryDimensions(Access& access, const std::string& texture, const std::string& result) {
    State& state = access.state;
    const auto shape = ShapeOf(access.image.dimension);
    const auto lod = shape.multisampled ? std::string() : AddressU32(access, 0u);
    std::array<std::string, 4> values{"0u", "0u", "0u", "0u"};
    std::uint32_t cursor = 0;
    const auto push = [&](const std::string& expression) { values[cursor++] = expression; };
    push(texture + ".get_width(" + lod + ")");
    const bool volume = access.image.dimension == RdnaImageDimension::Dim3D;
    const bool plane = shape.spatial >= 2u;
    if (plane) push(texture + ".get_height(" + lod + ")");
    if (volume) push(texture + ".get_depth(" + lod + ")");
    if (shape.arrayed) push(texture + ".get_array_size()");
    if (access.mem.imageDimension == RdnaImageDimension::Dim3D && access.image.dimension == RdnaImageDimension::Dim2D) values[2] = "1u";
    if (!shape.multisampled) values[3] = texture + ".get_num_mip_levels()";
    Line(state, result + " = uint4(" + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ");");
}

void EmitRead(Access& access, const std::string& texture, const std::string& result) {
    State& state = access.state;
    const auto shape = ShapeOf(access.image.dimension);
    if (access.image.resourceClass != ImageResourceClass::Sampled) Fail(access.inst, "reads a storage image");
    std::string condition;
    if (access.mem.imageDimension == RdnaImageDimension::Dim3D && (access.image.dimension == RdnaImageDimension::Dim2D || access.image.dimension == RdnaImageDimension::Dim1D)) condition = AddressU32(access, 2u) + " == 0u";
    const bool volumeSlice = access.mem.imageDimension == RdnaImageDimension::Dim2DArray && access.image.dimension == RdnaImageDimension::Dim3D;
    if (volumeSlice) condition += std::string(condition.empty() ? "" : " && ") + AddressU32(access, 2u) + " < " + texture + ".get_depth(" + LodU32(access) + ")";
    const auto coord = Temp(state);
    Open(state, condition.empty() ? std::string("{") : "if (" + condition + ") {");
    Line(state, "const auto " + coord + " = " + CoordU32(access) + ";");
    std::string call;
    if (shape.multisampled) {
        call = texture + ".read(" + SpatialPart(coord, shape) + (shape.arrayed ? ", " + coord + ".z" : "") + ", " + SampleIndex(access) + ")";
    } else if (shape.arrayed) {
        call = texture + ".read(" + HostPlane(SpatialPart(coord, shape), shape, false) + ", " + coord + (shape.spatial == 1u ? ".y" : ".z") + ", " + LodU32(access) + ")";
    } else {
        call = texture + ".read(" + HostPlane(coord, shape, false) + ", " + LodU32(access) + ")";
    }
    Line(state, result + " = " + ResultVector(access, call, access.image.numericClass, false, false) + ";");
    Close(state);
}

std::string RegisterSelector(Access& access, std::uint32_t source) {
    return Constant(access.state, PipelineSpecialization::ImageBase + access.mem.resource * PipelineSpecialization::ImageWords + 1u + source, source + 4u);
}

std::string StoreTexel(Access& access, const std::string& data, bool integer) {
    State& state = access.state;
    const auto& mem = access.mem;
    const auto dmask = EffectiveDmask(mem);
    std::array<std::string, 4> values;
    for (std::uint32_t component = 0; component < 4u; ++component) {
        std::string raw = "0u";
        for (std::uint32_t reverse = 4u; reverse != 0u; --reverse) {
            const auto source = reverse - 1u;
            const auto packedIndex = DmaskComponentIndex(dmask, source);
            std::string candidate = "0u";
            if (((dmask >> source) & 1u) != 0u) {
                candidate = data + "." + Swizzle(mem.dataBits == 16u ? packedIndex / 2u : packedIndex);
                if (mem.dataBits == 16u) {
                    if ((packedIndex & 1u) != 0u) candidate = "(" + candidate + " >> 16u)";
                    candidate = "(" + candidate + " & 0xffffu)";
                }
            }
            raw = "(" + RegisterSelector(access, source) + " == " + U32(component + 4u) + " ? " + candidate + " : " + raw + ")";
        }
        if (integer) {
            values[component] = raw;
        } else if (mem.dataBits == 16u) {
            state.features |= Half;
            values[component] = "aps5_f16_to_f32(" + raw + ")";
        } else {
            values[component] = "as_type<float>(" + raw + ")";
        }
    }
    return std::string(integer ? "uint4(" : "float4(") + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ")";
}

void EmitWrite(Access& access, const std::string& texture) {
    State& state = access.state;
    const auto shape = ShapeOf(access.image.dimension);
    if (access.image.resourceClass != ImageResourceClass::Storage) Fail(access.inst, "writes a sampled image");
    if (shape.multisampled) throw MslUnsupported("MSL backend: multisampled storage image writes are not emitted");
    const bool integer = access.image.numericClass == IrTextureNumericClass::Uint;
    if (access.mem.imageDimension == RdnaImageDimension::Dim2DArray && access.image.dimension == RdnaImageDimension::Dim3D) {
        if (access.image.mipMode == ImageMipMode::DynamicStorage) throw MslUnsupported("MSL backend: a 2D-array address stored to a 3D image with dynamic mips is not emitted");
        Open(state, "if (" + AddressU32(access, 2u) + " < " + texture + ".get_depth()) {");
    } else {
        Open(state, "{");
    }
    const auto coord = Temp(state);
    Line(state, "const auto " + coord + " = " + CoordU32(access) + ";");
    const auto texel = StoreTexel(access, Typed(state, access.inst, 2u), integer);
    if (shape.arrayed) Line(state, texture + ".write(" + texel + ", " + HostPlane(SpatialPart(coord, shape), shape, false) + ", " + coord + (shape.spatial == 1u ? ".y" : ".z") + ");");
    else Line(state, texture + ".write(" + texel + ", " + HostPlane(coord, shape, false) + ");");
    Close(state);
}

struct AtomicKind {
    std::string method;
    std::string scalar = "uint";
    std::uint32_t op = 0;
    bool native = false;
    bool wide = false;
};

AtomicKind AtomicOf(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::ImageAtomicIAdd32: return {"atomic_fetch_add", "uint", 2u, true};
    case IrOpcode::ImageAtomicISub32: return {"atomic_fetch_sub", "uint", 3u, true};
    case IrOpcode::ImageAtomicUMin32: return {"atomic_fetch_min", "uint", 5u, true};
    case IrOpcode::ImageAtomicUMax32: return {"atomic_fetch_max", "uint", 7u, true};
    case IrOpcode::ImageAtomicSMin32: return {"atomic_fetch_min", "int", 4u, true};
    case IrOpcode::ImageAtomicSMax32: return {"atomic_fetch_max", "int", 6u, true};
    case IrOpcode::ImageAtomicAnd32: return {"atomic_fetch_and", "uint", 8u, true};
    case IrOpcode::ImageAtomicOr32: return {"atomic_fetch_or", "uint", 9u, true};
    case IrOpcode::ImageAtomicXor32: return {"atomic_fetch_xor", "uint", 10u, true};
    case IrOpcode::ImageAtomicSwap32: return {"atomic_exchange", "uint", 0u, true};
    case IrOpcode::ImageAtomicCmpSwap32: return {{}, "uint", 1u};
    case IrOpcode::ImageAtomicInc32: return {{}, "uint", 11u};
    case IrOpcode::ImageAtomicDec32: return {{}, "uint", 12u};
    case IrOpcode::ImageAtomicFMin32: return {{}, "uint", 15u};
    case IrOpcode::ImageAtomicFMax32: return {{}, "uint", 16u};
    case IrOpcode::ImageAtomicFCmpSwap32: return {{}, "uint", 20u};
    case IrOpcode::ImageAtomicSwap64: return {{}, "ulong", 0u, false, true};
    case IrOpcode::ImageAtomicCmpSwap64: return {{}, "ulong", 1u, false, true};
    case IrOpcode::ImageAtomicIAdd64: return {{}, "ulong", 2u, false, true};
    case IrOpcode::ImageAtomicISub64: return {{}, "ulong", 3u, false, true};
    case IrOpcode::ImageAtomicSMin64: return {{}, "ulong", 4u, false, true};
    case IrOpcode::ImageAtomicSMax64: return {{}, "ulong", 6u, false, true};
    case IrOpcode::ImageAtomicUMin64: return {"atomic_min", "ulong", 5u, true, true};
    case IrOpcode::ImageAtomicUMax64: return {"atomic_max", "ulong", 7u, true, true};
    case IrOpcode::ImageAtomicAnd64: return {{}, "ulong", 8u, false, true};
    case IrOpcode::ImageAtomicOr64: return {{}, "ulong", 9u, false, true};
    case IrOpcode::ImageAtomicXor64: return {{}, "ulong", 10u, false, true};
    default: break;
    }
    Fail("an image atomic has no Metal lowering");
}

void EmitAtomic(Access& access, const std::string& texture, const std::string& result) {
    State& state = access.state;
    const auto opcode = access.inst.Opcode();
    const auto kind = AtomicOf(opcode);
    const auto shape = ShapeOf(access.image.dimension);
    const auto numeric = access.image.numericClass;
    if (access.image.resourceClass != ImageResourceClass::Storage || (numeric != IrTextureNumericClass::Uint && numeric != IrTextureNumericClass::Sint && !kind.wide) || shape.multisampled || access.image.dimension == RdnaImageDimension::Dim3D) throw MslUnsupported("MSL backend: image atomics need a single-sample 1D or 2D unsigned or signed integer storage image");
    const bool wide = kind.wide;
    if (wide && !kind.native) throw MslUnsupported("MSL backend: " + std::string(IrOpcodeName(opcode)) + " is not emitted (a Metal ulong texture only has atomic_min and atomic_max)");
    if (wide && access.inst.HasUses()) throw MslUnsupported("MSL backend: " + std::string(IrOpcodeName(opcode)) + " returns the old value, which Metal's ulong texture atomic_min and atomic_max do not");
    if (wide) state.features |= Atomic64;
    if (!kind.native) state.features |= UpdateTable;
    Open(state, "{");
    const auto coord = Temp(state);
    Line(state, "const auto " + coord + " = " + CoordU32(access) + ";");
    const auto position = shape.arrayed ? HostPlane(SpatialPart(coord, shape), shape, false) + ", " + coord + (shape.spatial == 1u ? ".y" : ".z") : HostPlane(coord, shape, false);
    std::string view = texture;
    if (wide || kind.scalar != ScalarName(access.image)) {
        view = Temp(state);
        const auto element = access.element;
        Line(state, "const " + TextureType(access.image, kind.scalar) + " " + view + " = " + TextureRef(state, access.image, element, kind.scalar) + ";");
    }
    const auto vec = [&](const std::string& value) { return kind.scalar + "4(" + value + ", " + kind.scalar + "(0), " + kind.scalar + "(0), " + kind.scalar + "(0))"; };
    const auto data = Typed(state, access.inst, 2u);
    const auto value = kind.scalar == "int" ? "int(" + data + ")" : data;
    const bool compare = kind.op == 1u;
    const bool comparing = compare || kind.op == 20u;
    const auto operand = comparing ? Typed(state, access.inst, 3u) : std::string("0u");
    const auto current = Temp(state);
    Line(state, "atomic_thread_fence(mem_flags::mem_texture, memory_order_seq_cst, thread_scope_device);");
    if (kind.native && wide) {
        Line(state, view + "." + kind.method + "(" + position + ", " + vec(value) + ");");
    } else if (kind.native) {
        Line(state, result + " = " + (kind.scalar == "int" ? "uint(" : "(") + view + "." + kind.method + "(" + position + ", " + vec(value) + ").x);");
    } else {
        const auto desired = Temp(state);
        Line(state, kind.scalar + "4 " + current + " = " + view + ".atomic_load(" + position + ");");
        Open(state, "while (true) {");
        if (compare) {
            Line(state, "if (" + current + ".x != " + operand + ") break;");
            Line(state, "const " + kind.scalar + " " + desired + " = " + data + ";");
        } else {
            Line(state, "const " + kind.scalar + " " + desired + " = " + (wide ? "aps5_update64(" : "aps5_update(") + U32(kind.op) + ", " + current + ".x, " + data + ", " + operand + ");");
            Line(state, "if (" + desired + " == " + current + ".x) break;");
        }
        Line(state, "if (" + view + ".atomic_compare_exchange_weak(" + position + ", &" + current + ", " + vec(desired) + ")) break;");
        Close(state);
        Line(state, result + " = " + current + ".x;");
    }
    Close(state);
}

struct SamplerParts {
    std::string sampler;
    std::string texture;
};

void EmitSampling(Access& access, const SamplerParts& parts, const std::string& result) {
    State& state = access.state;
    const auto& mem = access.mem;
    const auto& image = access.image;
    const auto shape = ShapeOf(image.dimension);
    const bool gather = access.inst.Opcode() == IrOpcode::ImageGatherRaw;
    if (shape.multisampled) throw MslUnsupported("MSL backend: multisampled images cannot be sampled");
    if (image.numericClass != IrTextureNumericClass::Float && !image.depthCompare) throw MslUnsupported("MSL backend: sampling integer images has no Metal sample function");
    const auto layout = LayoutOf(mem);
    const bool dref = HasFlag(mem, RdnaImageSampleFlagCompare);
    if (dref && (image.emulatedCompare & EmulatedCompare::Enabled) != 0u) throw MslUnsupported("MSL backend: depth comparison of a color texture is emulated in the shader and is not emitted");
    if (dref != image.depthCompare) throw MslUnsupported("MSL backend: depth comparison and the depth image heap disagree");
    const auto& samplers = state.program.Info().samplers;
    const auto& sampler = samplers.at(mem.sampler);
    const bool explicitLod = ImageSampleExplicitLod(mem.imageSampleFlags, state.program.Resources().stage);
    const auto texture = parts.texture;
    const auto coordinate = Temp(state);
    Line(state, "const auto " + coordinate + " = " + CoordF32(access, layout.coord, shape.coordinates, ShapeOf(mem.imageDimension).coordinates) + ";");
    const auto spatial = HostPlane(SpatialPart(coordinate, shape), shape, true);
    const std::string array = shape.arrayed ? LayerPart(coordinate, shape, true) : std::string();

    std::string options;
    if (gather) {
        if (layout.clamp != NoComponent) throw MslUnsupported("MSL backend: a gather with an LOD clamp is not emitted");
        if (HasFlag(mem, RdnaImageSampleFlagGatherHorizontal)) throw MslUnsupported("MSL backend: horizontal gather offsets are not emitted");
        if (shape.coordinates == 1u || (shape.spatial == 1u) || image.dimension == RdnaImageDimension::Dim3D) throw MslUnsupported("MSL backend: gathers need a 2D or 2D-array image");
    } else if (HasFlag(mem, RdnaImageSampleFlagDerivative)) {
        if (shape.spatial == 1u) throw MslUnsupported("MSL backend: gradients on 1D images have no Metal sample option");
        const auto gradX = CoordF32(access, layout.gradX, shape.spatial, ShapeOf(mem.imageDimension).spatial, true);
        const auto gradY = CoordF32(access, layout.gradY, shape.spatial, ShapeOf(mem.imageDimension).spatial, true);
        options = std::string(shape.spatial == 3u ? "gradient3d(" : "gradient2d(") + gradX + ", " + gradY + ")";
        if (layout.clamp != NoComponent) options += ", min_lod_clamp(" + AddressF32(access, layout.clamp) + ")";
    } else if (explicitLod) {
        auto lod = HasFlag(mem, RdnaImageSampleFlagLod) ? AddressF32(access, layout.lod) : std::string("0.0f");
        if (layout.clamp != NoComponent) lod = "max(" + lod + ", " + AddressF32(access, layout.clamp) + ")";
        options = "level(" + lod + ")";
    } else {
        if (layout.bias != NoComponent) options = "bias(" + AddressF32(access, layout.bias) + ")";
        if (layout.clamp != NoComponent) options += std::string(options.empty() ? "" : ", ") + "min_lod_clamp(" + AddressF32(access, layout.clamp) + ")";
    }

    const bool hasOffset = layout.offset != NoComponent;
    if (hasOffset && shape.spatial == 1u) throw MslUnsupported("MSL backend: texel offsets on 1D images have no Metal sample option");
    const auto constantOffset = hasOffset ? ConstantOffset(access, layout, shape) : std::nullopt;
    const bool foldOffset = hasOffset && FoldsTexelOffsets(sampler) && ResourceMaterializer::UnnormalizedSampleMismatch(image) == nullptr;
    if (hasOffset && !constantOffset && !foldOffset) throw MslUnsupported("MSL backend: a texel offset that is not constant has no Metal sample option");

    const auto call = [&](const std::string& coord, const std::string& layer, const std::string& opts, const std::optional<std::string>& offset) {
        std::string text = texture + (gather ? (dref ? ".gather_compare(" : ".gather(") : (dref ? ".sample_compare(" : ".sample(")) + parts.sampler + ", " + coord;
        if (shape.arrayed) text += ", " + layer;
        if (dref) {
            if (layout.dref == NoComponent || mem.imageAddressComponents <= layout.dref) Fail(access.inst, "has no depth reference in the image address");
            text += ", " + AddressF32(access, layout.dref);
        }
        if (!opts.empty()) text += ", " + opts;
        if (offset) text += ", " + *offset;
        if (gather && !dref) {
            if (!offset) text += ", int2(0)";
            const auto dmask = EffectiveDmask(mem);
            if (dmask != 1u && dmask != 2u && dmask != 4u && dmask != 8u) Fail(access.inst, "gathers a dmask that selects several components");
            static constexpr std::string_view Components[4] = {"x", "y", "z", "w"};
            text += std::string(", component::") + std::string(Components[DmaskComponent(access.inst, dmask, 0u)]);
        } else if (gather && !offset) {
            text += ", int2(0)";
        }
        return text + ")";
    };
    const auto foldedCoordinate = [&]() {
        const auto offset = PackedOffset(access, layout, shape);
        const auto converted = shape.spatial == 2u ? "float2(" + offset + ")" : "float3(" + offset + ")";
        return SpatialPart(coordinate, shape) + " + " + converted;
    };
    const auto assign = [&](const std::string& expression) {
        Line(state, result + " = " + ResultVector(access, expression, image.numericClass, dref && !gather, gather) + ";");
    };

    const bool baseLevelIfUnnormalized = !dref && !gather && SamplesBaseLevelIfUnnormalized(sampler);
    if (baseLevelIfUnnormalized) {
        const auto unnormalized = Constant(state, PipelineSpecialization::UnnormalizedBaseLevelBase + mem.sampler, 0u);
        Open(state, "if (" + unnormalized + " != 0u) {");
        assign(call(hasOffset ? foldedCoordinate() : spatial, array, "level(0.0f)", std::nullopt));
        Close(state, "} else {");
        ++state.depth;
        if (constantOffset) assign(call(spatial, array, options, constantOffset));
        else assign(call(foldOffset ? foldedCoordinate() : spatial, array, options, std::nullopt));
        Close(state);
    } else if (!hasOffset) {
        assign(call(spatial, array, options, std::nullopt));
    } else if (!constantOffset) {
        assign(call(foldedCoordinate(), array, options, std::nullopt));
    } else if (foldOffset) {
        const auto folded = Constant(state, PipelineSpecialization::FoldOffsetBase + mem.sampler, 0u);
        Open(state, "if (" + folded + " != 0u) {");
        assign(call(foldedCoordinate(), array, options, std::nullopt));
        Close(state, "} else {");
        ++state.depth;
        assign(call(spatial, array, options, constantOffset));
        Close(state);
    } else {
        assign(call(spatial, array, options, constantOffset));
    }
}

void EmitQueryLod(Access& access, const SamplerParts& parts, const std::string& result) {
    State& state = access.state;
    const auto shape = ShapeOf(access.image.dimension);
    if (shape.spatial == 1u || shape.multisampled) throw MslUnsupported("MSL backend: level-of-detail queries need a 2D or 3D image");
    const auto coord = Temp(state);
    Line(state, "const auto " + coord + " = " + CoordF32(access, 0u, shape.spatial, ShapeOf(access.mem.imageDimension).spatial) + ";");
    Line(state, result + " = uint4(as_type<uint>(" + parts.texture + ".calculate_clamped_lod(" + parts.sampler + ", " + coord + ")), as_type<uint>(" + parts.texture + ".calculate_unclamped_lod(" + parts.sampler + ", " + coord + ")), 0u, 0u);");
}

}

bool EmitImage(State& state, const IrValue& inst) {
    const auto opcode = inst.Opcode();
    const auto name = IrOpcodeName(opcode);
    if (!name.starts_with("Image") || opcode == IrOpcode::ImageSampleImplicitLod || opcode == IrOpcode::ImageSampleExplicitLod || opcode == IrOpcode::ImageBvhIntersectRay) return false;
    if (opcode == IrOpcode::GetImageResource || opcode == IrOpcode::MakeImageAddress) return false;
    const auto info = ImageOpcodeInfoOf(opcode);
    const auto& mem = Memory(state, inst);
    const auto& images = state.program.Info().images;
    if (mem.resource >= images.size()) Fail(inst, "names an image the program does not have");
    const auto& base = images[mem.resource];
    if (base.table != NoTable) throw MslUnsupported("MSL backend: " + std::string(name) + " reads an image through a bindless image table, which is not emitted yet");
    if (info.needsSampler && state.program.Info().samplers.at(mem.sampler).table != NoTable) throw MslUnsupported("MSL backend: " + std::string(name) + " uses a sampler from a bindless sampler table, which is not emitted yet");
    if (base.srgbDecode) throw MslUnsupported("MSL backend: " + std::string(name) + " decodes an sRGB image in the shader, which is not emitted");
    if (base.constantSwizzle) throw MslUnsupported("MSL backend: " + std::string(name) + " samples an image with a constant channel swizzle, which is not emitted");
    if (mem.imagePacked || mem.imageByElements != 0u) throw MslUnsupported("MSL backend: " + std::string(name) + " is a packed or element-wise image access, which is not emitted");
    if (mem.dataBits == 16u && base.depthBits) throw MslUnsupported("MSL backend: " + std::string(name) + " reads depth bits as 16-bit results, which is not emitted");
    const IrValue* resource = inst.Argument(0)->Resolve();
    if (resource->Opcode() != IrOpcode::GetImageResource) Fail(inst, "has a typed image handle with the wrong producer");
    const IrValue* address = inst.Argument(info.needsSampler ? 2u : 1u)->Resolve();
    if (address->Opcode() != IrOpcode::MakeImageAddress) Fail(inst, "has an image address that was not made by MakeImageAddress");

    const auto modes = ResourceMaterializer::RuntimeImageModes(base);
    const bool returns = opcode != IrOpcode::ImageWrite;
    std::string result;
    if (returns) {
        result = Temp(state);
        const auto type = TypeName(inst.Type());
        Line(state, type + " " + result + " = " + type + "(0);");
    }
    const IrValue* active = inst.Argument(inst.ArgumentCount() - 1u)->Resolve();
    const bool always = active->HasImmediate() && active->Type() == IrType::Bool && active->ImmediateBool();
    Open(state, always ? std::string("{") : "if (" + Typed(state, inst, inst.ArgumentCount() - 1u) + ") {");

    const auto emitMode = [&](const ImageResource& mode, std::uint32_t element, const std::function<void(Access&, const std::string&)>& body) {
        Access access{state, inst, mem, mode, *address, element};
        const auto texture = Temp(state);
        Line(state, "const " + TextureType(mode) + " " + texture + " = " + TextureRef(state, mode, element) + ";");
        body(access, texture);
    };
    const auto emitBody = [&](const ImageResource& mode) {
        const auto kind = DescriptorBindingForImage(mode);
        const auto element = ElementOf(state, kind, mem.resource);
        const auto sampled = [&](Access& access, const std::string& texture) {
            const bool point = access.image.numericClass == IrTextureNumericClass::Sint;
            if (opcode == IrOpcode::ImageQueryDimensions) EmitQueryDimensions(access, texture, result);
            else if (opcode == IrOpcode::ImageRead) EmitRead(access, texture, result);
            else if (opcode == IrOpcode::ImageQueryLod) EmitQueryLod(access, {SamplerRef(state, mem.sampler, point), texture}, result);
            else EmitSampling(access, {SamplerRef(state, mem.sampler, point), texture}, result);
        };
        if (opcode == IrOpcode::ImageWrite || info.access == ImageAccess::Atomic) {
            const auto storage = [&](Access& access, const std::string& texture) {
                if (opcode == IrOpcode::ImageWrite) EmitWrite(access, texture);
                else EmitAtomic(access, texture, result);
            };
            if (mode.mipMode != ImageMipMode::DynamicStorage) {
                emitMode(mode, element, storage);
                return;
            }
            if (mode.mipCount == 0u) Fail(inst, "has a dynamic storage image without mip descriptors");
            Access probe{state, inst, mem, mode, *address, element};
            const auto lod = Temp(state);
            Line(state, "const uint " + lod + " = " + LodU32(probe) + ";");
            const auto count = Constant(state, PipelineSpecialization::MipCountBase + mem.resource, mode.mipCount);
            Open(state, "switch (" + lod + ") {");
            for (std::uint32_t mip = 0; mip < mode.mipCount; ++mip) {
                Open(state, "case " + U32(mip) + ": {");
                Open(state, "if (" + U32(mip) + " < " + count + ") {");
                emitMode(mode, element + mip, storage);
                Close(state);
                Line(state, "break;");
                Close(state);
            }
            Open(state, "default: {");
            Line(state, "break;");
            Close(state);
            Close(state);
            return;
        }
        emitMode(mode, element, sampled);
    };

    std::vector<std::uint32_t> supported;
    for (std::uint32_t index = 0; index < modes.size(); ++index) {
        if (PlainMode(modes[index])) supported.push_back(index);
    }
    if (supported.empty()) throw MslUnsupported("MSL backend: " + std::string(name) + " has no image mode without format conversion");
    if (modes.size() == 1u) {
        emitBody(modes[supported.front()]);
    } else {
        const auto selector = Constant(state, PipelineSpecialization::ImageBase + mem.resource * PipelineSpecialization::ImageWords, 0u);
        Open(state, "switch (" + selector + ") {");
        std::optional<std::string> firstError;
        std::uint32_t emitted = 0;
        for (const auto index : supported) {
            const auto mark = state.body.size();
            const auto depth = state.depth;
            try {
                Open(state, "case " + U32(index) + ": {");
                emitBody(modes[index]);
                Line(state, "break;");
                Close(state);
                ++emitted;
            } catch (const MslUnsupported& error) {
                state.body.resize(mark);
                state.depth = depth;
                if (!firstError) firstError = error.what();
            }
        }
        if (emitted == 0u) throw MslUnsupported(*firstError);
        Open(state, "default: {");
        Line(state, "break;");
        Close(state);
        Close(state);
    }
    Close(state);
    if (returns) Define(state, inst, result);
    return true;
}

}
