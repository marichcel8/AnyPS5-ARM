#include "MslState.hpp"
#include "MetalAbi.hpp"
#include "PipelineSpecialization.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "RdnaDecoder/RdnaImageOpDecoder.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "ImageTableAbi.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
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

struct TableAccess {
    std::string element;
    std::string dword3;
    std::string sampler;
};

struct Access {
    State& state;
    const IrValue& inst;
    const MemoryInfo& mem;
    const ImageResource& image;
    const IrValue& address;
    std::uint32_t element;
    const TableAccess* table = nullptr;
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

std::uint32_t TypeCode(SpirvFormatComponentType type) {
    return static_cast<std::uint32_t>(type);
}

std::uint32_t LowMask(std::uint32_t bits) {
    return bits >= 32u ? 0xffffffffu : (1u << bits) - 1u;
}

SpirvBufferFormatInfo ConversionFormat(const ImageResource& image) {
    const auto format = image.conversionFormat;
    if (format == IrBufferFormat::Invalid || (image.resourceClass == ImageResourceClass::Storage && SampledTextureNumericClass(format) == IrTextureNumericClass::Sint)) return {};
    const auto info = GetFormatInfo(format);
    if (SampledTextureNumericClass(format) != IrTextureNumericClass::Uint || (info.type != SpirvFormatComponentType::Uint && info.type != SpirvFormatComponentType::Unorm && info.type != SpirvFormatComponentType::Float) || !info.packedBitfield || info.byteSize != 4u || info.componentCount == 0u || info.componentCount > 4u) throw MslUnsupported("MSL backend: an image conversion format is not a packed 32-bit unsigned integer, unorm or float format");
    return info;
}

SpirvBufferFormatInfo SintStorageFormat(const ImageResource& image) {
    const auto format = image.conversionFormat;
    if (image.resourceClass != ImageResourceClass::Storage || format == IrBufferFormat::Invalid || SampledTextureNumericClass(format) != IrTextureNumericClass::Sint) return {};
    const auto info = GetFormatInfo(format);
    if (image.numericClass != IrTextureNumericClass::Uint || info.type != SpirvFormatComponentType::Sint || info.packedBitfield || info.componentCount == 0u || info.byteSize == 12u) throw MslUnsupported("MSL backend: a storage image conversion format is not a SINT format written through a UINT view");
    return info;
}

bool NormalizedConversion(const SpirvBufferFormatInfo& info) {
    return info.type == SpirvFormatComponentType::Unorm || info.type == SpirvFormatComponentType::Float;
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

std::string TextureRef(State& state, const ImageResource& image, std::uint32_t element, const TableAccess* table, const std::string& scalar = {}) {
    const auto kind = DescriptorBindingForImage(image);
    const auto type = TextureType(image, scalar);
    if (image.table != NoTable) {
        if (table == nullptr) Fail("an image table access has no table element");
        return "((device const " + Wrapper(state, type) + "*)" + (image.resourceClass == ImageResourceClass::Storage ? "aps5_storage_table" : "aps5_image_table") + ")[" + table->element + "].t";
    }
    state.features |= Images;
    return "((device const " + Wrapper(state, type) + "*)aps5_images)[" + U32(MetalAbi::GuestImageIndex(static_cast<std::uint32_t>(kind), element)) + "].t";
}

bool PointSampled(const State& state, const ImageResource& image, std::uint32_t sampler) {
    if (image.table != NoTable || state.program.Info().samplers.at(sampler).table != NoTable) return image.numericClass == IrTextureNumericClass::Uint || image.numericClass == IrTextureNumericClass::Sint || image.conversionFormat != IrBufferFormat::Invalid || image.depthBits;
    return image.numericClass == IrTextureNumericClass::Sint || image.conversionFormat != IrBufferFormat::Invalid || image.depthBits;
}

std::string SamplerRef(State& state, const ImageResource& image, std::uint32_t sampler, const TableAccess* table) {
    const bool point = PointSampled(state, image, sampler);
    if (state.program.Info().samplers.at(sampler).table != NoTable) {
        if (table == nullptr || table->sampler.empty()) Fail("a sampler table access has no table element");
        return "((device const aps5_sampler*)aps5_sampler_table)[" + table->sampler + " + " + U32((image.depthCompare ? 2u : 0u) + (point ? 1u : 0u)) + "].s";
    }
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

std::string RegisterSelector(Access& access, std::uint32_t source) {
    if (access.image.table != NoTable) {
        if (access.table == nullptr) Fail(access.inst, "requests an image table swizzle outside a table access");
        return "extract_bits(" + access.table->dword3 + ", " + U32(source * 3u) + ", 3u)";
    }
    return Constant(access.state, PipelineSpecialization::ImageBase + access.mem.resource * PipelineSpecialization::ImageWords + 1u + source, source + 4u);
}

std::string Swizzled(const std::string& selector, const std::array<std::string, 4>& components, const std::string& one) {
    std::string value = "(" + selector + " == 1u ? " + one + " : 0u)";
    for (std::uint32_t index = 4u; index != 0u; --index) value = "(" + selector + " == " + U32(index + 3u) + " ? " + components[index - 1u] + " : " + value + ")";
    return value;
}

void RequireConvertedAccess(const Access& access, const SpirvBufferFormatInfo& info) {
    if (access.mem.dataBits == 16u) throw MslUnsupported("MSL backend: a converted unorm or float image with 16-bit data is not emitted");
    for (std::uint32_t component = 0; component < 4u; ++component) {
        const auto selector = (access.image.shaderSwizzle >> (component * 3u)) & 7u;
        if (selector >= 4u && selector - 4u >= info.componentCount) Fail(access.inst, "selects a channel the converted image format does not have");
    }
}

std::array<std::string, 4> UnpackedComponents(Access& access, const SpirvBufferFormatInfo& info, const std::string& raw) {
    const bool normalized = NormalizedConversion(info);
    if (normalized) access.state.features |= Formatted;
    std::array<std::string, 4> components{"0u", "0u", "0u", "0u"};
    for (std::uint32_t component = 0; component < 4u; ++component) {
        const auto physical = component % info.componentCount;
        auto value = "extract_bits(" + raw + ", " + U32(info.componentBitOffset[physical]) + ", " + U32(info.componentBits[physical]) + ")";
        if (normalized) value = "aps5_fmt_decode(" + value + ", " + U32(TypeCode(info.type)) + ", " + U32(info.componentBits[physical]) + ")";
        components[component] = value;
    }
    return components;
}

std::string UnpackTexel(Access& access, const std::string& texel) {
    const auto info = ConversionFormat(access.image);
    if (info.format == IrBufferFormat::Invalid) return texel;
    if (NormalizedConversion(info)) RequireConvertedAccess(access, info);
    State& state = access.state;
    const auto raw = Temp(state);
    Line(state, "const uint " + raw + " = (" + texel + ").x;");
    const auto unpacked = UnpackedComponents(access, info, raw);
    std::array<std::string, 4> parts;
    for (std::uint32_t component = 0; component < 4u; ++component) {
        const auto value = Temp(state);
        Line(state, "const uint " + value + " = " + unpacked[component] + ";");
        parts[component] = value;
    }
    const auto one = info.type == SpirvFormatComponentType::Uint ? std::string("1u") : std::string("0x3f800000u");
    std::array<std::string, 4> selected;
    for (std::uint32_t component = 0; component < 4u; ++component) selected[component] = Swizzled(RegisterSelector(access, component), parts, one);
    return "uint4(" + selected[0] + ", " + selected[1] + ", " + selected[2] + ", " + selected[3] + ")";
}

std::string UnpackGather(Access& access, const std::string& gathered) {
    const auto info = ConversionFormat(access.image);
    if (info.format == IrBufferFormat::Invalid) return gathered;
    if (NormalizedConversion(info)) throw MslUnsupported("MSL backend: gathers of a converted unorm or float image need filtering in the shader and are not emitted");
    State& state = access.state;
    const auto source = Temp(state);
    Line(state, "const uint4 " + source + " = " + gathered + ";");
    const auto selector = RegisterSelector(access, DmaskComponent(access.inst, EffectiveDmask(access.mem), 0u));
    std::array<std::string, 4> lanes;
    for (std::uint32_t lane = 0; lane < 4u; ++lane) lanes[lane] = Swizzled(selector, UnpackedComponents(access, info, source + "." + Swizzle(lane)), "1u");
    return "uint4(" + lanes[0] + ", " + lanes[1] + ", " + lanes[2] + ", " + lanes[3] + ")";
}

SpirvBufferFormatInfo PackedTexelFormat(const Access& access) {
    const auto format = access.image.packedFormat;
    if (format == IrBufferFormat::Invalid || access.image.table != NoTable || access.image.depthBits || access.image.conversionFormat != IrBufferFormat::Invalid) Fail(access.inst, "accesses packed texels of an image without a packed format");
    const auto info = GetFormatInfo(format);
    if (info.packedBitfield) Fail(access.inst, "accesses packed texels of a bitfield format");
    return info;
}

std::uint32_t TexelBits(const SpirvBufferFormatInfo& info) {
    std::uint32_t bits = 0;
    for (std::uint32_t component = 0; component < info.componentCount; ++component) bits += info.componentBits[component];
    return bits;
}

std::string PackedTexel(Access& access, const std::string& texel) {
    State& state = access.state;
    const auto info = PackedTexelFormat(access);
    const auto numeric = access.image.numericClass;
    const auto source = Temp(state);
    Line(state, "const auto " + source + " = " + texel + ";");
    std::array<std::string, 4> words{"0u", "0u", "0u", "0u"};
    std::uint32_t position = 0;
    for (std::uint32_t component = 0; component < info.componentCount; ++component) {
        const auto bits = info.componentBits[component];
        const bool exact = info.type == SpirvFormatComponentType::Uint || info.type == SpirvFormatComponentType::Sint || (info.type == SpirvFormatComponentType::Unorm && bits <= 16u) || (info.type == SpirvFormatComponentType::Float && bits == 32u);
        if (!exact) throw MslUnsupported("MSL backend: packed texels of a format whose bits are not recoverable from the view are not emitted");
        const auto scalar = source + "." + Swizzle(component);
        std::string value = info.type == SpirvFormatComponentType::Unorm ? "uint(" + scalar + " * " + std::to_string(LowMask(bits)) + ".0f + 0.5f)" : numeric == IrTextureNumericClass::Uint ? scalar : "as_type<uint>(" + scalar + ")";
        if (bits < 32u) value = "(" + value + " & " + U32(LowMask(bits)) + ")";
        if (position % 32u != 0u) value = "(" + value + " << " + U32(position % 32u) + ")";
        auto& word = words[position / 32u];
        word = word == "0u" ? value : word + " | " + value;
        position += bits;
    }
    const auto texelBits = TexelBits(info);
    if (access.mem.dataSigned && texelBits < 32u) words[0] = "as_type<uint>(as_type<int>((" + words[0] + ") << " + U32(32u - texelBits) + ") >> " + U32(32u - texelBits) + ")";
    return "uint4(" + words[0] + ", " + words[1] + ", " + words[2] + ", " + words[3] + ")";
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
    if (access.mem.imagePacked) Line(state, result + " = " + PackedTexel(access, call) + ";");
    else if (ConversionFormat(access.image).format != IrBufferFormat::Invalid) Line(state, result + " = " + ResultVector(access, UnpackTexel(access, call), IrTextureNumericClass::Uint, false, false) + ";");
    else Line(state, result + " = " + ResultVector(access, call, access.image.numericClass, false, false) + ";");
    Close(state);
}

void EmitByRead(Access& access, const std::string& texture, const std::string& result) {
    State& state = access.state;
    const auto& mem = access.mem;
    const auto elements = mem.imageByElements;
    const bool packed = mem.imagePacked;
    const auto components = packed ? 1u : mem.dataDwords / elements;
    if (access.image.resourceClass != ImageResourceClass::Sampled) Fail(access.inst, "reads a storage image");
    if (access.image.dimension != RdnaImageDimension::Dim2D || components == 0u || (packed ? mem.dataDwords != 1u : components * elements != mem.dataDwords) || mem.dataBits != 32u) throw MslUnsupported("MSL backend: a MIMG BY2/BY4 load outside the measured 2D, 32-bit data subset is not emitted");
    const auto texelBits = packed ? TexelBits(PackedTexelFormat(access)) : 0u;
    if (texelBits * elements > 32u) throw MslUnsupported("MSL backend: a MIMG PCK2/PCK4 load whose elements do not fit one dword is not emitted");
    Open(state, "{");
    const auto lod = Temp(state);
    const auto inside = Temp(state);
    const auto first = Temp(state);
    const auto row = Temp(state);
    Line(state, "const uint " + lod + "_raw = " + LodU32(access) + ";");
    Line(state, "const bool " + lod + "_in = " + lod + "_raw < " + texture + ".get_num_mip_levels();");
    Line(state, "const uint " + lod + " = " + lod + "_in ? " + lod + "_raw : 0u;");
    Line(state, "const uint " + lod + "_w = max(" + texture + ".get_width(0) >> " + lod + ", 1u);");
    Line(state, "const uint " + lod + "_h = max(" + texture + ".get_height(0) >> " + lod + ", 1u);");
    const auto x = AddressU32(access, 0u);
    const auto y = AddressU32(access, 1u);
    Line(state, "const bool " + inside + " = " + lod + "_in && " + x + " < " + lod + "_w && " + lod + "_w - " + x + " >= " + U32(elements) + " && " + y + " < " + lod + "_h;");
    Line(state, "const uint " + first + " = " + inside + " ? (" + x + " & " + U32(~(elements - 1u)) + ") : 0u;");
    Line(state, "const uint " + row + " = " + inside + " ? " + y + " : 0u;");
    MemoryInfo texelMemory = mem;
    texelMemory.dmask = (1u << components) - 1u;
    texelMemory.dataDwords = components;
    texelMemory.componentCount = components;
    texelMemory.imagePacked = false;
    texelMemory.imageByElements = 0u;
    Access texelAccess{state, access.inst, texelMemory, access.image, access.address, access.element, access.table};
    std::array<std::string, 4> words{"0u", "0u", "0u", "0u"};
    for (std::uint32_t element = 0; element < elements; ++element) {
        const auto read = texture + ".read(uint2(" + first + " + " + U32(element) + ", " + row + "), " + lod + ")";
        const auto texel = Temp(state);
        if (packed) {
            Line(state, "const uint4 " + texel + " = " + PackedTexel(access, read) + ";");
            const auto bits = "((" + texel + ".x & " + U32(LowMask(texelBits)) + ")" + (element == 0u ? std::string() : " << " + U32(element * texelBits)) + ")";
            words[0] = words[0] == "0u" ? bits : words[0] + " | " + bits;
            continue;
        }
        const auto value = ConversionFormat(access.image).format != IrBufferFormat::Invalid ? ResultVector(texelAccess, UnpackTexel(texelAccess, read), IrTextureNumericClass::Uint, false, false) : ResultVector(texelAccess, read, access.image.numericClass, false, false);
        Line(state, "const uint4 " + texel + " = " + value + ";");
        for (std::uint32_t component = 0; component < components; ++component) words[element * components + component] = texel + "." + Swizzle(component);
    }
    if (packed) words[0] = "(" + words[0] + ")";
    for (auto& word : words) {
        if (word != "0u") word = "(" + inside + " ? " + word + " : 0u)";
    }
    Line(state, result + " = uint4(" + words[0] + ", " + words[1] + ", " + words[2] + ", " + words[3] + ");");
    Close(state);
}

std::string StoreTexel(Access& access, const std::string& data, bool integer) {
    State& state = access.state;
    const auto& mem = access.mem;
    const auto dmask = EffectiveDmask(mem);
    const auto sint = SintStorageFormat(access.image);
    if (sint.format != IrBufferFormat::Invalid) state.features |= Formatted;
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
        if (component < sint.componentCount && sint.componentBits[component] < 32u) {
            values[component] = std::string(mem.dataBits == 16u ? "aps5_fmt_encode16(" : "aps5_fmt_encode(") + raw + ", 2u, " + U32(sint.componentBits[component]) + ")";
        } else if (integer) {
            values[component] = raw;
        } else if (mem.dataBits == 16u) {
            state.features |= Half;
            values[component] = "aps5_f16_to_f32(" + raw + ")";
        } else {
            values[component] = "as_type<float>(" + raw + ")";
        }
    }
    const auto texel = std::string(integer ? "uint4(" : "float4(") + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ")";
    const auto info = ConversionFormat(access.image);
    if (info.format == IrBufferFormat::Invalid) return texel;
    if (NormalizedConversion(info)) RequireConvertedAccess(access, info);
    state.features |= Formatted;
    const auto source = Temp(state);
    Line(state, "const uint4 " + source + " = " + texel + ";");
    std::string packed;
    for (std::uint32_t component = 0; component < info.componentCount; ++component) {
        auto value = "aps5_fmt_encode(" + source + "." + Swizzle(component) + ", " + U32(TypeCode(info.type)) + ", " + U32(info.componentBits[component]) + ")";
        if (info.componentBitOffset[component] != 0u) value = "(" + value + " << " + U32(info.componentBitOffset[component]) + ")";
        packed += (packed.empty() ? "" : " | ") + value;
    }
    return "uint4(" + packed + ", 0u, 0u, 0u)";
}

std::string PackedStoreTexel(Access& access, const std::string& data) {
    const auto info = PackedTexelFormat(access);
    const auto numeric = access.image.numericClass;
    if (info.byteSize == 12u) Fail(access.inst, "stores packed texels of a format the hardware does not write");
    std::array<std::string, 4> words;
    for (std::uint32_t word = 0; word < 4u; ++word) words[word] = ((access.mem.dmask >> word) & 1u) != 0u ? data + "." + Swizzle(DmaskComponentIndex(access.mem.dmask, word)) : std::string("0u");
    std::array<std::string, 4> values;
    for (std::uint32_t component = 0; component < 4u; ++component) {
        if (component >= info.componentCount) {
            values[component] = numeric == IrTextureNumericClass::Float ? "0.0f" : "0u";
            continue;
        }
        const auto bits = info.componentBits[component];
        const auto offset = info.componentBitOffset[component];
        const bool exact = (info.type == SpirvFormatComponentType::Uint && numeric == IrTextureNumericClass::Uint) || (bits == 32u && ((info.type == SpirvFormatComponentType::Sint && numeric == IrTextureNumericClass::Uint) || (info.type == SpirvFormatComponentType::Float && numeric == IrTextureNumericClass::Float)));
        if (!exact || offset % 32u + bits > 32u) throw MslUnsupported("MSL backend: packed texels of a format whose bits are not reproducible through the view are not stored");
        auto value = words[offset / 32u];
        if (offset % 32u != 0u) value = "(" + value + " >> " + U32(offset % 32u) + ")";
        if (bits < 32u) value = "(" + value + " & " + U32(LowMask(bits)) + ")";
        values[component] = numeric == IrTextureNumericClass::Float ? "as_type<float>(" + value + ")" : value;
    }
    return std::string(numeric == IrTextureNumericClass::Float ? "float4(" : "uint4(") + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ")";
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
    const auto texel = access.mem.imagePacked ? PackedStoreTexel(access, Typed(state, access.inst, 2u)) : StoreTexel(access, Typed(state, access.inst, 2u), integer);
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
        Line(state, "const " + TextureType(access.image, kind.scalar) + " " + view + " = " + TextureRef(state, access.image, element, access.table, kind.scalar) + ";");
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

void EmitOneDimensionalGather(Access& access, const SamplerParts& parts, const SampleLayout& layout, const std::string& coordinate, const std::string& result) {
    State& state = access.state;
    const auto& mem = access.mem;
    const auto shape = ShapeOf(access.image.dimension);
    if (HasFlag(mem, RdnaImageSampleFlagCompare) || !HasFlag(mem, RdnaImageSampleFlagLevelZero) || HasFlag(mem, RdnaImageSampleFlagGatherHorizontal)) throw MslUnsupported("MSL backend: this 1D gather variant is not emitted");
    if (access.image.numericClass != IrTextureNumericClass::Float || ConversionFormat(access.image).format != IrBufferFormat::Invalid) throw MslUnsupported("MSL backend: 1D gathers of integer or converted images are not emitted");
    const auto dmask = EffectiveDmask(mem);
    if (dmask != 1u && dmask != 2u && dmask != 4u && dmask != 8u) Fail(access.inst, "gathers a dmask that selects several components");
    const auto component = Swizzle(DmaskComponent(access.inst, dmask, 0u));
    const auto width = Temp(state);
    const auto left = Temp(state);
    const auto x = shape.arrayed ? coordinate + ".x" : coordinate;
    Line(state, "const float " + width + " = float(" + parts.texture + ".get_width(0));");
    std::string offset;
    if (layout.offset != NoComponent) offset = " + float(" + PackedOffset(access, layout, shape) + ")";
    Line(state, "const float " + left + " = floor(" + x + " * " + width + " - 0.5f)" + offset + ";");
    std::array<std::string, 2> values;
    for (std::uint32_t index = 0; index < 2u; ++index) {
        const auto position = "float2((" + left + (index == 0u ? " + 0.5f" : " + 1.5f") + ") / " + width + ", 0.5f)";
        const auto layer = shape.arrayed ? ", " + LayerPart(coordinate, shape, true) : std::string();
        values[index] = Temp(state);
        Line(state, "const float " + values[index] + " = " + parts.texture + ".sample(" + parts.sampler + ", " + position + layer + ", level(0.0f))." + component + ";");
    }
    Line(state, result + " = " + ResultVector(access, "float4(" + values[0] + ", " + values[1] + ", " + values[1] + ", " + values[0] + ")", IrTextureNumericClass::Float, false, true) + ";");
}

void EmitPackedHorizontalGather(Access& access, const std::string& texture, const std::string& result) {
    State& state = access.state;
    const auto& mem = access.mem;
    const auto dimension = access.image.dimension;
    if (dimension != RdnaImageDimension::Dim1D && dimension != RdnaImageDimension::Dim2D) throw MslUnsupported("MSL backend: a packed horizontal gather of an image dimension whose row and slice rules are not measured is not emitted");
    if (LayoutOf(mem).clamp != NoComponent || HasFlag(mem, RdnaImageSampleFlagCompare)) throw MslUnsupported("MSL backend: a packed horizontal gather with depth comparison or an LOD clamp is not emitted");
    if (ShapeOf(mem.imageDimension).coordinates < ShapeOf(dimension).coordinates) Fail(access.inst, "has an image address with too few coordinate components");
    const auto info = PackedTexelFormat(access);
    const auto texelBits = TexelBits(info);
    if (texelBits > 32u) {
        Line(state, result + " = uint4(0u);");
        return;
    }
    const bool plane = dimension == RdnaImageDimension::Dim2D;
    Open(state, "{");
    const auto width = Temp(state);
    const auto anchor = Temp(state);
    Line(state, "const int " + width + " = int(" + texture + ".get_width(0));");
    Line(state, "const int " + anchor + " = int(floor(" + AddressF32(access, 0u) + " * float(" + width + ") - 0.5f));");
    std::string row = "0";
    std::string rowInside;
    if (plane) {
        const auto height = Temp(state);
        const auto raw = Temp(state);
        Line(state, "const int " + height + " = int(" + texture + ".get_height(0));");
        Line(state, "const int " + raw + " = int(floor(" + AddressF32(access, 1u) + " * float(" + height + ")));");
        rowInside = raw + " < " + height;
        row = "clamp(" + raw + ", 0, " + height + " - 1)";
    }
    std::array<std::string, 4> stream{"0u", "0u", "0u", "0u"};
    for (std::uint32_t element = 0; element < 4u; ++element) {
        const auto column = "clamp(" + anchor + " + " + std::to_string(static_cast<std::int32_t>(element) - 1) + ", 0, " + width + " - 1)";
        const auto bits = Temp(state);
        Line(state, "const uint " + bits + " = " + PackedTexel(access, texture + ".read(uint2(uint(" + column + "), uint(" + row + ")), 0u)") + ".x;");
        const auto position = element * texelBits;
        const auto word = position / 32u;
        const auto shift = position % 32u;
        const auto add = [&](std::uint32_t index, const std::string& value) { stream[index] = stream[index] == "0u" ? value : stream[index] + " | " + value; };
        add(word, shift == 0u ? bits : "(" + bits + " << " + U32(shift) + ")");
        if (shift != 0u && shift + texelBits > 32u) add(word + 1u, "(" + bits + " >> " + U32(32u - shift) + ")");
    }
    std::array<std::string, 4> selected{"0u", "0u", "0u", "0u"};
    std::uint32_t next = 0;
    for (std::uint32_t bit = 0; bit < 4u; ++bit) {
        if (((mem.dmask >> bit) & 1u) == 0u) continue;
        selected[next++] = rowInside.empty() ? "(" + stream[bit] + ")" : "(" + rowInside + " ? (" + stream[bit] + ") : 0u)";
    }
    Line(state, result + " = uint4(" + selected[0] + ", " + selected[1] + ", " + selected[2] + ", " + selected[3] + ");");
    Close(state);
}

void EmitSampling(Access& access, const SamplerParts& parts, const std::string& result) {
    State& state = access.state;
    const auto& mem = access.mem;
    const auto& image = access.image;
    const auto shape = ShapeOf(image.dimension);
    const bool gather = access.inst.Opcode() == IrOpcode::ImageGatherRaw;
    if (shape.multisampled) throw MslUnsupported("MSL backend: multisampled images cannot be sampled");
    if (image.numericClass != IrTextureNumericClass::Float && !image.depthCompare && !gather) throw MslUnsupported("MSL backend: sampling integer images has no Metal sample function");
    const auto conversion = ConversionFormat(image);
    if (conversion.format != IrBufferFormat::Invalid && !gather) throw MslUnsupported("MSL backend: sampling a converted image needs filtering in the shader and is not emitted");
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
        if (image.dimension == RdnaImageDimension::Dim3D) throw MslUnsupported("MSL backend: gathers need a 1D, 2D or array image");
        if (shape.spatial == 1u) {
            EmitOneDimensionalGather(access, parts, layout, coordinate, result);
            return;
        }
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
            text += std::string(", component::") + std::string(Components[conversion.format != IrBufferFormat::Invalid ? 0u : DmaskComponent(access.inst, dmask, 0u)]);
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
        if (gather && conversion.format != IrBufferFormat::Invalid) Line(state, result + " = " + ResultVector(access, UnpackGather(access, expression), IrTextureNumericClass::Uint, false, true) + ";");
        else Line(state, result + " = " + ResultVector(access, expression, image.numericClass, dref && !gather, gather) + ";");
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

struct TableWord {
    std::string code;
    std::string address;
    std::string poisoned;
};

std::string MapAt(const std::string& index) {
    return "aps5_table_map[" + index + "]";
}

TableWord EmitTableWord(State& state, const IrValue& inst, std::uint32_t source, std::uint32_t table, std::size_t argument) {
    const IrValue* handle = inst.Argument(argument)->Resolve();
    const auto& sources = state.program.Resources().descriptorSources;
    if (handle->ArgumentCount() == 0u || source >= sources.size() || !sources[source].tableColumn.has_value()) Fail(inst, "has an image table handle without its column");
    const auto& column = *sources[source].tableColumn;
    if (column.stride == 0u) Fail(inst, "has an image table column without a stride");
    const auto header = ImageTableAbi::TableHeader(table);
    const auto field = [&](std::uint32_t index) { return MapAt(U32(header + index)); };
    const auto name = Temp(state);
    Line(state, "const uint " + name + "_offset = " + Typed(state, *handle, 0u) + ";");
    Line(state, "const uint " + name + "_relative = " + name + "_offset - " + U32(column.addend) + ";");
    Line(state, "const uint " + name + "_record = " + name + "_relative / " + U32(column.stride) + ";");
    Line(state, "const bool " + name + "_mapped = " + name + "_relative - " + name + "_record * " + U32(column.stride) + " == 0u && " + name + "_record < " + field(ImageTableAbi::TableKeyCount) + ";");
    Line(state, "const uint " + name + "_entry = " + MapAt(name + "_mapped ? " + field(ImageTableAbi::TableMapStart) + " + " + name + "_record : 0u") + ";");
    Line(state, "const uint " + name + "_fault = " + field(ImageTableAbi::TableFault) + ";");
    Line(state, "const ulong " + name + "_base = ulong(" + field(ImageTableAbi::TableBaseLow) + ") | (ulong(" + field(ImageTableAbi::TableBaseHigh) + ") << 32u);");
    const auto outsideSnapshot = U32(ImageTableAbi::PoisonCode(ImageTableAbi::OutsideSnapshotPoison));
    TableWord word{name + "_code", name + "_address", name + "_poisoned"};
    if (column.address) {
        const auto immediate = column.offset & ~3u;
        Line(state, "const uint " + word.code + " = " + name + "_fault != 0u ? " + name + "_fault : (" + name + "_mapped ? " + name + "_entry : " + outsideSnapshot + ");");
        Line(state, "const ulong " + word.address + " = " + name + "_base + ulong(" + name + "_offset & 0xfffffffcu) + ulong(long(int(" + U32(immediate) + ")));");
    } else {
        Line(state, "const ulong " + name + "_sum = ulong(" + name + "_offset) + ulong(" + U32(column.offset) + ");");
        Line(state, "const uint " + name + "_byte = uint(" + name + "_sum) & 0xfffffffcu;");
        Line(state, "const bool " + name + "_inside = (" + name + "_sum >> 32u) == 0ul && (" + field(ImageTableAbi::TableSizeHigh) + " != 0u || " + name + "_byte < (" + field(ImageTableAbi::TableSizeLow) + " & 0xfffffffcu));");
        Line(state, "const uint " + word.code + " = " + name + "_fault != 0u ? " + name + "_fault : (" + name + "_mapped ? " + name + "_entry : (" + name + "_inside ? " + outsideSnapshot + " : " + field(ImageTableAbi::TableOutsideCode) + "));");
        Line(state, "const ulong " + word.address + " = " + name + "_base + ulong(" + name + "_byte);");
    }
    Line(state, "const bool " + word.poisoned + " = (" + word.code + " & " + U32(ImageTableAbi::PoisonFlag) + ") != 0u;");
    return word;
}

void EmitTableFault(State& state, const IrValue& inst, const TableWord& word, const std::string& poison, const std::string& condition) {
    Line(state, "if (" + condition + ") aps5_table_fault(aps5_c, " + word.address + ", " + poison + ", " + U32(inst.Flags<MemoryFlags>().pc) + ");");
}

struct TableResult {
    std::string valid;
    std::string flags;
    bool unsupported = false;
};

TableResult EmitTableAccess(State& state, const IrValue& inst, const MemoryInfo& mem, const ImageResource& base, bool tableSampler, const std::string& exec, TableAccess& table) {
    state.features |= Tables | Fault;
    const auto flagged = [](const std::string& value) { return "(" + value + " & " + U32(ImageTableAbi::PoisonFlag) + ") != 0u"; };
    TableResult result{"true", {}, false};
    std::string live = "true";
    if (base.table != NoTable) {
        const auto word = EmitTableWord(state, inst, base.source, base.table, 0u);
        EmitTableFault(state, inst, word, word.code, exec + " && " + word.poisoned);
        if (base.tableOperation == TableOperation::Unsupported || state.program.Info().runtimeImageModes.at(mem.resource).empty()) {
            result.unsupported = true;
            return result;
        }
        const auto name = Temp(state);
        Line(state, "const bool " + name + "_selected = " + word.code + " != " + U32(ImageTableAbi::NullCode) + " && !" + word.poisoned + ";");
        Line(state, "const uint " + name + "_record = " + name + "_selected ? " + word.code + " : 0u;");
        Line(state, "const uint " + name + "_flags = " + MapAt(name + "_record + " + U32(ImageTableAbi::ImageRecordFlags)) + ";");
        Line(state, "const uint " + name + "_dword3 = " + MapAt(name + "_record + " + U32(ImageTableAbi::ImageRecordDword3)) + ";");
        Line(state, "const uint " + name + "_element = " + MapAt(name + "_record + " + U32(ImageTableAbi::ImageRecordElement)) + ";");
        Line(state, "const bool " + name + "_rejected = " + name + "_selected && " + flagged(name + "_flags") + ";");
        EmitTableFault(state, inst, word, name + "_flags", exec + " && " + name + "_rejected");
        Line(state, "const bool " + name + "_bounded = " + name + "_element < " + MapAt(U32(ImageTableAbi::HeaderElementCount)) + ";");
        EmitTableFault(state, inst, word, U32(ImageTableAbi::OutsideSnapshotPoison), exec + " && " + name + "_selected && !" + name + "_rejected && !" + name + "_bounded");
        Line(state, "const bool " + name + "_checked = " + name + "_selected && !" + name + "_rejected && " + name + "_bounded;");
        Line(state, "const uint " + name + "_driver = " + MapAt(name + "_checked ? " + MapAt(U32(ImageTableAbi::HeaderElementFaults)) + " + " + name + "_element : 0u") + ";");
        Line(state, "const bool " + name + "_dropped = " + name + "_checked && " + flagged(name + "_driver") + ";");
        EmitTableFault(state, inst, word, name + "_driver", exec + " && " + name + "_dropped");
        result.valid = "(" + name + "_checked && !" + name + "_dropped)";
        result.flags = name + "_flags";
        live = name + "_selected";
        table.element = MapAt(U32(ImageTableAbi::HeaderElementBase)) + " + " + name + "_element";
        table.dword3 = name + "_dword3";
    }
    if (tableSampler) {
        const auto& sampler = state.program.Info().samplers.at(mem.sampler);
        const auto word = EmitTableWord(state, inst, sampler.source, sampler.table, 1u);
        const auto guard = exec + " && " + live;
        EmitTableFault(state, inst, word, word.code, guard + " && " + word.poisoned);
        const auto name = Temp(state);
        Line(state, "const uint " + name + "_record = !" + word.poisoned + " ? " + word.code + " : 0u;");
        Line(state, "const uint " + name + "_flags = " + MapAt(name + "_record + " + U32(ImageTableAbi::SamplerRecordFlags)) + ";");
        Line(state, "const uint " + name + "_element = " + MapAt(name + "_record + " + U32(ImageTableAbi::SamplerRecordElement)) + ";");
        Line(state, "const bool " + name + "_rejected = !" + word.poisoned + " && " + flagged(name + "_flags") + ";");
        EmitTableFault(state, inst, word, name + "_flags", guard + " && " + name + "_rejected");
        Line(state, "const bool " + name + "_bounded = " + name + "_element + " + U32(ImageTableAbi::SamplerEntryElements) + " <= " + MapAt(U32(ImageTableAbi::HeaderSamplerCount)) + ";");
        EmitTableFault(state, inst, word, U32(ImageTableAbi::OutsideSnapshotPoison), guard + " && !" + word.poisoned + " && !" + name + "_rejected && !" + name + "_bounded");
        Line(state, "const bool " + name + "_checked = !" + word.poisoned + " && !" + name + "_rejected && " + name + "_bounded;");
        Line(state, "const uint " + name + "_driver = " + MapAt(name + "_checked ? " + MapAt(U32(ImageTableAbi::HeaderSamplerFaults)) + " + (" + name + "_element >> 2u) : 0u") + ";");
        Line(state, "const bool " + name + "_dropped = " + name + "_checked && " + flagged(name + "_driver") + ";");
        EmitTableFault(state, inst, word, name + "_driver", guard + " && " + name + "_dropped");
        result.valid = "(" + result.valid + " && " + name + "_checked && !" + name + "_dropped)";
        table.sampler = name + "_element";
    }
    return result;
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
    const bool tableSampler = info.needsSampler && state.program.Info().samplers.at(mem.sampler).table != NoTable;
    const bool tables = base.table != NoTable || tableSampler;
    if (base.srgbDecode) throw MslUnsupported("MSL backend: " + std::string(name) + " decodes an sRGB image in the shader, which is not emitted");
    if (base.constantSwizzle) throw MslUnsupported("MSL backend: " + std::string(name) + " samples an image with a constant channel swizzle, which is not emitted");
    if (mem.dataBits == 16u && base.depthBits) throw MslUnsupported("MSL backend: " + std::string(name) + " reads depth bits as 16-bit results, which is not emitted");
    const IrValue* resource = inst.Argument(0)->Resolve();
    if (resource->Opcode() != IrOpcode::GetImageResource) Fail(inst, "has a typed image handle with the wrong producer");
    const IrValue* address = inst.Argument(info.needsSampler ? 2u : 1u)->Resolve();
    if (address->Opcode() != IrOpcode::MakeImageAddress) Fail(inst, "has an image address that was not made by MakeImageAddress");

    const auto modes = base.table != NoTable ? state.program.Info().runtimeImageModes.at(mem.resource) : ResourceMaterializer::RuntimeImageModes(base);
    const bool returns = opcode != IrOpcode::ImageWrite;
    std::string result;
    if (returns) {
        result = Temp(state);
        const auto type = TypeName(inst.Type());
        Line(state, type + " " + result + " = " + type + "(0);");
    }
    const IrValue* active = inst.Argument(inst.ArgumentCount() - 1u)->Resolve();
    const bool always = active->HasImmediate() && active->Type() == IrType::Bool && active->ImmediateBool();
    TableAccess table;
    std::string tableSelector;
    if (tables) {
        Open(state, "{");
        const auto access = EmitTableAccess(state, inst, mem, base, tableSampler, always ? std::string("true") : Typed(state, inst, inst.ArgumentCount() - 1u), table);
        if (access.unsupported) {
            Close(state);
            if (returns) Define(state, inst, result);
            return true;
        }
        if (base.table != NoTable) tableSelector = "(" + access.valid + " ? ((" + access.flags + " >> " + U32(ImageTableAbi::ModeShift) + ") & " + U32(ImageTableAbi::ModeMask) + ") : 0xffffffffu)";
        else tableSelector = "(" + access.valid + " ? " + Constant(state, PipelineSpecialization::ImageBase + mem.resource * PipelineSpecialization::ImageWords, 0u) + " : 0xffffffffu)";
    }
    Open(state, always ? std::string("{") : "if (" + Typed(state, inst, inst.ArgumentCount() - 1u) + ") {");

    const auto emitMode = [&](const ImageResource& mode, std::uint32_t element, const std::function<void(Access&, const std::string&)>& body) {
        Access access{state, inst, mem, mode, *address, element, tables ? &table : nullptr};
        const auto texture = Temp(state);
        Line(state, "const " + TextureType(mode) + " " + texture + " = " + TextureRef(state, mode, element, access.table) + ";");
        body(access, texture);
    };
    const auto emitBody = [&](const ImageResource& mode) {
        if (mode.depthBits) throw MslUnsupported("MSL backend: " + std::string(name) + " reads the bits of a depth plane, which is not emitted");
        if (mode.packedFormat == IrBufferFormat::Fmask8_S2_F1) throw MslUnsupported("MSL backend: " + std::string(name) + " reads FMASK, which is not emitted");
        if (mode.srgbDecode) throw MslUnsupported("MSL backend: " + std::string(name) + " decodes an sRGB image in the shader, which is not emitted");
        if (mode.table != NoTable && ShapeOf(mode.dimension).multisampled) throw MslUnsupported("MSL backend: " + std::string(name) + " reads a multisampled image through an image table, which is not emitted");
        const auto kind = DescriptorBindingForImage(mode);
        const auto element = mode.table != NoTable ? 0u : ElementOf(state, kind, mem.resource);
        const auto sampled = [&](Access& access, const std::string& texture) {
            if (opcode == IrOpcode::ImageQueryDimensions) EmitQueryDimensions(access, texture, result);
            else if (opcode == IrOpcode::ImageRead && mem.imageByElements != 0u) EmitByRead(access, texture, result);
            else if (opcode == IrOpcode::ImageRead) EmitRead(access, texture, result);
            else if (mem.imagePacked) EmitPackedHorizontalGather(access, texture, result);
            else if (opcode == IrOpcode::ImageQueryLod) EmitQueryLod(access, {SamplerRef(state, access.image, mem.sampler, access.table), texture}, result);
            else EmitSampling(access, {SamplerRef(state, access.image, mem.sampler, access.table), texture}, result);
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
            Access probe{state, inst, mem, mode, *address, element, tables ? &table : nullptr};
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

    if (modes.empty()) throw MslUnsupported("MSL backend: " + std::string(name) + " has no runtime image mode");
    if (modes.size() == 1u && !tables) {
        emitBody(modes.front());
    } else {
        const auto selector = tables ? tableSelector : Constant(state, PipelineSpecialization::ImageBase + mem.resource * PipelineSpecialization::ImageWords, 0u);
        Open(state, "switch (" + selector + ") {");
        std::optional<std::string> firstError;
        std::uint32_t emitted = 0;
        for (std::uint32_t index = 0; index < modes.size(); ++index) {
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
    if (tables) Close(state);
    if (returns) Define(state, inst, result);
    return true;
}

}
