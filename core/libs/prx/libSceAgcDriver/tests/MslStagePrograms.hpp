#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLSTAGEPROGRAMS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLSTAGEPROGRAMS_HPP

#include "Recompiler.hpp"
#include "MslBackend/MslEmitter.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace MslStagePrograms {

inline constexpr std::uint32_t TargetSize = 64u;
inline constexpr std::uint64_t FetchAddress = 0x50000u;
inline constexpr std::uint32_t Format32x4Float = 77u;

inline const std::vector<std::uint32_t>& GridVertex() {
    static const std::vector<std::uint32_t> code{0x36020a81u, 0x2c040a81u, 0x36040481u, 0x7e020d01u, 0x7e040d02u, 0x7e060280u, 0x7e0802f2u, 0x7e0c02f0u, 0x4a120a81u, 0x7e120d09u, 0xd541000au, 0x03cded01u, 0xd541000bu, 0x03cded02u, 0xf80008cfu, 0x04030b0au, 0xf800020fu, 0x04060201u, 0xf800021fu, 0x04030309u, 0xbf810000u};
    return code;
}

inline const std::vector<std::uint32_t>& FetchVertex() {
    static const std::vector<std::uint32_t> code{0xbe802102u, 0xf80008cfu, 0x07060504u, 0xbf810000u};
    return code;
}

inline const std::vector<std::uint32_t>& InterpolatedPixel() {
    static const std::vector<std::uint32_t> code{0xc8080000u, 0xc8090001u, 0xc80c0100u, 0xc80d0101u, 0xc8120202u, 0x7e0a02f2u, 0xf800180fu, 0x05040302u, 0xbf810000u};
    return code;
}

inline const std::vector<std::uint32_t>& KillDepthPixel() {
    static const std::vector<std::uint32_t> code{0xc8080000u, 0xc8090001u, 0xc80e0402u, 0x7c2804ffu, 0x3e800000u, 0x7e0802f0u, 0x7e0a0280u, 0x7e0c02f2u, 0xf8000081u, 0x00000004u, 0xf800180fu, 0x06050302u, 0xbf810000u};
    return code;
}

inline const std::vector<std::uint32_t>& ConstantPixel() {
    static const std::vector<std::uint32_t> code{0x7e0402f0u, 0x7e0602f2u, 0x7e080280u, 0xf800180fu, 0x03040203u, 0xbf810000u};
    return code;
}

inline ShaderRecompiler::RecompileRequest Base(ShaderRecompiler::ShaderStage stage, std::uint64_t address, const std::vector<std::uint32_t>& code) {
    using namespace ShaderRecompiler;
    RecompileRequest request{};
    request.shader = {stage, address, code, 0, {}};
    request.context.waveSize = 32;
    request.backend = ShaderBackendKind::Msl;
    request.msl.simdgroupSize = 32;
    request.msl.bdaAbiVersion = 1;
    request.msl.maxThreadgroupSize = {1024u, 1024u, 1024u};
    request.msl.maxThreadsPerThreadgroup = 1024;
    request.msl.maxThreadgroupMemoryBytes = 32768;
    request.target = MslFrontendTarget(request.msl);
    request.layout = {0, 0, 0, 128};
    request.useCache = false;
    return request;
}

inline ShaderRecompiler::RecompileRequest GridVertexRequest(std::uint64_t address) {
    auto request = Base(ShaderRecompiler::ShaderStage::Vertex, address, GridVertex());
    request.context.vertex = ShaderRecompiler::ShaderVertexStageInfo{};
    return request;
}

inline ShaderRecompiler::RecompileRequest FetchVertexRequest(std::uint64_t address) {
    using namespace ShaderRecompiler;
    static const std::array<std::uint32_t, 4> fetch{0xe00c2000u, 0x80020405u, 0xbf8c3f70u, 0xbe802000u};
    static const std::array<std::uint32_t, 12> userData{0u, 0u, static_cast<std::uint32_t>(FetchAddress), 0u, 0u, 0u, 0u, 0u, 0x10000u, 16u << 16u, 3u, (Format32x4Float << 12u) | 0xfacu};
    static const std::array<MemoryRegion, 1> memory{MemoryRegion{FetchAddress, std::as_bytes(std::span(fetch))}};
    auto request = Base(ShaderStage::Vertex, address, FetchVertex());
    ShaderVertexStageInfo vertex{};
    vertex.resourcesNum = 1;
    vertex.resources[0].fields = {0x10000u, 16u << 16u, 3u, (Format32x4Float << 12u) | 0xfacu};
    vertex.resourcesDst[0] = {4, 4, 0, 0u};
    vertex.fetchAttribReg = 8;
    vertex.fetchBufferReg = 8;
    vertex.fetchEmbedded = true;
    request.context.vertex = vertex;
    request.context.userData = userData;
    request.context.memory = memory;
    return request;
}

inline ShaderRecompiler::ShaderPixelStageInfo PixelInfo(std::uint32_t interpolators) {
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = interpolators;
    for (std::uint32_t index = 0; index < interpolators; ++index) pixel.interpolatorSettings[index] = index;
    pixel.inputAddr = 0x2;
    pixel.hasPerspectiveCenterVgpr = true;
    pixel.wave32 = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    return pixel;
}

inline ShaderRecompiler::RecompileRequest InterpolatedPixelRequest(std::uint64_t address) {
    auto request = Base(ShaderRecompiler::ShaderStage::Fragment, address, InterpolatedPixel());
    request.context.pixel = PixelInfo(1u);
    return request;
}

inline ShaderRecompiler::RecompileRequest KillDepthPixelRequest(std::uint64_t address) {
    auto request = Base(ShaderRecompiler::ShaderStage::Fragment, address, KillDepthPixel());
    auto pixel = PixelInfo(2u);
    pixel.interpolatorSettings[1] |= 0x400u;
    pixel.depthExportEnable = true;
    pixel.pixelKillEnable = true;
    request.context.pixel = pixel;
    return request;
}

inline ShaderRecompiler::RecompileRequest ConstantPixelRequest(std::uint64_t address) {
    auto request = Base(ShaderRecompiler::ShaderStage::Fragment, address, ConstantPixel());
    auto pixel = PixelInfo(0u);
    pixel.inputAddr = 0u;
    pixel.hasPerspectiveCenterVgpr = false;
    request.context.pixel = pixel;
    return request;
}

}

#endif
