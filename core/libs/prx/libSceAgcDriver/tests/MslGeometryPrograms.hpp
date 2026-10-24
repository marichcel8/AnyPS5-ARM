#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLGEOMETRYPROGRAMS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLGEOMETRYPROGRAMS_HPP

#include "MeshGeometryFixture.hpp"
#include "Recompiler.hpp"
#include "MslBackend/MslEmitter.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace MslGeometryPrograms {

alignas(256) inline constexpr std::array<std::uint32_t, 29> LocalCode{
    0xd5480004, 0x02090502, 0xd5480005, 0x02090102, 0xd5480006, 0x02110902, 0xd5480007, 0x02111102,
    0x7e080d04, 0x7e0a0d05, 0x7e0c0d06, 0x7e0e0d07, 0x10100cff, 0x3f19999a, 0x061010ff, 0xbf666666,
    0x3e1008ff, 0x3e800000, 0xd54b0009, 0x03c5e105, 0x161406a0, 0xd9340000, 0x0000080a, 0xd8340008,
    0x0000070a, 0xd8340010, 0x0000070a, 0xbf8cc07f, 0xbefd2106,
};

alignas(256) inline constexpr std::array<std::uint32_t, 49> HullCode{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000060, 0xdbfc0000,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000030, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0x7da82881, 0xbe94047e, 0xb06a0060, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013, 0xd8dc1008,
    0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x34142484, 0xbefe0414,
    0xe0781000, 0x0403180a, 0x4a162481, 0x16161690, 0x4c1616ff, 0x00008000, 0xe0741000, 0x0202070b,
    0xbf810000,
};

alignas(256) inline constexpr std::array<std::uint32_t, 30> DomainCode{
    0xf8000941, 0x00000000, 0x34020e84, 0x16020283, 0xe0341000, 0x04030a01, 0xe0341010, 0x04030c01,
    0xe0341020, 0x04030e01, 0xbf8c3f70, 0x08040af2, 0x08040d02, 0x10201505, 0x3e201906, 0x3e201d02,
    0x10221705, 0x3e221b06, 0x3e221f02, 0xd5510012, 0x040a0d05, 0x102424ff, 0x40400000, 0x7e260280,
    0x7e2802f2, 0xf80008cf, 0x14131110, 0xf800020f, 0x14131214, 0xbf810000,
};

alignas(256) inline constexpr std::array<std::uint32_t, 41> QuadHullCode{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000080, 0xdbfc0000,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000040, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0x7da82881, 0xb06a0080, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013, 0xbf8cc07f, 0x7e300306,
    0x7e320306, 0x7e340306, 0x7e360306, 0x16142498, 0xe0781000, 0x0403180a, 0xe0741010, 0x0403180a,
    0xbf810000,
};

alignas(256) inline constexpr std::array<std::uint32_t, 38> QuadDomainCode{
    0xf8000941, 0x00000000, 0x34020e86, 0xe0341000, 0x04030a01, 0xe0341010, 0x04030c01, 0xe0341020,
    0x04030e01, 0xe0341030, 0x04031001, 0xbf8c3f70, 0x08040af2, 0x08060cf2, 0x10241502, 0x3e241905,
    0x10261702, 0x3e261b05, 0x10281d05, 0x3e282102, 0x102a1f05, 0x3e2a2302, 0x102c2503, 0x3e2c2906,
    0x102e2703, 0x3e2e2b06, 0xd5510018, 0x040a0d05, 0xd5510018, 0x04620718, 0x103030f4, 0x7e320280,
    0x7e3402f2, 0xf80008cf, 0x1a191716, 0xf800020f, 0x1a19181a, 0xbf810000,
};

alignas(256) inline constexpr std::array<std::uint32_t, 25> FastLaunchCode{
    0xd7650000, 0x000100c1, 0x7e000505, 0x93018300, 0x4a020001, 0xe0382000, 0x80020201, 0xe0382010,
    0x80020601, 0xb07c1003, 0xbf900009, 0xbf8c3f70, 0x7d880083, 0xbefe036a, 0xf80008cf, 0x05040302,
    0xf800020f, 0x09080706, 0x7d840080, 0xbefe036a, 0x7e1402ff, 0x00200400, 0xf8000941, 0x0000000a,
    0xbf810000,
};
alignas(256) inline constexpr auto Wave32GeometryCode = [] {
    auto code = AgcDriver::Test::MeshGeometryCode;
    code[10] = 0x04190a6bu;
    code[28] = 0x04250a02u;
    return code;
}();

alignas(256) inline constexpr std::array<std::uint32_t, 7> PixelCode{
    0xc8020002, 0xc8060102, 0xc80a0202, 0xc80e0302, 0xf800180f, 0x03020100, 0xbf810000,
};

inline constexpr std::array<std::uint32_t, 3> PatchLevels{1u, 0u, 5u};

constexpr std::uint16_t Corner(std::uint32_t patch, std::uint32_t x, std::uint32_t y) {
    return static_cast<std::uint16_t>((PatchLevels[patch] << 8u) | (patch << 4u) | (x << 2u) | y);
}

alignas(256) inline constexpr std::array<std::uint16_t, 9> CornerIndices{
    Corner(0, 0, 0), Corner(0, 2, 0), Corner(0, 1, 2),
    Corner(1, 0, 0), Corner(1, 2, 0), Corner(1, 1, 2),
    Corner(2, 0, 0), Corner(2, 2, 0), Corner(2, 1, 2),
};

inline constexpr ShaderRecompiler::TessellationConfiguration TrianglePatches{3u, 3u, 1u, 2u, 2u};
inline constexpr ShaderRecompiler::TessellationConfiguration QuadPatches{4u, 4u, 2u, 2u, 3u};
inline constexpr ShaderRecompiler::MeshConfiguration SmallSubgroup{4u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u};
inline constexpr ShaderRecompiler::MeshConfiguration FastLaunchSubgroup{1u, 1u, 1u, 3u, 1u, 32u, 128u, 0u, 4u};
alignas(256) inline constexpr std::array<std::uint16_t, 12> PatchIndices{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u};
inline const std::array<std::uint32_t, 16> HullUserData{0x1000u};
inline const std::array<std::uint32_t, 8> DomainUserData{};

inline ShaderRecompiler::MslTarget Target() {
    ShaderRecompiler::MslTarget msl{};
    msl.simdgroupSize = 32;
    msl.bdaAbiVersion = 1;
    msl.maxThreadgroupSize = {1024u, 1024u, 1024u};
    msl.maxThreadsPerThreadgroup = 1024;
    msl.maxThreadgroupMemoryBytes = 32768;
    msl.mesh = ShaderRecompiler::MeshTargetLimits{{1024u, 1u, 1u}, 1024u, 32768u, 256u, 512u, 128u, 32768u, 32u, 32u};
    msl.tessellation = ShaderRecompiler::TessellationTargetLimits{32u, 128u, 128u, 120u, 4096u, 128u, 128u};
    return msl;
}

inline ShaderRecompiler::RecompileRequest Tessellation(ShaderRecompiler::ShaderStage stage, std::span<const std::uint32_t> code, const ShaderRecompiler::TessellationConfiguration& configuration, std::span<const std::byte> memoryBytes) {
    using namespace ShaderRecompiler;
    static thread_local std::array<MemoryRegion, 1> memory{};
    memory[0] = MemoryRegion{reinterpret_cast<std::uintptr_t>(code.data()), memoryBytes};
    const bool hull = stage == ShaderStage::TessellationControl;
    RecompileRequest request{};
    request.shader = {stage, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}};
    request.context.waveSize = 64;
    request.context.userDataBaseRegister = hull ? 0u : 8u;
    request.context.userData = hull ? std::span<const std::uint32_t>(HullUserData) : stage == ShaderStage::TessellationEvaluation ? std::span<const std::uint32_t>(DomainUserData) : std::span<const std::uint32_t>();
    request.context.vertex = ShaderVertexStageInfo{};
    request.context.memory = memory;
    request.graphics = GraphicsCompileContext{hull ? 0u : 8u, {}, std::nullopt, configuration, {reinterpret_cast<std::uintptr_t>(PatchIndices.data()), configuration.inputControlPoints * 3u, 2, 1}};
    request.backend = ShaderBackendKind::Msl;
    request.msl = Target();
    request.target = MslFrontendTarget(request.msl);
    request.layout = {0, 0, 0, TessellationLevelPushOffsetBytes};
    request.useCache = false;
    return request;
}

template<std::size_t Size>
ShaderRecompiler::RecompileRequest Tessellation(ShaderRecompiler::ShaderStage stage, const std::array<std::uint32_t, Size>& code, const ShaderRecompiler::TessellationConfiguration& configuration) {
    return Tessellation(stage, std::span<const std::uint32_t>(code), configuration, std::as_bytes(std::span(code)));
}

inline ShaderRecompiler::RecompileRequest Mesh(std::span<const std::uint32_t> code, const ShaderRecompiler::MeshConfiguration& configuration, std::uint32_t waveSize) {
    using namespace ShaderRecompiler;
    static thread_local std::array<MemoryRegion, 1> memory{};
    static thread_local std::array<std::uint32_t, 12> userData{};
    userData = {};
    userData[8] = 0x10000u;
    userData[9] = 32u << 16u;
    userData[10] = 64u;
    userData[11] = 0x01016facu;
    memory[0] = MemoryRegion{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)};
    RecompileRequest request{};
    request.shader = {ShaderStage::Mesh, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}};
    request.context.waveSize = waveSize;
    request.context.userData = userData;
    request.context.vertex = ShaderVertexStageInfo{};
    request.context.memory = memory;
    request.graphics = GraphicsCompileContext{0, {}, configuration, std::nullopt, {0, 12, 0, 1}};
    request.backend = ShaderBackendKind::Msl;
    request.msl = Target();
    request.target = MslFrontendTarget(request.msl);
    request.layout = {0, 0, 0, MeshDrawPushOffsetBytes};
    request.useCache = false;
    return request;
}

inline ShaderRecompiler::RecompileRequest Pixel() {
    using namespace ShaderRecompiler;
    static const std::array<MemoryRegion, 1> memory{MemoryRegion{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}};
    RecompileRequest request{};
    request.shader = {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}};
    request.context.waveSize = 32;
    request.context.memory = memory;
    ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = 1;
    pixel.interpolatorSettings[0] = 0x400u;
    pixel.wave32 = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    request.context.pixel = pixel;
    request.backend = ShaderBackendKind::Msl;
    request.msl = Target();
    request.target = MslFrontendTarget(request.msl);
    request.layout = {0, 0, 0, 128};
    request.useCache = false;
    return request;
}

}

#endif
