#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLIMAGEPROGRAMS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLIMAGEPROGRAMS_HPP

#include "MslComputePrograms.hpp"
#include <array>
#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <vector>

namespace MslImagePrograms {

inline constexpr std::uint32_t Threads = MslComputePrograms::Threads;
inline constexpr std::uint32_t Width = 64u;
inline constexpr std::uint64_t TextureBase = 0x20000000ull;
inline constexpr std::uint32_t Format8888UNorm = 56;
inline constexpr std::uint32_t Format32UInt = 20;
inline constexpr std::uint32_t Format32Float = 22;
inline constexpr std::uint32_t Type2D = 9;
inline constexpr std::uint32_t WaitCnt = 0xbf8c3f70u;
inline constexpr std::uint32_t EndProgram = MslComputePrograms::EndProgram;

enum class Kind { Load, Sample, Store, AtomicAdd, ResInfo, CompileOnly };

struct Program {
    std::string_view name;
    std::vector<std::uint32_t> code;
    Kind kind;
    std::uint32_t format;
    std::vector<std::string_view> markers;
};

constexpr std::uint32_t MovConstant(std::uint32_t vgpr, std::uint32_t source) {
    return 0x7e000200u | (vgpr << 17u) | source;
}

constexpr std::uint32_t Mimg(std::uint32_t opcode, std::uint32_t dmask) {
    return 0xf0000000u | (opcode << 18u) | (dmask << 8u) | 8u;
}

constexpr std::uint32_t MimgOperands(std::uint32_t address, std::uint32_t data, std::uint32_t resource, std::uint32_t sampler) {
    return address | (data << 8u) | (resource << 16u) | (sampler << 21u);
}

inline constexpr std::uint32_t Zero = 0x80u;
inline constexpr std::uint32_t One = 0x81u;
inline constexpr std::uint32_t Three = 0x83u;
inline constexpr std::uint32_t Half = 0xf0u;
inline constexpr std::uint32_t FloatOne = 0xf2u;
inline constexpr std::uint32_t Tid = 0x100u;
inline constexpr std::uint32_t Literal = 0xffu;
inline constexpr std::uint32_t StoreFirstLane = 0x80000200u;

inline std::vector<std::uint32_t> Sampling(std::uint32_t opcode, std::uint32_t dmask, std::initializer_list<std::uint32_t> address) {
    std::vector<std::uint32_t> code;
    std::uint32_t vgpr = 8;
    for (const auto word : address) {
        if (word == 0x00000201u) {
            code.push_back(MovConstant(vgpr, Literal));
            code.push_back(word);
        } else {
            code.push_back(MovConstant(vgpr, word));
        }
        ++vgpr;
    }
    code.push_back(Mimg(opcode, dmask));
    code.push_back(MimgOperands(8u, 2u, 1u, 3u));
    code.push_back(WaitCnt);
    code.push_back(MslComputePrograms::BufferStore);
    code.push_back(StoreFirstLane);
    code.push_back(EndProgram);
    return code;
}

inline std::vector<Program> Programs() {
    constexpr std::uint32_t PointFive = Half;
    constexpr std::uint32_t Offset = 0x00000201u;
    return {
        {"image load",
         {MovConstant(1u, Zero), Mimg(0x00u, 0xfu), MimgOperands(0u, 2u, 1u, 0u), WaitCnt, MslComputePrograms::BufferStore, StoreFirstLane, EndProgram},
         Kind::Load, Format8888UNorm, {".read(", "access::sample"}},
        {"image sample",
         {0x7e040d00u, 0x060404f0u, 0x101004ffu, 0x3c800000u, MovConstant(9u, PointFive), Mimg(0x27u, 0xfu), MimgOperands(8u, 2u, 1u, 3u), WaitCnt, MslComputePrograms::BufferStore, StoreFirstLane, EndProgram},
         Kind::Sample, Format8888UNorm, {".sample(", "level("}},
        {"image store",
         {MovConstant(1u, Zero), MovConstant(2u, FloatOne), MovConstant(3u, Half), MovConstant(4u, Zero), MovConstant(5u, FloatOne), Mimg(0x08u, 0xfu), MimgOperands(0u, 2u, 1u, 0u), EndProgram},
         Kind::Store, Format8888UNorm, {".write(", "access::write"}},
        {"image atomic add",
         {0x36000083u, MovConstant(1u, Zero), MovConstant(2u, One), Mimg(0x11u, 0x1u), MimgOperands(0u, 2u, 1u, 0u), EndProgram},
         Kind::AtomicAdd, Format32UInt, {"atomic_fetch_add", "access::read_write"}},
        {"image resinfo",
         {MovConstant(6u, Tid), MovConstant(7u, Zero), Mimg(0x0eu, 0xfu), MimgOperands(7u, 2u, 1u, 0u), WaitCnt, MslComputePrograms::BufferStore, 0x80000206u, EndProgram},
         Kind::ResInfo, Format8888UNorm, {".get_width(", "get_num_mip_levels"}},
        {"sample lod", Sampling(0x24u, 0xfu, {PointFive, PointFive, Zero}), Kind::CompileOnly, Format8888UNorm, {"level("}},
        {"sample bias", Sampling(0x25u, 0xfu, {Zero, PointFive, PointFive}), Kind::CompileOnly, Format8888UNorm, {".sample("}},
        {"sample gradient", Sampling(0x22u, 0xfu, {Zero, Zero, Zero, Zero, PointFive, PointFive}), Kind::CompileOnly, Format8888UNorm, {"gradient2d("}},
        {"sample lod clamp", Sampling(0x21u, 0xfu, {PointFive, PointFive, Zero}), Kind::CompileOnly, Format8888UNorm, {"max(0.0f"}},
        {"sample offset", Sampling(0x37u, 0xfu, {Offset, PointFive, PointFive}), Kind::CompileOnly, Format8888UNorm, {"int2(1, 2)"}},
        {"sample compare", Sampling(0x2fu, 0x1u, {One, PointFive, PointFive}), Kind::CompileOnly, Format32Float, {"sample_compare(", "depth2d<float"}},
        {"gather", Sampling(0x47u, 0x1u, {PointFive, PointFive}), Kind::CompileOnly, Format8888UNorm, {".gather(", "component::x"}},
        {"gather compare", Sampling(0x4fu, 0x1u, {One, PointFive, PointFive}), Kind::CompileOnly, Format32Float, {"gather_compare("}},
        {"sample implicit", Sampling(0x20u, 0xfu, {PointFive, PointFive}), Kind::CompileOnly, Format8888UNorm, {"level(0.0f)"}},
    };
}

inline std::array<std::uint32_t, 8> Texture(std::uint32_t format, std::uint32_t width) {
    return {
        static_cast<std::uint32_t>(TextureBase >> 8u),
        static_cast<std::uint32_t>((TextureBase >> 40u) & 0xffu) | (format << 20u) | (((width - 1u) & 3u) << 30u),
        ((width - 1u) >> 2u),
        0xfacu | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

inline std::array<std::uint32_t, 4> Sampler() {
    return {0u, 0xfffu << 12u, 1u << 26u, 0u};
}

inline std::vector<std::uint32_t> UserData(std::uint32_t format) {
    auto words = MslComputePrograms::UserData();
    words.resize(16u, 0u);
    const auto texture = Texture(format, Width);
    const auto sampler = Sampler();
    for (std::size_t index = 0; index < texture.size(); ++index) words[4u + index] = texture[index];
    for (std::size_t index = 0; index < sampler.size(); ++index) words[12u + index] = sampler[index];
    return words;
}

inline ShaderRecompiler::RecompileRequest Request(const Program& program, std::uint64_t address, const std::vector<std::uint32_t>& userData) {
    using namespace ShaderRecompiler;
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, address, program.code, 0, {}};
    request.context.waveSize = 32;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{Threads, 1u, 1u}, Threads, {false, false, false}, false, 1u};
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

inline constexpr std::uint32_t StoredTexel = 0xff0080ffu;

inline float LoadedRed(std::uint32_t lane) {
    return static_cast<float>(lane) / 255.0f;
}

}

#endif
