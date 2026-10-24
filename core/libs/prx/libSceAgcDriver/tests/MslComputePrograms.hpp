#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLCOMPUTEPROGRAMS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLCOMPUTEPROGRAMS_HPP

#include "Recompiler.hpp"
#include "MslBackend/MslEmitter.hpp"
#include <cstdint>
#include <string_view>
#include <vector>

namespace MslComputePrograms {

inline constexpr std::uint32_t Threads = 64u;
inline constexpr std::uint32_t WideThreads = 128u;

inline constexpr std::uint32_t HostThreads(std::uint32_t waveSize, std::uint32_t threads) {
    return waveSize == 64u ? (threads + 63u) / 64u * 32u : threads;
}
inline constexpr std::uint32_t BufferGuestBase = 0x10000000u;
inline constexpr std::uint64_t BdaGuestBase = 0x120000000ull;
inline constexpr std::uint32_t BufferStore = 0xe0702000u;
inline constexpr std::uint32_t BufferStoreOperands = 0x80000100u;
inline constexpr std::uint32_t EndProgram = 0xbf810000u;

enum class Expect { Plus5, Swapped, FirstLanePlus, LowHalfSeven, Half, Times4, CounterAt0, QuadSwap, RowShift, Mirror, SwapHalf, ReverseHalf, RotateHalf, ReadLane5, Ballot32, Ballot64, ReadLane37, FirstUpperPlus, Carry64, MaxPair64, OrPair64, AllOne };

inline constexpr std::uint32_t Wave32 = 1u;
inline constexpr std::uint32_t Wave64 = 2u;
inline constexpr std::uint32_t BothWaves = Wave32 | Wave64;

struct Program {
    std::string_view name;
    std::vector<std::uint32_t> code;
    std::string_view marker;
    Expect expect;
    bool bda = false;
    std::uint32_t waves = Wave32;
    std::uint32_t threads = Threads;
};

inline std::uint32_t Expected(Expect expect, std::uint32_t lane) {
    switch (expect) {
    case Expect::Plus5: return lane + 5u;
    case Expect::Swapped: return lane ^ 1u;
    case Expect::FirstLanePlus: return (lane & ~31u) + lane;
    case Expect::LowHalfSeven: return lane < 16u ? 7u : 3u;
    case Expect::Half: return lane / 2u;
    case Expect::Times4: return lane * 4u;
    case Expect::CounterAt0: return lane == 0u ? Threads : 0u;
    case Expect::QuadSwap: return lane ^ 1u;
    case Expect::RowShift: return (lane & 15u) == 0u ? 100u : lane - 1u;
    case Expect::Mirror: return lane ^ 15u;
    case Expect::SwapHalf: return lane ^ 16u;
    case Expect::ReverseHalf: return (lane & ~31u) | (31u - (lane & 31u));
    case Expect::RotateHalf: return (lane & ~31u) | ((lane + 31u) & 31u);
    case Expect::ReadLane5: return (lane & ~31u) + 5u;
    case Expect::Ballot32: return lane < 32u ? 0xffe00000u : 0xffffffffu;
    case Expect::Ballot64: return lane < 64u ? 0x001fffffu : 0u;
    case Expect::ReadLane37: return (lane & ~63u) + 37u;
    case Expect::FirstUpperPlus: return lane < 64u ? (lane > 32u ? 33u + lane : 0u) : (lane & ~63u) + lane;
    case Expect::Carry64: return lane == 1u ? 1u : 0u;
    case Expect::MaxPair64: return lane == 0u ? Threads - 1u : (lane == 1u ? 1u : 0u);
    case Expect::OrPair64: return lane < 2u ? 0xffffffffu : 0u;
    case Expect::AllOne: return 1u;
    }
    return 0u;
}

inline std::vector<Program> Programs() {
    return {
        {"vector add", {0x4a020085u, BufferStore, BufferStoreOperands, EndProgram}, "aps5_buffer_store", Expect::Plus5},
        {"lds exchange", {0x34040082u, 0xd8340000u, 0x00000002u, 0xbf8cc07fu, 0xbf8a0000u, 0x3a060484u, 0xd8d80000u, 0x01000003u, 0xbf8cc07fu, BufferStore, BufferStoreOperands, EndProgram}, "threadgroup uint aps5_lds", Expect::Swapped},
        {"readfirstlane", {0x7e080500u, 0x4a020004u, BufferStore, BufferStoreOperands, EndProgram}, "simd_broadcast_first", Expect::FirstLanePlus},
        {"exec mask select", {0x7e020283u, 0x7d880090u, 0xbe883c6au, 0x7e020287u, 0x887e087eu, BufferStore, BufferStoreOperands, EndProgram}, "7u : 3u", Expect::LowHalfSeven},
        {"execz branch", {0x7e020283u, 0x7d880090u, 0xbe883c6au, 0xbf880001u, 0x7e020287u, 0x887e087eu, BufferStore, BufferStoreOperands, EndProgram}, "simd_any", Expect::LowHalfSeven},
        {"float convert", {0x7e020d00u, 0x100202f0u, 0x7e020f01u, BufferStore, BufferStoreOperands, EndProgram}, "float(", Expect::Half},
        {"scalar loop", {0xbe880380u, 0x7e020280u, 0x4a020300u, 0x80088108u, 0xbf0a8408u, 0xbf85fffcu, BufferStore, BufferStoreOperands, EndProgram}, "while (true)", Expect::Times4},
        {"buffer atomic", {0x7e020281u, 0xe0c80000u, 0x80000100u, EndProgram}, "aps5_atomic(", Expect::CounterAt0},
        {"global store", {0x34040082u, 0x4a020085u, 0xdc708000u, 0x00040102u, EndProgram}, "aps5_bda_store", Expect::Plus5, true},
        {"dpp quad swap", {0x7e0202fau, 0xff00b100u, BufferStore, BufferStoreOperands, EndProgram}, "simd_shuffle_xor", Expect::QuadSwap, false, BothWaves, WideThreads},
        {"dpp row shift", {0x7e0202ffu, 0x00000064u, 0x7e0202fau, 0xff011100u, BufferStore, BufferStoreOperands, EndProgram}, "simd_shuffle_up", Expect::RowShift, false, BothWaves, WideThreads},
        {"dpp row mirror", {0x7e0202fau, 0xff014000u, BufferStore, BufferStoreOperands, EndProgram}, "simd_shuffle_xor", Expect::Mirror, false, BothWaves, WideThreads},
        {"ds swizzle", {0xd8d4401fu, 0x01000000u, 0xbf8cc07fu, BufferStore, BufferStoreOperands, EndProgram}, "simd_shuffle_xor", Expect::SwapHalf, false, BothWaves, WideThreads},
        {"ds bpermute", {0x3a04009fu, 0x34040482u, 0xdacc0000u, 0x01000002u, 0xbf8cc07fu, BufferStore, BufferStoreOperands, EndProgram}, "aps5_ballot32", Expect::ReverseHalf, false, BothWaves, WideThreads},
        {"ds permute", {0x4a040081u, 0x3604049fu, 0x34040482u, 0xdac80000u, 0x01000002u, 0xbf8cc07fu, BufferStore, BufferStoreOperands, EndProgram}, "aps5_permute", Expect::RotateHalf, false, BothWaves, WideThreads},
        {"permlane16", {0xbe8403ffu, 0x89abcdefu, 0xbe8503ffu, 0x01234567u, 0xd7770001u, 0x00140900u, BufferStore, BufferStoreOperands, EndProgram}, "simd_shuffle", Expect::Mirror, false, BothWaves, WideThreads},
        {"readlane", {0xd7600004u, 0x00010b00u, 0x7e020204u, BufferStore, BufferStoreOperands, EndProgram}, "simd_shuffle(", Expect::ReadLane5, false, Wave32, WideThreads},
        {"ballot", {0xd4c10008u, 0x00020094u, 0x7e020208u, BufferStore, BufferStoreOperands, EndProgram}, "aps5_ballot32", Expect::Ballot32, false, Wave32, WideThreads},
        {"wave64 ballot", {0xd4c10008u, 0x00020094u, 0x7e020209u, 0x3a020208u, BufferStore, BufferStoreOperands, EndProgram}, "aps5_ballot32", Expect::Ballot64, false, Wave64, WideThreads},
        {"wave64 readlane", {0xd7600004u, 0x00014b00u, 0x7e020204u, BufferStore, BufferStoreOperands, EndProgram}, ">= 32u ? ", Expect::ReadLane37, false, Wave64, WideThreads},
        {"wave64 readfirstlane", {0xd4c10008u, 0x000200a0u, 0xbefe0408u, 0x7e080500u, 0x4a020004u, BufferStore, BufferStoreOperands, EndProgram}, "ctz(", Expect::FirstUpperPlus, false, Wave64, WideThreads},
        {"wave64 execz branch", {0x7e020283u, 0xd4c40008u, 0x00020090u, 0xbe8a2408u, 0xbf880001u, 0x7e020287u, 0xbefe040au, BufferStore, BufferStoreOperands, EndProgram}, "simd_any", Expect::LowHalfSeven, false, Wave64, WideThreads},
        {"buffer atomic add64", {0x7e0202ffu, 0x04000000u, 0x7e040280u, 0xe1480000u, 0x80000100u, EndProgram}, "aps5_atomic64(", Expect::Carry64, false, BothWaves},
        {"buffer atomic umax64", {0x7e040281u, 0x7e020300u, 0xe1600000u, 0x80000100u, EndProgram}, "aps5_atomic64(", Expect::MaxPair64, false, BothWaves},
        {"buffer atomic or64", {0x7e060281u, 0x34020700u, 0x7e040301u, 0xe1680000u, 0x80000100u, EndProgram}, "aps5_atomic64(", Expect::OrPair64, false, BothWaves},
        {"global atomic add64", {0x7e0602ffu, 0x04000000u, 0x7e080280u, 0x7e040280u, 0xdd488000u, 0x00040302u, EndProgram}, "aps5_bda_atomic8", Expect::Carry64, true, BothWaves},
        {"lds atomic add64", {0x7e0602ffu, 0x04000000u, 0x7e080280u, 0x7e0a0280u, 0x7e0c0280u, 0x7e0e0280u, 0xd9340000u, 0x00000605u, 0xbf8cc07fu, 0xbf8a0000u, 0xd9000000u, 0x00000305u, 0xbf8cc07fu, 0xbf8a0000u, 0xd8d80004u, 0x01000005u, 0xbf8cc07fu, 0x34040082u, BufferStore, BufferStoreOperands, EndProgram}, "aps5_atomic64_lds", Expect::AllOne, false, BothWaves},
    };
}

inline std::vector<std::uint32_t> UserData(std::uint32_t threads = Threads) {
    return {BufferGuestBase, 0x00040000u, threads, 0x00027facu, static_cast<std::uint32_t>(BdaGuestBase), static_cast<std::uint32_t>(BdaGuestBase >> 32u)};
}

inline ShaderRecompiler::RecompileRequest Request(const Program& program, std::uint64_t address, const std::vector<std::uint32_t>& userData, std::uint32_t waveSize = 32u) {
    using namespace ShaderRecompiler;
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, address, program.code, 0, {}};
    request.context.waveSize = waveSize;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{program.threads, 1u, 1u}, program.threads, {false, false, false}, false, 1u};
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

}

#endif
