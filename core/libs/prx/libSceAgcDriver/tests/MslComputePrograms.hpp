#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLCOMPUTEPROGRAMS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLCOMPUTEPROGRAMS_HPP

#include "Recompiler.hpp"
#include "MslBackend/MslEmitter.hpp"
#include <cstdint>
#include <string_view>
#include <vector>

namespace MslComputePrograms {

inline constexpr std::uint32_t Threads = 64u;
inline constexpr std::uint32_t BufferGuestBase = 0x10000000u;
inline constexpr std::uint64_t BdaGuestBase = 0x120000000ull;
inline constexpr std::uint32_t BufferStore = 0xe0702000u;
inline constexpr std::uint32_t BufferStoreOperands = 0x80000100u;
inline constexpr std::uint32_t EndProgram = 0xbf810000u;

enum class Expect { Plus5, Swapped, FirstLanePlus, LowHalfSeven, Half, Times4, CounterAt0 };

struct Program {
    std::string_view name;
    std::vector<std::uint32_t> code;
    std::string_view marker;
    Expect expect;
    bool bda = false;
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
    };
}

inline std::vector<std::uint32_t> UserData() {
    return {BufferGuestBase, 0x00040000u, Threads, 0x00027facu, static_cast<std::uint32_t>(BdaGuestBase), static_cast<std::uint32_t>(BdaGuestBase >> 32u)};
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

}

#endif
