#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLPRECISIONPROGRAMS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_MSLPRECISIONPROGRAMS_HPP

#include "MslComputePrograms.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace MslPrecisionPrograms {

using MslComputePrograms::BufferStore;
using MslComputePrograms::BufferStoreOperands;
using MslComputePrograms::EndProgram;
using MslComputePrograms::Threads;

inline constexpr std::uint32_t BufferLoad = 0xe0302000u;
inline constexpr std::uint32_t BufferLoadOperands = 0x80000100u;
inline constexpr std::uint32_t WaitLoads = 0xbf8c3f70u;

enum class Check { MaxOne, IsNan, Float64, Dot2, Timing };

struct Program {
    std::string_view name;
    std::vector<std::uint32_t> body;
    std::vector<std::string_view> present;
    std::vector<std::string_view> absent;
    Check check;
    bool relaxed = true;
};

inline MslComputePrograms::Program Compute(const Program& program) {
    std::vector<std::uint32_t> code{BufferLoad, BufferLoadOperands, WaitLoads};
    code.insert(code.end(), program.body.begin(), program.body.end());
    code.insert(code.end(), {BufferStore, BufferStoreOperands, EndProgram});
    return {program.name, std::move(code), {}, MslComputePrograms::Expect::Plus5};
}

inline std::vector<std::uint32_t> FmaLoop(std::uint32_t iterations) {
    return {0xb0080000u | iterations, 0x7e0402ffu, 0x3f7ff000u, 0x7e0602f0u, 0xd54b0001u, 0x040e0501u, 0x80888108u, 0xbf078008u, 0xbf85fffbu};
}

inline std::vector<Program> Programs() {
    const std::vector<std::uint32_t> isNan{0x7c1a0301u, 0xd5010001u, 0x01a90280u};
    const std::vector<std::uint32_t> maxOne{0xd5100001u, 0x0001e501u};
    auto observed = FmaLoop(0x1000u);
    observed.insert(observed.end(), maxOne.begin(), maxOne.end());
    return {
        {"nan max", maxOne, {"aps5_i_max("}, {"aps5_s_"}, Check::MaxOne},
        {"nan compare", isNan, {"aps5_i_une(", "math_mode(relaxed)"}, {"aps5_s_"}, Check::IsNan},
        {"fp64 fma sqrt", {0x7e042101u, 0xd54c0002u, 0x03ca0502u, 0x7e046902u, 0x7e021f02u}, {"aps5_f64_fma(", "aps5_f64_sqrt(", "aps5_f64_to_f32("}, {"double"}, Check::Float64},
        {"dot2", {0xcc134001u, 0x1a020301u}, {"aps5_dot2_f32_f16("}, {}, Check::Dot2},
        {"fma loop fast", FmaLoop(0x1000u), {"fma("}, {"#pragma", "aps5_i_", "aps5_s_"}, Check::Timing},
        {"fma loop observed", observed, {"aps5_i_fma(", "aps5_i_max("}, {"aps5_s_"}, Check::Timing},
        {"fma loop strict", FmaLoop(0x1000u), {"aps5_s_fma(", "math_mode(safe)"}, {"aps5_i_"}, Check::Timing, false},
    };
}

inline ShaderRecompiler::RecompileRequest Request(const Program& program, const MslComputePrograms::Program& compute, std::uint64_t address) {
    static const auto userData = MslComputePrograms::UserData();
    auto request = MslComputePrograms::Request(compute, address, userData);
    request.msl.relaxedFloat = program.relaxed;
    request.target = ShaderRecompiler::MslFrontendTarget(request.msl);
    return request;
}

inline std::uint32_t Bits(float value) { return std::bit_cast<std::uint32_t>(value); }

inline std::uint32_t FloatInput(std::uint32_t lane) {
    static constexpr std::uint32_t Special[] = {0x7fc00000u, 0xffc00001u, 0x7f800000u, 0xff800000u, 0x80000000u, 0x00000000u, 0x00000001u, 0x807fffffu, 0x00800000u, 0x7f7fffffu, 0xff7fffffu, 0x3f800000u, 0xbf800000u, 0x7fc12345u, 0x5f800000u, 0x1f800000u};
    if (lane < std::size(Special)) return Special[lane];
    std::uint32_t state = lane * 0x9e3779b9u;
    state ^= state >> 15u;
    state *= 0x2c1b3c6du;
    state ^= state >> 12u;
    return (state & 0x807fffffu) | ((state % 200u + 27u) << 23u);
}

inline std::uint32_t HalfInput(std::uint32_t lane) {
    static constexpr std::uint32_t Special[] = {0x7e003c00u, 0x7c003c00u, 0xfc007c00u, 0x80000000u, 0x00010001u, 0x7bff7bffu, 0xfbff7bffu, 0x3c00bc00u, 0x03ff0400u, 0x3555b555u};
    if (lane < std::size(Special)) return Special[lane];
    std::uint32_t state = lane * 0x85ebca6bu;
    state ^= state >> 13u;
    state *= 0xc2b2ae35u;
    state ^= state >> 16u;
    return state & 0xbfffbfffu;
}

inline std::uint32_t Input(Check check, std::uint32_t lane) {
    if (check == Check::Dot2) return HalfInput(lane);
    if (check == Check::Timing) return Bits(static_cast<float>(lane) * 0.25f);
    return FloatInput(lane);
}

inline double Half(std::uint32_t bits) {
    const auto exponent = static_cast<int>((bits >> 10u) & 0x1fu);
    const auto mantissa = static_cast<double>(bits & 0x3ffu);
    const double sign = (bits & 0x8000u) != 0u ? -1.0 : 1.0;
    if (exponent == 31) return mantissa != 0.0 ? std::nan("") : sign * INFINITY;
    if (exponent == 0) return sign * std::ldexp(mantissa, -24);
    return sign * std::ldexp(mantissa + 1024.0, exponent - 25);
}

inline bool Matches(Check check, std::uint32_t input, std::uint32_t output) {
    const float value = std::bit_cast<float>(input);
    const float result = std::bit_cast<float>(output);
    switch (check) {
    case Check::MaxOne: return output == Bits(std::isnan(value) ? 1.0f : std::fmax(value, 1.0f));
    case Check::IsNan: return output == (std::isnan(value) ? 1u : 0u);
    case Check::Float64: {
        if (std::isnan(value)) return std::isnan(result);
        const double wide = value;
        return output == Bits(static_cast<float>(std::sqrt(std::fma(wide, wide, 1.0))));
    }
    case Check::Dot2: {
        const double low = Half(input & 0xffffu);
        const double high = Half(input >> 16u);
        const double exact = low * low + high * high;
        if (std::isnan(exact)) return std::isnan(result);
        const float rounded = static_cast<float>(exact);
        if (std::fpclassify(rounded) == FP_SUBNORMAL || rounded == 0.0f) return result == rounded || std::fabs(result) <= std::ldexp(1.0f, -126);
        return output == Bits(rounded);
    }
    case Check::Timing: return true;
    }
    return false;
}

}

#endif
