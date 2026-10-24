#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace MslSoftFloat {

using uint = std::uint32_t;
using ulong = std::uint64_t;
using std::clamp;
using std::max;
using std::min;

static inline uint clz(uint value) {
    return static_cast<uint>(std::countl_zero(value));
}

#define constant const
#define APS5_MSL_SOURCE(...) __VA_ARGS__
#include "MslBackend/src/MslSoftFloat64.inc"
#undef APS5_MSL_SOURCE
#undef constant

}

namespace {

using namespace MslSoftFloat;

constexpr ulong DefaultNan = 0xfff8000000000000ull;
constexpr int Rounds = 200000;

ulong state = 0x9e3779b97f4a7c15ull;

ulong Next() {
    state ^= state << 13u;
    state ^= state >> 7u;
    state ^= state << 17u;
    return state;
}

ulong RandomDouble() {
    const ulong bits = Next();
    const ulong sign = bits & 0x8000000000000000ull;
    ulong fraction = Next() & 0x000fffffffffffffull;
    switch (Next() % 8u) {
    case 0: fraction = 0u; break;
    case 1: fraction = 1u; break;
    case 2: fraction = 0x000fffffffffffffull; break;
    case 3: fraction &= 0x000ff00000000000ull; break;
    default: break;
    }
    ulong exponent;
    switch (Next() % 10u) {
    case 0: exponent = 0u; break;
    case 1: exponent = 0x7ffu; break;
    case 2: exponent = 1u + Next() % 4u; break;
    case 3: exponent = 0x7fau + Next() % 5u; break;
    case 4:
    case 5: exponent = 1000u + Next() % 50u; break;
    default: exponent = Next() % 0x800u; break;
    }
    return sign | (exponent << 52u) | fraction;
}

ulong Near(ulong bits, ulong around) {
    return (bits & 0x800fffffffffffffull) | ((around + Next() % 60u - 30u) & 0x7ffu) << 52u;
}

double D(ulong bits) {
    return std::bit_cast<double>(bits);
}

ulong B(double value) {
    return std::bit_cast<ulong>(value);
}

bool IsNan(ulong bits) {
    return (bits & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
}

ulong Quiet(ulong bits) {
    return bits | 0x0008000000000000ull;
}

ulong NanRule(std::initializer_list<ulong> inputs, ulong native) {
    for (const auto input : inputs) {
        if (IsNan(input)) return Quiet(input);
    }
    return IsNan(native) ? DefaultNan : native;
}

std::string Hex(ulong value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(value));
    return text;
}

void Expect(const char* name, ulong got, ulong expected, std::initializer_list<ulong> inputs) {
    if (got == expected) return;
    std::string message = std::string(name) + "(";
    bool first = true;
    for (const auto input : inputs) {
        message += (first ? "" : ", ") + Hex(input);
        first = false;
    }
    throw std::runtime_error(message + ") = " + Hex(got) + ", expected " + Hex(expected));
}

ulong ReferenceMinMax(ulong a, ulong b, bool maximum) {
    const bool ordered = maximum ? D(a) > D(b) : D(a) < D(b);
    ulong result = ordered ? a : b;
    const bool zeroA = (a & 0x7fffffffffffffffull) == 0u;
    const bool zeroB = (b & 0x7fffffffffffffffull) == 0u;
    const bool negativeA = (a >> 63u) != 0u;
    if (zeroA && zeroB) result = maximum ? (negativeA ? b : a) : (negativeA ? a : b);
    if (IsNan(a)) result = b;
    if (IsNan(b)) result = a;
    if (IsNan(b) && (b & 0x0008000000000000ull) == 0u) result = Quiet(b);
    if (IsNan(a) && (a & 0x0008000000000000ull) == 0u) result = Quiet(a);
    return result;
}

ulong ReferenceTrigPreop(ulong x, uint segment) {
    static constexpr uint TwoOverPi[40] = {
        0xa2f9836eu, 0x4e441529u, 0xfc2757d1u, 0xf534ddc0u, 0xdb629599u, 0x3c439041u, 0xfe5163abu, 0xdebbc561u,
        0xb7246e3au, 0x424dd2e0u, 0x06492eeau, 0x09d1921cu, 0xfe1deb1cu, 0xb129a73eu, 0xe88235f5u, 0x2ebb4484u,
        0xe99c7026u, 0xb45f7e41u, 0x3991d639u, 0x835339f4u, 0x9c845f8bu, 0xbdf9283bu, 0x1ff897ffu, 0xde05980fu,
        0xef2f118bu, 0x5a0a6d1fu, 0x6d367ecfu, 0x27cb09b7u, 0x4f463f66u, 0x9e5fea2du, 0x7527bac7u, 0xebe5f17bu,
        0x3d0739f7u, 0x8a5292eau, 0x6bfb5fb1u, 0x1f8d5d08u, 0x56033046u, 0x00000000u, 0x00000000u, 0x00000000u,
    };
    const uint exponent = static_cast<uint>(x >> 52u) & 0x7ffu;
    const uint extra = std::max(exponent, 1077u) - 1077u;
    const uint shift = std::min((segment & 31u) * 53u + extra, 1201u);
    const uint word = shift >> 5u;
    const uint offset = shift & 31u;
    const auto funnel = [&](uint upper, uint lower) { return offset == 0u ? upper : (upper << offset) | (lower >> (32u - offset)); };
    const double high = static_cast<double>(funnel(TwoOverPi[word], TwoOverPi[word + 1u]));
    const double low = static_cast<double>(funnel(TwoOverPi[word + 1u], TwoOverPi[word + 2u]) >> 11u);
    const double mantissa = high * 2097152.0 + low;
    const uint bias = exponent >= 1968u ? 1023u + 600u - 53u + 128u : 1023u + 600u - 53u;
    const double scale = D(static_cast<ulong>(bias - shift) << 52u);
    return B(mantissa * scale * D(0x1a70000000000000ull));
}



struct Pair {
    double high;
    double low;
};

Pair TwoSum(double lhs, double rhs) {
    const double sum = lhs + rhs;
    const double rhsPart = sum - lhs;
    const double lhsPart = sum - rhsPart;
    return {sum, (lhs - lhsPart) + (rhs - rhsPart)};
}

double AddRoundToOdd(double lhs, double rhs) {
    const Pair sum = TwoSum(lhs, rhs);
    if (sum.low == 0.0) return sum.high;
    ulong bits = B(sum.high);
    if (((bits ^ B(sum.low)) >> 63u) != 0u) bits -= 1u;
    return D(bits | 1u);
}

uint ReferenceDot2(uint a, uint b, uint c) {
    struct Half {
        uint bits;
        uint sign;
        bool nan;
        bool infinite;
        bool zero;
        double value;
    };
    const auto half = [](uint packed, uint shift) {
        Half h{};
        h.bits = (packed >> shift) & 0xffffu;
        const uint magnitude = h.bits & 0x7fffu;
        h.sign = (h.bits & 0x8000u) << 16u;
        h.nan = magnitude > 0x7c00u;
        h.infinite = magnitude == 0x7c00u;
        h.zero = magnitude == 0u;
        const double unsignedValue = magnitude < 0x400u ? static_cast<double>(magnitude) * D(0x3e70000000000000ull) : D(static_cast<ulong>((magnitude << 10u) + (1008u << 20u)) << 32u);
        h.value = h.sign != 0u ? -unsignedValue : unsignedValue;
        return h;
    };
    const Half aLow = half(a, 0u);
    const Half bLow = half(b, 0u);
    const Half aHigh = half(a, 16u);
    const Half bHigh = half(b, 16u);
    const uint magnitudeC = c & 0x7fffffffu;
    const uint signC = c & 0x80000000u;
    const bool nanC = magnitudeC > 0x7f800000u;
    const bool infiniteC = magnitudeC == 0x7f800000u;
    const bool flushedC = magnitudeC < 0x00800000u;
    const ulong highC = flushedC ? 0u : (magnitudeC >> 3u) + (896u << 20u);
    const ulong lowC = flushedC ? 0u : magnitudeC << 29u;
    const double valueC = D(((highC | signC) << 32u) | lowC);
    const double productLow = aLow.value * bLow.value;
    const double productHigh = aHigh.value * bHigh.value;
    const Pair upper = TwoSum(productHigh, valueC);
    const Pair total = TwoSum(productLow, upper.high);
    const ulong sum = B(AddRoundToOdd(total.high, AddRoundToOdd(total.low, upper.low)));
    const uint sumHigh = static_cast<uint>(sum >> 32u);
    const uint sumLow = static_cast<uint>(sum);
    const uint exponent = (sumHigh >> 20u) & 0x7ffu;
    const uint mantissa = ((sumHigh & 0x000fffffu) << 3u) | (sumLow >> 29u);
    const uint base = ((exponent - 896u) << 23u) + mantissa;
    const uint rest = sumLow & 0x1fffffffu;
    const bool roundUp = rest > 0x10000000u || (rest == 0x10000000u && (base & 1u) != 0u);
    uint result = (sumHigh & 0x80000000u) | (base + (roundUp ? 1u : 0u));
    const uint signLow = aLow.sign ^ bLow.sign;
    const uint signHigh = aHigh.sign ^ bHigh.sign;
    if ((sum & 0x7fffffffffffffffull) == 0u) result = signLow & signHigh & signC;
    const auto infiniteProduct = [](const Half& lhs, const Half& rhs) { return (lhs.infinite && !rhs.zero) || (rhs.infinite && !lhs.zero); };
    const auto invalidProduct = [](const Half& lhs, const Half& rhs) { return (lhs.infinite && rhs.zero) || (rhs.infinite && lhs.zero); };
    const bool infiniteLow = infiniteProduct(aLow, bLow);
    const bool infiniteHigh = infiniteProduct(aHigh, bHigh);
    bool invalid = invalidProduct(aLow, bLow) || invalidProduct(aHigh, bHigh);
    invalid = invalid || (infiniteLow && infiniteHigh && signLow != signHigh);
    invalid = invalid || (infiniteLow && infiniteC && signLow != signC);
    invalid = invalid || (infiniteHigh && infiniteC && signHigh != signC);
    if (infiniteLow || infiniteHigh || infiniteC) result = (infiniteLow ? signLow : infiniteHigh ? signHigh : signC) | 0x7f800000u;
    if (invalid) result = 0xffc00000u;
    const auto quietHalf = [](const Half& h) { return h.sign | 0x7fc00000u | ((h.bits & 0x3ffu) << 13u); };
    if (nanC) result = c | 0x00400000u;
    if (bHigh.nan) result = quietHalf(bHigh);
    if (aHigh.nan) result = quietHalf(aHigh);
    if (bLow.nan) result = quietHalf(bLow);
    if (aLow.nan) result = quietHalf(aLow);
    return result;
}

uint RandomHalves() {
    const auto half = [] {
        const uint bits = static_cast<uint>(Next());
        switch (Next() % 6u) {
        case 0: return bits & 0x83ffu;
        case 1: return (bits & 0x8000u) | 0x7c00u | (Next() % 3u == 0u ? bits & 0x3ffu : 0u);
        case 2: return bits & 0x8000u;
        default: return bits & 0xffffu;
        }
    };
    return half() | (half() << 16u);
}

uint RandomSingle() {
    const uint bits = static_cast<uint>(Next());
    switch (Next() % 6u) {
    case 0: return bits & 0x807fffffu;
    case 1: return (bits & 0x80000000u) | 0x7f800000u | (Next() % 3u == 0u ? bits & 0x7fffffu : 0u);
    case 2: return (bits & 0x807fffffu) | ((100u + static_cast<uint>(Next() % 60u)) << 23u);
    default: return bits;
    }
}

void CheckDot2() {
    for (int round = 0; round < Rounds; ++round) {
        const uint a = RandomHalves();
        const uint b = RandomHalves();
        const uint c = RandomSingle();
        const uint expected = ReferenceDot2(a, b, c);
        const uint got = aps5_dot2_exact(a, b, c);
        const uint magnitude = got & 0x7fffffffu;
        if (magnitude != 0u && magnitude < 0x00800000u) continue;
        Expect("dot2", got, expected, {a, b, c});
    }
}

void CheckArithmetic() {
    for (int round = 0; round < Rounds; ++round) {
        const ulong a = RandomDouble();
        const ulong b = round % 3 == 0 ? Near(RandomDouble(), (a >> 52u) & 0x7ffu) : RandomDouble();
        const ulong c = round % 2 == 0 ? Near(RandomDouble(), ((a >> 52u) & 0x7ffu) + ((b >> 52u) & 0x7ffu) - 1023u) : RandomDouble();
        Expect("add", aps5_f64_add(a, b), NanRule({a, b}, B(D(a) + D(b))), {a, b});
        Expect("mul", aps5_f64_mul(a, b), NanRule({a, b}, B(D(a) * D(b))), {a, b});
        Expect("fma", aps5_f64_fma(a, b, c), NanRule({a, b, c}, B(std::fma(D(a), D(b), D(c)))), {a, b, c});
        Expect("min", aps5_f64_min(a, b), ReferenceMinMax(a, b, false), {a, b});
        Expect("max", aps5_f64_max(a, b), ReferenceMinMax(a, b, true), {a, b});
        const int scale = static_cast<int>(Next() % 257u) - 128;
        const double scaledA = std::ldexp(D(a), scale);
        const double scaledC = std::ldexp(D(c), scale);
        const bool exactA = std::isfinite(scaledA) && std::ldexp(scaledA, -scale) == D(a);
        const bool exactC = std::isfinite(scaledC) && std::ldexp(scaledC, -scale) == D(c);
        if (exactA && exactC && !IsNan(b)) {
            Expect("fma_scale", aps5_f64_fma_scale(a, b, c, scale), NanRule({a, b, c}, B(std::fma(scaledA, D(b), scaledC))), {a, b, c, static_cast<ulong>(scale)});
        }
    }
    Expect("fma_scale subnormal", aps5_f64_fma_scale(0x3ff8000000000000ull, 0x3ff0000000000000ull, 0ull, -1074), 0x0000000000000002ull, {});
    Expect("fma_scale tie", aps5_f64_fma_scale(0x3ff0000000000000ull, 0x3ff0000000000000ull, 0ull, -1075), 0ull, {});
    Expect("fma_scale overflow", aps5_f64_fma_scale(0x7fe0000000000000ull, 0x3ff0000000000000ull, 0ull, 1), 0x7ff0000000000000ull, {});
    Expect("add cancel", aps5_f64_add(0x3ff0000000000000ull, 0xbff0000000000000ull), 0ull, {});
    Expect("add negative zero", aps5_f64_add(0x8000000000000000ull, 0x8000000000000000ull), 0x8000000000000000ull, {});
    Expect("fma zero sign", aps5_f64_fma(0x8000000000000000ull, 0x3ff0000000000000ull, 0x8000000000000000ull), 0x8000000000000000ull, {});
    Expect("fma inf", aps5_f64_fma(0x7ff0000000000000ull, 0x3ff0000000000000ull, 0xfff0000000000000ull), DefaultNan, {});
}

void CheckUnary() {
    for (int round = 0; round < Rounds; ++round) {
        const ulong x = RandomDouble();
        const double value = D(x);
        const bool nan = IsNan(x);
        Expect("sqrt", aps5_f64_sqrt(x), NanRule({x}, B(std::sqrt(value))), {x});
        Expect("rcp", aps5_f64_rcp(x), NanRule({x}, B(1.0 / value)), {x});
        Expect("rsq", aps5_f64_rsq(x), NanRule({x}, B(1.0 / std::sqrt(value))), {x});
        Expect("trunc", aps5_f64_trunc(x), nan ? Quiet(x) : B(std::trunc(value)), {x});
        Expect("floor", aps5_f64_floor(x), nan ? Quiet(x) : B(std::floor(value)), {x});
        Expect("ceil", aps5_f64_ceil(x), nan ? Quiet(x) : B(std::ceil(value)), {x});
        Expect("round_even", aps5_f64_round_even(x), nan ? Quiet(x) : B(std::nearbyint(value)), {x});
        const double fraction = std::min(value - std::floor(value), D(0x3fefffffffffffffull));
        Expect("fract", aps5_f64_fract(x), NanRule({x}, B(fraction)), {x});
        const bool special = ((x >> 52u) & 0x7ffu) == 0x7ffu;
        int exponent = 0;
        const double mantissa = std::frexp(value, &exponent);
        Expect("frexp_mant", aps5_f64_frexp_mant(x), special ? (nan ? Quiet(x) : x) : B(mantissa), {x});
        Expect("frexp_exp", aps5_f64_frexp_exp(x), special ? 0u : static_cast<uint>(exponent), {x});
        Expect("saturate", aps5_f64_saturate(x), value > 0.0 ? (value < 1.0 ? x : B(1.0)) : 0u, {x});
        const int shift = static_cast<int>(Next() % 4400u) - 2200;
        Expect("ldexp", aps5_f64_ldexp(x, static_cast<uint>(shift)), nan ? Quiet(x) : B(std::ldexp(value, shift)), {x, static_cast<ulong>(shift)});
        const auto segment = static_cast<uint>(Next());
        Expect("trig_preop", aps5_f64_trig_preop(x, segment), ReferenceTrigPreop(x, segment), {x, segment});
        const float narrowed = static_cast<float>(value);
        const uint narrowedBits = nan ? (static_cast<uint>(x >> 32u) & 0x80000000u) | 0x7fc00000u | static_cast<uint>((x & 0x000fffffffffffffull) >> 29u) : std::bit_cast<uint>(narrowed);
        Expect("to_f32", aps5_f64_to_f32(x), narrowedBits, {x});
        const double low = -2147483648.0;
        const uint toI32 = nan ? 0u : value <= low ? 0x80000000u : value >= 2147483648.0 ? 0x7fffffffu : static_cast<uint>(static_cast<std::int32_t>(std::trunc(value)));
        Expect("to_i32", aps5_f64_to_i32(x), toI32, {x});
        const uint toU32 = nan || value <= 0.0 ? 0u : value >= 4294967296.0 ? 0xffffffffu : static_cast<uint>(std::trunc(value));
        Expect("to_u32", aps5_f64_to_u32(x), toU32, {x});
    }
    for (int round = 0; round < Rounds; ++round) {
        const uint bits = static_cast<uint>(Next());
        const uint shaped = round % 4 == 0 ? (bits & 0x807fffffu) : round % 4 == 1 ? (bits | 0x7f800000u) : bits;
        const bool nan = (shaped & 0x7fffffffu) > 0x7f800000u;
        const ulong widened = nan ? (static_cast<ulong>(shaped & 0x80000000u) << 32u) | 0x7ff8000000000000ull | (static_cast<ulong>(shaped & 0x007fffffu) << 29u) : B(static_cast<double>(std::bit_cast<float>(shaped)));
        Expect("from_f32", aps5_f64_from_f32(shaped), widened, {shaped});
        Expect("from_i32", aps5_f64_from_i32(bits), B(static_cast<double>(static_cast<std::int32_t>(bits))), {bits});
        Expect("from_u32", aps5_f64_from_u32(bits), B(static_cast<double>(bits)), {bits});
    }
}

}

int main() {
    try {
        if (std::numeric_limits<double>::is_iec559 == false) throw std::runtime_error("the host has no IEEE double to compare against");
        CheckArithmetic();
        CheckUnary();
        CheckDot2();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MslSoftFloat: %s\n", error.what());
        return 1;
    }
    std::puts("MslSoftFloat: the MSL fp64 soft-float and v_dot2_f32_f16 match IEEE arithmetic and the GCN NaN rules");
    return 0;
}
