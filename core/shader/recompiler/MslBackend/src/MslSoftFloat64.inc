APS5_MSL_SOURCE(
struct aps5_u128 { ulong hi; ulong lo; };
static inline aps5_u128 aps5_u128_of(ulong hi, ulong lo) { aps5_u128 r; r.hi = hi; r.lo = lo; return r; }
static inline uint aps5_clz64(ulong x) { uint high = uint(x >> 32u); return high != 0u ? clz(high) : 32u + clz(uint(x)); }
static inline uint aps5_clz128(aps5_u128 x) { return x.hi != 0ul ? aps5_clz64(x.hi) : 64u + aps5_clz64(x.lo); }
static inline aps5_u128 aps5_mul64(ulong a, ulong b) {
    ulong a0 = a & 0xfffffffful;
    ulong a1 = a >> 32u;
    ulong b0 = b & 0xfffffffful;
    ulong b1 = b >> 32u;
    ulong p00 = a0 * b0;
    ulong p01 = a0 * b1;
    ulong p10 = a1 * b0;
    ulong p11 = a1 * b1;
    ulong middle = (p00 >> 32u) + (p01 & 0xfffffffful) + (p10 & 0xfffffffful);
    return aps5_u128_of(p11 + (p01 >> 32u) + (p10 >> 32u) + (middle >> 32u), (middle << 32u) | (p00 & 0xfffffffful));
}
static inline aps5_u128 aps5_u128_add(aps5_u128 a, aps5_u128 b) { ulong lo = a.lo + b.lo; return aps5_u128_of(a.hi + b.hi + (lo < a.lo ? 1ul : 0ul), lo); }
static inline aps5_u128 aps5_u128_sub(aps5_u128 a, aps5_u128 b) { return aps5_u128_of(a.hi - b.hi - (a.lo < b.lo ? 1ul : 0ul), a.lo - b.lo); }
static inline bool aps5_u128_less(aps5_u128 a, aps5_u128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
static inline aps5_u128 aps5_u128_shl(aps5_u128 x, uint n) {
    if (n == 0u) return x;
    if (n >= 128u) return aps5_u128_of(0ul, 0ul);
    if (n >= 64u) return aps5_u128_of(x.lo << (n - 64u), 0ul);
    return aps5_u128_of((x.hi << n) | (x.lo >> (64u - n)), x.lo << n);
}
static inline aps5_u128 aps5_u128_shr_jam(aps5_u128 x, uint n) {
    if (n == 0u) return x;
    if (n >= 128u) return aps5_u128_of(0ul, (x.hi | x.lo) != 0ul ? 1ul : 0ul);
    if (n >= 64u) {
        ulong dropped = x.lo | (n == 64u ? 0ul : x.hi << (128u - n));
        return aps5_u128_of(0ul, (n == 64u ? x.hi : x.hi >> (n - 64u)) | (dropped != 0ul ? 1ul : 0ul));
    }
    ulong lost = x.lo << (64u - n);
    return aps5_u128_of(x.hi >> n, ((x.hi << (64u - n)) | (x.lo >> n)) | (lost != 0ul ? 1ul : 0ul));
}
static inline ulong aps5_shr_jam64(ulong x, uint n) {
    if (n == 0u) return x;
    if (n >= 63u) return x != 0ul ? 1ul : 0ul;
    return (x >> n) | ((x << (64u - n)) != 0ul ? 1ul : 0ul);
}
static inline bool aps5_f64_is_nan(ulong x) { return (x & 0x7ffffffffffffffful) > 0x7ff0000000000000ul; }
static inline bool aps5_f64_is_inf(ulong x) { return (x & 0x7ffffffffffffffful) == 0x7ff0000000000000ul; }
static inline bool aps5_f64_is_zero(ulong x) { return (x & 0x7ffffffffffffffful) == 0ul; }
static inline bool aps5_f64_is_signaling(ulong x) { return aps5_f64_is_nan(x) && (x & 0x0008000000000000ul) == 0ul; }
static inline ulong aps5_f64_quiet(ulong x) { return x | 0x0008000000000000ul; }
static inline ulong aps5_f64_sign(ulong x) { return x & 0x8000000000000000ul; }
static inline int aps5_f64_exponent(ulong x) { return int((x >> 52u) & 0x7fful); }
static inline ulong aps5_f64_round_pack(ulong sign, int exponent, ulong sig) {
    uint round = uint(sig & 0x3fful);
    if (exponent < 0) {
        sig = aps5_shr_jam64(sig, exponent < -100 ? 100u : uint(-exponent));
        exponent = 0;
        round = uint(sig & 0x3fful);
    } else if (exponent > 0x7fd || (exponent == 0x7fd && sig + 0x200ul >= 0x8000000000000000ul)) {
        return sign | 0x7ff0000000000000ul;
    }
    sig = (sig + 0x200ul) >> 10u;
    if (round == 0x200u) sig &= ~1ul;
    if (sig == 0ul) exponent = 0;
    return sign + (ulong(exponent) << 52u) + sig;
}
static inline ulong aps5_f64_round_wide(ulong sign, aps5_u128 value, int exponent) {
    if ((value.hi | value.lo) == 0ul) return sign;
    uint top = 127u - aps5_clz128(value);
    ulong sig;
    if (top >= 62u) {
        aps5_u128 shifted = aps5_u128_shr_jam(value, top - 62u);
        sig = shifted.lo;
    } else {
        sig = value.lo << (62u - top);
    }
    return aps5_f64_round_pack(sign, exponent + int(top), sig);
}
struct aps5_f64_parts { ulong sign; ulong mantissa; int exponent; };
static inline aps5_f64_parts aps5_f64_unpack(ulong x) {
    aps5_f64_parts p;
    p.sign = aps5_f64_sign(x);
    int field = aps5_f64_exponent(x);
    ulong fraction = x & 0x000ffffffffffffful;
    if (field == 0) {
        uint shift = aps5_clz64(fraction) - 11u;
        p.mantissa = fraction << shift;
        p.exponent = -1074 - int(shift);
    } else {
        p.mantissa = fraction | 0x0010000000000000ul;
        p.exponent = field - 1075;
    }
    return p;
}
struct aps5_wide { ulong sign; aps5_u128 value; int exponent; };
static inline aps5_wide aps5_wide_sum(ulong signX, aps5_u128 x, int exponentX, ulong signY, aps5_u128 y, int exponentY) {
    aps5_wide result;
    if ((y.hi | y.lo) == 0ul) { result.sign = signX; result.value = x; result.exponent = exponentX; return result; }
    if ((x.hi | x.lo) == 0ul) { result.sign = signY; result.value = y; result.exponent = exponentY; return result; }
    int shiftX = int(aps5_clz128(x)) - 2;
    int shiftY = int(aps5_clz128(y)) - 2;
    x = shiftX >= 0 ? aps5_u128_shl(x, uint(shiftX)) : aps5_u128_shr_jam(x, uint(-shiftX));
    y = shiftY >= 0 ? aps5_u128_shl(y, uint(shiftY)) : aps5_u128_shr_jam(y, uint(-shiftY));
    exponentX -= shiftX;
    exponentY -= shiftY;
    result.exponent = exponentX;
    if (exponentX >= exponentY) {
        y = aps5_u128_shr_jam(y, uint(min(exponentX - exponentY, 200)));
    } else {
        result.exponent = exponentY;
        x = aps5_u128_shr_jam(x, uint(min(exponentY - exponentX, 200)));
    }
    result.sign = signX;
    if (signX == signY) {
        result.value = aps5_u128_add(x, y);
    } else if (aps5_u128_less(x, y)) {
        result.sign = signY;
        result.value = aps5_u128_sub(y, x);
    } else {
        result.value = aps5_u128_sub(x, y);
        if ((result.value.hi | result.value.lo) == 0ul) result.sign = 0ul;
    }
    return result;
}
static inline ulong aps5_f64_sum(ulong signX, aps5_u128 x, int exponentX, ulong signY, aps5_u128 y, int exponentY, int scale) {
    aps5_wide total = aps5_wide_sum(signX, x, exponentX, signY, y, exponentY);
    return aps5_f64_round_wide(total.sign, total.value, total.exponent + scale + 1022);
}
static inline ulong aps5_f64_nan2(ulong a, ulong b, ulong result) {
    if (aps5_f64_is_nan(a)) return aps5_f64_quiet(a);
    if (aps5_f64_is_nan(b)) return aps5_f64_quiet(b);
    return aps5_f64_is_nan(result) ? 0xfff8000000000000ul : result;
}
static inline ulong aps5_f64_add(ulong a, ulong b) {
    if (aps5_f64_is_nan(a) || aps5_f64_is_nan(b)) return aps5_f64_nan2(a, b, 0ul);
    if (aps5_f64_is_inf(a)) return aps5_f64_is_inf(b) && aps5_f64_sign(a) != aps5_f64_sign(b) ? 0xfff8000000000000ul : a;
    if (aps5_f64_is_inf(b)) return b;
    if (aps5_f64_is_zero(a)) return aps5_f64_is_zero(b) ? (a & b) : b;
    if (aps5_f64_is_zero(b)) return a;
    aps5_f64_parts pa = aps5_f64_unpack(a);
    aps5_f64_parts pb = aps5_f64_unpack(b);
    return aps5_f64_sum(pa.sign, aps5_u128_of(0ul, pa.mantissa), pa.exponent, pb.sign, aps5_u128_of(0ul, pb.mantissa), pb.exponent, 0);
}
static inline ulong aps5_f64_mul(ulong a, ulong b) {
    if (aps5_f64_is_nan(a) || aps5_f64_is_nan(b)) return aps5_f64_nan2(a, b, 0ul);
    ulong sign = aps5_f64_sign(a ^ b);
    if (aps5_f64_is_inf(a) || aps5_f64_is_inf(b)) return aps5_f64_is_zero(a) || aps5_f64_is_zero(b) ? 0xfff8000000000000ul : sign | 0x7ff0000000000000ul;
    if (aps5_f64_is_zero(a) || aps5_f64_is_zero(b)) return sign;
    aps5_f64_parts pa = aps5_f64_unpack(a);
    aps5_f64_parts pb = aps5_f64_unpack(b);
    return aps5_f64_round_wide(sign, aps5_mul64(pa.mantissa, pb.mantissa), pa.exponent + pb.exponent + 1022);
}
static inline ulong aps5_f64_scale(ulong x, int scale) {
    if (aps5_f64_is_nan(x)) return aps5_f64_quiet(x);
    if (aps5_f64_is_inf(x) || aps5_f64_is_zero(x)) return x;
    aps5_f64_parts p = aps5_f64_unpack(x);
    return aps5_f64_round_wide(p.sign, aps5_u128_of(0ul, p.mantissa), p.exponent + clamp(scale, -2200, 2200) + 1022);
}
static inline ulong aps5_f64_fma_scale(ulong a, ulong b, ulong c, int scale) {
    if (aps5_f64_is_nan(a)) return aps5_f64_quiet(a);
    if (aps5_f64_is_nan(b)) return aps5_f64_quiet(b);
    if (aps5_f64_is_nan(c)) return aps5_f64_quiet(c);
    ulong sign = aps5_f64_sign(a ^ b);
    if (aps5_f64_is_inf(a) || aps5_f64_is_inf(b)) {
        if (aps5_f64_is_zero(a) || aps5_f64_is_zero(b)) return 0xfff8000000000000ul;
        if (aps5_f64_is_inf(c) && aps5_f64_sign(c) != sign) return 0xfff8000000000000ul;
        return sign | 0x7ff0000000000000ul;
    }
    if (aps5_f64_is_inf(c)) return c;
    scale = clamp(scale, -2200, 2200);
    if (aps5_f64_is_zero(a) || aps5_f64_is_zero(b)) {
        if (aps5_f64_is_zero(c)) return sign & c;
        return aps5_f64_scale(c, scale);
    }
    aps5_f64_parts pa = aps5_f64_unpack(a);
    aps5_f64_parts pb = aps5_f64_unpack(b);
    aps5_u128 product = aps5_mul64(pa.mantissa, pb.mantissa);
    if (aps5_f64_is_zero(c)) return aps5_f64_round_wide(sign, product, pa.exponent + pb.exponent + scale + 1022);
    aps5_f64_parts pc = aps5_f64_unpack(c);
    return aps5_f64_sum(sign, product, pa.exponent + pb.exponent, pc.sign, aps5_u128_of(0ul, pc.mantissa), pc.exponent, scale);
}
static inline ulong aps5_f64_fma(ulong a, ulong b, ulong c) { return aps5_f64_fma_scale(a, b, c, 0); }
static inline bool aps5_f64_less(ulong a, ulong b) {
    if (aps5_f64_is_nan(a) || aps5_f64_is_nan(b)) return false;
    if (aps5_f64_is_zero(a) && aps5_f64_is_zero(b)) return false;
    bool negativeA = aps5_f64_sign(a) != 0ul;
    bool negativeB = aps5_f64_sign(b) != 0ul;
    if (negativeA != negativeB) return negativeA;
    return negativeA ? a > b : a < b;
}
static inline ulong aps5_f64_min_max(ulong a, ulong b, bool maximum) {
    ulong result = (maximum ? aps5_f64_less(b, a) : aps5_f64_less(a, b)) ? a : b;
    if (aps5_f64_is_zero(a) && aps5_f64_is_zero(b)) result = maximum ? (aps5_f64_sign(a) != 0ul ? b : a) : (aps5_f64_sign(a) != 0ul ? a : b);
    if (aps5_f64_is_nan(a)) result = b;
    if (aps5_f64_is_nan(b)) result = a;
    if (aps5_f64_is_signaling(b)) result = aps5_f64_quiet(b);
    if (aps5_f64_is_signaling(a)) result = aps5_f64_quiet(a);
    return result;
}
static inline ulong aps5_f64_min(ulong a, ulong b) { return aps5_f64_min_max(a, b, false); }
static inline ulong aps5_f64_max(ulong a, ulong b) { return aps5_f64_min_max(a, b, true); }
static inline ulong aps5_f64_saturate(ulong x) {
    if (aps5_f64_is_nan(x) || aps5_f64_sign(x) != 0ul || aps5_f64_is_zero(x)) return 0ul;
    return x < 0x3ff0000000000000ul ? x : 0x3ff0000000000000ul;
}
static inline ulong aps5_f64_ldexp(ulong x, uint exponent) {
    if (aps5_f64_is_nan(x)) return aps5_f64_quiet(x);
    return aps5_f64_scale(x, int(exponent));
}
static inline ulong aps5_f64_trunc(ulong x) {
    if (aps5_f64_is_nan(x)) return aps5_f64_quiet(x);
    int field = aps5_f64_exponent(x);
    if (field >= 1075) return x;
    if (field < 1023) return aps5_f64_sign(x);
    return x & ~((1ul << uint(1075 - field)) - 1ul);
}
static inline ulong aps5_f64_floor(ulong x) {
    ulong t = aps5_f64_trunc(x);
    return aps5_f64_sign(x) != 0ul && t != x && !aps5_f64_is_nan(x) ? aps5_f64_add(t, 0xbff0000000000000ul) : t;
}
static inline ulong aps5_f64_ceil(ulong x) {
    ulong t = aps5_f64_trunc(x);
    return aps5_f64_sign(x) == 0ul && t != x && !aps5_f64_is_nan(x) ? aps5_f64_add(t, 0x3ff0000000000000ul) : t;
}
static inline ulong aps5_f64_from_u64(ulong sign, ulong value) {
    if (value == 0ul) return sign;
    return aps5_f64_round_wide(sign, aps5_u128_of(0ul, value), 1022);
}
static inline ulong aps5_f64_round_even(ulong x) {
    if (aps5_f64_is_nan(x)) return aps5_f64_quiet(x);
    int field = aps5_f64_exponent(x);
    ulong sign = aps5_f64_sign(x);
    if (field >= 1075) return x;
    if (field < 1022) return sign;
    ulong mantissa = (x & 0x000ffffffffffffful) | 0x0010000000000000ul;
    uint shift = uint(1075 - field);
    ulong half = 1ul << (shift - 1u);
    ulong rest = mantissa & ((1ul << shift) - 1ul);
    ulong whole = mantissa >> shift;
    if (rest > half || (rest == half && (whole & 1ul) != 0ul)) whole += 1ul;
    return aps5_f64_from_u64(sign, whole);
}
static inline ulong aps5_f64_fract(ulong x) {
    if (aps5_f64_is_nan(x)) return aps5_f64_quiet(x);
    if (aps5_f64_is_inf(x)) return 0xfff8000000000000ul;
    ulong fraction = aps5_f64_add(x, aps5_f64_floor(x) ^ 0x8000000000000000ul);
    return fraction > 0x3feffffffffffffful ? 0x3feffffffffffffful : fraction;
}
static inline ulong aps5_f64_frexp_mant(ulong x) {
    int field = aps5_f64_exponent(x);
    if (field == 0x7ff) return aps5_f64_is_nan(x) ? aps5_f64_quiet(x) : x;
    if (aps5_f64_is_zero(x)) return x;
    aps5_f64_parts p = aps5_f64_unpack(x);
    return p.sign | (1022ul << 52u) | (p.mantissa & 0x000ffffffffffffful);
}
static inline uint aps5_f64_frexp_exp(ulong x) {
    int field = aps5_f64_exponent(x);
    if (field == 0x7ff || aps5_f64_is_zero(x)) return 0u;
    aps5_f64_parts p = aps5_f64_unpack(x);
    return uint(p.exponent + 53);
}
static inline ulong aps5_f64_rcp(ulong x) {
    if (aps5_f64_is_nan(x)) return aps5_f64_quiet(x);
    ulong sign = aps5_f64_sign(x);
    if (aps5_f64_is_zero(x)) return sign | 0x7ff0000000000000ul;
    if (aps5_f64_is_inf(x)) return sign;
    aps5_f64_parts p = aps5_f64_unpack(x);
    ulong remainder = 0x0010000000000000ul;
    int exponent = -p.exponent - 52;
    if (remainder < p.mantissa) {
        remainder <<= 1u;
        exponent -= 1;
    }
    ulong quotient = 0ul;
    for (uint bit = 0u; bit < 55u; ++bit) {
        quotient <<= 1u;
        if (remainder >= p.mantissa) {
            remainder -= p.mantissa;
            quotient |= 1ul;
        }
        remainder <<= 1u;
    }
    ulong sig = (quotient << 8u) | (remainder != 0ul ? 1ul : 0ul);
    return aps5_f64_round_pack(sign, exponent + 1022, sig);
}
static inline ulong aps5_f64_sqrt(ulong x) {
    if (aps5_f64_is_nan(x)) return aps5_f64_quiet(x);
    if (aps5_f64_is_zero(x)) return x;
    if (aps5_f64_sign(x) != 0ul) return 0xfff8000000000000ul;
    if (aps5_f64_is_inf(x)) return x;
    aps5_f64_parts p = aps5_f64_unpack(x);
    int exponent = p.exponent + 52;
    ulong mantissa = p.mantissa;
    if ((exponent & 1) != 0) {
        mantissa <<= 1u;
        exponent -= 1;
    }
    aps5_u128 radicand = aps5_u128_shl(aps5_u128_of(0ul, mantissa), 58u);
    ulong root = 0ul;
    ulong remainder = 0ul;
    for (int pair = 55; pair >= 0; --pair) {
        uint at = uint(pair) * 2u;
        ulong digits = at >= 64u ? (radicand.hi >> (at - 64u)) & 3ul : (radicand.lo >> at) & 3ul;
        remainder = (remainder << 2u) | digits;
        ulong trial = (root << 2u) | 1ul;
        if (remainder >= trial) {
            remainder -= trial;
            root = (root << 1u) | 1ul;
        } else {
            root <<= 1u;
        }
    }
    ulong sig = (root << 7u) | (remainder != 0ul ? 1ul : 0ul);
    return aps5_f64_round_pack(0ul, 1077 + (exponent - 110) / 2, sig);
}
static inline ulong aps5_f64_rsq(ulong x) {
    if (aps5_f64_is_nan(x)) return aps5_f64_quiet(x);
    ulong sign = aps5_f64_sign(x);
    if (aps5_f64_is_zero(x)) return sign | 0x7ff0000000000000ul;
    if (sign != 0ul) return 0xfff8000000000000ul;
    if (aps5_f64_is_inf(x)) return 0ul;
    return aps5_f64_rcp(aps5_f64_sqrt(x));
}
constant uint aps5_two_over_pi[40] = {
    0xa2f9836eu, 0x4e441529u, 0xfc2757d1u, 0xf534ddc0u, 0xdb629599u, 0x3c439041u, 0xfe5163abu, 0xdebbc561u,
    0xb7246e3au, 0x424dd2e0u, 0x06492eeau, 0x09d1921cu, 0xfe1deb1cu, 0xb129a73eu, 0xe88235f5u, 0x2ebb4484u,
    0xe99c7026u, 0xb45f7e41u, 0x3991d639u, 0x835339f4u, 0x9c845f8bu, 0xbdf9283bu, 0x1ff897ffu, 0xde05980fu,
    0xef2f118bu, 0x5a0a6d1fu, 0x6d367ecfu, 0x27cb09b7u, 0x4f463f66u, 0x9e5fea2du, 0x7527bac7u, 0xebe5f17bu,
    0x3d0739f7u, 0x8a5292eau, 0x6bfb5fb1u, 0x1f8d5d08u, 0x56033046u, 0x00000000u, 0x00000000u, 0x00000000u
};
static inline ulong aps5_f64_trig_preop(ulong x, uint segment) {
    uint field = uint(aps5_f64_exponent(x));
    uint extra = max(field, 1077u) - 1077u;
    uint shift = min((segment & 31u) * 53u + extra, 1201u);
    uint word = shift >> 5u;
    uint offset = shift & 31u;
    uint first = aps5_two_over_pi[word];
    uint second = aps5_two_over_pi[word + 1u];
    uint third = aps5_two_over_pi[word + 2u];
    uint high = offset == 0u ? first : (first << offset) | (second >> (32u - offset));
    uint low = (offset == 0u ? second : (second << offset) | (third >> (32u - offset))) >> 11u;
    ulong mantissa = (ulong(high) << 21u) | ulong(low);
    int bias = field >= 1968u ? 1023 + 600 - 53 + 128 : 1023 + 600 - 53;
    if (mantissa == 0ul) return 0ul;
    return aps5_f64_round_wide(0ul, aps5_u128_of(0ul, mantissa), bias - int(shift) - 1623 + 1084 - 62);
}
static inline ulong aps5_f64_from_f32(uint bits) {
    ulong sign = ulong(bits & 0x80000000u) << 32u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return sign | 0x7ff8000000000000ul | (ulong(magnitude & 0x007fffffu) << 29u);
    if (magnitude == 0x7f800000u) return sign | 0x7ff0000000000000ul;
    if (magnitude == 0u) return sign;
    if (magnitude < 0x00800000u) return aps5_f64_round_wide(sign, aps5_u128_of(0ul, ulong(magnitude)), -149 + 1084 - 62);
    return sign | (ulong(magnitude) << 29u) + (896ul << 52u);
}
static inline uint aps5_f32_round_pack(uint sign, int exponent, uint sig) {
    if (exponent < 0) {
        sig = uint(aps5_shr_jam64(ulong(sig), exponent < -100 ? 100u : uint(-exponent)));
        exponent = 0;
    } else if (exponent >= 0xfe) {
        return sign | 0x7f800000u;
    }
    uint round = sig & 0x7fu;
    sig = (sig + 0x40u) >> 7u;
    if (round == 0x40u) sig &= ~1u;
    if (sig == 0u) exponent = 0;
    return sign + (uint(exponent) << 23u) + sig;
}
static inline uint aps5_f32_round_wide(uint sign, aps5_u128 value, int exponent) {
    if ((value.hi | value.lo) == 0ul) return sign;
    uint top = 127u - aps5_clz128(value);
    uint sig = top >= 30u ? uint(aps5_u128_shr_jam(value, top - 30u).lo) : uint(value.lo) << (30u - top);
    return aps5_f32_round_pack(sign, exponent + int(top) + 126, sig);
}
static inline uint aps5_f64_to_f32(ulong x) {
    uint sign = uint(x >> 32u) & 0x80000000u;
    if (aps5_f64_is_nan(x)) return sign | 0x7fc00000u | uint((x & 0x000ffffffffffffful) >> 29u);
    if (aps5_f64_is_inf(x)) return sign | 0x7f800000u;
    if (aps5_f64_is_zero(x)) return sign;
    aps5_f64_parts p = aps5_f64_unpack(x);
    return aps5_f32_round_wide(sign, aps5_u128_of(0ul, p.mantissa), p.exponent);
}
static inline ulong aps5_f64_from_i32(uint bits) {
    int value = int(bits);
    return value < 0 ? aps5_f64_from_u64(0x8000000000000000ul, ulong(0u - bits)) : aps5_f64_from_u64(0ul, ulong(bits));
}
static inline ulong aps5_f64_from_u32(uint bits) { return aps5_f64_from_u64(0ul, ulong(bits)); }
static inline uint aps5_f64_to_i32(ulong x) {
    if (aps5_f64_is_nan(x)) return 0u;
    ulong magnitude = x & 0x7ffffffffffffffful;
    bool negative = aps5_f64_sign(x) != 0ul;
    if (negative && magnitude >= 0x41e0000000000000ul) return 0x80000000u;
    if (!negative && magnitude >= 0x41e0000000000000ul) return 0x7fffffffu;
    int field = aps5_f64_exponent(x);
    if (field < 1023) return 0u;
    ulong whole = ((x & 0x000ffffffffffffful) | 0x0010000000000000ul) >> uint(1075 - field);
    return negative ? 0u - uint(whole) : uint(whole);
}
static inline uint aps5_f64_to_u32(ulong x) {
    if (aps5_f64_is_nan(x) || aps5_f64_sign(x) != 0ul || aps5_f64_is_zero(x)) return 0u;
    if (x >= 0x41f0000000000000ul) return 0xffffffffu;
    int field = aps5_f64_exponent(x);
    if (field < 1023) return 0u;
    return uint(((x & 0x000ffffffffffffful) | 0x0010000000000000ul) >> uint(1075 - field));
}
struct aps5_half_term { uint sign; uint mantissa; int exponent; bool nan; bool infinite; bool zero; uint bits; };
static inline aps5_half_term aps5_half_term_of(uint packed, uint shift) {
    aps5_half_term h;
    h.bits = (packed >> shift) & 0xffffu;
    uint magnitude = h.bits & 0x7fffu;
    h.sign = (h.bits & 0x8000u) << 16u;
    h.nan = magnitude > 0x7c00u;
    h.infinite = magnitude == 0x7c00u;
    h.zero = magnitude == 0u;
    h.mantissa = magnitude < 0x400u ? magnitude : (magnitude & 0x3ffu) | 0x400u;
    h.exponent = magnitude < 0x400u ? -24 : int(magnitude >> 10u) - 25;
    return h;
}
static inline uint aps5_half_quiet(aps5_half_term h) { return h.sign | 0x7fc00000u | ((h.bits & 0x3ffu) << 13u); }
static inline uint aps5_dot2_exact(uint a, uint b, uint c) {
    aps5_half_term aLow = aps5_half_term_of(a, 0u);
    aps5_half_term bLow = aps5_half_term_of(b, 0u);
    aps5_half_term aHigh = aps5_half_term_of(a, 16u);
    aps5_half_term bHigh = aps5_half_term_of(b, 16u);
    if (aLow.nan) return aps5_half_quiet(aLow);
    if (bLow.nan) return aps5_half_quiet(bLow);
    if (aHigh.nan) return aps5_half_quiet(aHigh);
    if (bHigh.nan) return aps5_half_quiet(bHigh);
    uint magnitudeC = c & 0x7fffffffu;
    uint signC = c & 0x80000000u;
    if (magnitudeC > 0x7f800000u) return c | 0x00400000u;
    uint signLow = aLow.sign ^ bLow.sign;
    uint signHigh = aHigh.sign ^ bHigh.sign;
    bool infiniteLow = (aLow.infinite && !bLow.zero) || (bLow.infinite && !aLow.zero);
    bool infiniteHigh = (aHigh.infinite && !bHigh.zero) || (bHigh.infinite && !aHigh.zero);
    bool infiniteC = magnitudeC == 0x7f800000u;
    bool invalid = (aLow.infinite && bLow.zero) || (bLow.infinite && aLow.zero) || (aHigh.infinite && bHigh.zero) || (bHigh.infinite && aHigh.zero);
    invalid = invalid || (infiniteLow && infiniteHigh && signLow != signHigh);
    invalid = invalid || (infiniteLow && infiniteC && signLow != signC);
    invalid = invalid || (infiniteHigh && infiniteC && signHigh != signC);
    if (invalid) return 0xffc00000u;
    if (infiniteLow) return signLow | 0x7f800000u;
    if (infiniteHigh) return signHigh | 0x7f800000u;
    if (infiniteC) return c;
    aps5_u128 low = aps5_u128_of(0ul, ulong(aLow.mantissa * bLow.mantissa));
    aps5_u128 high = aps5_u128_of(0ul, ulong(aHigh.mantissa * bHigh.mantissa));
    aps5_wide products = aps5_wide_sum(ulong(signLow) << 32u, low, aLow.exponent + bLow.exponent, ulong(signHigh) << 32u, high, aHigh.exponent + bHigh.exponent);
    aps5_u128 addend = aps5_u128_of(0ul, magnitudeC < 0x00800000u ? 0ul : ulong((magnitudeC & 0x007fffffu) | 0x00800000u));
    aps5_wide total = aps5_wide_sum(products.sign, products.value, products.exponent, ulong(signC) << 32u, addend, int(magnitudeC >> 23u) - 150);
    if ((total.value.hi | total.value.lo) == 0ul) return signLow & signHigh & signC;
    return aps5_f32_round_wide(uint(total.sign >> 32u), total.value, total.exponent);
}
)
