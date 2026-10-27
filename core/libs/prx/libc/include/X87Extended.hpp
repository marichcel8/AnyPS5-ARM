#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_X87EXTENDED_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_X87EXTENDED_HPP

#include <cmath>
#include <cstdint>
#include <cstring>

// The guest's long double is the x87 80-bit format, which x86-64 hosts share. On native arm64 the
// host's long double is a double, and values cross between the formats here, rounded to double
// precision like the translator's x87 registers.
namespace X87Extended {

inline long double Load(const void* source) {
#if !defined(__aarch64__)
    long double value;
    std::memcpy(&value, source, sizeof(value));
    return value;
#else
    std::uint64_t mantissa = 0;
    std::uint16_t top = 0;
    std::memcpy(&mantissa, source, 8);
    std::memcpy(&top, static_cast<const std::uint8_t*>(source) + 8, 2);
    const int exponent = top & 0x7fff;
    const double sign = (top & 0x8000) != 0 ? -1.0 : 1.0;
    if (exponent == 0x7fff) return (mantissa << 1) == 0 ? sign * HUGE_VAL : std::copysign(std::nan(""), sign);
    if (mantissa == 0) return sign * 0.0;
    return sign * std::ldexp(static_cast<double>(mantissa), (exponent == 0 ? 1 : exponent) - 16383 - 63);
#endif
}

inline void Store(void* target, long double value) {
#if !defined(__aarch64__)
    std::memcpy(target, &value, sizeof(value));
#else
    std::uint64_t mantissa = 0;
    std::uint16_t top = std::signbit(value) ? 0x8000 : 0;
    const double magnitude = std::fabs(static_cast<double>(value));
    if (std::isnan(magnitude)) {
        mantissa = 0xc000000000000000ull;
        top |= 0x7fff;
    } else if (std::isinf(magnitude)) {
        mantissa = 0x8000000000000000ull;
        top |= 0x7fff;
    } else if (magnitude != 0.0) {
        int exponent = 0;
        const double fraction = std::frexp(magnitude, &exponent);
        mantissa = static_cast<std::uint64_t>(std::ldexp(fraction, 64));
        top |= static_cast<std::uint16_t>(exponent - 1 + 16383);
    }
    std::memcpy(target, &mantissa, 8);
    std::memcpy(static_cast<std::uint8_t*>(target) + 8, &top, 2);
#endif
}

}

#endif
