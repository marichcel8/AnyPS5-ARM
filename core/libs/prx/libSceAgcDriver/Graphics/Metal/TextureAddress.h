#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define AGC_POINTER(type) device type*
#define AGC_PARAMS constant AgcDetileParams&
#define AGC_POPCOUNT(value) popcount(value)
#define AGC_UINT4(x, y, z, w) uint4(x, y, z, w)
#else
#include <bit>
#include <cstdint>
using uint = std::uint32_t;
using uchar = std::uint8_t;
using ushort = std::uint16_t;
struct uint4 {
    uint x;
    uint y;
    uint z;
    uint w;
    bool operator==(const uint4&) const = default;
};
using ulong = std::uint64_t;
#define AGC_POINTER(type) std::uint64_t
#define AGC_PARAMS const AgcDetileParams&
#define AGC_POPCOUNT(value) static_cast<uint>(std::popcount(static_cast<uint>(value)))
#define AGC_UINT4(x, y, z, w) uint4{x, y, z, w}
#endif

#define AGC_FAMILY_LINEAR 0u
#define AGC_FAMILY_STANDARD 1u
#define AGC_FAMILY_EQUATION 2u

struct AgcDetileParams {
    AGC_POINTER(uchar) tiled;
    AGC_POINTER(uchar) linear;
    uint width;
    uint height;
    uint pitchBytes;
    uint blocksPerRow;
    uint tail;
    uint tailX;
    uint tailY;
    uint slice;
    uint pipeBankXor;
    uint family;
    uint blockBytes;
    uint blockLog2Width;
    uint blockLog2Height;
    uint columnBegin;
    uint rowBegin;
    uint rangeBegin;
    uint rangeEnd;
    uint destinationSlice;
    uint linearPitchBytes;
    uint pad;
    uint equation[16];
};

inline uint agc_equation_bit(uint mask, uint x, uint y, uint slice) {
    const uint selected = (x & (mask & 0xfffu)) ^ ((y << 12) & (mask & 0xfff000u)) ^ ((slice << 24) & (mask & 0xff000000u));
    return AGC_POPCOUNT(selected) & 1u;
}

inline uint agc_equation_offset(AGC_PARAMS p, uint x, uint y) {
    uint offset = 0u;
    for (uint bit = 0u; bit < 16u; ++bit) offset |= agc_equation_bit(p.equation[bit], x, y, p.slice) << bit;
    return offset;
}

inline uint agc_standard_offset(uint x, uint y, uint elementBytes) {
    switch (elementBytes) {
        case 1u:
            return ((y << 4) & 0x1f0u) ^ ((y << 5) & 0x400u) ^ (x & 0x00fu) ^ ((x << 5) & 0x200u) ^ ((x << 6) & 0x800u);
        case 2u:
            return ((y << 4) & 0x070u) ^ ((y << 5) & 0x100u) ^ ((y << 6) & 0x400u) ^ ((x << 1) & 0x00eu) ^ ((x << 4) & 0x080u) ^ ((x << 5) & 0x200u) ^ ((x << 6) & 0x800u);
        case 4u:
            return ((y << 4) & 0x070u) ^ ((y << 5) & 0x100u) ^ ((y << 6) & 0x400u) ^ ((x << 2) & 0x00cu) ^ ((x << 5) & 0x080u) ^ ((x << 6) & 0x200u) ^ ((x << 7) & 0x800u);
        case 8u:
            return ((y << 4) & 0x030u) ^ ((y << 6) & 0x100u) ^ ((y << 7) & 0x400u) ^ ((x << 3) & 0x008u) ^ ((x << 5) & 0x0c0u) ^ ((x << 6) & 0x200u) ^ ((x << 7) & 0x800u);
        default:
            return ((y << 4) & 0x030u) ^ ((y << 6) & 0x100u) ^ ((y << 7) & 0x400u) ^ ((x << 6) & 0x0c0u) ^ ((x << 7) & 0x200u) ^ ((x << 8) & 0x800u);
    }
}

inline uint agc_standard64_extra(uint x, uint y, uint elementBytes) {
    switch (elementBytes) {
        case 1u:
            return ((x << 7) & 0x2000u) ^ ((x << 8) & 0x8000u) ^ ((y << 6) & 0x1000u) ^ ((y << 7) & 0x4000u);
        case 2u:
            return ((x << 7) & 0x2000u) ^ ((x << 8) & 0x8000u) ^ ((y << 7) & 0x1000u) ^ ((y << 8) & 0x4000u);
        case 4u:
            return ((x << 8) & 0x2000u) ^ ((x << 9) & 0x8000u) ^ ((y << 7) & 0x1000u) ^ ((y << 8) & 0x4000u);
        case 8u:
            return ((x << 8) & 0x2000u) ^ ((x << 9) & 0x8000u) ^ ((y << 8) & 0x1000u) ^ ((y << 9) & 0x4000u);
        default:
            return ((x << 9) & 0x2000u) ^ ((x << 10) & 0x8000u) ^ ((y << 8) & 0x1000u) ^ ((y << 9) & 0x4000u);
    }
}

inline uint agc_tiled_offset(AGC_PARAMS p, uint x, uint y, uint elementBytes) {
    if (p.family == AGC_FAMILY_LINEAR) return y * p.pitchBytes + x * elementBytes;
    uint swizzleX = x;
    uint swizzleY = y;
    uint blockX = 0u;
    uint blockY = 0u;
    if (p.tail != 0u) {
        swizzleX += p.tailX;
        swizzleY += p.tailY;
    } else {
        blockX = x >> p.blockLog2Width;
        blockY = y >> p.blockLog2Height;
    }
    uint within;
    if (p.family == AGC_FAMILY_EQUATION) {
        within = agc_equation_offset(p, swizzleX, swizzleY) ^ p.pipeBankXor;
    } else {
        within = agc_standard_offset(swizzleX, swizzleY, elementBytes);
        if (p.blockBytes > 4096u) within ^= agc_standard64_extra(swizzleX, swizzleY, elementBytes);
        within &= p.blockBytes - 1u;
    }
    return (blockY * p.blocksPerRow + blockX) * p.blockBytes + within;
}

#define AGC_TRANSFORM_NONE 0u
#define AGC_TRANSFORM_REPACK101111 1u
#define AGC_TRANSFORM_EXPAND_SNORM2101010 2u
#define AGC_TRANSFORM_EXPAND_SINT2101010 3u

inline uint agc_repack101111(uint word) {
    return ((word >> 21u) & 0x7ffu) | (((word >> 10u) & 0x7ffu) << 11u) | ((word & 0x3ffu) << 22u);
}

inline int agc_sign_extend(uint value, uint bits) {
    return static_cast<int>(value << (32u - bits)) >> (32u - bits);
}

inline uint agc_snorm_to_16(int value, int maxValue) {
    const int clamped = value < -maxValue ? -maxValue : value;
    const int scaled = clamped * 32767;
    const int rounded = (scaled >= 0 ? scaled + maxValue / 2 : scaled - maxValue / 2) / maxValue;
    return static_cast<uint>(rounded) & 0xffffu;
}

inline uint4 agc_transform(uint4 value, uint transform) {
    switch (transform) {
        case AGC_TRANSFORM_REPACK101111:
            return AGC_UINT4(agc_repack101111(value.x), 0u, 0u, 0u);
        case AGC_TRANSFORM_EXPAND_SNORM2101010: {
            const uint word = value.x;
            const uint red = agc_snorm_to_16(agc_sign_extend(word & 0x3ffu, 10u), 511);
            const uint green = agc_snorm_to_16(agc_sign_extend((word >> 10u) & 0x3ffu, 10u), 511);
            const uint blue = agc_snorm_to_16(agc_sign_extend((word >> 20u) & 0x3ffu, 10u), 511);
            const uint alpha = agc_snorm_to_16(agc_sign_extend(word >> 30u, 2u), 1);
            return AGC_UINT4(red | (green << 16u), blue | (alpha << 16u), 0u, 0u);
        }
        case AGC_TRANSFORM_EXPAND_SINT2101010: {
            const uint word = value.x;
            const uint red = static_cast<uint>(agc_sign_extend(word & 0x3ffu, 10u)) & 0xffffu;
            const uint green = static_cast<uint>(agc_sign_extend((word >> 10u) & 0x3ffu, 10u)) & 0xffffu;
            const uint blue = static_cast<uint>(agc_sign_extend((word >> 20u) & 0x3ffu, 10u)) & 0xffffu;
            const uint alpha = static_cast<uint>(agc_sign_extend(word >> 30u, 2u)) & 0xffffu;
            return AGC_UINT4(red | (green << 16u), blue | (alpha << 16u), 0u, 0u);
        }
        default:
            return value;
    }
}

#ifdef __METAL_VERSION__
inline uint4 agc_load_element(device const uchar* base, uint offset, uint elementBytes) {
    device const uchar* source = base + offset;
    switch (elementBytes) {
        case 1u: return AGC_UINT4(uint(source[0]), 0u, 0u, 0u);
        case 2u: return AGC_UINT4(uint(*reinterpret_cast<device const ushort*>(source)), 0u, 0u, 0u);
        case 4u: return AGC_UINT4(*reinterpret_cast<device const uint*>(source), 0u, 0u, 0u);
        case 8u: {
            device const uint* words = reinterpret_cast<device const uint*>(source);
            return AGC_UINT4(words[0], words[1], 0u, 0u);
        }
        default: {
            device const uint* words = reinterpret_cast<device const uint*>(source);
            return AGC_UINT4(words[0], words[1], words[2], words[3]);
        }
    }
}

inline void agc_store_element(device uchar* base, uint offset, uint elementBytes, uint4 value) {
    device uchar* destination = base + offset;
    switch (elementBytes) {
        case 1u:
            destination[0] = uchar(value.x);
            break;
        case 2u:
            *reinterpret_cast<device ushort*>(destination) = ushort(value.x);
            break;
        case 4u:
            *reinterpret_cast<device uint*>(destination) = value.x;
            break;
        case 8u: {
            device uint* words = reinterpret_cast<device uint*>(destination);
            words[0] = value.x;
            words[1] = value.y;
            break;
        }
        default: {
            device uint* words = reinterpret_cast<device uint*>(destination);
            words[0] = value.x;
            words[1] = value.y;
            words[2] = value.z;
            words[3] = value.w;
            break;
        }
    }
}
#else
#include <cstring>
inline uint4 agc_load_element(const uchar* base, uint offset, uint elementBytes) {
    uint words[4] = {0u, 0u, 0u, 0u};
    if (elementBytes == 1u) {
        words[0] = base[offset];
    } else if (elementBytes == 2u) {
        ushort half;
        std::memcpy(&half, base + offset, 2);
        words[0] = half;
    } else {
        std::memcpy(words, base + offset, elementBytes);
    }
    return uint4{words[0], words[1], words[2], words[3]};
}

inline void agc_store_element(uchar* base, uint offset, uint elementBytes, uint4 value) {
    const uint words[4] = {value.x, value.y, value.z, value.w};
    if (elementBytes == 1u) {
        base[offset] = static_cast<uchar>(words[0]);
    } else if (elementBytes == 2u) {
        const ushort half = static_cast<ushort>(words[0]);
        std::memcpy(base + offset, &half, 2);
    } else {
        std::memcpy(base + offset, words, elementBytes);
    }
}
#endif
