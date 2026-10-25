#include "MslState.hpp"
#include "BdaAbi.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include <bit>
#include <string>

namespace ShaderRecompiler::Msl {

namespace {

std::string Reason(BdaAbi::FaultReason reason) {
    return U32(static_cast<std::uint32_t>(reason));
}

std::string Replace(std::string text, std::string_view key, std::string_view value) {
    for (auto at = text.find(key); at != std::string::npos; at = text.find(key, at + value.size())) text.replace(at, key.size(), value);
    return text;
}

constexpr std::string_view BaseText = R"(#include <metal_stdlib>
using namespace metal;
struct aps5_guest_buffer { device uchar* data; uint size; uint flags; };
struct aps5_bda_header { uint version; uint count; uint entry_bytes; uint reserved; };
struct aps5_bda_range { ulong begin; ulong end; device uchar* base; uint permissions; uint reserved; };
struct aps5_ctx { device const aps5_bda_header* table; device atomic_uint* fault; device const aps5_guest_buffer* buffers; uint stage; bool stop; };
static inline bool aps5_nan(uint bits) { return (bits & 0x7fffffffu) > 0x7f800000u; }
static inline uint aps5_bits(float value) { return as_type<uint>(value); }
static inline float aps5_float(uint bits) { return as_type<float>(bits); }
static inline uint aps5_msb(uint value) { return value == 0u ? 0xffffffffu : 31u - clz(value); }
static inline uint aps5_lsb(uint value) { return value == 0u ? 0xffffffffu : ctz(value); }
static inline uint aps5_med3(uint a, uint b, uint c) { return max(min(a, b), min(max(a, b), c)); }
static inline int aps5_smed3(int a, int b, int c) { return max(min(a, b), min(max(a, b), c)); }
static inline float aps5_fract(float value) { float magnitude = abs(value); float part = magnitude - floor(magnitude); return value < 0.0f && part != 0.0f ? 1.0f - part : part; }
static inline float aps5_saturate(uint bits) { return as_type<float>(bits - 1u < 0x7f800000u ? min(bits, 0x3f800000u) : 0u); }
static inline uint aps5_pick(uint lhs, uint rhs, bool maximum) {
    uint either = lhs | rhs;
    if ((either & 0x7fffffffu) == 0u) return maximum ? (lhs & rhs) : either;
    float left = as_type<float>(lhs);
    float right = as_type<float>(rhs);
    return (maximum ? left >= right : left < right) ? lhs : rhs;
}
static inline float aps5_med3f(float x, float y, float z) {
    uint a = as_type<uint>(x);
    uint b = as_type<uint>(y);
    uint c = as_type<uint>(z);
    uint low = aps5_nan(b) ? a : aps5_pick(a, b, false);
    uint lowest = aps5_nan(c) ? low : aps5_pick(low, c, false);
    uint median = aps5_pick(low, aps5_pick(aps5_pick(a, b, true), c, false), true);
    return as_type<float>(aps5_nan(a) || aps5_nan(b) || aps5_nan(c) ? lowest : median);
}
static inline uint aps5_wqm(uint value) {
    uint any = value | (value >> 1u) | (value >> 2u) | (value >> 3u);
    any &= 0x11111111u;
    return any * 0xfu;
}
)";

constexpr std::string_view FloatClassText = R"(static inline bool aps5_class(uint bits, uint mask) {
    uint magnitude = bits & 0x7fffffffu;
    bool negative = (bits >> 31u) != 0u;
    uint kind;
    if (magnitude > 0x7f800000u) kind = (bits & 0x00400000u) != 0u ? 1u : 0u;
    else if (magnitude == 0x7f800000u) kind = negative ? 2u : 9u;
    else if (magnitude >= 0x00800000u) kind = negative ? 3u : 8u;
    else if (magnitude != 0u) kind = negative ? 4u : 7u;
    else kind = negative ? 5u : 6u;
    return ((mask >> kind) & 1u) != 0u;
}
)";

constexpr std::string_view HalfText = R"(static inline uint aps5_f32_to_f16(uint bits) {
#pragma METAL fp math_mode(relaxed)
    float value = as_type<float>(bits);
    uint sign = (bits >> 16u) & 0x8000u;
    if (aps5_nan(bits)) return sign | 0x7e00u | ((bits & 0x007fffffu) >> 13u);
    return sign | (uint(as_type<ushort>(half(value))) & 0x7fffu);
}
static inline uint aps5_f32_to_f16_rtz(uint bits) {
#pragma METAL fp math_mode(relaxed)
    float value = as_type<float>(bits);
    if (aps5_nan(bits)) return aps5_f32_to_f16(bits);
    uint nearest = aps5_f32_to_f16(bits);
    float back = float(as_type<half>(ushort(nearest)));
    return abs(back) > abs(value) ? nearest - 1u : nearest;
}
static inline float aps5_f16_to_f32(uint bits) {
    bits &= 0xffffu;
    if ((bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0u) return as_type<float>(uint(int(bits << 16u) >> 3) | 0x7f800000u);
    return float(as_type<half>(ushort(bits)));
}
static inline uint aps5_half2(uint bits, uint other, bool multiply) {
    half2 a = as_type<half2>(bits);
    half2 b = as_type<half2>(other);
    return as_type<uint>(multiply ? a * b : a + b);
}
)";

constexpr std::string_view ExactHalfText = R"(static inline uint aps5_f32_to_f16(uint bits) {
    uint sign = (bits >> 16u) & 0x8000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return sign | 0x7e00u | ((bits & 0x007fffffu) >> 13u);
    if (magnitude >= 0x477ff000u) return sign | 0x7c00u;
    if (magnitude <= 0x33000000u) return sign;
    uint exponent = magnitude >> 23u;
    uint mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    uint shift = exponent < 113u ? 126u - exponent : 13u;
    uint rounded = exponent < 113u ? mantissa >> shift : ((exponent - 112u) << 10u) + ((mantissa >> 13u) & 0x3ffu);
    uint rest = mantissa & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    return sign | (rounded + (rest > halfway || (rest == halfway && (rounded & 1u) != 0u) ? 1u : 0u));
}
static inline uint aps5_f32_to_f16_rtz(uint bits) {
    uint sign = (bits >> 16u) & 0x8000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return aps5_f32_to_f16(bits);
    if (magnitude == 0x7f800000u) return sign | 0x7c00u;
    if (magnitude >= 0x47800000u) return sign | 0x7bffu;
    if (magnitude < 0x33800000u) return sign;
    uint exponent = magnitude >> 23u;
    uint mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    return sign | (exponent < 113u ? mantissa >> (126u - exponent) : ((exponent - 112u) << 10u) | ((mantissa >> 13u) & 0x3ffu));
}
static inline float aps5_f16_to_f32(uint bits) {
    uint sign = (bits & 0x8000u) << 16u;
    uint exponent = (bits >> 10u) & 0x1fu;
    uint mantissa = bits & 0x3ffu;
    if (exponent == 31u) return as_type<float>(sign | 0x7f800000u | (mantissa << 13u));
    if (exponent != 0u) return as_type<float>(sign | ((exponent + 112u) << 23u) | (mantissa << 13u));
    if (mantissa == 0u) return as_type<float>(sign);
    uint msb = 31u - clz(mantissa);
    return as_type<float>(sign | ((msb + 103u) << 23u) | ((mantissa << (23u - msb)) & 0x7fffffu));
}
static inline uint aps5_half_op(uint a, uint b, bool multiply) {
#pragma METAL fp math_mode(safe)
    float x = aps5_f16_to_f32(a);
    float y = aps5_f16_to_f32(b);
    return aps5_f32_to_f16(as_type<uint>(multiply ? x * y : x + y));
}
static inline uint aps5_half2(uint bits, uint other, bool multiply) {
    return aps5_half_op(bits & 0xffffu, other & 0xffffu, multiply) | (aps5_half_op(bits >> 16u, other >> 16u, multiply) << 16u);
}
)";

constexpr std::string_view TableFaultText = R"(static inline void aps5_table_fault(thread aps5_ctx& c, ulong address, uint code, uint pc) {
    if (c.fault == nullptr) return;
    uint expected = 0u;
    while (!atomic_compare_exchange_weak_explicit(&c.fault[0], &expected, 1u, memory_order_relaxed, memory_order_relaxed)) {
        if (expected != 0u) return;
    }
    atomic_store_explicit(&c.fault[1], $REASON, memory_order_relaxed);
    atomic_store_explicit(&c.fault[2], uint(address), memory_order_relaxed);
    atomic_store_explicit(&c.fault[3], uint(address >> 32u), memory_order_relaxed);
    atomic_store_explicit(&c.fault[4], code & 0x7fffffffu, memory_order_relaxed);
    atomic_store_explicit(&c.fault[5], c.stage, memory_order_relaxed);
    atomic_store_explicit(&c.fault[6], pc, memory_order_relaxed);
    atomic_store_explicit(&c.fault[7], 0u, memory_order_relaxed);
    atomic_store_explicit(&c.fault[0], 2u, memory_order_relaxed);
}
)";

constexpr std::string_view FaultText = R"(static inline void aps5_fault(thread aps5_ctx& c, uint reason, ulong address, uint bytes, uint pc) {
    c.stop = true;
    if (c.fault == nullptr) return;
    for (;;) {
        uint expected = 0u;
        if (atomic_compare_exchange_weak_explicit(&c.fault[0], &expected, 1u, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&c.fault[1], reason, memory_order_relaxed);
            atomic_store_explicit(&c.fault[2], uint(address), memory_order_relaxed);
            atomic_store_explicit(&c.fault[3], uint(address >> 32u), memory_order_relaxed);
            atomic_store_explicit(&c.fault[4], bytes, memory_order_relaxed);
            atomic_store_explicit(&c.fault[5], c.stage, memory_order_relaxed);
            atomic_store_explicit(&c.fault[6], pc, memory_order_relaxed);
            atomic_store_explicit(&c.fault[7], 0u, memory_order_relaxed);
            atomic_store_explicit(&c.fault[0], 2u, memory_order_relaxed);
            return;
        }
        if (expected != 0u) return;
    }
}
)";

constexpr std::string_view BdaText = R"(static inline device uchar* aps5_bda_lookup(thread aps5_ctx& c, ulong address, uint bytes, uint need, thread uint& reason) {
    device const aps5_bda_header* table = c.table;
    if (table == nullptr || table->version != 1u || table->entry_bytes != 32u || table->reserved != 0u) { reason = $INVALID_TABLE; return nullptr; }
    ulong end = address + ulong(bytes);
    if (bytes == 0u || end <= address) { reason = $OVERFLOW; return nullptr; }
    device const aps5_bda_range* ranges = reinterpret_cast<device const aps5_bda_range*>(table + 1);
    uint lower = 0u;
    uint upper = table->count;
    while (lower < upper) {
        uint middle = lower + ((upper - lower) >> 1u);
        if (address < ranges[middle].begin) upper = middle;
        else lower = middle + 1u;
    }
    if (lower == 0u) { reason = $UNMAPPED; return nullptr; }
    aps5_bda_range range = ranges[lower - 1u];
    if (range.reserved != 0u || range.begin >= range.end) { reason = $INVALID_TABLE; return nullptr; }
    if (end > range.end) { reason = $UNMAPPED; return nullptr; }
    if ((range.permissions & need) != need) { reason = $PERMISSION; return nullptr; }
    if (range.base == nullptr) { reason = $INVALID_TABLE; return nullptr; }
    return range.base + (address - range.begin);
}
static inline bool aps5_bda_add(thread aps5_ctx& c, ulong base, ulong offset, bool subtract, uint pc, thread ulong& result) {
    result = subtract ? base - offset : base + offset;
    bool overflow = subtract ? offset > base : result < base;
    if (overflow) aps5_fault(c, $OVERFLOW, base, 0u, pc);
    return !overflow;
}
static inline uint aps5_bda_load(thread aps5_ctx& c, ulong address, uint bytes, uint pc) {
    uint reason = 0u;
    device uchar* pointer = aps5_bda_lookup(c, address, bytes, $READ, reason);
    if (pointer != nullptr && (address & ulong(bytes - 1u)) == 0ul) {
        if (bytes == 4u) return *reinterpret_cast<device uint*>(pointer);
        if (bytes == 2u) return uint(*reinterpret_cast<device ushort*>(pointer));
        return uint(*pointer);
    }
    uint value = 0u;
    for (uint byte = 0u; byte < bytes; ++byte) {
        device uchar* part = aps5_bda_lookup(c, address + byte, 1u, $READ, reason);
        if (part == nullptr) aps5_fault(c, reason, address + byte, 1u, pc);
        else value |= uint(*part) << (byte * 8u);
    }
    return value;
}
)";

constexpr std::string_view BdaStoreText = R"(static inline void aps5_bda_store(thread aps5_ctx& c, ulong address, uint bytes, uint value, uint pc) {
    uint reason = 0u;
    device uchar* pointer = aps5_bda_lookup(c, address, bytes, $WRITE, reason);
    if (pointer != nullptr && (address & ulong(bytes - 1u)) == 0ul) {
        if (bytes == 4u) *reinterpret_cast<device uint*>(pointer) = value;
        else if (bytes == 2u) *reinterpret_cast<device ushort*>(pointer) = ushort(value);
        else *pointer = uchar(value);
        aps5_note_write(c, address);
        return;
    }
    for (uint byte = 0u; byte < bytes; ++byte) {
        device uchar* part = aps5_bda_lookup(c, address + byte, 1u, $WRITE, reason);
        if (part == nullptr) {
            aps5_fault(c, reason, address + byte, 1u, pc);
        } else {
            *part = uchar(value >> (byte * 8u));
            aps5_note_write(c, address + byte);
        }
    }
}
static inline device atomic_uint* aps5_bda_atomic(thread aps5_ctx& c, ulong address, uint need, uint pc) {
    if ((address & 3ul) != 0ul) { aps5_fault(c, $UNALIGNED, address, 4u, pc); return nullptr; }
    uint reason = 0u;
    device uchar* pointer = aps5_bda_lookup(c, address, 4u, need, reason);
    if (pointer == nullptr) { aps5_fault(c, reason, address, 4u, pc); return nullptr; }
    aps5_note_write(c, address);
    return reinterpret_cast<device atomic_uint*>(pointer);
}
)";

constexpr std::string_view NoWritesText = R"(static inline void aps5_note_write(thread aps5_ctx&, ulong) {}
)";

constexpr std::string_view BdaWritesText = R"(static inline void aps5_note_write(thread aps5_ctx& c, ulong address) {
    if (c.fault == nullptr) return;
    uint key = uint(address >> $CHUNK_SHIFT) + 1u;
    uint bit = 1u << (uint(address >> $PAGE_SHIFT) & $PAGE_MASK);
    uint hash = (key * 0x9e3779b1u) >> $HASH_SHIFT;
    bool noted = false;
    for (uint probe = 0u; probe < $PROBES && !noted; ++probe) {
        uint slot = (hash + probe) & $SLOT_MASK;
        uint word = $SLOTS_WORD + 2u * slot;
        noted = atomic_load_explicit(&c.fault[word], memory_order_relaxed) == key;
        if (!noted) {
            uint expected = 0u;
            bool claimed = atomic_compare_exchange_weak_explicit(&c.fault[word], &expected, key, memory_order_relaxed, memory_order_relaxed);
            noted = claimed || expected == key;
            if (claimed) {
                uint index = atomic_fetch_add_explicit(&c.fault[$COUNT_WORD], 1u, memory_order_relaxed);
                if (index < $SLOTS) atomic_store_explicit(&c.fault[$LIST_WORD + index], slot, memory_order_relaxed);
            }
        }
        if (noted && (atomic_load_explicit(&c.fault[word + 1u], memory_order_relaxed) & bit) == 0u) atomic_fetch_or_explicit(&c.fault[word + 1u], bit, memory_order_relaxed);
    }
    if (!noted) atomic_store_explicit(&c.fault[$OVERFLOW_WORD], 1u, memory_order_relaxed);
}
)";

constexpr std::string_view DescriptorsText = R"(struct aps5_vsharp { ulong base; uint records; uint stride; uint word3; bool swizzle; uint index; bool valid; };
static inline aps5_vsharp aps5_vsharp_make(uint w0, uint w1, uint w2, uint w3, uint index) {
    aps5_vsharp r;
    r.base = ulong(w0) | (ulong(w1 & 0xffffu) << 32u);
    r.records = w2;
    r.stride = extract_bits(w1, 16u, 14u);
    r.word3 = w3;
    r.swizzle = r.stride != 0u && (w1 >> 31u) != 0u;
    r.index = index;
    r.valid = w2 != 0u && (w0 != 0u || (w1 & 0xffffu) != 0u);
    return r;
}
static inline bool aps5_vsharp_byte(aps5_vsharp r, uint offset, uint soffset, uint bytes, thread uint& byte) {
    uint shift = extract_bits(r.word3, 21u, 2u) + 3u;
    uint indices = 1u << shift;
    uint swizzled = ((r.index >> shift) * r.stride + (offset & ~3u)) * indices + ((r.index & (indices - 1u)) << 2u) + (offset & 3u);
    byte = (r.swizzle ? swizzled : r.index * r.stride + offset) + soffset;
    uint mode = extract_bits(r.word3, 28u, 2u);
    bool inBounds;
    if (mode == 0u) {
        inBounds = r.index < r.records && offset < r.stride;
    } else if (mode == 1u) {
        inBounds = r.index < r.records;
    } else if (mode == 2u) {
        inBounds = r.records != 0u;
    } else {
        uint raw = r.records - soffset;
        bool fitsStride = r.stride >= bytes && offset <= r.stride - bytes;
        bool fitsRaw = raw >= bytes && offset <= raw - bytes;
        inBounds = soffset <= r.records && (r.swizzle ? (r.index < raw && fitsStride) : fitsRaw);
    }
    return r.valid && extract_bits(r.word3, 12u, 7u) != 0u && inBounds;
}
)";

constexpr std::string_view DirectBuffersText = R"(static inline uint aps5_buffer_load(thread aps5_ctx& c, uint element, aps5_vsharp r, uint byte, uint bytes) {
    aps5_guest_buffer buffer = c.buffers[element];
    if (ulong(byte) + ulong(bytes) > ulong(buffer.size)) return 0u;
    device uchar* pointer = buffer.data + byte;
    if (((r.base + ulong(byte)) & ulong(bytes - 1u)) == 0ul) {
        if (bytes == 4u) return *reinterpret_cast<device uint*>(pointer);
        if (bytes == 2u) return uint(*reinterpret_cast<device ushort*>(pointer));
        return uint(*pointer);
    }
    uint value = 0u;
    for (uint part = 0u; part < bytes; ++part) value |= uint(pointer[part]) << (part * 8u);
    return value;
}
static inline void aps5_buffer_store(thread aps5_ctx& c, uint element, aps5_vsharp r, uint byte, uint bytes, uint value) {
    aps5_guest_buffer buffer = c.buffers[element];
    if (ulong(byte) + ulong(bytes) > ulong(buffer.size)) return;
    device uchar* pointer = buffer.data + byte;
    if (((r.base + ulong(byte)) & ulong(bytes - 1u)) == 0ul) {
        if (bytes == 4u) *reinterpret_cast<device uint*>(pointer) = value;
        else if (bytes == 2u) *reinterpret_cast<device ushort*>(pointer) = ushort(value);
        else *pointer = uchar(value);
        return;
    }
    for (uint part = 0u; part < bytes; ++part) pointer[part] = uchar(value >> (part * 8u));
}
static inline device atomic_uint* aps5_buffer_atomic(thread aps5_ctx& c, uint element, aps5_vsharp r, uint byte, uint pc) {
    ulong guest = r.base + ulong(byte);
    if ((guest & 3ul) != 0ul) { aps5_fault(c, $UNALIGNED, guest, 4u, pc); return nullptr; }
    aps5_guest_buffer buffer = c.buffers[element];
    if (ulong(byte) + 4ul > ulong(buffer.size)) return nullptr;
    return reinterpret_cast<device atomic_uint*>(buffer.data + byte);
}
)";

constexpr std::string_view GpuBuffersText = R"(static inline aps5_vsharp aps5_vsharp_gpu(thread aps5_ctx& c, uint w0, uint w1, uint w2, uint w3, uint index, uint pc) {
    aps5_vsharp r = aps5_vsharp_make(w0, w1, w2, w3, index);
    bool invalid = extract_bits(w3, 30u, 2u) != 0u;
    if (invalid) aps5_fault(c, $INVALID_DESCRIPTOR, r.base, 0u, pc);
    r.valid = !invalid && r.base != 0ul;
    return r;
}
static inline uint aps5_vsharp_load(thread aps5_ctx& c, aps5_vsharp r, uint byte, uint bytes, uint pc) {
    ulong address = 0ul;
    if (!aps5_bda_add(c, r.base, ulong(byte), false, pc, address)) return 0u;
    return aps5_bda_load(c, address, bytes, pc);
}
static inline void aps5_vsharp_store(thread aps5_ctx& c, aps5_vsharp r, uint byte, uint bytes, uint value, uint pc) {
    ulong address = 0ul;
    if (aps5_bda_add(c, r.base, ulong(byte), false, pc, address)) aps5_bda_store(c, address, bytes, value, pc);
}
static inline device atomic_uint* aps5_vsharp_atomic(thread aps5_ctx& c, aps5_vsharp r, uint byte, uint pc) {
    ulong address = 0ul;
    if (!aps5_bda_add(c, r.base, ulong(byte), false, pc, address)) return nullptr;
    return aps5_bda_atomic(c, address, $READ_WRITE, pc);
}
)";

constexpr std::string_view FormatText = R"(static inline uint aps5_fmt_unorm(uint value, uint bits) {
    if (value == 0u) return 0u;
    uint msb = 31u - clz(value);
    uint leading = bits - 1u - msb;
    uint rotated = ((value << leading) | (value >> (bits - leading))) & ((1u << bits) - 1u);
    uint pattern = rotated << (32u - bits);
    for (uint shift = bits; shift < 32u; shift *= 2u) pattern |= pattern >> shift;
    return ((msb + 126u - bits) << 23u) + (pattern >> 8u) + ((pattern >> 7u) & 1u);
}
static inline uint aps5_fmt_half(uint value) {
    uint sign = (value & 0x8000u) << 16u;
    uint exponent = (value >> 10u) & 0x1fu;
    uint mantissa = value & 0x3ffu;
    if (exponent == 31u) return sign | 0x7f800000u | (mantissa << 13u);
    if (exponent != 0u) return sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    if (mantissa == 0u) return sign;
    uint msb = 31u - clz(mantissa);
    return sign | ((msb + 103u) << 23u) | ((mantissa << (23u - msb)) & 0x7fffffu);
}
static inline uint aps5_fmt_ufloat(uint raw, uint bits) {
    uint width = bits == 11u ? 6u : 5u;
    uint mantissa = raw & ((1u << width) - 1u);
    uint exponent = (raw >> width) & 0x1fu;
    if (exponent == 31u) return 0x7f800000u | (mantissa << (23u - width));
    if (exponent != 0u) return ((exponent + 112u) << 23u) | (mantissa << (23u - width));
    if (mantissa == 0u) return 0u;
    uint msb = 31u - clz(mantissa);
    return ((msb + 113u - width) << 23u) | ((mantissa << (23u - msb)) & 0x7fffffu);
}
static inline uint aps5_fmt_decode(uint raw, uint type, uint bits) {
    switch (type) {
    case 1u:
    case 2u: return raw;
    case 3u: return aps5_fmt_unorm(raw, bits);
    case 4u: {
        bool negative = int(raw) < 0;
        uint magnitude = min(negative ? 0u - raw : raw, (1u << (bits - 1u)) - 1u);
        return (negative ? 0x80000000u : 0u) | aps5_fmt_unorm(magnitude, bits - 1u);
    }
    case 5u: return as_type<uint>(float(raw));
    case 6u: return as_type<uint>(float(int(raw)));
    case 7u: return bits == 32u ? raw : bits == 16u ? aps5_fmt_half(raw) : aps5_fmt_ufloat(raw, bits);
    default: return 0u;
    }
}
static inline uint aps5_fmt_norm16(uint value, uint maximum) {
    if (value == 0u) return 0u;
    uint first = 9u + (32u - clz(maximum)) - (31u - clz(value));
    if ((value << first) < 1024u * maximum) first += 1u;
    uint shift = min(first, 24u);
    uint scaled = value << shift;
    uint rounded = scaled / maximum + (scaled % maximum > (maximum >> 1u) ? 1u : 0u);
    return ((25u - shift) << 10u) + rounded - 1024u;
}
static inline uint aps5_fmt_int16(uint value) {
    if (value == 0u) return 0u;
    uint msb = 31u - clz(value);
    bool wide = msb > 10u;
    uint right = wide ? msb - 10u : 0u;
    uint quotient = (value << (wide ? 0u : 10u - msb)) >> right;
    uint unit = 1u << right;
    uint remainder = value & (unit - 1u);
    uint halfway = unit >> 1u;
    bool up = wide && (remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u));
    return min(((msb + 15u) << 10u) + quotient + (up ? 1u : 0u) - 1024u, 0x7bffu);
}
static inline uint aps5_fmt_small(uint bits, uint width, bool signedFormat) {
    uint infinity = 0x1fu << width;
    bool negative = (bits & 0x80000000u) != 0u;
    uint exponent = (bits >> 23u) & 0xffu;
    uint mantissa = bits & 0x7fffffu;
    uint top = mantissa >> (23u - width);
    int biased = int(exponent) - 112;
    uint magnitude = 0u;
    if (exponent == 0xffu) magnitude = infinity | (top == 0u && mantissa != 0u ? 1u : top);
    else if (biased >= 31) magnitude = infinity - 1u;
    else if (biased >= 1) magnitude = (uint(biased) << width) | top;
    else if (biased >= -int(width)) magnitude = (mantissa | 0x800000u) >> min(24u - width - uint(biased), 31u);
    if (!signedFormat) return negative && !(exponent == 0xffu && mantissa != 0u) ? 0u : magnitude;
    return (negative ? 1u << (5u + width) : 0u) | magnitude;
}
static inline uint aps5_fmt_norm_store(uint bits, uint maximum, bool signedFormat) {
    uint magnitude = bits & 0x7fffffffu;
    uint exponent = (bits >> 23u) & 0xffu;
    bool denormal = exponent == 0u;
    uint mantissa = (bits & 0x7fffffu) | (denormal ? 0u : 0x800000u);
    uint extra = (denormal ? 149u : 150u - exponent) - 24u;
    uint low = mantissa * maximum;
    uint high = mulhi(mantissa, maximum);
    uint fraction = low & 0xffffffu;
    uint fixed = (((low >> 24u) | (high << 8u)) << 16u) | (fraction >> 8u);
    bool sticky = (fraction & 0xffu) != 0u;
    uint safe = min(extra, 16u);
    uint quotient = (fixed >> 16u) >> safe;
    uint remainder = fixed - ((quotient << 16u) << safe);
    uint halfway = (1u << 15u) << safe;
    bool up = remainder > halfway || (remainder == halfway && (sticky || (quotient & 1u) != 0u));
    uint result = extra > 16u ? 0u : quotient + (up ? 1u : 0u);
    if (magnitude >= 0x3f800000u && magnitude <= 0x7f800000u) result = maximum;
    bool positive = signedFormat ? magnitude != 0u && magnitude <= 0x7f800000u : int(bits) > 0 && bits <= 0x7f800000u;
    if (!positive) result = 0u;
    return signedFormat && (bits & 0x80000000u) != 0u ? 0u - result : result;
}
static inline uint aps5_fmt_decode16(uint raw, uint type, uint bits) {
    if (type != 7u && bits > 16u) {
        if (type == 1u) return min(raw, 0xffffu);
        if (type == 2u) return uint(clamp(int(raw), -32768, 32767)) & 0xffffu;
        return 0u;
    }
    bool negative = int(raw) < 0;
    uint magnitude = negative ? 0u - raw : raw;
    uint sign = negative ? 0x8000u : 0u;
    switch (type) {
    case 1u:
    case 2u: return raw & 0xffffu;
    case 3u: return aps5_fmt_norm16(raw, (1u << bits) - 1u);
    case 4u: return sign | aps5_fmt_norm16(min(magnitude, (1u << (bits - 1u)) - 1u), (1u << (bits - 1u)) - 1u);
    case 5u: return aps5_fmt_int16(raw);
    case 6u: return sign | aps5_fmt_int16(magnitude);
    case 7u: return bits == 16u ? raw & 0xffffu : aps5_fmt_small(aps5_fmt_decode(raw, 7u, bits), 10u, true);
    default: return 0u;
    }
}
static inline uint aps5_fmt_encode(uint data, uint type, uint bits) {
    uint mask = bits >= 32u ? 0xffffffffu : (1u << bits) - 1u;
    switch (type) {
    case 1u: return bits >= 32u ? data : min(data, mask);
    case 2u: return bits >= 32u ? data : uint(clamp(int(data), int(~(mask >> 1u)), int(mask >> 1u))) & mask;
    case 3u:
    case 5u: return aps5_fmt_norm_store(data, mask, false);
    case 4u:
    case 6u: return aps5_fmt_norm_store(data, mask >> 1u, true) & mask;
    case 7u: return bits == 32u ? data : bits == 16u ? aps5_fmt_small(data, 10u, true) : aps5_fmt_small(data, bits - 5u, false);
    default: return 0u;
    }
}
static inline uint aps5_fmt_encode16(uint value, uint type, uint bits) {
    if (type == 1u) return aps5_fmt_encode(value, type, bits);
    if (type == 2u) return aps5_fmt_encode(uint(int(value << 16u) >> 16u), type, bits);
    if (type == 7u && bits == 16u) return value;
    return aps5_fmt_encode(aps5_fmt_half(value), type, bits);
}
$TABLE
static inline uint3 aps5_fmt_info(uint format) {
    if (format == 0u || format > $FORMATS) return uint3(0u);
    return uint3(aps5_fmt_table[format * 3u], aps5_fmt_table[format * 3u + 1u], aps5_fmt_table[format * 3u + 2u]);
}
)";

constexpr std::string_view FormatAccessText = R"(static inline uint4 aps5_fmt_loadNAME(thread aps5_ctx& c, uint element, aps5_vsharp r, uint offset, uint soffset, uint3 info, uint sel, uint outputs, bool d16, bool check, uint pc) {
    uint count = info.x & 7u;
    if (count == 0u) {
        BAD;
        return uint4(0u);
    }
    bool packed = (info.x & 8u) != 0u;
    uint type = extract_bits(info.x, 4u, 4u);
    uint required = 0u;
    for (uint output = 0u; output < outputs; ++output) {
        uint source = extract_bits(sel, output * 3u, 3u);
        if (check && (source == 2u || source == 3u)) BAD;
        if (source >= 4u) required |= 1u << ((source - 4u) % count);
    }
    uint bytes[4] = {0u, 0u, 0u, 0u};
    bool inBounds = true;
    for (uint part = 0u; part < count; ++part) {
        if ((required & (1u << part)) == 0u) continue;
        uint width = packed ? 4u : extract_bits(info.x, 8u + part * 6u, 6u) >> 3u;
        inBounds = inBounds && aps5_vsharp_byte(r, offset + extract_bits(info.y, part * 8u, 8u), soffset, width, bytes[part]);
    }
    uint values[4] = {0u, 0u, 0u, 0u};
    for (uint part = 0u; inBounds && part < count; ++part) {
        if ((required & (1u << part)) == 0u) continue;
        uint bits = extract_bits(info.x, 8u + part * 6u, 6u);
        uint raw = READ(bytes[part], packed ? 4u : bits >> 3u);
        if (bits != 32u) {
            uint shift = packed ? extract_bits(info.z, part * 8u, 8u) : 0u;
            raw = type == 2u || type == 4u || type == 6u ? uint(extract_bits(int(raw), shift, bits)) : extract_bits(raw, shift, bits);
        }
        values[part] = d16 ? aps5_fmt_decode16(raw, type, bits) : aps5_fmt_decode(raw, type, bits);
    }
    uint one = type == 1u || type == 2u ? 1u : d16 ? 0x3c00u : 0x3f800000u;
    uint4 result = uint4(0u);
    for (uint output = 0u; output < outputs; ++output) {
        uint source = extract_bits(sel, output * 3u, 3u);
        result[output] = source == 0u ? 0u : source == 1u ? one : values[(source - 4u) % count];
    }
    return result;
}
static inline void aps5_fmt_storeNAME(thread aps5_ctx& c, uint element, aps5_vsharp r, uint offset, uint soffset, uint3 info, uint components, uint4 data, bool d16, uint pc) {
    uint count = info.x & 7u;
    if (count == 0u) {
        BAD;
        return;
    }
    bool packed = (info.x & 8u) != 0u;
    uint type = extract_bits(info.x, 4u, 4u);
    uint bytes[4] = {0u, 0u, 0u, 0u};
    bool inBounds = true;
    for (uint part = 0u; part < count; ++part) {
        uint width = packed ? 4u : extract_bits(info.x, 8u + part * 6u, 6u) >> 3u;
        inBounds = inBounds && aps5_vsharp_byte(r, offset + extract_bits(info.y, part * 8u, 8u), soffset, width, bytes[part]);
    }
    if (!inBounds) return;
    uint word = 0u;
    for (uint part = 0u; part < count; ++part) {
        uint bits = extract_bits(info.x, 8u + part * 6u, 6u);
        uint encoded = part >= components ? 0u : d16 ? aps5_fmt_encode16(data[part], type, bits) : aps5_fmt_encode(data[part], type, bits);
        if (packed) word |= (encoded & ((1u << bits) - 1u)) << extract_bits(info.z, part * 8u, 8u);
        else WRITE(bytes[part], bits >> 3u, encoded);
    }
    if (packed) WRITE(bytes[0], 4u, word);
}
)";

constexpr std::string_view LocalMemoryText = R"(static inline uint aps5_readNAME(SPACE uint* words, uint count, uint byte, uint bytes) {
    uint last = byte + bytes - 1u;
    if (last < byte || (last >> 2u) >= count) return 0u;
    uint shift = (byte & 3u) * 8u;
    uint mask = bytes == 4u ? 0xffffffffu : (1u << (bytes * 8u)) - 1u;
    uint index = byte >> 2u;
    if ((byte & 3u) + bytes <= 4u) return (words[index] >> shift) & mask;
    ulong pair = ulong(words[index]) | (ulong(words[index + 1u]) << 32u);
    return uint(pair >> shift) & mask;
}
static inline void aps5_writeNAME(SPACE uint* words, uint count, uint byte, uint bytes, uint value) {
    uint last = byte + bytes - 1u;
    if (last < byte || (last >> 2u) >= count) return;
    uint shift = (byte & 3u) * 8u;
    ulong mask = (bytes == 4u ? 0xfffffffful : ((1ul << (bytes * 8u)) - 1ul)) << shift;
    ulong data = (ulong(value) << shift) & mask;
    uint index = byte >> 2u;
    if (uint(mask) == 0xffffffffu) {
        words[index] = uint(data);
        return;
    }
    if (uint(mask) != 0u) MERGE(index, uint(mask), uint(data));
    if (uint(mask >> 32u) != 0u) MERGE(index + 1u, uint(mask >> 32u), uint(data >> 32u));
}
)";

constexpr std::string_view AtomicsText = R"(static inline uint aps5_float_key(uint bits) { return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u; }
static inline bool aps5_signaling(uint bits) { return aps5_nan(bits) && (bits & 0x00400000u) == 0u; }
static inline uint aps5_float_minmax(uint old, uint source, bool maximum) {
    bool better = maximum ? aps5_float_key(source) > aps5_float_key(old) : aps5_float_key(source) < aps5_float_key(old);
    uint result = better ? source : old;
    if (aps5_nan(old)) result = source;
    if (aps5_nan(source)) result = old;
    if (aps5_nan(source) && aps5_nan(old)) result = source;
    if (aps5_signaling(old)) result = old | 0x00400000u;
    if (aps5_signaling(source)) result = source | 0x00400000u;
    return result;
}
static inline uint aps5_flush(uint bits, bool enabled) { return enabled && (bits & 0x7fffffffu) < 0x00800000u ? bits & 0x80000000u : bits; }
static inline bool aps5_float_equal(uint left, uint right) { return (left == right && !aps5_nan(left)) || ((left | right) & 0x7fffffffu) == 0u; }
static inline uint aps5_update(uint op, uint old, uint a, uint b) {
    switch (op) {
    case 0u: return a;
    case 1u: return old == b ? a : old;
    case 2u: return old + a;
    case 3u: return old - a;
    case 4u: return uint(min(int(old), int(a)));
    case 5u: return min(old, a);
    case 6u: return uint(max(int(old), int(a)));
    case 7u: return max(old, a);
    case 8u: return old & a;
    case 9u: return old | a;
    case 10u: return old ^ a;
    case 11u: return old >= a ? 0u : old + 1u;
    case 12u: return (old == 0u || old > a) ? a : old - 1u;
    case 13u: return old >= a ? old - a : 0u;
    case 14u: return a - old;
    case 15u: return aps5_float_minmax(aps5_flush(old, b != 0u), aps5_flush(a, b != 0u), false);
    case 16u: return aps5_float_minmax(aps5_flush(old, b != 0u), aps5_flush(a, b != 0u), true);
    case 17u: {
        uint sum = aps5_flush(as_type<uint>(as_type<float>(aps5_flush(old, (b & 1u) != 0u)) + as_type<float>(aps5_flush(a, (b & 1u) != 0u))), (b & 2u) != 0u);
        if (aps5_nan(sum)) sum = 0xffc00000u;
        if (aps5_nan(old)) sum = old | 0x00400000u;
        if (aps5_nan(a)) sum = a | 0x00400000u;
        return sum;
    }
    case 18u: return (old & ~a) | b;
    case 19u: return old >= a ? old - a : old + b;
    case 20u: return aps5_float_equal(aps5_flush(old, false), b) ? a : old;
    case 21u: return aps5_float_equal(aps5_flush(old, true), aps5_flush(b, true)) ? a : old;
    }
    return old;
}
)";

constexpr std::string_view AtomicSpaceText = R"(static inline uint aps5_atomicNAME(SPACE atomic_uint* pointer, uint op, uint a, uint b) {
    switch (op) {
    case 0u: return atomic_exchange_explicit(pointer, a, memory_order_relaxed);
    case 2u: return atomic_fetch_add_explicit(pointer, a, memory_order_relaxed);
    case 3u: return atomic_fetch_sub_explicit(pointer, a, memory_order_relaxed);
    case 4u: return uint(atomic_fetch_min_explicit(reinterpret_cast<SPACE atomic_int*>(pointer), int(a), memory_order_relaxed));
    case 5u: return atomic_fetch_min_explicit(pointer, a, memory_order_relaxed);
    case 6u: return uint(atomic_fetch_max_explicit(reinterpret_cast<SPACE atomic_int*>(pointer), int(a), memory_order_relaxed));
    case 7u: return atomic_fetch_max_explicit(pointer, a, memory_order_relaxed);
    case 8u: return atomic_fetch_and_explicit(pointer, a, memory_order_relaxed);
    case 9u: return atomic_fetch_or_explicit(pointer, a, memory_order_relaxed);
    case 10u: return atomic_fetch_xor_explicit(pointer, a, memory_order_relaxed);
    }
    uint old = atomic_load_explicit(pointer, memory_order_relaxed);
    for (;;) {
        uint next = aps5_update(op, old, a, b);
        if (next == old) return old;
        if (atomic_compare_exchange_weak_explicit(pointer, &old, next, memory_order_relaxed, memory_order_relaxed)) return old;
    }
}
)";

constexpr std::string_view Atomic64Text = R"(static inline ulong aps5_update64(uint op, ulong old, ulong a, ulong b) {
    switch (op) {
    case 0u: return a;
    case 1u: return old == b ? a : old;
    case 2u: return old + a;
    case 3u: return old - a;
    case 4u: return ulong(min(long(old), long(a)));
    case 5u: return min(old, a);
    case 6u: return ulong(max(long(old), long(a)));
    case 7u: return max(old, a);
    case 8u: return old & a;
    case 9u: return old | a;
    case 10u: return old ^ a;
    case 11u: return old >= a ? 0ul : old + 1ul;
    case 12u: return (old == 0ul || old > a) ? a : old - 1ul;
    case 14u: return a - old;
    }
    return old;
}
)";

constexpr std::string_view Atomic64SpaceText = R"(static inline ulong aps5_load64NAME(SPACE atomic_uint* p) {
    uint high = atomic_load_explicit(p + 1, memory_order_relaxed);
    uint low = 0u;
    for (;;) {
        low = atomic_load_explicit(p, memory_order_relaxed);
        uint again = atomic_load_explicit(p + 1, memory_order_relaxed);
        if (again == high) break;
        high = again;
    }
    return (ulong(high) << 32u) | ulong(low);
}
static inline ulong aps5_atomic64NAME(SPACE atomic_uint* p, uint op, ulong a, bool wanted) {
    const uint low = uint(a);
    const uint high = uint(a >> 32u);
    uint oldLow = 0u;
    uint oldHigh = 0u;
    switch (op) {
    case 2u:
        oldLow = atomic_fetch_add_explicit(p, low, memory_order_relaxed);
        oldHigh = atomic_fetch_add_explicit(p + 1, high + (oldLow + low < oldLow ? 1u : 0u), memory_order_relaxed);
        break;
    case 3u:
        oldLow = atomic_fetch_sub_explicit(p, low, memory_order_relaxed);
        oldHigh = atomic_fetch_sub_explicit(p + 1, high + (oldLow < low ? 1u : 0u), memory_order_relaxed);
        break;
    case 8u:
        oldLow = atomic_fetch_and_explicit(p, low, memory_order_relaxed);
        oldHigh = atomic_fetch_and_explicit(p + 1, high, memory_order_relaxed);
        break;
    case 9u:
        oldLow = atomic_fetch_or_explicit(p, low, memory_order_relaxed);
        oldHigh = atomic_fetch_or_explicit(p + 1, high, memory_order_relaxed);
        break;
    case 10u:
        oldLow = atomic_fetch_xor_explicit(p, low, memory_order_relaxed);
        oldHigh = atomic_fetch_xor_explicit(p + 1, high, memory_order_relaxed);
        break;
MINMAX    }
    return (ulong(oldHigh) << 32u) | ulong(oldLow);
}
)";

constexpr std::string_view Atomic64MinMaxText = R"(    case 5u:
    case 7u: {
        ulong old = wanted ? aps5_load64(p) : 0ul;
        if (op == 5u) atomic_min_explicit(reinterpret_cast<device atomic_ulong*>(p), a, memory_order_relaxed);
        else atomic_max_explicit(reinterpret_cast<device atomic_ulong*>(p), a, memory_order_relaxed);
        return old;
    }
)";

constexpr std::string_view Atomic64PointerText = R"(static inline device atomic_uint* aps5_bda_atomic8(thread aps5_ctx& c, ulong address, uint need, uint pc) {
    if ((address & 7ul) != 0ul) { aps5_fault(c, $UNALIGNED, address, 8u, pc); return nullptr; }
    uint reason = 0u;
    device uchar* pointer = aps5_bda_lookup(c, address, 8u, need, reason);
    if (pointer == nullptr) { aps5_fault(c, reason, address, 8u, pc); return nullptr; }
    aps5_note_write(c, address);
    return reinterpret_cast<device atomic_uint*>(pointer);
}
)";

constexpr std::string_view Atomic64DirectText = R"(static inline device atomic_uint* aps5_buffer_atomic8(thread aps5_ctx& c, uint element, aps5_vsharp r, uint byte, uint pc) {
    ulong guest = r.base + ulong(byte);
    if ((guest & 7ul) != 0ul) { aps5_fault(c, $UNALIGNED, guest, 8u, pc); return nullptr; }
    aps5_guest_buffer buffer = c.buffers[element];
    if (ulong(byte) + 8ul > ulong(buffer.size)) return nullptr;
    return reinterpret_cast<device atomic_uint*>(buffer.data + byte);
}
)";

constexpr std::string_view Atomic64GpuText = R"(static inline device atomic_uint* aps5_vsharp_atomic8(thread aps5_ctx& c, aps5_vsharp r, uint byte, uint pc) {
    ulong address = 0ul;
    if (!aps5_bda_add(c, r.base, ulong(byte), false, pc, address)) return nullptr;
    return aps5_bda_atomic8(c, address, $READ_WRITE, pc);
}
)";

constexpr std::string_view WaveText = R"(static inline uint aps5_ballot32(bool predicate) { return uint(static_cast<simd_vote::vote_t>(simd_ballot(predicate))); }
)";

constexpr std::string_view VertexWaveText = R"(#define simd_shuffle(value, lane) (value)
#define simd_shuffle_xor(value, mask) (value)
#define simd_shuffle_up(value, delta) (value)
#define simd_shuffle_down(value, delta) (value)
#define simd_broadcast_first(value) (value)
#define simd_any(predicate) (predicate)
static inline uint aps5_ballot32(bool predicate) { return predicate ? 1u : 0u; }
)";

constexpr std::string_view PermuteText = R"(static inline uint aps5_permute(uint value, uint address, bool active, uint lane) {
    const uint request = active ? (((address >> 2u) & 31u) | 0x100u) : 0u;
    uint result = 0u;
    for (ushort source = 0; source < 32; ++source) {
        const uint target = simd_shuffle(request, source);
        const uint sent = simd_shuffle(value, source);
        if ((target & 0x100u) != 0u && (target & 31u) == lane) result = sent;
    }
    return result;
}
)";

#define APS5_MSL_SOURCE(...) #__VA_ARGS__
constexpr std::string_view SoftFloatText =
#include "MslSoftFloat64.inc"
    ;
#undef APS5_MSL_SOURCE

constexpr std::string_view Dot2Text = R"(static inline uint aps5_dot2_widen(uint pair, uint shift) {
#pragma METAL fp math_mode(safe)
    uint bits = (pair >> shift) & 0xffffu;
    uint magnitude = bits & 0x7fffu;
    uint normal = (magnitude << 13u) + (112u << 23u);
    uint subnormal = as_type<uint>(float(magnitude) * as_type<float>(0x33800000u));
    return (magnitude < 0x400u ? subnormal : normal) | ((bits & 0x8000u) << 16u);
}
static inline uint2 aps5_dot2_fast(uint a, uint b, uint c) {
#pragma METAL fp math_mode(safe)
    uint magnitudeC = c & 0x7fffffffu;
    bool flushedC = magnitudeC < 0x00800000u;
    bool fast = magnitudeC < 0x7f800000u && (flushedC || magnitudeC >= 0x0d800000u);
    fast = fast && (a & 0x7c00u) != 0x7c00u && (a & 0x7c000000u) != 0x7c000000u && (b & 0x7c00u) != 0x7c00u && (b & 0x7c000000u) != 0x7c000000u;
    float valueC = as_type<float>(flushedC ? (c & 0x80000000u) : c);
    float productLow = as_type<float>(aps5_dot2_widen(a, 0u)) * as_type<float>(aps5_dot2_widen(b, 0u));
    float productHigh = as_type<float>(aps5_dot2_widen(a, 16u)) * as_type<float>(aps5_dot2_widen(b, 16u));
    float upper = productHigh + valueC;
    float upperRight = upper - productHigh;
    float upperLeft = upper - upperRight;
    float upperLeftError = productHigh - upperLeft;
    float upperRightError = valueC - upperRight;
    float upperError = upperLeftError + upperRightError;
    float total = productLow + upper;
    float totalRight = total - productLow;
    float totalLeft = total - totalRight;
    float totalLeftError = productLow - totalLeft;
    float totalRightError = upper - totalRight;
    float totalError = totalLeftError + totalRightError;
    float nearest = totalError + upperError;
    float nearestRight = nearest - totalError;
    float nearestLeft = nearest - nearestRight;
    float nearestLeftError = totalError - nearestLeft;
    float nearestRightError = upperError - nearestRight;
    float error = nearestLeftError + nearestRightError;
    uint nearestBits = as_type<uint>(nearest);
    uint errorBits = as_type<uint>(error);
    bool inexact = (errorBits & 0x7fffffffu) != 0u;
    bool even = (nearestBits & 1u) == 0u;
    bool sameSign = ((nearestBits ^ errorBits) & 0x80000000u) == 0u;
    uint odd = nearestBits + (sameSign ? 1u : 0xffffffffu);
    float roundedToOdd = as_type<float>(inexact && even ? odd : nearestBits);
    uint result = as_type<uint>(total + roundedToOdd);
    uint magnitude = result & 0x7fffffffu;
    fast = fast && magnitude >= 0x0d800000u && magnitude < 0x7f000000u;
    return uint2(result, fast ? 1u : 0u);
}
static inline uint aps5_dot2_f32_f16(uint a, uint b, uint c) {
    uint2 fast = aps5_dot2_fast(a, b, c);
    return fast.y != 0u ? fast.x : aps5_dot2_exact(a, b, c);
}
)";

constexpr std::string_view StrictTrigText = R"(static inline uint aps5_s_tiny(uint a, uint b, uint r) {
#pragma METAL fp math_mode(safe)
    uint ea = a & 0x7f800000u;
    uint eb = b & 0x7f800000u;
    if (ea == 0u || eb == 0u || ea == 0x7f800000u || eb == 0x7f800000u) return r;
    uint scaled = as_type<uint>((as_type<float>(a) * as_type<float>(0x5f800000u)) * as_type<float>(b));
    uint magnitude = scaled & 0x7fffffffu;
    return magnitude >= 0x20800000u ? (scaled & 0x80000000u) | (magnitude - 0x20000000u) : r;
}
static inline uint aps5_s_tiny_fma(uint a, uint b, uint c, uint r) {
#pragma METAL fp math_mode(safe)
    if ((c & 0x7fffffffu) == 0u) return aps5_s_tiny(a, b, r);
    if ((a & 0x7fffffffu) >= 0x5f800000u || (b & 0x7f800000u) == 0x7f800000u || (c & 0x7fffffffu) >= 0x5f800000u) return r;
    float scale = as_type<float>(0x5f800000u);
    uint scaled = as_type<uint>(fma(as_type<float>(a) * scale, as_type<float>(b), as_type<float>(c) * scale));
    uint magnitude = scaled & 0x7fffffffu;
    return magnitude >= 0x20800000u && magnitude < 0x7f800000u ? (scaled & 0x80000000u) | (magnitude - 0x20000000u) : r;
}
static inline float aps5_s_sin_series(float t) {
#pragma METAL fp math_mode(safe)
    float square = t * t;
    float p = as_type<float>(0xc17183a8u);
    p = fma(p, square, as_type<float>(0x42283c1au));
    p = fma(p, square, as_type<float>(0xc2996966u));
    p = fma(p, square, as_type<float>(0x42a335e3u));
    p = fma(p, square, as_type<float>(0xc2255de7u));
    p = fma(p, square, as_type<float>(0x40c90fdbu));
    return t * p;
}
static inline float aps5_s_cos_series(float t) {
#pragma METAL fp math_mode(safe)
    float square = t * t;
    float p = as_type<float>(0x40fce9c5u);
    p = fma(p, square, as_type<float>(0xc1d368f9u));
    p = fma(p, square, as_type<float>(0x4270fa83u));
    p = fma(p, square, as_type<float>(0xc2aae9e4u));
    p = fma(p, square, as_type<float>(0x4281e0f8u));
    p = fma(p, square, as_type<float>(0xc19de9e6u));
    return fma(p, square, 1.0f);
}
static inline float aps5_s_far(float value, float series) {
    uint v = as_type<uint>(value);
    uint s = as_type<uint>(series);
    return min(v - s, s - v) > 4u ? series : value;
}
static inline float aps5_s_near(float value, float distance, float linear) {
#pragma METAL fp math_mode(safe)
    return (as_type<uint>(distance) & 0x7fffffffu) < 0x39800000u ? distance * linear : value;
}
static inline uint aps5_s_sin_turns(uint a) {
#pragma METAL fp math_mode(safe)
    float x = as_type<float>(a);
    float cycle = (a & 0x7fffffffu) == 0u ? x : x - rint(x);
    float absolute = abs(cycle);
    bool nearHalf = absolute > 0.25f;
    float distance = nearHalf ? 0.5f - absolute : absolute;
    float series = distance > 0.125f ? aps5_s_cos_series(0.25f - distance) : aps5_s_sin_series(distance);
    float value = aps5_s_far(sin(absolute * as_type<float>(0x40c90fdbu)), series);
    float result = aps5_s_near(value, distance, as_type<float>(nearHalf ? 0x40c90fdbu : 0x40c90fd5u));
    return as_type<uint>(result) | (as_type<uint>(cycle) & 0x80000000u);
}
static inline uint aps5_s_cos_turns(uint a) {
#pragma METAL fp math_mode(safe)
    float x = as_type<float>(a);
    float absolute = abs(x - rint(x));
    float distance = 0.25f - absolute;
    float series = absolute > 0.375f ? -aps5_s_cos_series(0.5f - absolute) : absolute > 0.125f ? aps5_s_sin_series(distance) : aps5_s_cos_series(absolute);
    float value = aps5_s_far(cos(absolute * as_type<float>(0x40c90fdbu)), series);
    return as_type<uint>(aps5_s_near(value, distance, as_type<float>(0x40c90fdbu)));
}
)";

std::string FormatTable() {
    constexpr auto count = static_cast<std::uint32_t>(IrBufferFormat::Format32_32_32_32Float);
    std::string text = "constant uint aps5_fmt_table[" + std::to_string(3u * (count + 1u)) + "] = {0u, 0u, 0u";
    for (std::uint32_t index = 1; index <= count; ++index) {
        const auto info = GetFormatInfo(static_cast<IrBufferFormat>(index));
        std::uint32_t layout = info.componentCount | (info.packedBitfield ? 8u : 0u) | (static_cast<std::uint32_t>(info.type) << 4u);
        std::uint32_t bytes = 0;
        std::uint32_t bits = 0;
        for (std::uint32_t component = 0; component < info.componentCount; ++component) {
            layout |= info.componentBits[component] << (8u + component * 6u);
            bytes |= GetFormatComponentByteOffset(info, component) << (component * 8u);
            bits |= info.componentBitOffset[component] << (component * 8u);
        }
        text += ", " + U32(layout) + ", " + U32(bytes) + ", " + U32(bits);
    }
    return Replace(std::string(FormatText), "$TABLE", text + "};") ;
}

std::string PreciseText(bool strict) {
    const std::string prefix = strict ? "aps5_s_" : "aps5_i_";
    const std::string pragma = strict ? "#pragma METAL fp math_mode(safe)\n" : "#pragma METAL fp math_mode(relaxed)\n";
    std::string text;
    const auto define = [&](std::string_view result, std::string_view name, std::uint32_t count, std::string_view body) {
        static constexpr std::string_view Parameters[3] = {"uint a", "uint b", "uint c"};
        static constexpr std::string_view Floats[3] = {"    float x = as_type<float>(a);\n", "    float y = as_type<float>(b);\n", "    float z = as_type<float>(c);\n"};
        text += "static inline " + std::string(result) + " " + prefix + std::string(name) + "(";
        for (std::uint32_t index = 0; index < count; ++index) text += (index == 0u ? "" : ", ") + std::string(Parameters[index]);
        text += ") {\n" + pragma;
        for (std::uint32_t index = 0; index < count; ++index) text += Floats[index];
        text += body;
        text += "}\n";
    };
    const auto value = [&](std::string_view name, std::uint32_t count, std::string_view expression) { define("uint", name, count, "    return as_type<uint>(" + std::string(expression) + ");\n"); };
    const auto test = [&](std::string_view name, std::string_view expression) { define("bool", name, 2u, "    return " + std::string(expression) + ";\n"); };
    value("add", 2u, "x + y");
    value("sub", 2u, "x - y");
    if (strict) {
        text += StrictTrigText;
        define("uint", "mul", 2u, "    uint r = as_type<uint>(x * y);\n    return (r & 0x7fffffffu) == 0u ? aps5_s_tiny(a, b, r) : r;\n");
        define("uint", "mad", 3u, "    uint product = as_type<uint>(x * y);\n    if ((product & 0x7fffffffu) == 0u) product = aps5_s_tiny(a, b, product);\n    return as_type<uint>(as_type<float>(product) + z);\n");
        define("uint", "fma", 3u, "    uint r = as_type<uint>(fma(x, y, z));\n    return (r & 0x7fffffffu) == 0u ? aps5_s_tiny_fma(a, b, c, r) : r;\n");
    } else {
        value("mul", 2u, "x * y");
        define("uint", "mad", 3u, "    float product = x * y;\n    return as_type<uint>(product + z);\n");
        value("fma", 3u, "fma(x, y, z)");
    }
    value("recip", 1u, "1.0f / x");
    value("rsqrt", 1u, "rsqrt(x)");
    value("sqrt", 1u, "sqrt(x)");
    if (strict) {
        define("uint", "sin", 1u, "    return aps5_s_sin_turns(a);\n");
        define("uint", "cos", 1u, "    return aps5_s_cos_turns(a);\n");
    } else {
        value("sin", 1u, "sinpi(2.0f * x)");
        value("cos", 1u, "cospi(2.0f * x)");
    }
    value("exp2", 1u, "exp2(x)");
    value("log2", 1u, "log2(x)");
    value("rint", 1u, "rint(x)");
    value("floor", 1u, "floor(x)");
    value("ceil", 1u, "ceil(x)");
    value("trunc", 1u, "trunc(x)");
    define("uint", "fract", 1u, "    float magnitude = abs(x);\n    float part = magnitude - floor(magnitude);\n    return as_type<uint>(x < 0.0f && part != 0.0f ? 1.0f - part : part);\n");
    test("eq", "!aps5_nan(a) && !aps5_nan(b) && x == y");
    test("one", "!aps5_nan(a) && !aps5_nan(b) && x != y");
    test("lt", "!aps5_nan(a) && !aps5_nan(b) && x < y");
    test("gt", "!aps5_nan(a) && !aps5_nan(b) && x > y");
    test("le", "!aps5_nan(a) && !aps5_nan(b) && x <= y");
    test("ge", "!aps5_nan(a) && !aps5_nan(b) && x >= y");
    test("ueq", "aps5_nan(a) || aps5_nan(b) || x == y");
    test("une", "aps5_nan(a) || aps5_nan(b) || x != y");
    test("ult", "aps5_nan(a) || aps5_nan(b) || x < y");
    test("ugt", "aps5_nan(a) || aps5_nan(b) || x > y");
    test("ule", "aps5_nan(a) || aps5_nan(b) || x <= y");
    test("uge", "aps5_nan(a) || aps5_nan(b) || x >= y");
    define("uint", "min", 2u, "    if (aps5_nan(b)) return a;\n    if (((a | b) & 0x7fffffffu) == 0u) return a | b;\n    return x < y ? a : b;\n");
    define("uint", "max", 2u, "    if (aps5_nan(b)) return a;\n    if (((a | b) & 0x7fffffffu) == 0u) return a & b;\n    return x >= y ? a : b;\n");
    text += "static inline uint " + prefix + "min3(uint a, uint b, uint c) { return " + prefix + "min(" + prefix + "min(a, b), c); }\n";
    text += "static inline uint " + prefix + "max3(uint a, uint b, uint c) { return " + prefix + "max(" + prefix + "max(a, b), c); }\n";
    define("uint", "pick", 2u, "    if (((a | b) & 0x7fffffffu) == 0u) return a | b;\n    return x < y ? a : b;\n");
    define("uint", "pickmax", 2u, "    if (((a | b) & 0x7fffffffu) == 0u) return a & b;\n    return x >= y ? a : b;\n");
    text += "static inline uint " + prefix + "med3(uint a, uint b, uint c) {\n"
            "    uint low = aps5_nan(b) ? a : " + prefix + "pick(a, b);\n"
            "    uint lowest = aps5_nan(c) ? low : " + prefix + "pick(low, c);\n"
            "    uint median = " + prefix + "pickmax(low, " + prefix + "pick(" + prefix + "pickmax(a, b), c));\n"
            "    return aps5_nan(a) || aps5_nan(b) || aps5_nan(c) ? lowest : median;\n"
            "}\n";
    return text;
}

}

std::string Prelude(const State& state) {
    const auto has = [&](std::uint32_t feature) { return (state.features & feature) != 0u; };
    std::string text(BaseText);
    if (has(FloatClass)) text += FloatClassText;
    if (has(IeeeFloat)) text += PreciseText(false);
    if (has(StrictFloat)) text += PreciseText(true);
    if (has(Float64 | Dot2)) {
        text += SoftFloatText;
        text += '\n';
    }
    if (has(Dot2)) text += Dot2Text;
    if (has(Half)) text += state.target.relaxedFloat ? HalfText : ExactHalfText;
    if (has(Wave | Lane) && state.program.Resources().stage == IrShaderStage::Vertex) text += VertexWaveText;
    else if (has(Wave)) text += WaveText;
    if (has(Permute)) text += PermuteText;
    if (has(Fault | Bda | GuestBuffers)) text += FaultText;
    if (has(Tables)) text += Replace(std::string(TableFaultText), "$REASON", Reason(BdaAbi::FaultReason::ImageTable));
    if (has(Bda)) {
        auto bda = std::string(BdaText);
        bda = Replace(bda, "$INVALID_TABLE", Reason(BdaAbi::FaultReason::InvalidTable));
        bda = Replace(bda, "$OVERFLOW", Reason(BdaAbi::FaultReason::Overflow));
        bda = Replace(bda, "$UNMAPPED", Reason(BdaAbi::FaultReason::Unmapped));
        bda = Replace(bda, "$PERMISSION", Reason(BdaAbi::FaultReason::Permission));
        bda = Replace(bda, "$READ", U32(BdaAbi::Read));
        text += bda;
        if (has(BdaWrites)) {
            auto writes = std::string(BdaWritesText);
            writes = Replace(writes, "$CHUNK_SHIFT", U32(BdaAbi::WrittenChunkShift));
            writes = Replace(writes, "$PAGE_SHIFT", U32(BdaAbi::WrittenPageShift));
            writes = Replace(writes, "$PAGE_MASK", U32(BdaAbi::WrittenChunkPages - 1u));
            writes = Replace(writes, "$HASH_SHIFT", U32(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
            writes = Replace(writes, "$PROBES", U32(BdaAbi::WrittenPageProbes));
            writes = Replace(writes, "$SLOT_MASK", U32(BdaAbi::WrittenPageSlots - 1u));
            writes = Replace(writes, "$SLOTS_WORD", U32(BdaAbi::WrittenSlotsWord));
            writes = Replace(writes, "$COUNT_WORD", U32(BdaAbi::WrittenCountWord));
            writes = Replace(writes, "$LIST_WORD", U32(BdaAbi::WrittenListWord));
            writes = Replace(writes, "$SLOTS", U32(BdaAbi::WrittenPageSlots));
            writes = Replace(writes, "$OVERFLOW_WORD", U32(BdaAbi::WrittenOverflowWord));
            text += writes;
        } else {
            text += NoWritesText;
        }
        auto store = std::string(BdaStoreText);
        store = Replace(store, "$UNALIGNED", Reason(BdaAbi::FaultReason::Unaligned));
        store = Replace(store, "$WRITE", U32(BdaAbi::Write));
        text += store;
    }
    if (has(Formatted)) text += Replace(FormatTable(), "$FORMATS", U32(static_cast<std::uint32_t>(IrBufferFormat::Format32_32_32_32Float)));
    if (has(Descriptors)) {
        text += DescriptorsText;
        if (has(GuestBuffers)) text += Replace(std::string(DirectBuffersText), "$UNALIGNED", Reason(BdaAbi::FaultReason::Unaligned));
        if (has(Bda)) {
            auto gpu = Replace(std::string(GpuBuffersText), "$INVALID_DESCRIPTOR", Reason(BdaAbi::FaultReason::InvalidDescriptor));
            text += Replace(gpu, "$READ_WRITE", U32(BdaAbi::Read | BdaAbi::Write));
        }
        if (has(Formatted)) {
            if (has(GuestBuffers)) {
                auto direct = Replace(Replace(std::string(FormatAccessText), "NAME", ""), "BAD", "{}");
                direct = Replace(direct, "READ(bytes[part], packed ? 4u : bits >> 3u)", "(packed || bits == 32u ? aps5_buffer_load(c, element, r, bytes[part], 4u) : aps5_buffer_load(c, element, r, bytes[part] & ~3u, 4u) >> ((bytes[part] & 3u) * 8u))");
                direct = Replace(direct, "READ(", "aps5_buffer_load(c, element, r, ");
                text += Replace(direct, "WRITE(", "aps5_buffer_store(c, element, r, ");
            }
            if (has(Bda)) {
                auto gpu = Replace(Replace(std::string(FormatAccessText), "NAME", "_gpu"), "BAD", "aps5_fault(c, " + Reason(BdaAbi::FaultReason::InvalidDescriptor) + ", r.base, 0u, pc)");
                gpu = Replace(gpu, "READ(bytes[part], packed ? 4u : bits >> 3u)", "aps5_vsharp_load(c, r, bytes[part], packed ? 4u : bits >> 3u, pc)");
                gpu = Replace(gpu, "WRITE(bytes[part], bits >> 3u, encoded)", "aps5_vsharp_store(c, r, bytes[part], bits >> 3u, encoded, pc)");
                text += Replace(gpu, "WRITE(bytes[0], 4u, word)", "aps5_vsharp_store(c, r, bytes[0], 4u, word, pc)");
            }
        }
    }
    if (has(Lds)) {
        auto lds = Replace(Replace(std::string(LocalMemoryText), "SPACE", "threadgroup"), "NAME", "_lds");
        lds = Replace(lds, "MERGE(index, ", "aps5_merge_threadgroup(&words[index], ");
        lds = Replace(lds, "MERGE(index + 1u, ", "aps5_merge_threadgroup(&words[index + 1u], ");
        text += "static inline void aps5_merge_threadgroup(threadgroup uint* word, uint mask, uint value) {\n"
                "    atomic_fetch_and_explicit(reinterpret_cast<threadgroup atomic_uint*>(word), ~mask, memory_order_relaxed);\n"
                "    atomic_fetch_or_explicit(reinterpret_cast<threadgroup atomic_uint*>(word), value, memory_order_relaxed);\n"
                "}\n";
        text += lds;
    }
    if (has(Scratch)) {
        auto scratch = Replace(Replace(std::string(LocalMemoryText), "SPACE", "thread"), "NAME", "_scratch");
        scratch = Replace(scratch, "MERGE(index, uint(mask), uint(data));", "words[index] = (words[index] & ~uint(mask)) | uint(data);");
        scratch = Replace(scratch, "MERGE(index + 1u, uint(mask >> 32u), uint(data >> 32u));", "words[index + 1u] = (words[index + 1u] & ~uint(mask >> 32u)) | uint(data >> 32u);");
        text += scratch;
    }
    if (has(Lds | GuestBuffers | Bda | UpdateTable)) {
        text += AtomicsText;
        text += Replace(Replace(std::string(AtomicSpaceText), "SPACE", "device"), "NAME", "");
        if (has(Lds)) text += Replace(Replace(std::string(AtomicSpaceText), "SPACE", "threadgroup"), "NAME", "_lds");
    }
    if (has(Atomic64)) {
        text += Atomic64Text;
        if (has(Bda)) text += Replace(std::string(Atomic64PointerText), "$UNALIGNED", Reason(BdaAbi::FaultReason::Unaligned));
        if (has(GuestBuffers)) text += Replace(std::string(Atomic64DirectText), "$UNALIGNED", Reason(BdaAbi::FaultReason::Unaligned));
        if (has(Bda) && has(Descriptors)) text += Replace(std::string(Atomic64GpuText), "$READ_WRITE", U32(BdaAbi::Read | BdaAbi::Write));
        auto device = Replace(Replace(std::string(Atomic64SpaceText), "SPACE", "device"), "NAME", "");
        text += Replace(device, "MINMAX", std::string(Atomic64MinMaxText));
        if (has(Lds)) text += Replace(Replace(Replace(std::string(Atomic64SpaceText), "SPACE", "threadgroup"), "NAME", "_lds"), "MINMAX", "");
    }
    for (std::size_t index = 0; index < state.imageTypes.size(); ++index) text += "struct aps5_img" + std::to_string(index) + " { " + state.imageTypes[index] + " t; };\n";
    if (has(Samplers | Tables)) text += "struct aps5_sampler { sampler s; };\n";
    for (const auto& [id, fallback] : state.constants) {
        const auto name = std::to_string(id);
        text += "constant uint aps5_fc" + name + " [[function_constant(" + name + ")]];\n";
        text += "constant uint aps5_c" + name + " = is_function_constant_defined(aps5_fc" + name + ") ? aps5_fc" + name + " : " + U32(fallback) + ";\n";
    }
    return text;
}

}
