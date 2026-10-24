#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_COMPILEKEY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_COMPILEKEY_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>

namespace AgcDriver::Graphics::Metal {

struct CompileKey {
    std::uint64_t low = 0;
    std::uint64_t high = 0;

    [[nodiscard]] bool Empty() const { return low == 0 && high == 0; }
    bool operator==(const CompileKey&) const = default;
};

struct CompileKeyHash {
    std::size_t operator()(const CompileKey& key) const { return static_cast<std::size_t>(key.low ^ (key.high * 0x9e3779b97f4a7c15ull)); }
};

class CompileKeyBuilder {
public:
    explicit CompileKeyBuilder(std::uint64_t domain) : low(0x243f6a8885a308d3ull ^ domain), high(0x13198a2e03707344ull + domain * 0x9e3779b97f4a7c15ull) {}

    CompileKeyBuilder& Add(std::uint64_t value) {
        low = (low ^ value) * 0xff51afd7ed558ccdull;
        low ^= low >> 32u;
        high = (high + value + 0x9e3779b97f4a7c15ull) * 0xc4ceb9fe1a85ec53ull;
        high ^= high >> 29u;
        return *this;
    }

    CompileKeyBuilder& Add(const CompileKey& key) { return Add(key.low).Add(key.high); }

    CompileKeyBuilder& AddBytes(const void* data, std::size_t bytes) {
        Add(bytes);
        const auto* cursor = static_cast<const unsigned char*>(data);
        for (; bytes >= 8; bytes -= 8, cursor += 8) {
            std::uint64_t word;
            std::memcpy(&word, cursor, 8);
            Add(word);
        }
        if (bytes != 0) {
            std::uint64_t word = 0;
            std::memcpy(&word, cursor, bytes);
            Add(word);
        }
        return *this;
    }

    CompileKeyBuilder& AddString(std::string_view text) { return AddBytes(text.data(), text.size()); }

    [[nodiscard]] CompileKey Finish() const {
        CompileKey key{low, high};
        key.low ^= key.high >> 31u;
        key.high ^= key.low >> 27u;
        key.low *= 0xd6e8feb86659fd93ull;
        key.high *= 0xff51afd7ed558ccdull;
        key.low ^= key.low >> 32u;
        key.high ^= key.high >> 32u;
        if (key.Empty()) key.low = 1;
        return key;
    }

private:
    std::uint64_t low;
    std::uint64_t high;
};

inline std::string CompileKeyHex(const CompileKey& key) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string text(32, '0');
    for (unsigned index = 0; index < 16; ++index) {
        text[15 - index] = digits[(key.high >> (index * 4u)) & 0xfu];
        text[31 - index] = digits[(key.low >> (index * 4u)) & 0xfu];
    }
    return text;
}

}

#endif
