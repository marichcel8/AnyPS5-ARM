#include "Elf.hpp"

#include <algorithm>
#include <string>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace Translator {

namespace {

constexpr std::uint32_t PtLoad = 1;
constexpr std::uint32_t PtDynamic = 2;
constexpr std::uint32_t PtTls = 7;
constexpr std::uint32_t PtGnuEhFrame = 0x6474e550;
constexpr std::int64_t DtNull = 0, DtNeeded = 1, DtPltRelSz = 2, DtStrTab = 5, DtSymTab = 6, DtRela = 7, DtRelaSz = 8;
constexpr std::int64_t DtInit = 12, DtStrSz = 10, DtJmpRel = 23, DtInitArray = 25, DtInitArraySz = 27;
constexpr std::int64_t DtOsInit = 0x6000000c, DtOsInitArray = 0x60000019, DtOsInitArraySz = 0x6000001b;
constexpr std::int64_t DtOsJmpRel = 0x61000029, DtOsPltRelSz = 0x6100002d, DtOsRela = 0x6100002f, DtOsRelaSz = 0x61000031;
constexpr std::int64_t DtOsStrTab = 0x61000035, DtOsStrSz = 0x61000037, DtOsSymTab = 0x61000039, DtOsSymTabSz = 0x6100003f;

template <typename T>
T Read(const std::vector<std::uint8_t>& bytes, std::uint64_t offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) throw std::runtime_error("ELF read out of bounds");
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

}

Elf::Elf(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    bytes.assign(std::istreambuf_iterator<char>(file), {});
    if (bytes.size() < 64 || std::memcmp(bytes.data(), "\x7f" "ELF\x02\x01", 6) != 0) throw std::runtime_error("not a 64-bit little-endian ELF");
    if (Read<std::uint16_t>(bytes, 18) != 62) throw std::runtime_error("not an x86-64 ELF");
    entry = Read<std::uint64_t>(bytes, 24);
    const auto phoff = Read<std::uint64_t>(bytes, 32);
    const auto phentsize = Read<std::uint16_t>(bytes, 54);
    const auto phnum = Read<std::uint16_t>(bytes, 56);
    if (phentsize != 56 || phoff + static_cast<std::uint64_t>(phnum) * 56 > bytes.size()) throw std::runtime_error("bad program header table");
    programHeaders.assign(bytes.begin() + static_cast<std::ptrdiff_t>(phoff), bytes.begin() + static_cast<std::ptrdiff_t>(phoff + phnum * 56ull));
    std::uint64_t dynamicOffset = 0, dynamicSize = 0, unwindHeader = 0;
    for (std::uint16_t i = 0; i < phnum; ++i) {
        const auto header = phoff + static_cast<std::uint64_t>(i) * phentsize;
        const auto type = Read<std::uint32_t>(bytes, header);
        const auto flags = Read<std::uint32_t>(bytes, header + 4);
        const auto offset = Read<std::uint64_t>(bytes, header + 8);
        const auto vaddr = Read<std::uint64_t>(bytes, header + 16);
        const auto fileSize = Read<std::uint64_t>(bytes, header + 32);
        const auto memorySize = Read<std::uint64_t>(bytes, header + 40);
        // A PT_LOAD without permissions is the PS5 dynamic data, which is not mapped; SysV tags of
        // homebrew images still point into it by virtual address.
        if (type == PtLoad && flags != 0) loads.push_back({vaddr, offset, fileSize, memorySize, flags});
        if (type == PtLoad && flags == 0) dynamicData.push_back({vaddr, offset, fileSize, memorySize, flags});
        if (type == PtTls) {
            tls = {vaddr, offset, fileSize, memorySize, flags};
            tlsAlign = std::max<std::uint64_t>(Read<std::uint64_t>(bytes, header + 48), 1);
        }
        if (type == PtGnuEhFrame) unwindHeader = vaddr;
        if (type == PtDynamic) {
            dynamicOffset = offset;
            dynamicSize = fileSize;
        }
    }
    std::sort(loads.begin(), loads.end(), [](const Segment& left, const Segment& right) { return left.vaddr < right.vaddr; });
    if (loads.empty()) throw std::runtime_error("no loadable segments");
    if (unwindHeader != 0) ReadUnwindTable(unwindHeader);
    for (std::uint64_t at = dynamicOffset; dynamicSize != 0 && at + 16 <= dynamicOffset + dynamicSize; at += 16) {
        const auto tag = Read<std::int64_t>(bytes, at);
        if (tag == DtNull) break;
        tags.emplace(tag, Read<std::uint64_t>(bytes, at + 8));
    }
    if (tags.empty()) return;
    stringsOffset = TableOffset(DtOsStrTab, DtStrTab);
    stringsSize = TableSize(DtOsStrSz, DtStrSz);
    for (auto [it, end] = tags.equal_range(DtNeeded); it != end; ++it) needed.push_back(String(it->second));
    const auto symbolsOffset = TableOffset(DtOsSymTab, DtSymTab);
    // SysV has no symbol table size; the string table follows the symbol table in every PS5 image.
    const auto symbolsSize = HasTag(DtOsSymTabSz) ? Tag(DtOsSymTabSz) : (stringsOffset > symbolsOffset ? stringsOffset - symbolsOffset : 0);
    for (std::uint64_t at = symbolsOffset; at + 24 <= symbolsOffset + symbolsSize && at + 24 <= bytes.size(); at += 24) {
        Symbol symbol;
        const auto name = Read<std::uint32_t>(bytes, at);
        symbol.name = name < stringsSize ? String(name) : std::string();
        symbol.info = Read<std::uint8_t>(bytes, at + 4);
        symbol.section = Read<std::uint16_t>(bytes, at + 6);
        symbol.value = Read<std::uint64_t>(bytes, at + 8);
        symbol.size = Read<std::uint64_t>(bytes, at + 16);
        symbols.push_back(std::move(symbol));
    }
    if (HasTag(DtOsRela) || HasTag(DtRela)) ReadRelocations(TableOffset(DtOsRela, DtRela), TableSize(DtOsRelaSz, DtRelaSz));
    if (HasTag(DtOsJmpRel) || HasTag(DtJmpRel)) ReadRelocations(TableOffset(DtOsJmpRel, DtJmpRel), TableSize(DtOsPltRelSz, DtPltRelSz));
}

std::uint64_t Elf::ImageBegin() const {
    return loads.front().vaddr & ~0x3fffull;
}

std::uint64_t Elf::ImageEnd() const {
    std::uint64_t end = 0;
    for (const auto& segment : loads) end = std::max(end, segment.vaddr + segment.memorySize);
    return (end + 0x3fff) & ~0x3fffull;
}

std::vector<std::uint64_t> Elf::InitFunctions() const {
    std::vector<std::uint64_t> functions;
    if (HasTag(DtOsInit) || HasTag(DtInit)) functions.push_back(HasTag(DtOsInit) ? Tag(DtOsInit) : Tag(DtInit));
    const auto array = HasTag(DtOsInitArray) ? Tag(DtOsInitArray) : Tag(DtInitArray);
    const auto size = HasTag(DtOsInitArraySz) ? Tag(DtOsInitArraySz) : Tag(DtInitArraySz);
    for (std::uint64_t at = 0; array != 0 && at + 8 <= size; at += 8) {
        if (const auto* slot = At(array + at, 8)) {
            std::uint64_t value;
            std::memcpy(&value, slot, 8);
            for (const auto& relocation : relocations) {
                if (relocation.offset == array + at && relocation.type == 8) value = static_cast<std::uint64_t>(relocation.addend);
            }
            if (value != 0 && value != ~0ull) functions.push_back(value);
        }
    }
    return functions;
}

void Elf::ReadUnwindTable(std::uint64_t header) {
    struct Reader {
        const Elf& elf;
        std::uint64_t at;
        bool failed = false;

        std::uint64_t Bytes(unsigned count) {
            const auto* data = elf.At(at, count);
            if (data == nullptr) {
                failed = true;
                return 0;
            }
            std::uint64_t value = 0;
            std::memcpy(&value, data, count);
            at += count;
            return value;
        }
        std::uint64_t Unsigned() {
            std::uint64_t value = 0;
            for (unsigned shift = 0; shift < 64 && !failed; shift += 7) {
                const auto byte = Bytes(1);
                value |= (byte & 0x7f) << shift;
                if ((byte & 0x80) == 0) break;
            }
            return value;
        }
        std::int64_t Signed() {
            std::uint64_t value = 0;
            unsigned shift = 0;
            std::uint64_t byte = 0;
            do {
                byte = Bytes(1);
                value |= (byte & 0x7f) << shift;
                shift += 7;
            } while ((byte & 0x80) != 0 && shift < 64 && !failed);
            if (shift < 64 && (byte & 0x40) != 0) value |= ~std::uint64_t{0} << shift;
            return static_cast<std::int64_t>(value);
        }
        std::uint64_t Encoded(std::uint8_t encoding, std::uint64_t dataBase, bool applyBase = true) {
            const auto field = at;
            std::uint64_t value = 0;
            switch (encoding & 0x0f) {
            case 0x00: value = Bytes(8); break;
            case 0x01: value = Unsigned(); break;
            case 0x02: value = Bytes(2); break;
            case 0x03: value = Bytes(4); break;
            case 0x04: value = Bytes(8); break;
            case 0x09: value = static_cast<std::uint64_t>(Signed()); break;
            case 0x0a: value = static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int16_t>(Bytes(2)))); break;
            case 0x0b: value = static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(Bytes(4)))); break;
            case 0x0c: value = Bytes(8); break;
            default: failed = true; return 0;
            }
            if (!applyBase) return value;
            switch (encoding & 0x70) {
            case 0x00: return value;
            case 0x10: return value + field;
            case 0x30: return value + dataBase;
            default: failed = true; return 0;
            }
        }
    };
    Reader reader{*this, header};
    if (reader.Bytes(1) != 1) return;
    const auto frameEncoding = static_cast<std::uint8_t>(reader.Bytes(1));
    reader.Bytes(2);
    if (frameEncoding == 0xff) return;
    reader.at = reader.Encoded(frameEncoding, header);
    std::map<std::uint64_t, std::uint8_t> cieEncodings;
    for (unsigned entries = 0; entries < (1u << 24) && !reader.failed; ++entries) {
        const auto start = reader.at;
        std::uint64_t length = reader.Bytes(4);
        if (length == 0 || reader.failed) break;
        if (length == 0xffffffffu) length = reader.Bytes(8);
        const auto body = reader.at;
        const auto next = body + length;
        const auto idField = reader.at;
        const auto id = reader.Bytes(4);
        if (id == 0) {
            const auto version = reader.Bytes(1);
            std::string augmentation;
            for (std::uint64_t c = reader.Bytes(1); c != 0 && !reader.failed && augmentation.size() < 16; c = reader.Bytes(1)) augmentation.push_back(static_cast<char>(c));
            reader.Unsigned();
            reader.Signed();
            if (version == 1) reader.Bytes(1);
            else reader.Unsigned();
            std::uint8_t encoding = 0;
            if (!augmentation.empty() && augmentation[0] == 'z') {
                reader.Unsigned();
                for (std::size_t i = 1; i < augmentation.size() && !reader.failed; ++i) {
                    if (augmentation[i] == 'R') encoding = static_cast<std::uint8_t>(reader.Bytes(1));
                    else if (augmentation[i] == 'P') reader.Encoded(static_cast<std::uint8_t>(reader.Bytes(1)) & 0x7f, header);
                    else if (augmentation[i] == 'L') reader.Bytes(1);
                    else if (augmentation[i] != 'S' && augmentation[i] != 'B') break;
                }
            }
            cieEncodings[start] = encoding;
        } else {
            const auto cie = cieEncodings.find(idField - id);
            if (cie != cieEncodings.end()) {
                const auto begin = reader.Encoded(cie->second, header);
                const auto range = reader.Encoded(cie->second & 0x0f, header, false);
                if (!reader.failed && range != 0 && InExecutable(begin)) functionRanges[begin] = std::max(functionRanges[begin], begin + range);
            }
        }
        reader.at = next;
    }
}

const std::uint8_t* Elf::At(std::uint64_t vaddr, std::uint64_t size) const {
    for (const auto& segment : loads) {
        if (vaddr >= segment.vaddr && vaddr - segment.vaddr <= segment.fileSize && size <= segment.fileSize - (vaddr - segment.vaddr))
            return bytes.data() + segment.fileOffset + (vaddr - segment.vaddr);
    }
    return nullptr;
}

bool Elf::InExecutable(std::uint64_t vaddr) const {
    for (const auto& segment : loads) {
        if (segment.Executable() && vaddr >= segment.vaddr && vaddr < segment.vaddr + segment.fileSize) return true;
    }
    return false;
}

std::uint64_t Elf::Translate(std::uint64_t vaddr) const {
    for (const auto* segments : {&loads, &dynamicData}) {
        for (const auto& segment : *segments) {
            if (vaddr >= segment.vaddr && vaddr < segment.vaddr + segment.fileSize) return segment.fileOffset + (vaddr - segment.vaddr);
        }
    }
    throw std::runtime_error("dynamic table address outside the loaded segments");
}

std::uint64_t Elf::Tag(std::int64_t tag, std::uint64_t fallback) const {
    const auto found = tags.find(tag);
    return found == tags.end() ? fallback : found->second;
}

bool Elf::HasTag(std::int64_t tag) const {
    return tags.count(tag) != 0;
}

std::uint64_t Elf::TableOffset(std::int64_t osTag, std::int64_t sysvTag) const {
    if (HasTag(osTag)) return Tag(osTag);
    if (HasTag(sysvTag)) return Translate(Tag(sysvTag));
    return 0;
}

std::uint64_t Elf::TableSize(std::int64_t osTag, std::int64_t sysvTag) const {
    return HasTag(osTag) ? Tag(osTag) : Tag(sysvTag);
}

std::string Elf::String(std::uint64_t offset) const {
    if (offset >= stringsSize) throw std::runtime_error("dynamic string offset out of range");
    const auto* begin = reinterpret_cast<const char*>(bytes.data() + stringsOffset + offset);
    return std::string(begin, strnlen(begin, stringsSize - offset));
}

void Elf::ReadRelocations(std::uint64_t offset, std::uint64_t size) {
    for (std::uint64_t at = offset; at + 24 <= offset + size; at += 24) {
        const auto info = Read<std::uint64_t>(bytes, at + 8);
        relocations.push_back({Read<std::uint64_t>(bytes, at), static_cast<std::uint32_t>(info), static_cast<std::uint32_t>(info >> 32), Read<std::int64_t>(bytes, at + 16)});
    }
}

}
