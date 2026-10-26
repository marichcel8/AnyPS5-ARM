#ifndef TOOLS_TRANSLATOR_SRC_ELF_HPP
#define TOOLS_TRANSLATOR_SRC_ELF_HPP

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace Translator {

struct Segment {
    std::uint64_t vaddr;
    std::uint64_t fileOffset;
    std::uint64_t fileSize;
    std::uint64_t memorySize;
    std::uint32_t flags;

    bool Executable() const { return (flags & 1u) != 0; }
};

struct Relocation {
    std::uint64_t offset;
    std::uint32_t type;
    std::uint32_t symbol;
    std::int64_t addend;
};

struct Symbol {
    std::string name;
    std::uint64_t value;
    std::uint64_t size;
    std::uint8_t info;
    std::uint8_t visibility;
    std::uint16_t section;
};

// The initializers and finalizers of an image: DT_INIT and DT_FINI addresses, and the virtual
// addresses and sizes of the DT_INIT_ARRAY and DT_FINI_ARRAY pointer arrays (0 when absent).
struct ImageInitializers {
    std::uint64_t init = 0;
    std::uint64_t fini = 0;
    std::uint64_t initArray = 0;
    std::uint64_t initArraySize = 0;
    std::uint64_t finiArray = 0;
    std::uint64_t finiArraySize = 0;
};

// A guest ELF as the relinker reads it: PT_LOAD segments mapped at their virtual addresses, and the
// dynamic tables either through SysV tags (virtual addresses) or the PS5 DT_OS_* tags (file offsets).
class Elf {
public:
    explicit Elf(const std::filesystem::path& path);

    const std::vector<std::uint8_t>& Bytes() const { return bytes; }
    const std::vector<Segment>& Loads() const { return loads; }
    // PT_TLS: the initialization image (vaddr, fileSize) and the block size and alignment.
    const Segment& Tls() const { return tls; }
    std::uint64_t TlsAlign() const { return tlsAlign; }
    // The size of the static TLS block, the PT_TLS memory size rounded up to its alignment.
    std::uint64_t TlsBlockSize() const { return (tls.memorySize + tlsAlign - 1) / tlsAlign * tlsAlign; }
    // A PS5 prx or other shared object (ET_SCE_DYNAMIC, ET_DYN) rather than an executable.
    bool IsModule() const { return type == 0xfe18 || type == 3; }
    // The raw program header table (56-byte entries), for dl_iterate_phdr.
    const std::vector<std::uint8_t>& ProgramHeaders() const { return programHeaders; }
    std::uint64_t Entry() const { return entry; }
    std::uint64_t ImageBegin() const;
    std::uint64_t ImageEnd() const;

    const std::vector<Relocation>& Relocations() const { return relocations; }
    const std::vector<Symbol>& Symbols() const { return symbols; }
    const std::vector<std::string>& Needed() const { return needed; }
    std::vector<std::uint64_t> InitFunctions() const;
    ImageInitializers Initializers() const;
    const std::map<std::uint64_t, std::uint64_t>& FunctionRanges() const { return functionRanges; }

    // The bytes of the loaded image at a virtual address, or null outside the file-backed part.
    const std::uint8_t* At(std::uint64_t vaddr, std::uint64_t size) const;
    bool InExecutable(std::uint64_t vaddr) const;

private:
    std::uint64_t Translate(std::uint64_t vaddr) const;
    std::uint64_t Tag(std::int64_t tag, std::uint64_t fallback = 0) const;
    bool HasTag(std::int64_t tag) const;
    std::uint64_t TableOffset(std::int64_t osTag, std::int64_t sysvTag) const;
    std::uint64_t TableSize(std::int64_t osTag, std::int64_t sysvTag) const;
    std::string String(std::uint64_t offset) const;
    void ReadRelocations(std::uint64_t offset, std::uint64_t size);
    void ReadUnwindTable(std::uint64_t header);

    std::vector<std::uint8_t> bytes;
    std::vector<Segment> loads;
    std::vector<Segment> dynamicData;
    Segment tls{};
    std::vector<std::uint8_t> programHeaders;
    std::uint64_t tlsAlign = 1;
    std::multimap<std::int64_t, std::uint64_t> tags;
    std::uint64_t entry = 0;
    std::uint16_t type = 0;
    std::uint64_t stringsOffset = 0;
    std::uint64_t stringsSize = 0;
    std::vector<Relocation> relocations;
    std::vector<Symbol> symbols;
    std::vector<std::string> needed;
    std::map<std::uint64_t, std::uint64_t> functionRanges;
};

}

#endif
