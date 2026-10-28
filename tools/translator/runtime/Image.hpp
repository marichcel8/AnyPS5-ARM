#ifndef TOOLS_TRANSLATOR_RUNTIME_IMAGE_HPP
#define TOOLS_TRANSLATOR_RUNTIME_IMAGE_HPP

#include "GuestState.hpp"

#include <cstdint>

// Emitted by the translator: the table of translated functions of an image, sorted by guest offset.
struct TranslatedFunction {
    std::uint64_t offset;
    void (*code)(GuestState*);
};

// A {guest offset, count} row of an image's __DATA,__aps5_prof section (quick tier counters).
struct ProfileRow {
    std::uint64_t offset;
    std::uint64_t count;
};

// A translated guest image, the executable or a module: the runtime finds translated functions and
// writes profiles over every registered image.
struct TranslatedImage {
    std::uint64_t base;
    std::uint64_t size;
    const TranslatedFunction* functions;
    std::uint64_t functionCount;
    const char* profilePath;
    ProfileRow* profileRows;
    std::uint64_t profileRowCount;
    const char* name;
};

extern "C" {
void aps5_register_image(const TranslatedImage* image);
void aps5_unregister_image(std::uint64_t base);
// The translated function at a guest address of any registered image, or null.
const void* aps5_translated(std::uint64_t address);
int aps5_invoke_translated(const void*, const void*, const void*, const void*);
}

#endif
