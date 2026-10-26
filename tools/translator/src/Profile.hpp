#ifndef TOOLS_TRANSLATOR_SRC_PROFILE_HPP
#define TOOLS_TRANSLATOR_SRC_PROFILE_HPP

#include <cstdint>
#include <map>
#include <string>

namespace llvm {
class Function;
}

namespace Translator {

void Instrument(llvm::Function& function, std::uint64_t offset);
std::map<std::uint64_t, std::uint64_t> ReadProfile(const std::string& path);

}

#endif
