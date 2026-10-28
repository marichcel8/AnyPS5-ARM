#ifndef CORE_SHADER_RECOMPILER_MSLBACKEND_SRC_MSLWRAP_HPP
#define CORE_SHADER_RECOMPILER_MSLBACKEND_SRC_MSLWRAP_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ShaderRecompiler::MslWrap {

struct Member {
    std::string type;
    std::string name;
    std::string attribute;
    std::uint32_t count = 1u;
};

struct Parameter {
    std::string declaration;
    std::string name;
    std::string attribute;
};

struct Span {
    std::size_t begin = std::string_view::npos;
    std::size_t end = std::string_view::npos;
    std::string_view body;
};

struct Entry {
    std::size_t at = 0;
    Span inputs;
    Span outputs;
    std::vector<Parameter> parameters;
    std::vector<Member> outputMembers;
    std::uint32_t vertexBytes = 0;
    std::string_view body;
};

[[noreturn]] void Unsupported(std::string_view reason);
std::string_view Trim(std::string_view text);
std::vector<Member> Members(std::string_view body);
std::uint32_t Components(std::string_view type);
std::string Scalar(std::string_view type);
bool Arrayed(const Member& member);
Entry ParseEntry(std::string_view source, std::string_view signature);
std::string OutputStruct(std::string_view name, const std::vector<Member>& members);
std::string CopyOutputs(const std::vector<Member>& members, std::string_view target, std::string_view source, std::string_view indent);
std::string Join(const std::vector<std::string>& items, std::string_view separator);

}

#endif
