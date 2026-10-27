#include "MslWrap.hpp"
#include "MslBackend/MslEmitter.hpp"
#include <charconv>

namespace ShaderRecompiler::MslWrap {

namespace {

std::vector<std::string_view> Lines(std::string_view text) {
    std::vector<std::string_view> lines;
    while (!text.empty()) {
        const auto end = text.find('\n');
        const auto line = Trim(text.substr(0, end));
        if (!line.empty()) lines.push_back(line);
        if (end == std::string_view::npos) break;
        text.remove_prefix(end + 1u);
    }
    return lines;
}

Span Struct(std::string_view source, std::string_view name) {
    Span span;
    const auto head = "struct " + std::string(name) + " {\n";
    span.begin = source.find(head);
    if (span.begin == std::string_view::npos) return span;
    const auto close = source.find("\n};\n", span.begin);
    if (close == std::string_view::npos) Unsupported("struct " + std::string(name) + " is not closed");
    span.end = close + 4u;
    span.body = source.substr(span.begin + head.size(), close - span.begin - head.size());
    return span;
}

void Attributed(std::string_view text, std::string_view& before, std::string& attribute, std::string_view& after) {
    const auto open = text.find("[[");
    const auto close = text.find("]]", open == std::string_view::npos ? 0u : open);
    if (open == std::string_view::npos || close == std::string_view::npos) Unsupported("\"" + std::string(text) + "\" has no attribute");
    before = Trim(text.substr(0, open));
    attribute = std::string(text.substr(open + 2u, close - open - 2u));
    after = Trim(text.substr(close + 2u));
}

std::string LastToken(std::string_view declaration) {
    const auto at = declaration.find_last_of(" *&");
    return std::string(at == std::string_view::npos ? declaration : declaration.substr(at + 1u));
}

std::uint32_t MemberBytes(const Member& member) {
    const auto scalar = Scalar(member.type);
    if (scalar != "float" && scalar != "uint" && scalar != "int") Unsupported("output " + member.name + " has type " + member.type);
    const auto components = Components(member.type);
    return (components == 3u ? 4u : components) * 4u * member.count;
}

}

void Unsupported(std::string_view reason) {
    throw MslUnsupported(std::string(reason));
}

std::string_view Trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\n')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n' || text.back() == ',')) text.remove_suffix(1);
    return text;
}

std::vector<Member> Members(std::string_view body) {
    std::vector<Member> members;
    for (auto line : Lines(body)) {
        if (!line.ends_with(';')) Unsupported("struct member \"" + std::string(line) + "\" is not one declaration");
        line.remove_suffix(1);
        std::string_view before;
        std::string_view after;
        Member member;
        Attributed(line, before, member.attribute, after);
        const auto space = before.find_last_of(' ');
        if (space == std::string_view::npos) Unsupported("struct member \"" + std::string(line) + "\" has no type");
        member.type = std::string(Trim(before.substr(0, space)));
        member.name = std::string(before.substr(space + 1u));
        if (!after.empty()) {
            if (after.size() < 3u || after.front() != '[' || after.back() != ']') Unsupported("struct member \"" + std::string(line) + "\" has an unexpected suffix");
            const auto digits = after.substr(1u, after.size() - 2u);
            if (std::from_chars(digits.data(), digits.data() + digits.size(), member.count).ec != std::errc{}) Unsupported("struct member \"" + std::string(line) + "\" has an unreadable array size");
        }
        members.push_back(std::move(member));
    }
    return members;
}

std::uint32_t Components(std::string_view type) {
    const auto last = type.empty() ? '\0' : type.back();
    return last >= '2' && last <= '4' ? static_cast<std::uint32_t>(last - '0') : 1u;
}

std::string Scalar(std::string_view type) {
    return std::string(Components(type) == 1u ? type : type.substr(0, type.size() - 1u));
}

bool Arrayed(const Member& member) {
    return member.count > 1u || member.attribute == "clip_distance";
}

Entry ParseEntry(std::string_view source, std::string_view signature) {
    Entry entry;
    entry.at = source.find(signature);
    if (entry.at == std::string_view::npos || (entry.at != 0u && source[entry.at - 1u] != '\n')) Unsupported("the source has no entry point \"" + std::string(signature) + "\"");
    const auto open = source.find(")\n{\n", entry.at);
    if (open == std::string_view::npos || !source.ends_with("}\n")) Unsupported("the entry point is not one function at the end of the source");
    entry.inputs = Struct(source, "aps5_stage_in");
    entry.outputs = Struct(source, "aps5_stage_out");
    if (entry.outputs.begin == std::string_view::npos) Unsupported("the source has no stage output struct");
    if (entry.inputs.begin != std::string_view::npos && (entry.inputs.begin > entry.outputs.begin || !Trim(source.substr(entry.inputs.end, entry.outputs.begin - entry.inputs.end)).empty())) Unsupported("the stage input struct does not precede the output struct");
    if (!Trim(source.substr(entry.outputs.end, entry.at - entry.outputs.end)).empty()) Unsupported("the stage output struct does not precede the entry point");
    if (source.find("struct aps5_fetch {") != std::string_view::npos || source.find("aps5_rect_") != std::string_view::npos || source.find("aps5_ts_") != std::string_view::npos) Unsupported("the source already declares wrapper names");
    for (const auto line : Lines(source.substr(entry.at + signature.size(), open - entry.at - signature.size()))) {
        std::string_view before;
        std::string_view after;
        Parameter parameter;
        Attributed(line, before, parameter.attribute, after);
        if (!after.empty()) Unsupported("parameter \"" + std::string(line) + "\" has an unexpected suffix");
        parameter.declaration = std::string(before);
        parameter.name = LastToken(before);
        entry.parameters.push_back(std::move(parameter));
    }
    for (auto& member : Members(entry.outputs.body)) {
        if (member.attribute == "render_target_array_index" || member.attribute == "viewport_array_index") Unsupported("the function exports a layer or viewport, which mesh vertices cannot carry");
        if (member.attribute == "point_size") continue;
        if (member.attribute != "position" && member.attribute != "clip_distance" && !member.attribute.starts_with("user(")) Unsupported("output " + member.name + " has attribute " + member.attribute);
        entry.vertexBytes += MemberBytes(member);
        entry.outputMembers.push_back(std::move(member));
    }
    entry.vertexBytes = (entry.vertexBytes + 15u) & ~15u;
    entry.body = source.substr(open + 4u, source.size() - 2u - open - 4u);
    return entry;
}

std::string OutputStruct(std::string_view name, const std::vector<Member>& members) {
    std::string text = "struct " + std::string(name) + " {\n";
    for (const auto& member : members) text += "    " + member.type + " " + member.name + " [[" + member.attribute + "]]" + (Arrayed(member) ? " [" + std::to_string(member.count) + "]" : std::string()) + ";\n";
    return text + "};\n";
}

std::string CopyOutputs(const std::vector<Member>& members, std::string_view target, std::string_view source, std::string_view indent) {
    std::string text;
    for (const auto& member : members) {
        const auto to = std::string(target) + "." + member.name;
        const auto from = std::string(source) + "." + member.name;
        if (Arrayed(member)) text += std::string(indent) + "for (uint aps5_k = 0u; aps5_k < " + std::to_string(member.count) + "u; ++aps5_k) " + to + "[aps5_k] = " + from + "[aps5_k];\n";
        else text += std::string(indent) + to + " = " + from + ";\n";
    }
    return text;
}

std::string Join(const std::vector<std::string>& items, std::string_view separator) {
    std::string joined;
    for (std::size_t index = 0; index < items.size(); ++index) joined += (index == 0u ? "" : std::string(separator)) + items[index];
    return joined;
}

}
