#include "MslImagePrograms.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

using namespace ShaderRecompiler;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckProgram(const MslImagePrograms::Program& program, std::uint64_t address) {
    const auto userData = MslImagePrograms::UserData(program.format);
    const auto result = Recompile(MslImagePrograms::Request(program, address, userData));
    const auto source = std::string(result.msl.Source());
    const auto name = std::string(program.name);
    if (std::getenv("APS5_MSL_DUMP") != nullptr) std::printf("// ---- %s\n%s\n", name.c_str(), source.c_str());
    Require(result.backend == ShaderBackendKind::Msl, name + ": the result is not an MSL artifact");
    Require(result.spirv.empty(), name + ": an MSL artifact also carries SPIR-V");
    Require(source.find("device const ulong* aps5_images [[buffer(7)]]") != std::string::npos, name + ": the kernel takes no image table");
    for (const auto marker : program.markers) Require(source.find(marker) != std::string::npos, name + ": the source lacks " + std::string(marker));
    const bool sampling = program.kind == MslImagePrograms::Kind::Sample || program.kind == MslImagePrograms::Kind::CompileOnly;
    Require((source.find("device const ulong* aps5_samplers [[buffer(8)]]") != std::string::npos) == sampling, name + ": the sampler table is bound " + (sampling ? "nowhere" : "without a sampler"));
    Require(source.find("#pragma") == std::string::npos, name + ": the source carries a pragma");
}

void CheckUnsupportedRejected() {
    const auto programs = MslImagePrograms::Programs();
    auto program = programs[3];
    for (auto& word : program.code) {
        if (word == MslImagePrograms::Mimg(0x11u, 0x1u)) word = MslImagePrograms::Mimg(0x0fu, 0x1u);
    }
    bool rejected = false;
    try {
        static_cast<void>(Recompile(MslImagePrograms::Request(program, 0x7f000u, MslImagePrograms::UserData(program.format))));
    } catch (const MslUnsupported&) {
        rejected = true;
    } catch (const std::exception&) {
        rejected = true;
    }
    Require(rejected, "an image op the Metal backend cannot express recompiled to MSL");
}

}

int main() {
    try {
        std::uint64_t address = 0x40000u;
        for (const auto& program : MslImagePrograms::Programs()) {
            CheckProgram(program, address);
            address += 0x1000u;
        }
        CheckUnsupportedRejected();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MslImages: %s\n", error.what());
        return 1;
    }
    std::puts("MslImages: image reads, writes, atomics, queries, sampling and gathers recompile to MSL texture calls");
    return 0;
}
