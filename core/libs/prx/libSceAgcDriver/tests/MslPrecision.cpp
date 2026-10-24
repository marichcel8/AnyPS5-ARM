#include "MslPrecisionPrograms.hpp"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

using namespace ShaderRecompiler;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckProgram(const MslPrecisionPrograms::Program& program, std::uint64_t address) {
    const auto compute = MslPrecisionPrograms::Compute(program);
    const auto name = std::string(program.name);
    RecompileResult result;
    try {
        result = Recompile(MslPrecisionPrograms::Request(program, compute, address));
    } catch (const std::exception& error) {
        throw std::runtime_error(name + ": " + error.what());
    }
    const auto source = std::string(result.msl.Source());
    Require(result.backend == ShaderBackendKind::Msl, name + ": the result is not an MSL artifact");
    for (const auto marker : program.present) Require(source.find(marker) != std::string::npos, name + ": the source lacks " + std::string(marker));
    for (const auto marker : program.absent) Require(source.find(marker) == std::string::npos, name + ": the source carries " + std::string(marker));
}

void CheckInputs() {
    using MslPrecisionPrograms::Check;
    Require(MslPrecisionPrograms::Matches(Check::MaxOne, 0x7fc00000u, 0x3f800000u), "max with a NaN does not expect the other operand");
    Require(!MslPrecisionPrograms::Matches(Check::IsNan, 0x7f800000u, 1u), "an infinity counts as NaN");
    Require(MslPrecisionPrograms::Matches(Check::Float64, 0x40400000u, MslPrecisionPrograms::Bits(std::sqrt(10.0f))), "the fp64 reference does not round sqrt(x * x + 1)");
    Require(MslPrecisionPrograms::Matches(Check::Dot2, 0x3c004000u, MslPrecisionPrograms::Bits(5.0f)), "the dot2 reference does not sum the squared halves");
}

}

int main() {
    try {
        CheckInputs();
        std::uint64_t address = 0x60000u;
        for (const auto& program : MslPrecisionPrograms::Programs()) {
            CheckProgram(program, address);
            address += 0x1000u;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MslPrecision: %s\n", error.what());
        return 1;
    }
    std::puts("MslPrecision: float programs emit exact helpers only where the guest observes NaN, fp64 as soft-float and dot2 exactly");
    return 0;
}
