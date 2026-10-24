#include "MslComputePrograms.hpp"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

using namespace ShaderRecompiler;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckProgram(const MslComputePrograms::Program& program, std::uint64_t address, std::uint32_t waveSize) {
    const auto userData = MslComputePrograms::UserData();
    const auto result = Recompile(MslComputePrograms::Request(program, address, userData, waveSize));
    const auto source = std::string(result.msl.Source());
    const auto name = std::string(program.name) + " (wave" + std::to_string(waveSize) + ")";
    Require(result.backend == ShaderBackendKind::Msl, name + ": the result is not an MSL artifact");
    Require(result.spirv.empty(), name + ": an MSL artifact also carries SPIR-V");
    Require(source.find("kernel void " + std::string(MslEntryPoint) + "(") != std::string::npos, name + ": the source has no " + std::string(MslEntryPoint) + " kernel");
    Require(source.find("[[max_total_threads_per_threadgroup(64)]]") != std::string::npos, name + ": the kernel does not declare its threadgroup size");
    Require(source.find(program.marker) != std::string::npos, name + ": the source lacks " + std::string(program.marker));
    Require(source.find("#pragma") == std::string::npos, name + ": the source carries a pragma");
    if (!program.bda) return;
    Require(result.bdaAbiVersion != 0u, name + ": a program with physical addresses carries no BDA ABI version");
    Require(source.find("aps5_note_write(thread aps5_ctx& c, ulong address)") != std::string::npos, name + ": a BDA store does not note written pages");
}

void CheckWave64Rejected() {
    const auto programs = MslComputePrograms::Programs();
    const auto userData = MslComputePrograms::UserData();
    auto request = MslComputePrograms::Request(programs.front(), 0x7f000u, userData, 64u);
    request.context.compute = ShaderComputeStageInfo{{128u, 1u, 1u}, 128u, {false, false, false}, false, 1u};
    bool rejected = false;
    try {
        static_cast<void>(Recompile(request));
    } catch (const std::exception&) {
        rejected = true;
    }
    Require(rejected, "a wave64 program with 128 threads recompiled to MSL although a wave there needs two guest lanes per thread");
}

}

int main() {
    try {
        std::uint64_t address = 0x40000u;
        for (const auto& program : MslComputePrograms::Programs()) {
            for (const auto waveSize : {32u, 64u}) {
                if ((program.waves & (waveSize == 32u ? MslComputePrograms::Wave32 : MslComputePrograms::Wave64)) == 0u) continue;
                CheckProgram(program, address, waveSize);
                address += 0x1000u;
            }
        }
        CheckWave64Rejected();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MslCompute: %s\n", error.what());
        return 1;
    }
    std::puts("MslCompute: wave32 and wave64 compute programs recompile to MSL kernels with the expected memory, SIMD-group and cross-lane paths");
    return 0;
}
