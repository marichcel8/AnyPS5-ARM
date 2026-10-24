#ifndef CORE_SHADER_RECOMPILER_MSLBACKEND_SRC_MSLSTATE_HPP
#define CORE_SHADER_RECOMPILER_MSLBACKEND_SRC_MSLSTATE_HPP

#include "MslBackend/MslEmitter.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler::Msl {

enum Feature : std::uint32_t {
    ShaderData = 1u << 0u,
    PushData = 1u << 1u,
    FlattenedSrt = 1u << 2u,
    GuestBuffers = 1u << 3u,
    Bda = 1u << 4u,
    Fault = 1u << 5u,
    Lds = 1u << 6u,
    Scratch = 1u << 7u,
    VectorWindow = 1u << 8u,
    WorkgroupId = 1u << 9u,
    LocalId = 1u << 10u,
    GlobalId = 1u << 11u,
    LocalIndex = 1u << 12u,
    Lane = 1u << 13u,
    BdaWrites = 1u << 14u,
    Half = 1u << 15u,
    FloatClass = 1u << 16u,
    Descriptors = 1u << 17u,
    Wave = 1u << 18u,
    Permute = 1u << 20u,
    Images = 1u << 21u,
    Samplers = 1u << 22u,
    IeeeFloat = 1u << 23u,
    StrictFloat = 1u << 24u,
    Float64 = 1u << 25u,
    Dot2 = 1u << 26u,
    Atomic64 = 1u << 27u,
    UpdateTable = 1u << 28u
};

enum : std::uint8_t {
    PrecisionFast = 0,
    PrecisionIeee = 1,
    PrecisionStrict = 2
};

struct State {
    const IrProgram& program;
    const ShaderStageInputInfo& input;
    const IrBindingLayout& layout;
    const MslTarget& target;
    std::string body;
    std::uint32_t depth = 1;
    std::uint32_t features = 0;
    std::uint32_t temps = 0;
    std::uint32_t ldsDwords = 0;
    std::uint32_t scratchDwords = 0;
    std::uint32_t windowDwords = 0;
    std::uint32_t waveLds = 0;
    bool stopAllowed = false;
    std::uint32_t lanes = 1;
    std::unordered_set<const IrValue*> hoisted;
    std::unordered_set<const IrValue*> defined;
    std::vector<const IrValue*> hoistOrder;
    std::map<std::uint32_t, std::uint32_t> constants;
    std::vector<std::string> imageTypes;
    std::unordered_map<const IrValue*, std::uint8_t> precision;
    std::string exit = "return;";
    std::map<std::string, std::string> stageInputs;
    std::map<std::string, std::string> stageOutputs;
    std::map<std::string, std::string> stageArguments;
    std::map<std::string, std::string> stageGlobals;
    bool validMask = false;
    std::uint32_t zeroClip = 0xffffffffu;
    std::map<std::uint32_t, std::uint32_t> localFetches;
    std::vector<std::string> prologue;
};

[[noreturn]] void Fail(const IrValue& inst, std::string_view reason);
[[noreturn]] void Fail(std::string_view reason);

std::uint32_t& CurrentHalf();
std::string HalfSuffix();
std::string Name(const IrValue& value);
std::string TypeName(IrType type);
std::string Literal(const IrValue& value);
std::string Value(State& state, const IrValue* value);
std::string Arg(State& state, const IrValue& inst, std::size_t index);
std::string Typed(State& state, const IrValue& inst, std::size_t index);
std::string U32(std::uint32_t value);
std::string Temp(State& state);
void Line(State& state, std::string_view text);
void Open(State& state, std::string_view text);
void Close(State& state, std::string_view text = "}");
void Define(State& state, const IrValue& inst, std::string_view expression);
void DefineZero(State& state, const IrValue& inst);
const MemoryInfo& Memory(const State& state, const IrValue& inst);

void AnalyzePrecision(State& state);
std::uint8_t Precision(const State& state, const IrValue& inst);
bool EmitAlu(State& state, const IrValue& inst);
bool EmitMemory(State& state, const IrValue& inst);
bool EmitWave(State& state, const IrValue& inst);
bool IsCrossLane(IrOpcode opcode);
std::string WaveLane(State& state);
std::string WaveAny(State& state, const std::string& zero, const std::string& one);
std::string HalfArg(State& state, const IrValue& inst, std::size_t index, std::uint32_t half);
void DefineHalves(State& state, const IrValue& inst, std::string_view zero, std::string_view one);
bool EmitImage(State& state, const IrValue& inst);
void EmitInstruction(State& state, const IrValue& inst);
void EmitBlockBody(State& state, const IrBlock& block);
void EmitFlow(State& state);
std::string Prelude(const State& state);
void PrepareStage(State& state);
std::string StageBuiltin(State& state, StageInputKind kind, std::uint32_t component);
bool EmitStage(State& state, const IrValue& inst);
std::string StageHeader(const State& state, const std::string& parameters);
void DeclareVertexOutputs(State& state);
std::string StageConstant(State& state, std::uint32_t id, std::uint32_t fallback);

bool IsGeometry(IrShaderStage stage);
void PrepareGeometry(State& state);
std::string GeometryBuiltin(State& state, StageInputKind kind, std::uint32_t component);
bool EmitGeometry(State& state, const IrValue& inst);
std::string LocalAttribute(State& state, std::uint32_t location, std::uint32_t component);
bool MeshAuxExport(State& state, const PositionExportComponent& output, const std::string& raw);
void MeshPrimitive(State& state, const std::string& value);
void CheckGeometry(const State& state);
std::string GeometryProgram(State& state, const std::string& prelude, const std::string& declarations, const std::string& parameters);

template <class Body>
void ForEachHalf(const State& state, Body&& body) {
    struct Restore {
        std::uint32_t previous = CurrentHalf();
        ~Restore() { CurrentHalf() = previous; }
    } restore;
    for (std::uint32_t half = 0; half < state.lanes; ++half) {
        CurrentHalf() = half;
        body(half);
    }
}

}

#endif
