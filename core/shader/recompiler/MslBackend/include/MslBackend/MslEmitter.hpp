#ifndef CORE_SHADER_RECOMPILER_MSLBACKEND_INCLUDE_MSLBACKEND_MSLEMITTER_HPP
#define CORE_SHADER_RECOMPILER_MSLBACKEND_INCLUDE_MSLBACKEND_MSLEMITTER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "Recompiler.hpp"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ShaderRecompiler {

struct MslUnsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class MslEmitter {
public:
    [[nodiscard]] std::string Emit(const IrProgram& program, const ShaderStageInputInfo& inputInfo, const CompiledBindingLayout& bindings, const MslTarget& target) const;
};

[[nodiscard]] SpirvTarget MslFrontendTarget(const MslTarget& target);

enum class MslGeometryKind : std::uint32_t {
    Local = 1,
    Hull = 2,
    Domain = 3,
    Mesh = 4
};

struct MslGeometryLayout {
    MslGeometryKind kind = MslGeometryKind::Local;
    std::uint32_t localStride = 0;
    std::uint32_t controlStride = 0;
    std::uint32_t patchStride = 0;
    std::uint32_t inputControlPoints = 0;
    std::uint32_t outputControlPoints = 0;
    std::uint32_t domain = 0;
    std::uint32_t partitioning = 0;
    std::uint32_t outputTopology = 0;
    std::uint32_t meshVertices = 0;
    std::uint32_t meshPrimitives = 0;
    std::uint32_t meshThreads = 0;
    std::uint32_t meshPrimitiveSize = 0;

    [[nodiscard]] std::uint32_t LocalWords() const { return (localStride + 3u) / 4u + 1u; }
    [[nodiscard]] std::uint32_t ControlWords() const { return (controlStride + 3u) / 4u; }
    [[nodiscard]] std::uint32_t PatchWords() const { return (patchStride + 3u) / 4u; }
};

inline constexpr std::uint32_t MslGeometryLayoutFields = 13u;

[[nodiscard]] std::optional<MslGeometryLayout> ParseMslGeometryLayout(std::string_view source);

}

#endif
