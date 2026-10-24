#ifndef CORE_SHADER_RECOMPILER_MSLBACKEND_INCLUDE_MSLBACKEND_MSLEMITTER_HPP
#define CORE_SHADER_RECOMPILER_MSLBACKEND_INCLUDE_MSLBACKEND_MSLEMITTER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "Recompiler.hpp"
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

struct MslUnsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class MslEmitter {
public:
    [[nodiscard]] std::string Emit(const IrProgram& program, const ShaderStageInputInfo& inputInfo, const CompiledBindingLayout& bindings, const MslTarget& target) const;
};

[[nodiscard]] SpirvTarget MslFrontendTarget(const MslTarget& target);

}

#endif
