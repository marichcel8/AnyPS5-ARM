#include "Recompiler.hpp"

#include <atomic>

namespace ShaderRecompiler {

namespace {

struct BackendDefaults {
    ShaderBackendKind backend = ShaderBackendKind::Spirv;
    MslTarget msl{};
};

std::atomic<const BackendDefaults*>& CurrentBackendDefaults() {
    static const BackendDefaults initial{};
    static std::atomic<const BackendDefaults*> current{&initial};
    return current;
}

}

void SetDefaultShaderBackend(ShaderBackendKind backend, const MslTarget& msl) {
    CurrentBackendDefaults().store(new BackendDefaults{backend, msl}, std::memory_order_release);
}

ShaderBackendKind DefaultShaderBackend() {
    return CurrentBackendDefaults().load(std::memory_order_acquire)->backend;
}

MslTarget DefaultMslTarget() {
    return CurrentBackendDefaults().load(std::memory_order_acquire)->msl;
}

}
