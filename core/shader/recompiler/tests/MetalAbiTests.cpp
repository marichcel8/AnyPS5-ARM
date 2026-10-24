#include "MetalAbi.hpp"
#include "CacheKey.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

namespace ShaderRecompiler {

bool DebugProbeActive() { return false; }
bool RayTracingStrict() { return false; }
bool RayTracingMiss() { return false; }

}

namespace {

void Require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "failed: %s\n", message);
        std::exit(1);
    }
}

constexpr ShaderRecompiler::DescriptorRole Roles[] = {
    ShaderRecompiler::DescriptorRole::GuestBuffers,
    ShaderRecompiler::DescriptorRole::GuestImages,
    ShaderRecompiler::DescriptorRole::GuestSamplers,
    ShaderRecompiler::DescriptorRole::Gds,
    ShaderRecompiler::DescriptorRole::BdaPagetable,
    ShaderRecompiler::DescriptorRole::FaultBuffer,
    ShaderRecompiler::DescriptorRole::FlattenedSrt,
    ShaderRecompiler::DescriptorRole::ShaderData,
    ShaderRecompiler::DescriptorRole::ImageTable,
    ShaderRecompiler::DescriptorRole::SamplerTable,
    ShaderRecompiler::DescriptorRole::ImageTableMap,
    ShaderRecompiler::DescriptorRole::ImageStorageTable};

}

int main() {
    using namespace ShaderRecompiler;

    std::set<std::uint32_t> slots;
    for (const auto role : Roles) {
        const auto slot = MetalAbi::SlotFor(role);
        Require(slot.has_value(), "every descriptor role has a Metal slot");
        Require(slot->slot < MetalAbi::BufferSlot::StageLink, "role slots precede the stage link slot");
        Require(slots.insert(MetalAbi::SlotIndex(slot->slot)).second, "role slots are distinct");
    }
    Require(slots.size() == std::size(Roles), "all roles mapped");
    Require(!slots.contains(MetalAbi::SlotIndex(MetalAbi::BufferSlot::PushConstants)), "push constants have their own slot");
    for (std::uint32_t binding = 0; binding < static_cast<std::uint32_t>(RuntimeAbi::Binding::Count); ++binding) {
        const auto abi = static_cast<RuntimeAbi::Binding>(binding);
        const bool typedHeap = binding >= RuntimeAbi::FirstImageBinding && binding < static_cast<std::uint32_t>(RuntimeAbi::Binding::Samplers);
        Require(MetalAbi::SlotFor(abi).has_value() != typedHeap, "runtime bindings map unless they are typed image heaps");
    }
    Require(MetalAbi::GuestImageIndex(RuntimeAbi::FirstImageBinding, 0) == 0u, "first image table entry");
    Require(MetalAbi::GuestImageIndex(RuntimeAbi::FirstStorageImageBinding, 0) == (RuntimeAbi::FirstStorageImageBinding - RuntimeAbi::FirstImageBinding) * RuntimeAbi::SampledHeapCapacity, "storage heaps follow sampled heaps");

    RecompileRequest spirv{};
    spirv.shader.stage = ShaderStage::Compute;
    spirv.context.compute = ShaderComputeStageInfo{};
    RecompileRequest msl = spirv;
    msl.backend = ShaderBackendKind::Msl;
    msl.msl.metalVersion = 400u;
    msl.msl.maxThreadgroupSize = {1024u, 1024u, 1024u};

    std::vector<std::uint64_t> spirvKey;
    std::vector<std::uint64_t> mslKey;
    RecompileCacheKey::Build(spirv, spirvKey);
    RecompileCacheKey::Build(msl, mslKey);
    Require(spirvKey != mslKey, "backends have distinct cache keys");
    auto tuned = msl;
    tuned.msl.relaxedFloat = !tuned.msl.relaxedFloat;
    std::vector<std::uint64_t> tunedKey;
    RecompileCacheKey::Build(tuned, tunedKey);
    Require(tunedKey != mslKey, "the MSL target is part of the key");
    auto spirvOnlyMsl = spirv;
    spirvOnlyMsl.msl.metalVersion = 7u;
    std::vector<std::uint64_t> ignoredKey;
    RecompileCacheKey::Build(spirvOnlyMsl, ignoredKey);
    Require(ignoredKey == spirvKey, "an unused MSL target does not change a SPIR-V key");

    RecompileResult one;
    one.backend = ShaderBackendKind::Msl;
    one.msl = SharedMsl(std::string("kernel void aps5_main() {}"));
    RecompileResult two = one;
    two.msl = SharedMsl(std::string("kernel void aps5_main() {}"));
    RecompileResult other;
    other.backend = ShaderBackendKind::Msl;
    other.msl = SharedMsl(std::string("kernel void aps5_main() { }"));
    Require(!one.Identity().Empty(), "an MSL result has an identity");
    Require(one.Identity() == two.Identity(), "equal sources have equal identities");
    Require(one.Identity() != other.Identity(), "different sources have different identities");
    RecompileResult empty;
    empty.backend = ShaderBackendKind::Msl;
    Require(empty.Identity().Empty(), "an empty module has no identity");

    const RequestSerializer serializer;
    const auto text = serializer.Serialize(msl);
    const auto replay = serializer.Deserialize(text);
    Require(replay.request.backend == ShaderBackendKind::Msl && replay.request.msl.metalVersion == 400u && replay.request.msl.maxThreadgroupSize[0] == 1024u, "MSL request round trip");
    const auto spirvReplay = serializer.Deserialize(serializer.Serialize(spirv));
    Require(spirvReplay.request.backend == ShaderBackendKind::Spirv, "SPIR-V request round trip");

    return 0;
}
