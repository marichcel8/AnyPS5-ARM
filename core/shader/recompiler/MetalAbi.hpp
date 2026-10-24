#ifndef CORE_SHADER_RECOMPILER_METALABI_HPP
#define CORE_SHADER_RECOMPILER_METALABI_HPP

#include "Recompiler.hpp"
#include "RuntimeAbi.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace ShaderRecompiler::MetalAbi {

inline constexpr std::uint32_t Version = 1u;
inline constexpr std::uint32_t MaxBufferSlots = 31u;
inline constexpr std::uint32_t MaxTextureSlots = 128u;
inline constexpr std::uint32_t MaxSamplerSlots = 16u;

enum class BufferSlot : std::uint32_t {
    ShaderData = 0u,
    PushConstants = 1u,
    BdaPagetable = 2u,
    FaultBuffer = 3u,
    FlattenedSrt = 4u,
    Gds = 5u,
    GuestBuffers = 6u,
    GuestImages = 7u,
    GuestSamplers = 8u,
    ImageTable = 9u,
    ImageStorageTable = 10u,
    SamplerTable = 11u,
    ImageTableMap = 12u,
    StageLink = 13u,
    Count = 14u
};

enum class TableEntry : std::uint8_t {
    None,
    Inline,
    BufferPointer,
    ResourceId
};

struct GuestBufferEntry {
    std::uint64_t gpuAddress;
    std::uint32_t sizeBytes;
    std::uint32_t flags;
};

struct ResourceIdEntry {
    std::uint64_t resourceId;
};

inline constexpr std::uint32_t PushConstantBytes = RuntimeAbi::PushConstantDwords * sizeof(std::uint32_t);
inline constexpr std::uint32_t GuestBufferTableBytes = RuntimeAbi::BufferCapacity * sizeof(GuestBufferEntry);
inline constexpr std::uint32_t GuestImageTableEntries = (RuntimeAbi::FirstStorageImageBinding - RuntimeAbi::FirstImageBinding) * RuntimeAbi::SampledHeapCapacity + (static_cast<std::uint32_t>(RuntimeAbi::Binding::Samplers) - RuntimeAbi::FirstStorageImageBinding) * RuntimeAbi::StorageHeapCapacity;
inline constexpr std::uint32_t GuestImageTableBytes = GuestImageTableEntries * sizeof(ResourceIdEntry);
inline constexpr std::uint32_t GuestSamplerTableBytes = RuntimeAbi::SamplerHeapCapacity * sizeof(ResourceIdEntry);
inline constexpr std::uint32_t PositionFlipYConstant = 60000u;
inline constexpr std::uint32_t DepthNegativeOneToOneConstant = 60001u;

struct ArgumentSlot {
    BufferSlot slot;
    TableEntry entry;
};

constexpr std::optional<ArgumentSlot> SlotFor(DescriptorRole role) {
    switch (role) {
    case DescriptorRole::ShaderData: return ArgumentSlot{BufferSlot::ShaderData, TableEntry::Inline};
    case DescriptorRole::BdaPagetable: return ArgumentSlot{BufferSlot::BdaPagetable, TableEntry::Inline};
    case DescriptorRole::FaultBuffer: return ArgumentSlot{BufferSlot::FaultBuffer, TableEntry::Inline};
    case DescriptorRole::FlattenedSrt: return ArgumentSlot{BufferSlot::FlattenedSrt, TableEntry::Inline};
    case DescriptorRole::Gds: return ArgumentSlot{BufferSlot::Gds, TableEntry::Inline};
    case DescriptorRole::GuestBuffers: return ArgumentSlot{BufferSlot::GuestBuffers, TableEntry::BufferPointer};
    case DescriptorRole::GuestImages: return ArgumentSlot{BufferSlot::GuestImages, TableEntry::ResourceId};
    case DescriptorRole::GuestSamplers: return ArgumentSlot{BufferSlot::GuestSamplers, TableEntry::ResourceId};
    case DescriptorRole::ImageTable: return ArgumentSlot{BufferSlot::ImageTable, TableEntry::ResourceId};
    case DescriptorRole::ImageStorageTable: return ArgumentSlot{BufferSlot::ImageStorageTable, TableEntry::ResourceId};
    case DescriptorRole::SamplerTable: return ArgumentSlot{BufferSlot::SamplerTable, TableEntry::ResourceId};
    case DescriptorRole::ImageTableMap: return ArgumentSlot{BufferSlot::ImageTableMap, TableEntry::Inline};
    }
    return std::nullopt;
}

constexpr std::optional<ArgumentSlot> SlotFor(RuntimeAbi::Binding binding) {
    switch (binding) {
    case RuntimeAbi::Binding::Buffers: return SlotFor(DescriptorRole::GuestBuffers);
    case RuntimeAbi::Binding::Samplers: return SlotFor(DescriptorRole::GuestSamplers);
    case RuntimeAbi::Binding::Gds: return SlotFor(DescriptorRole::Gds);
    case RuntimeAbi::Binding::BdaPagetable: return SlotFor(DescriptorRole::BdaPagetable);
    case RuntimeAbi::Binding::FaultBuffer: return SlotFor(DescriptorRole::FaultBuffer);
    case RuntimeAbi::Binding::FlattenedSrt: return SlotFor(DescriptorRole::FlattenedSrt);
    case RuntimeAbi::Binding::ShaderData: return SlotFor(DescriptorRole::ShaderData);
    case RuntimeAbi::Binding::SamplerTable: return SlotFor(DescriptorRole::SamplerTable);
    case RuntimeAbi::Binding::ImageTableMap: return SlotFor(DescriptorRole::ImageTableMap);
    case RuntimeAbi::Binding::ImageTable: return SlotFor(DescriptorRole::ImageTable);
    case RuntimeAbi::Binding::ImageStorageTable: return SlotFor(DescriptorRole::ImageStorageTable);
    case RuntimeAbi::Binding::Count: return std::nullopt;
    }
    return std::nullopt;
}

constexpr std::uint32_t GuestImageIndex(std::uint32_t binding, std::uint32_t element) {
    const auto heap = binding >= RuntimeAbi::FirstStorageImageBinding ? RuntimeAbi::StorageHeapCapacity : RuntimeAbi::SampledHeapCapacity;
    return (binding - RuntimeAbi::FirstImageBinding) * heap + element;
}

constexpr std::uint32_t SlotIndex(BufferSlot slot) {
    return static_cast<std::uint32_t>(slot);
}

static_assert(SlotIndex(BufferSlot::Count) <= MaxBufferSlots);
static_assert(RuntimeAbi::SampledHeapCapacity == RuntimeAbi::StorageHeapCapacity);
static_assert(GuestImageTableEntries == RuntimeAbi::ImageBindingCount * RuntimeAbi::SampledHeapCapacity);
static_assert(GuestImageIndex(RuntimeAbi::FirstImageBinding + RuntimeAbi::ImageBindingCount - 1u, RuntimeAbi::StorageHeapCapacity - 1u) == GuestImageTableEntries - 1u);
static_assert(std::is_standard_layout_v<GuestBufferEntry> && std::is_trivially_copyable_v<GuestBufferEntry> && sizeof(GuestBufferEntry) == 16u);
static_assert(std::is_standard_layout_v<ResourceIdEntry> && std::is_trivially_copyable_v<ResourceIdEntry> && sizeof(ResourceIdEntry) == 8u);
static_assert(offsetof(GuestBufferEntry, sizeBytes) == 8u);

}

#endif
