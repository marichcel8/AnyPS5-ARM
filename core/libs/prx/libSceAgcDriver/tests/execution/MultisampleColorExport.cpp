#include "DepthFastClearHarness.hpp"
#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTextureLayout.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::ColorTileMode;
using AgcDriver::Graphics::MultisampleTextureLayout;
using AgcDriver::Graphics::Require;
using DepthFastClearHarness::Height;
using DepthFastClearHarness::Shaders;
using DepthFastClearHarness::Width;

constexpr std::size_t BlockBytes = 1u << 20u;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(65536, BlockBytes));
#endif
        Require(block != nullptr, "multisample color export: cannot allocate the guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }
    void Clear() { std::memset(block, 0, BlockBytes); }

private:
    std::uint8_t* block = nullptr;
};

struct Counts {
    std::uint32_t full = 0;
    std::uint32_t empty = 0;
    std::uint32_t mixed = 0;
};

Counts DrawAndCount(AgcDriver::VulkanDevice& device, const Shaders& compiled, std::uint8_t* arena, std::uint32_t samples, ColorTileMode tile, const std::string& what) {
    static std::uint32_t slot = 0;
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderRecompiler::ShaderStage::Vertex, &compiled.vertex, 0},
        {ShaderRecompiler::ShaderStage::Fragment, &compiled.pixel, PixelPushOffset(compiled.vertexPush, device.Target())}
    }};
    const MultisampleTextureLayout layout(AgcDriver::Graphics::ColorTextureTileMode(tile), 4u, samples, Width, Height, 1u);
    Require(layout.Bytes() <= 65536u && (slot + 1u) * 65536u <= BlockBytes, what + ": the layout does not fit its slot of the guest block");
    auto* block = arena + static_cast<std::size_t>(slot++) * 65536u;
    std::memset(block, 0, layout.Bytes());
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(block), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, static_cast<std::size_t>(layout.Bytes()), 0xe4u};
    state.color.tileMode = tile;
    state.color.samples = samples;
    state.color.coverageSamples = samples;
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.samples = samples;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, 3, 0, 1, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
    Counts counts;
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            std::uint32_t covered = 0;
            for (std::uint32_t sample = 0; sample < samples; ++sample) {
                std::uint32_t texel = 0;
                std::memcpy(&texel, block + layout.Offset(x, y, 0, sample), sizeof(texel));
                Require(texel == 0u || texel == 0xffffffffu, what + ": pixel " + std::to_string(x) + "," + std::to_string(y) + " sample " + std::to_string(sample) + " holds " + std::to_string(texel) + ", expected 0 or 0xffffffff");
                if (texel != 0u) ++covered;
            }
            if (covered == samples) ++counts.full;
            else if (covered == 0u) ++counts.empty;
            else ++counts.mixed;
        }
    }
    return counts;
}

alignas(16) const std::array<std::array<float, 4>, 3> Full{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};
alignas(16) const std::array<std::array<float, 4>, 3> Half{{{-1.0f, -1.0f, 0.5f, 1.0f}, {1.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 1.0f, 0.5f, 1.0f}}};

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock block;
        const auto fullShaders = DepthFastClearHarness::Compile(*device, Full.data());
        const auto halfShaders = DepthFastClearHarness::Compile(*device, Half.data());
        for (const auto samples : {2u, 4u, 8u}) {
            for (const auto tile : {ColorTileMode::Z64KBX, ColorTileMode::RenderTarget}) {
                const std::string what = std::to_string(samples) + " samples, tile mode " + std::to_string(static_cast<std::uint32_t>(tile));
                try {
                    const auto full = DrawAndCount(*device, fullShaders, block.Data(), samples, tile, what + ", full triangle");
                    Require(full.full == Width * Height && full.mixed == 0u && full.empty == 0u, what + ": the full triangle left " + std::to_string(full.empty) + " empty and " + std::to_string(full.mixed) + " mixed pixels in guest memory");
                    const auto half = DrawAndCount(*device, halfShaders, block.Data(), samples, tile, what + ", half triangle");
                    Require(half.full > 0u && half.empty > 0u && half.mixed > 0u, what + ": the half triangle left " + std::to_string(half.full) + " full, " + std::to_string(half.empty) + " empty and " + std::to_string(half.mixed) + " mixed pixels, so the samples were not exported one by one");
                    Require(half.mixed < 2u * (Width + Height), what + ": the half triangle left " + std::to_string(half.mixed) + " mixed pixels, more than its edge can cross");
                } catch (const AgcDriver::Graphics::UnrenderableDraw& refusal) {
                    std::printf("%s: skipped, %s\n", what.c_str(), refusal.what());
                }
            }
        }
        std::puts("multisample color export tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
