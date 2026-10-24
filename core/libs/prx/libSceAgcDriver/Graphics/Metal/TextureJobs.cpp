#include "prx/libSceAgcDriver/Graphics/Metal/TextureJobs.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics::Metal {

namespace {

void require(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error("AGC graphics: " + reason);
}

std::uint32_t thickEquationMode(TextureTileMode tileMode) {
    switch (tileMode) {
        case TextureTileMode::kStandard4KB: return 0x105u;
        case TextureTileMode::kS4KBX: return 0x115u;
        case TextureTileMode::kStandard64KB: return 0x109u;
        case TextureTileMode::kS64KBX: return 0x119u;
        case TextureTileMode::kD64KBX: return 0x11au;
        default: return 0u;
    }
}

}

std::uint32_t BlockBytesForTileMode(TextureTileMode tileMode) {
    switch (tileMode) {
        case TextureTileMode::kLinear: return 0u;
        case TextureTileMode::kStandard256B:
        case TextureTileMode::kD256B: return 256u;
        case TextureTileMode::kStandard4KB:
        case TextureTileMode::kD4KB:
        case TextureTileMode::kS4KBX:
        case TextureTileMode::kD4KBX: return 4096u;
        case TextureTileMode::kStandard64KB:
        case TextureTileMode::kD64KB:
        case TextureTileMode::kS64KBT:
        case TextureTileMode::kD64KBT:
        case TextureTileMode::kZ64KBX:
        case TextureTileMode::kS64KBX:
        case TextureTileMode::kD64KBX:
        case TextureTileMode::kR64KBX: return 65536u;
    }
    throw std::runtime_error("AGC graphics: Metal texture detiling met an unknown tile mode");
}

AgcDetileParams MakeDetileParams(const GuestTextureResource& descriptor, const SurfaceGeometry& geometry, std::uint32_t level, std::uint32_t layer, std::uint64_t surfaceAddress) {
    const auto* format = FindFormat(descriptor.format);
    require(format != nullptr && format->uploadable, "texture format " + std::to_string(descriptor.format) + " cannot be detiled on Metal");
    const auto elementBytes = format->guestElementBytes;
    require(std::has_single_bit(elementBytes) && elementBytes <= 16u, "unsupported element size for texture detiling");
    const auto& mip = geometry.mips.at(level);
    AgcDetileParams params{};
    params.tiled = surfaceAddress + geometry.GuestLayerOffset(layer) + mip.tiledOffset;
    params.width = mip.width;
    params.height = mip.height;
    params.pitchBytes = mip.pitchBytes;
    params.blocksPerRow = mip.blocksPerRow;
    params.tail = mip.tail ? 1u : 0u;
    params.tailX = mip.tailX;
    params.tailY = mip.tailY;
    params.slice = layer;
    params.pipeBankXor = descriptor.pipeBankXor;
    params.rangeBegin = 0u;
    params.rangeEnd = 0xffffffffu;
    params.destinationSlice = layer;
    params.family = AGC_FAMILY_LINEAR;
    if (descriptor.tileMode == TextureTileMode::kLinear) return params;
    params.blockBytes = BlockBytesForTileMode(descriptor.tileMode);
    params.family = AGC_FAMILY_STANDARD;
    if (geometry.thick) {
        const auto mode = thickEquationMode(descriptor.tileMode);
        require(mode != 0u, "3D textures are only detiled thick from SW_4KB_S, SW_4KB_S_X, SW_64KB_S, SW_64KB_S_X or SW_64KB_D_X");
        const auto* equation = FindTextureSwizzleEquation(mode, elementBytes);
        require(equation != nullptr, "no thick swizzle equation for the element size");
        const auto extent = ThickBlockExtent(descriptor.tileMode, elementBytes);
        params.family = AGC_FAMILY_EQUATION;
        std::copy(equation->bits.begin(), equation->bits.end(), params.equation);
        params.blockLog2Width = static_cast<std::uint32_t>(std::countr_zero(extent[0]));
        params.blockLog2Height = static_cast<std::uint32_t>(std::countr_zero(extent[1]));
        return params;
    }
    const auto block = ThinBlockLayout(descriptor.tileMode, elementBytes);
    params.blockLog2Width = static_cast<std::uint32_t>(std::countr_zero(block[1]));
    params.blockLog2Height = static_cast<std::uint32_t>(std::countr_zero(block[2]));
    if (const auto mode = EquationSwizzleMode(descriptor.tileMode); mode != 0u) {
        const auto* equation = FindTextureSwizzleEquation(mode, elementBytes);
        require(equation != nullptr, "no swizzle equation for tile mode " + std::to_string(mode) + " at " + std::to_string(elementBytes) + " bytes per element");
        params.family = AGC_FAMILY_EQUATION;
        std::copy(equation->bits.begin(), equation->bits.end(), params.equation);
    }
    return params;
}

SurfacePlan PlanSurface(const GuestTextureResource& descriptor, std::uint64_t surfaceAddress) {
    const auto* format = FindFormat(descriptor.format);
    require(format != nullptr && format->uploadable, "texture format " + std::to_string(descriptor.format) + " cannot be detiled on Metal");
    SurfacePlan plan;
    plan.geometry = DescribeSurface(descriptor);
    plan.elementBytes = format->guestElementBytes;
    plan.texelBytes = format->texelBytes;
    plan.volume = plan.geometry.imageDepth > 1u;
    const auto& geometry = plan.geometry;
    plan.jobs.reserve(static_cast<std::size_t>(geometry.layers) * descriptor.mipCount);
    for (std::uint32_t layer = 0; layer < geometry.layers; ++layer) {
        for (std::uint32_t level = 0; level < descriptor.mipCount; ++level) {
            if (!geometry.HasLayer(level, layer)) continue;
            MipJob job;
            job.level = level;
            job.layer = layer;
            job.destinationLayer = geometry.CopyLayer(layer);
            job.destinationDepth = static_cast<std::uint32_t>(geometry.CopyDepth(layer));
            job.params = MakeDetileParams(descriptor, geometry, level, layer, surfaceAddress);
            job.columns = geometry.mips[level].width;
            job.rows = geometry.mips[level].height;
            job.linearOffset = geometry.LinearLayerOffset(layer) + geometry.mips[level].linearOffset;
            plan.jobs.push_back(job);
        }
    }
    return plan;
}

}
