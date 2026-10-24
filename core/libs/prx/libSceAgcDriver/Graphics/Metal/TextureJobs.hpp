#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_TEXTUREJOBS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_TEXTUREJOBS_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/TextureAddress.h"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <cstdint>
#include <vector>

namespace AgcDriver::Graphics::Metal {

struct MipJob {
    std::uint32_t level = 0;
    std::uint32_t layer = 0;
    std::uint32_t destinationLayer = 0;
    std::uint32_t destinationDepth = 0;
    std::uint32_t columns = 0;
    std::uint32_t rows = 0;
    std::uint64_t linearOffset = 0;
    AgcDetileParams params{};
};

struct SurfacePlan {
    SurfaceGeometry geometry;
    std::uint32_t elementBytes = 0;
    std::uint32_t texelBytes = 0;
    bool volume = false;
    std::vector<MipJob> jobs;
};

std::uint32_t BlockBytesForTileMode(TextureTileMode tileMode);
AgcDetileParams MakeDetileParams(const GuestTextureResource& descriptor, const SurfaceGeometry& geometry, std::uint32_t level, std::uint32_t layer, std::uint64_t surfaceAddress);
SurfacePlan PlanSurface(const GuestTextureResource& descriptor, std::uint64_t surfaceAddress);

}

#endif
