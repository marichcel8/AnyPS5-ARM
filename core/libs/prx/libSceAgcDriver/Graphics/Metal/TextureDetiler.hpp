#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_TEXTUREDETILER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_TEXTUREDETILER_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/MetalCompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureAddress.h"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureFormat.hpp"
#include <cstdint>
#include <memory>
#include <string>

namespace AgcDriver::Graphics::Metal {

enum class DetileKernel : std::uint8_t {
    TiledToArray2D,
    TiledToArray1D,
    TiledToVolume,
    Array2DToTiled,
    VolumeToTiled,
    TiledToLinear,
    LinearToTiled,
    Count
};

struct DetileCounters {
    std::uint64_t dispatches = 0;
    std::uint64_t pipelines = 0;
    std::uint64_t arenaChunks = 0;
    std::uint64_t arenaBytes = 0;
    std::uint64_t stalls = 0;
    std::uint64_t stallNanos = 0;
};

class TextureDetiler {
public:
    static std::unique_ptr<TextureDetiler> Create(Device& device, MetalCompiler& compiler, std::string* failure);
    ~TextureDetiler();
    TextureDetiler(const TextureDetiler&) = delete;
    TextureDetiler& operator=(const TextureDetiler&) = delete;

    void Encode(void* computeEncoder, DetileKernel kernel, std::uint32_t elementBytes, TextureTransform transform, std::uint32_t texelBytes, const AgcDetileParams& params, void* texture, std::uint32_t columns, std::uint32_t rows);
    void Warm(DetileKernel kernel, std::uint32_t elementBytes, TextureTransform transform, std::uint32_t texelBytes);
    void Seal(QueueKind queue, std::uint64_t serial);
    DetileCounters Counters() const;

    struct State;

private:
    explicit TextureDetiler(std::unique_ptr<State> state);

    std::unique_ptr<State> state;
};

}

#endif
