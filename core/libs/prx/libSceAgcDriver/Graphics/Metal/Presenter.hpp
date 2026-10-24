#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_PRESENTER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_PRESENTER_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/FramePacer.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/PowerPolicy.hpp"
#include <cstdint>
#include <memory>
#include <string>

namespace AgcDriver::Graphics::Metal {

enum class SourceEncoding : std::uint8_t { Unorm, LinearHalf, Pq };
enum class UpscaleMode : std::uint8_t { Auto, Off, On };

struct PresentSource {
    void* texture = nullptr;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    SourceEncoding encoding = SourceEncoding::Unorm;
    float referenceWhiteNits = 203.0f;
    float peakNits = 1000.0f;
};

struct PresenterOptions {
    double targetFps = 0.0;
    UpscaleMode upscale = UpscaleMode::Auto;
    PowerProfile profile = PowerProfile::Auto;
    std::uint32_t drawables = 2;

    static PresenterOptions FromEnvironment();
};

enum class PresentOutcome : std::uint8_t { Presented, Suspended, NoDrawable, Invalid };

struct PresentResult {
    PresentOutcome outcome = PresentOutcome::Invalid;
    bool upscaled = false;
    std::uint64_t serial = 0;
    double pacedMs = 0.0;
    double waitMs = 0.0;
    double encodeMs = 0.0;
};

struct PresenterStats {
    std::uint64_t presented = 0;
    std::uint64_t suspended = 0;
    std::uint64_t noDrawable = 0;
    std::uint64_t upscaled = 0;
    std::uint64_t scalerRebuilds = 0;
    double refreshHz = 0.0;
    double effectiveHz = 0.0;
    double pacedMs = 0.0;
    double waitMs = 0.0;
    double encodeMs = 0.0;
    PowerAdvice advice;
    PacerStats pacer;
};

class Presenter {
public:
    static std::unique_ptr<Presenter> Create(Device& device, void* metalLayer, const PresenterOptions& options, std::string* failure);
    ~Presenter();
    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;

    PresentResult Present(const PresentSource& source);
    bool RenderTo(const PresentSource& source, void* bgra8Target, bool allowUpscale);

    void SetDrawableSize(std::uint32_t width, std::uint32_t height);
    void SetOccluded(bool occluded);
    void SetRefreshRate(double hz);
    void SetTargetFps(double fps);
    PresenterStats Stats() const;

    struct State;

private:
    explicit Presenter(std::unique_ptr<State> state);

    std::unique_ptr<State> state;
};

}

#endif
