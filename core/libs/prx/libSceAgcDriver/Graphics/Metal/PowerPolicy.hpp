#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_POWERPOLICY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_POWERPOLICY_HPP

#include <cstdint>

namespace AgcDriver::Graphics::Metal {

enum class ThermalState : std::uint8_t { Nominal, Fair, Serious, Critical };
enum class PowerProfile : std::uint8_t { Auto, Saver, Performance };

struct PowerInputs {
    ThermalState thermal = ThermalState::Nominal;
    bool lowPowerMode = false;
    bool onBattery = false;
    bool occluded = false;
};

struct PowerAdvice {
    std::uint32_t level = 0;
    double maxFps = 0.0;
    double renderScale = 1.0;
    bool allowUpscaler = true;
    bool suspend = false;
};

struct PowerTuning {
    std::int64_t relaxHoldNs = 20'000'000'000;
    double saverFps = 30.0;
    double seriousFps = 30.0;
    double criticalFps = 30.0;
    double fairScale = 0.875;
    double seriousScale = 0.75;
    double criticalScale = 0.5;
};

class PowerPolicy {
public:
    explicit PowerPolicy(PowerProfile profile = PowerProfile::Auto, PowerTuning tuning = {});

    PowerAdvice Update(const PowerInputs& inputs, std::int64_t nowNs);
    const PowerAdvice& Current() const { return advice; }
    void SetProfile(PowerProfile next);

private:
    std::uint32_t wantedLevel(const PowerInputs& inputs) const;
    PowerAdvice adviceFor(std::uint32_t level, bool suspend) const;

    PowerProfile profile;
    PowerTuning tuning;
    PowerAdvice advice;
    std::int64_t relaxSince = 0;
    bool relaxing = false;
};

}

#endif
