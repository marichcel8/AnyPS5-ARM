#include "prx/libSceAgcDriver/Graphics/Metal/PowerPolicy.hpp"
#include <algorithm>

namespace AgcDriver::Graphics::Metal {

PowerPolicy::PowerPolicy(PowerProfile profile, PowerTuning tuning) : profile(profile), tuning(tuning) {
    advice = adviceFor(0, false);
}

void PowerPolicy::SetProfile(PowerProfile next) {
    profile = next;
    relaxing = false;
}

std::uint32_t PowerPolicy::wantedLevel(const PowerInputs& inputs) const {
    std::uint32_t thermal = static_cast<std::uint32_t>(inputs.thermal);
    if (profile == PowerProfile::Performance && thermal == 1) thermal = 0;
    std::uint32_t level = thermal;
    if (profile != PowerProfile::Performance && inputs.lowPowerMode) level = std::max(level, 2u);
    if (profile == PowerProfile::Saver || (profile == PowerProfile::Auto && inputs.onBattery)) level = std::max(level, 1u);
    return level;
}

PowerAdvice PowerPolicy::adviceFor(std::uint32_t level, bool suspend) const {
    PowerAdvice next;
    next.level = level;
    next.suspend = suspend;
    switch (level) {
    case 0:
        break;
    case 1:
        next.renderScale = tuning.fairScale;
        break;
    case 2:
        next.maxFps = tuning.seriousFps;
        next.renderScale = tuning.seriousScale;
        break;
    default:
        next.maxFps = tuning.criticalFps;
        next.renderScale = tuning.criticalScale;
        next.allowUpscaler = false;
        break;
    }
    if (profile == PowerProfile::Saver && next.maxFps == 0.0) next.maxFps = tuning.saverFps;
    return next;
}

PowerAdvice PowerPolicy::Update(const PowerInputs& inputs, std::int64_t nowNs) {
    const auto wanted = wantedLevel(inputs);
    auto level = advice.level;
    if (wanted > level) {
        level = wanted;
        relaxing = false;
    } else if (wanted < level) {
        if (!relaxing) {
            relaxing = true;
            relaxSince = nowNs;
        } else if (nowNs - relaxSince >= tuning.relaxHoldNs) {
            level -= 1;
            relaxSince = nowNs;
        }
    } else {
        relaxing = false;
    }
    advice = adviceFor(level, inputs.occluded);
    return advice;
}

}
