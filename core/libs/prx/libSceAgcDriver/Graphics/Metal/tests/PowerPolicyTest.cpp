#include "prx/libSceAgcDriver/Graphics/Metal/PowerPolicy.hpp"
#include <cstdio>
#include <cstdlib>

namespace {

using namespace AgcDriver::Graphics::Metal;

constexpr std::int64_t Second = 1'000'000'000;

void Check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

PowerInputs Thermal(ThermalState state) {
    PowerInputs inputs;
    inputs.thermal = state;
    return inputs;
}

void NominalIsUntouched() {
    PowerPolicy policy;
    const auto advice = policy.Update(Thermal(ThermalState::Nominal), 0);
    Check(advice.level == 0 && advice.maxFps == 0.0 && advice.renderScale == 1.0 && advice.allowUpscaler && !advice.suspend, "nominal changes something");
}

void EscalatesAtOnce() {
    PowerPolicy policy;
    Check(policy.Update(Thermal(ThermalState::Fair), 0).level == 1, "fair");
    const auto serious = policy.Update(Thermal(ThermalState::Serious), 1);
    Check(serious.level == 2 && serious.maxFps == 30.0 && serious.renderScale == 0.75, "serious");
    const auto critical = policy.Update(Thermal(ThermalState::Critical), 2);
    Check(critical.level == 3 && critical.renderScale == 0.5 && !critical.allowUpscaler, "critical");
}

void RelaxesOneStepPerHold() {
    PowerPolicy policy;
    policy.Update(Thermal(ThermalState::Critical), 0);
    Check(policy.Update(Thermal(ThermalState::Nominal), 1 * Second).level == 3, "relaxed without waiting");
    Check(policy.Update(Thermal(ThermalState::Nominal), 10 * Second).level == 3, "relaxed before the hold");
    Check(policy.Update(Thermal(ThermalState::Nominal), 22 * Second).level == 2, "first step down");
    Check(policy.Update(Thermal(ThermalState::Nominal), 30 * Second).level == 2, "second step too soon");
    Check(policy.Update(Thermal(ThermalState::Nominal), 43 * Second).level == 1, "second step down");
    Check(policy.Update(Thermal(ThermalState::Nominal), 64 * Second).level == 0, "back to nominal");
}

void PressureReturnCancelsRelax() {
    PowerPolicy policy;
    policy.Update(Thermal(ThermalState::Serious), 0);
    policy.Update(Thermal(ThermalState::Nominal), 1 * Second);
    policy.Update(Thermal(ThermalState::Serious), 15 * Second);
    Check(policy.Update(Thermal(ThermalState::Nominal), 25 * Second).level == 2, "the hold restarts after the pressure came back");
}

void LowPowerModeCaps() {
    PowerPolicy policy;
    PowerInputs inputs;
    inputs.lowPowerMode = true;
    const auto advice = policy.Update(inputs, 0);
    Check(advice.level == 2 && advice.maxFps == 30.0, "low power mode");
}

void ProfilesBend() {
    PowerPolicy saver(PowerProfile::Saver);
    const auto advice = saver.Update(Thermal(ThermalState::Nominal), 0);
    Check(advice.level == 1 && advice.maxFps == 30.0, "saver");
    PowerPolicy performance(PowerProfile::Performance);
    Check(performance.Update(Thermal(ThermalState::Fair), 0).level == 0, "performance ignores fair");
    PowerInputs lowPower;
    lowPower.lowPowerMode = true;
    Check(performance.Update(lowPower, 1).level == 0, "performance ignores low power mode");
    Check(performance.Update(Thermal(ThermalState::Serious), 2).level == 2, "performance still reacts to serious");
    PowerPolicy automatic;
    PowerInputs battery;
    battery.onBattery = true;
    const auto onBattery = automatic.Update(battery, 0);
    Check(onBattery.level == 1 && onBattery.maxFps == 0.0, "battery in auto");
}

void OcclusionSuspends() {
    PowerPolicy policy;
    PowerInputs inputs;
    inputs.occluded = true;
    Check(policy.Update(inputs, 0).suspend, "occluded does not suspend");
    Check(!policy.Update(PowerInputs{}, 1).suspend, "visible still suspended");
}

}

int main() {
    NominalIsUntouched();
    EscalatesAtOnce();
    RelaxesOneStepPerHold();
    PressureReturnCancelsRelax();
    LowPowerModeCaps();
    ProfilesBend();
    OcclusionSuspends();
    std::puts("power policy ok");
    return 0;
}
