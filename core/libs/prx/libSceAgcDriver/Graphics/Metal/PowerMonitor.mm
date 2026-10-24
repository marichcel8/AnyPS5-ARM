#include "prx/libSceAgcDriver/Graphics/Metal/PowerMonitor.hpp"
#import <Foundation/Foundation.h>
#import <IOKit/ps/IOPowerSources.h>
#import <IOKit/ps/IOPSKeys.h>
#include <atomic>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr std::int64_t BatteryPollNs = 5'000'000'000;

ThermalState mapThermal(NSProcessInfoThermalState state) {
    switch (state) {
    case NSProcessInfoThermalStateFair: return ThermalState::Fair;
    case NSProcessInfoThermalStateSerious: return ThermalState::Serious;
    case NSProcessInfoThermalStateCritical: return ThermalState::Critical;
    default: return ThermalState::Nominal;
    }
}

bool readOnBattery() {
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (info == nullptr) return false;
    CFStringRef type = IOPSGetProvidingPowerSourceType(info);
    const bool battery = type != nullptr && CFStringCompare(type, CFSTR(kIOPSBatteryPowerValue), 0) == kCFCompareEqualTo;
    CFRelease(info);
    return battery;
}

}

struct PowerMonitor::State {
    std::atomic<std::uint8_t> thermal{0};
    std::atomic<bool> lowPower{false};
    bool onBattery = false;
    std::int64_t nextBatteryPoll = 0;
    __strong id thermalToken = nil;
    __strong id powerToken = nil;

    void refresh() {
        NSProcessInfo* info = NSProcessInfo.processInfo;
        thermal.store(static_cast<std::uint8_t>(mapThermal(info.thermalState)), std::memory_order_relaxed);
        lowPower.store(info.lowPowerModeEnabled, std::memory_order_relaxed);
    }
};

PowerMonitor::PowerMonitor() : state(std::make_unique<State>()) {
    auto* raw = state.get();
    state->refresh();
    NSNotificationCenter* center = NSNotificationCenter.defaultCenter;
    state->thermalToken = [center addObserverForName:NSProcessInfoThermalStateDidChangeNotification object:nil queue:nil usingBlock:^(NSNotification*) { raw->refresh(); }];
    state->powerToken = [center addObserverForName:NSProcessInfoPowerStateDidChangeNotification object:nil queue:nil usingBlock:^(NSNotification*) { raw->refresh(); }];
}

PowerMonitor::~PowerMonitor() {
    NSNotificationCenter* center = NSNotificationCenter.defaultCenter;
    if (state->thermalToken != nil) [center removeObserver:state->thermalToken];
    if (state->powerToken != nil) [center removeObserver:state->powerToken];
}

PowerInputs PowerMonitor::Sample(std::int64_t nowNs) {
    PowerInputs inputs;
    inputs.thermal = static_cast<ThermalState>(state->thermal.load(std::memory_order_relaxed));
    inputs.lowPowerMode = state->lowPower.load(std::memory_order_relaxed);
    if (nowNs >= state->nextBatteryPoll) {
        state->onBattery = readOnBattery();
        state->nextBatteryPoll = nowNs + BatteryPollNs;
    }
    inputs.onBattery = state->onBattery;
    return inputs;
}

}
