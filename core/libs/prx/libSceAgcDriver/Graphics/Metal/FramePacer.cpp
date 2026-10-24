#include "prx/libSceAgcDriver/Graphics/Metal/FramePacer.hpp"
#include <algorithm>
#include <cmath>

namespace AgcDriver::Graphics::Metal {

void FramePacer::Configure(double refreshHz, double targetHz) {
    if (!(refreshHz >= 1.0)) refreshHz = 60.0;
    refreshNs = static_cast<std::int64_t>(std::llround(1.0e9 / refreshHz));
    std::uint32_t next = 1;
    if (targetHz >= 1.0 && targetHz < refreshHz) next = static_cast<std::uint32_t>(std::max(1.0, std::round(refreshHz / targetHz)));
    if (next != divisor) anchored = false;
    divisor = next;
}

void FramePacer::Reset() {
    anchored = false;
}

double FramePacer::EffectiveHz() const {
    return 1.0e9 / static_cast<double>(IntervalNs());
}

std::int64_t FramePacer::Schedule(std::int64_t nowNs) {
    ++stats.scheduled;
    if (divisor == 1) {
        last = nowNs;
        anchored = true;
        return nowNs;
    }
    const auto interval = IntervalNs();
    if (!anchored) {
        anchored = true;
        last = nowNs;
        return nowNs;
    }
    const auto next = last + interval;
    const auto tolerance = refreshNs / 2;
    std::int64_t slot = next;
    if (nowNs > next + tolerance) {
        ++stats.reanchored;
        ++stats.latePresents;
        slot = nowNs;
        last = nowNs;
    } else if (nowNs > next) {
        slot = nowNs;
        last = next;
    } else {
        last = next;
    }
    stats.sleptNs += std::max<std::int64_t>(0, slot - nowNs);
    return slot;
}

void FramePacer::NoteDisplayed(std::int64_t displayedNs) {
    if (divisor == 1 || !anchored || displayedNs <= 0) return;
    const auto drift = displayedNs - last;
    if (drift > 0 && drift < refreshNs) last += drift;
}

}
