#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_FRAMEPACER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_FRAMEPACER_HPP

#include <cstdint>

namespace AgcDriver::Graphics::Metal {

struct PacerStats {
    std::uint64_t scheduled = 0;
    std::uint64_t reanchored = 0;
    std::uint64_t latePresents = 0;
    std::int64_t sleptNs = 0;
};

class FramePacer {
public:
    void Configure(double refreshHz, double targetHz);
    std::int64_t Schedule(std::int64_t nowNs);
    void NoteDisplayed(std::int64_t displayedNs);
    void Reset();

    std::uint32_t Divisor() const { return divisor; }
    std::int64_t RefreshNs() const { return refreshNs; }
    std::int64_t IntervalNs() const { return refreshNs * divisor; }
    double EffectiveHz() const;
    const PacerStats& Stats() const { return stats; }

private:
    std::int64_t refreshNs = 16666667;
    std::uint32_t divisor = 1;
    std::int64_t last = 0;
    bool anchored = false;
    PacerStats stats;
};

}

#endif
