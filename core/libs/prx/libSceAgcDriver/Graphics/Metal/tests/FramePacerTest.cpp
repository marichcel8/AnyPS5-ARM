#include "prx/libSceAgcDriver/Graphics/Metal/FramePacer.hpp"
#include <cstdio>
#include <cstdlib>

namespace {

using namespace AgcDriver::Graphics::Metal;

constexpr std::int64_t Ms = 1'000'000;

void Check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

void NativeNeverSleeps() {
    FramePacer pacer;
    pacer.Configure(60.0, 0.0);
    Check(pacer.Divisor() == 1, "native divisor");
    for (std::int64_t now = 0; now < 100 * Ms; now += 3 * Ms) Check(pacer.Schedule(now) == now, "native schedule sleeps");
}

void DivisorFollowsTarget() {
    FramePacer pacer;
    pacer.Configure(60.0, 30.0);
    Check(pacer.Divisor() == 2, "60 to 30");
    pacer.Configure(120.0, 30.0);
    Check(pacer.Divisor() == 4, "120 to 30");
    pacer.Configure(120.0, 60.0);
    Check(pacer.Divisor() == 2, "120 to 60");
    pacer.Configure(60.0, 45.0);
    Check(pacer.Divisor() == 1, "45 on 60 rounds to every refresh");
    pacer.Configure(60.0, 120.0);
    Check(pacer.Divisor() == 1, "a target above refresh stays native");
    pacer.Configure(0.0, 30.0);
    Check(pacer.RefreshNs() == 16666667, "a bad refresh falls back to 60 Hz");
}

void GuestFasterThanTargetIsHeldOnTheGrid() {
    FramePacer pacer;
    pacer.Configure(60.0, 30.0);
    const auto interval = pacer.IntervalNs();
    std::int64_t now = 0;
    std::int64_t first = pacer.Schedule(now);
    Check(first == 0, "the first present is immediate");
    std::int64_t previous = first;
    for (int frame = 0; frame < 200; ++frame) {
        now = previous + 1 * Ms;
        const auto slot = pacer.Schedule(now);
        Check(slot - previous == interval, "a fast guest is held to the grid");
        previous = slot;
    }
    Check(pacer.Stats().reanchored == 0, "no reanchor on a steady guest");
}

void SlowFrameDoesNotBurst() {
    FramePacer pacer;
    pacer.Configure(60.0, 30.0);
    const auto interval = pacer.IntervalNs();
    pacer.Schedule(0);
    const auto late = pacer.Schedule(interval * 5);
    Check(late == interval * 5, "a late frame presents immediately");
    Check(pacer.Stats().reanchored == 1, "a late frame reanchors");
    const auto after = pacer.Schedule(late + 2 * Ms);
    Check(after == late + interval, "the grid restarts from the late frame");
}

void SmallLatenessKeepsTheGrid() {
    FramePacer pacer;
    pacer.Configure(60.0, 30.0);
    const auto interval = pacer.IntervalNs();
    pacer.Schedule(0);
    const auto slightly = pacer.Schedule(interval + 3 * Ms);
    Check(slightly == interval + 3 * Ms, "slight lateness presents at once");
    Check(pacer.Stats().reanchored == 0, "slight lateness is not a reanchor");
    const auto next = pacer.Schedule(slightly + 1 * Ms);
    Check(next == 2 * interval, "the grid is kept after slight lateness");
}

void DisplayedTimePullsThePhase() {
    FramePacer pacer;
    pacer.Configure(60.0, 30.0);
    const auto interval = pacer.IntervalNs();
    pacer.Schedule(0);
    const auto slot = pacer.Schedule(interval);
    pacer.NoteDisplayed(slot + 4 * Ms);
    const auto next = pacer.Schedule(slot + 1 * Ms);
    Check(next == slot + 4 * Ms + interval, "the next slot follows the displayed time");
}

}

int main() {
    NativeNeverSleeps();
    DivisorFollowsTarget();
    GuestFasterThanTargetIsHeldOnTheGrid();
    SlowFrameDoesNotBurst();
    SmallLatenessKeepsTheGrid();
    DisplayedTimePullsThePhase();
    std::puts("frame pacer ok");
    return 0;
}
