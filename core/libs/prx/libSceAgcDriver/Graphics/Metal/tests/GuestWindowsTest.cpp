#include "prx/libSceAgcDriver/Graphics/Metal/GuestWindows.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace {

using namespace AgcDriver::Graphics::Metal;

constexpr std::uint64_t HostPage = 16384;

void Check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

class FakeBackend final : public WindowBackend {
public:
    WrappedMemory Wrap(HostSpan span, bool writable) override {
        Check(span.base % HostPage == 0 && span.bytes % HostPage == 0, "a window is not host page aligned");
        if (span.base == refuse) return {};
        const auto id = ++next;
        live.emplace(id, span);
        return {reinterpret_cast<void*>(id), 0x100000000000ull + id * 0x100000000ull, !writable && span.base == copyBase};
    }

    void Retire(const WrappedMemory& memory, HostSpan span) override {
        const auto it = live.find(reinterpret_cast<std::uintptr_t>(memory.handle));
        Check(it != live.end() && it->second.base == span.base && it->second.bytes == span.bytes, "retired an unknown window");
        live.erase(it);
        ++retired;
    }

    std::map<std::uintptr_t, HostSpan> live;
    std::uintptr_t next = 0;
    std::uint64_t refuse = ~0ull;
    std::uint64_t copyBase = ~0ull;
    std::uint64_t retired = 0;
};

void TestCovering() {
    const auto span = HostPagesCovering(0x201000, 0x5000, HostPage);
    Check(span.base == 0x200000 && span.bytes == 0x8000, "a 4 KiB aligned range rounds out to host pages");
    const auto exact = HostPagesCovering(0x204000, 0x4000, HostPage);
    Check(exact.base == 0x204000 && exact.bytes == 0x4000, "an aligned range stays as it is");
    const auto one = HostPagesCovering(0x207fff, 1, HostPage);
    Check(one.base == 0x204000 && one.bytes == HostPage, "one byte takes its host page");
}

void TestPlan() {
    const std::uint64_t limit = 64 * HostPage;
    Check(PlanWindows({0, limit}, limit, HostPage).size() == 1, "a span at the limit is one window");
    const auto plan = PlanWindows({0x40000000, 200 * HostPage}, limit, HostPage);
    const auto step = WindowStep(limit, HostPage);
    Check(step == 32 * HostPage, "the step is half the limit");
    Check(plan.front().base == 0x40000000 && plan.back().End() == 0x40000000 + 200 * HostPage, "the plan covers the span");
    for (std::size_t index = 0; index < plan.size(); ++index) {
        Check(plan[index].bytes <= limit, "no window exceeds the limit");
        if (index > 0) Check(plan[index].base == plan[index - 1].base + step, "windows advance by the step");
    }
    for (std::uint64_t address = 0x40000000; address + step <= 0x40000000 + 200 * HostPage; address += 4096) {
        bool covered = false;
        for (const auto& window : plan) covered = covered || window.Contains(address, step);
        Check(covered, "every access up to the step fits one window");
    }
}

void TestResolveAndTable() {
    FakeBackend backend;
    GuestWindows windows(backend, HostPage, 64 * HostPage);
    const GuestRange first{0x201000, 0x5000, true};
    const GuestRange second{0x206000, 0x3000, false};
    Check(windows.Map(first) && windows.Map(second), "ranges map");
    auto snapshot = windows.Snapshot();
    Check(snapshot->windows.size() == 2 && backend.live.size() == 2, "one window per range");
    const auto resolved = Resolve(*snapshot, 0x201010, 16);
    Check(static_cast<bool>(resolved), "an address inside the first range resolves");
    Check(resolved.deviceAddress == resolved.window->gpuAddress + 0x1010, "the device address keeps the offset into the host page");
    const auto shared = Resolve(*snapshot, 0x207000, 0x1000);
    Check(static_cast<bool>(shared), "the host page shared by both ranges resolves");
    Check(!Resolve(*snapshot, 0x300000, 4), "an unmapped address does not resolve");
    Check(snapshot->table.size() == 2, "one table entry per range");
    Check(snapshot->table[0].begin == first.address && snapshot->table[0].end == first.address + first.bytes, "the table holds the guest range, not the host pages");
    Check(snapshot->table[0].deviceAddress == Resolve(*snapshot, first.address, 1).deviceAddress, "the table entry points at the guest bytes");
    Check(snapshot->table[0].permissions == (ShaderRecompiler::BdaAbi::Read | ShaderRecompiler::BdaAbi::Write), "a writable range is writable");
    Check(snapshot->table[1].permissions == ShaderRecompiler::BdaAbi::Read, "a read-only range is read-only");
    const auto serial = windows.Serial();
    Check(windows.Map(first) && windows.Serial() == serial, "mapping an unchanged range publishes nothing");
    windows.Unmap(first.address, first.bytes);
    snapshot = windows.Snapshot();
    Check(snapshot->windows.size() == 1 && backend.retired == 1, "unmapping retires only that range's window");
    Check(static_cast<bool>(Resolve(*snapshot, 0x206000, 0x3000)), "the neighbour sharing a host page stays");
    const GuestRange protectedSecond{0x206000, 0x3000, true};
    Check(windows.Map(protectedSecond), "a protection change maps again");
    Check(backend.retired == 2 && windows.Snapshot()->table[0].permissions == (ShaderRecompiler::BdaAbi::Read | ShaderRecompiler::BdaAbi::Write), "a protection change replaces the window");
    std::vector<std::byte> bytes(BdaTableBytes(windows.Snapshot()->table.size()));
    WriteBdaTable(bytes, windows.Snapshot()->table);
    ShaderRecompiler::BdaAbi::Header header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    Check(header.version == ShaderRecompiler::BdaAbi::Version && header.count == 1 && header.entryBytes == sizeof(ShaderRecompiler::BdaAbi::Range), "the table header");
    windows.Clear();
    Check(backend.live.empty(), "clearing retires everything");
}

void TestSplitTable() {
    FakeBackend backend;
    const std::uint64_t limit = 64 * HostPage;
    GuestWindows windows(backend, HostPage, limit);
    const GuestRange large{0x80001000, 300 * HostPage, true};
    Check(windows.Map(large), "a range above the limit maps");
    const auto snapshot = windows.Snapshot();
    Check(snapshot->windows.size() > 1, "it takes several windows");
    std::uint64_t cursor = large.address;
    for (const auto& entry : snapshot->table) {
        Check(entry.begin == cursor && entry.end > entry.begin, "the table entries tile the range");
        const auto resolved = Resolve(*snapshot, entry.begin, entry.end - entry.begin);
        Check(static_cast<bool>(resolved) && resolved.deviceAddress == entry.deviceAddress, "each entry lies in one window");
        cursor = entry.end;
    }
    Check(cursor == large.address + large.bytes, "the entries reach the end");
    const auto step = WindowStep(limit, HostPage);
    for (std::uint64_t address = large.address; address + step <= cursor; address += 3 * 4096) Check(static_cast<bool>(Resolve(*snapshot, address, step)), "any access up to the step resolves");
}

void TestReplaceAndFailure() {
    FakeBackend backend;
    GuestWindows windows(backend, HostPage, 64 * HostPage);
    std::vector<GuestRange> ranges;
    for (std::uint64_t index = 0; index < 1000; ++index) ranges.push_back({0x10000000 + index * 0x10000, 0x3000, (index & 1) != 0});
    windows.Replace(ranges);
    Check(backend.live.size() == 1000 && windows.Snapshot()->table.size() == 1000, "replace maps every range");
    ranges.erase(ranges.begin() + 10, ranges.begin() + 20);
    ranges.push_back({0x30000000, 0x1000, false});
    windows.Replace(ranges);
    Check(backend.live.size() == 991 && backend.retired == 10, "replace retires only what left");
    backend.refuse = 0x40000000;
    Check(!windows.Map({0x40000000, 0x1000, false}), "a refused wrap reports failure");
    Check(windows.Counters().wrapFailures == 1, "the failure is counted");
    Check(!Resolve(*windows.Snapshot(), 0x40000000, 1), "a refused range does not resolve");
    backend.copyBase = 0x50000000;
    Check(windows.Map({0x50000000, 0x1000, false}) && windows.Counters().copiedBytes == HostPage, "a copied window is counted apart");
}

}

int main() {
    TestCovering();
    TestPlan();
    TestResolveAndTable();
    TestSplitTable();
    TestReplaceAndFailure();
    std::puts("metal guest windows: ok");
    return 0;
}
