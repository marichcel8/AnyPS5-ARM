#include "prx/libSceAgcDriver/Graphics/Metal/LabelTable.hpp"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using namespace AgcDriver::Graphics::Metal;

void Check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

void Labels() {
    LabelTable table;
    Check(!table.Overlaps(0x1000, 8) && table.Size() == 0, "an empty table overlaps nothing");
    table.Add({0x1000, 8, 1, QueueKind::Graphics, 5, false});
    table.Add({0x2000, 4, 2, QueueKind::Compute, 3});
    table.Add({0x1004, 4, 3, QueueKind::Graphics, 7});
    Check(table.Size() == 3, "three labels are tracked");
    Check(table.Overlaps(0x1000, 4) && table.Overlaps(0x1007, 1) && !table.Overlaps(0x1008, 4) && !table.Overlaps(0x0ffc, 4), "overlap is exact at the edges");
    Check(!table.Overlaps(0x1000, 0), "an empty range never overlaps");
    const auto serials = table.SerialsOver(0x1000, 8);
    Check(serials[static_cast<std::size_t>(QueueKind::Graphics)] == 7 && serials[static_cast<std::size_t>(QueueKind::Compute)] == 0, "the newest serial per queue is reported");
    const auto newest = table.Newest(0x1000, 8, 1);
    Check(newest && newest->stamp == 3, "the newest stamp wins");
    Check(!table.Newest(0x1000, 8, 3), "nothing is newer than the newest");
    table.Settle({5, 0});
    Check(table.Size() == 2 && table.Overlaps(0x1004, 4) && !table.Overlaps(0x1000, 4), "settling drops completed labels per queue");
    table.Add({0x3000, 4, 4, QueueKind::Graphics, 9, true});
    table.Settle({20, 20});
    Check(table.Overlaps(0x3000, 4), "a deferred label outlives its serial until it lands");
    table.Land(QueueKind::Graphics, 8);
    Check(table.Overlaps(0x3000, 4), "landing another serial keeps it");
    table.Land(QueueKind::Graphics, 9);
    Check(!table.Overlaps(0x3000, 4), "landing the serial drops it");
    table.Settle({7, 3});
    Check(table.Size() == 0 && !table.Overlaps(0x1004, 4) && !table.Overlaps(0x2000, 4), "settling everything empties the table");
}

void Actions() {
    CompletionActions actions;
    std::vector<int> order;
    Check(actions.Push(QueueKind::Graphics, 4, [&] { order.push_back(4); }), "the first action at a serial is fresh");
    Check(!actions.Push(QueueKind::Graphics, 4, [&] { order.push_back(41); }), "a second action at the same serial is not fresh");
    Check(actions.Push(QueueKind::Graphics, 2, [&] { order.push_back(2); }), "another serial is fresh");
    Check(actions.Push(QueueKind::Compute, 1, [&] { order.push_back(100); }), "queues are independent");
    Check(actions.Size() == 4, "four actions are pending");
    for (auto& action : actions.Take(QueueKind::Graphics, 3)) action();
    Check(order == std::vector<int>{2} && actions.Size() == 3, "only completed serials run");
    for (auto& action : actions.Take(QueueKind::Graphics, 10)) action();
    Check((order == std::vector<int>{2, 4, 41}), "actions run in serial then submission order");
    Check(actions.Take(QueueKind::Graphics, 10).empty(), "actions run once");
    Check(actions.Push(QueueKind::Graphics, 4, [] {}), "a drained serial is fresh again");
    actions.Clear();
    Check(actions.Size() == 0, "clear drops everything");
}

void Threads() {
    LabelTable table;
    CompletionActions actions;
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 4; ++worker) {
        workers.emplace_back([&, worker] {
            for (std::uint64_t index = 1; index <= 2000; ++index) {
                table.Add({0x10000ull * (worker + 1) + index * 4, 4, index, QueueKind::Graphics, index});
                actions.Push(QueueKind::Graphics, index, [] {});
                if (index % 16 == 0) {
                    table.Settle({index - 8, 0});
                    actions.Take(QueueKind::Graphics, index - 8);
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    table.Settle({2000, 0});
    actions.Take(QueueKind::Graphics, 2000);
    Check(table.Size() == 0 && actions.Size() == 0, "concurrent use drains cleanly");
}

}

int main() {
    Labels();
    Actions();
    Threads();
    std::puts("agc_metal_sync_table ok");
    return 0;
}
