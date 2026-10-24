#include "prx/libSceAgcDriver/Graphics/Metal/CompileCache.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/ShaderCompileService.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>

namespace {

using namespace AgcDriver::Graphics::Metal;

void Check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

class FakeBackend final : public CompileBackend {
public:
    struct Worker final : CompileWorker {
        std::uint32_t index = 0;
    };

    std::unique_ptr<CompileWorker> CreateWorker(std::uint32_t index) override {
        auto worker = std::make_unique<Worker>();
        worker->index = index;
        ++created;
        return worker;
    }

    void Prepare(CompileWorker&, CompilePriority priority) override {
        std::lock_guard lock(mutex);
        prepared.push_back(priority);
    }

    std::atomic<int> created{0};
    std::mutex mutex;
    std::vector<CompilePriority> prepared;
};

class Gate {
public:
    void Open() {
        {
            std::lock_guard lock(mutex);
            open = true;
        }
        changed.notify_all();
    }

    void Wait() {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return open; });
    }

private:
    std::mutex mutex;
    std::condition_variable changed;
    bool open = false;
};

CompileKey Key(std::uint64_t value) {
    return CompileKeyBuilder(1).Add(value).Finish();
}

CompileOutcome Product(int value, std::uint64_t cost = 1) {
    CompileOutcome outcome;
    outcome.product = std::make_shared<int>(value);
    outcome.costBytes = cost;
    return outcome;
}

void TestKeys() {
    Check(Key(1) == Key(1) && !(Key(1) == Key(2)), "equal inputs give equal keys");
    Check(!(CompileKeyBuilder(1).Add(1).Finish() == CompileKeyBuilder(2).Add(1).Finish()), "the domain separates keys");
    Check(CompileKeyBuilder(1).AddString("ab").Add(3).Finish() != CompileKeyBuilder(1).AddString("a").AddString("b3").Finish(), "string lengths separate keys");
    Check(CompileKeyHex(Key(7)).size() == 32, "hex keys have 32 digits");
    Check(!Key(0).Empty(), "a finished key is never empty");
}

void TestRunsAndDeduplicates() {
    auto backend = std::make_shared<FakeBackend>();
    ShaderCompileService service(backend, {2, 1, 1 << 20});
    std::atomic<int> runs{0};
    auto first = service.Submit(Key(1), CompilePriority::Needed, [&](CompileWorker&) {
        ++runs;
        return Product(42);
    });
    auto second = service.Submit(Key(1), CompilePriority::Needed, [&](CompileWorker&) {
        ++runs;
        return Product(43);
    });
    Check(first == second, "the same key returns the same ticket");
    Check(first->Wait(5000) && first->Ready(), "the task finishes");
    Check(*first->ProductAs<int>() == 42 && runs == 1, "the task ran once");
    auto again = service.Submit(Key(1), CompilePriority::Urgent, [&](CompileWorker&) {
        ++runs;
        return Product(44);
    });
    Check(again == first && runs == 1, "a finished result is served without a second compile");
    Check(service.Find(Key(1)) == first && service.Find(Key(99)) == nullptr, "Find returns known tickets only");
    Check(backend->created == 2, "each worker built its context once");
    Check(service.WaitIdle(5000), "the service goes idle");
    const auto stats = service.Stats();
    Check(stats.submitted == 3 && stats.deduplicated == 2 && stats.completed == 1, "stats count submits, dedups and completions");
}

void TestPriorityOrder() {
    auto backend = std::make_shared<FakeBackend>();
    ShaderCompileService service(backend, {1, 1, 1 << 20});
    Gate gate;
    std::mutex orderMutex;
    std::vector<int> order;
    const auto record = [&](int id) {
        return [&, id](CompileWorker&) {
            std::lock_guard lock(orderMutex);
            order.push_back(id);
            return Product(id);
        };
    };
    auto blocker = service.Submit(Key(100), CompilePriority::Needed, [&](CompileWorker&) {
        gate.Wait();
        return Product(0);
    });
    while (service.Stats().running == 0) std::this_thread::yield();
    auto a = service.Submit(Key(1), CompilePriority::Prefetch, record(1));
    auto b = service.Submit(Key(2), CompilePriority::Needed, record(2));
    auto c = service.Submit(Key(3), CompilePriority::Urgent, record(3));
    auto d = service.Submit(Key(4), CompilePriority::Prefetch, record(4));
    service.Submit(Key(4), CompilePriority::Urgent, record(99));
    Check(service.Stats().promoted == 1, "a queued job can be promoted");
    gate.Open();
    Check(a->Wait(5000) && b->Wait(5000) && c->Wait(5000) && d->Wait(5000) && blocker->Wait(5000), "all jobs finish");
    const std::vector<int> expected{3, 4, 2, 1};
    Check(order == expected || order == std::vector<int>({4, 3, 2, 1}), "urgent, then needed, then prefetch in order");
    Check(service.Stats().promoted == 1 && d->Priority() == CompilePriority::Urgent, "the promoted ticket reports its new priority");
}

void TestPrefetchLimit() {
    auto backend = std::make_shared<FakeBackend>();
    ShaderCompileService service(backend, {3, 1, 1 << 20});
    std::atomic<int> concurrent{0};
    std::atomic<int> peak{0};
    std::vector<std::shared_ptr<CompileTicket>> tickets;
    for (int index = 0; index < 6; ++index) {
        tickets.push_back(service.Submit(Key(200 + index), CompilePriority::Prefetch, [&](CompileWorker&) {
            const int now = ++concurrent;
            int seen = peak.load();
            while (now > seen && !peak.compare_exchange_weak(seen, now)) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            --concurrent;
            return Product(1);
        }));
    }
    for (auto& ticket : tickets) Check(ticket->Wait(5000), "prefetch jobs finish");
    Check(peak == 1, "prefetch jobs run one at a time");
    std::atomic<int> neededPeak{0};
    std::atomic<int> neededNow{0};
    std::atomic<int> arrived{0};
    std::vector<std::shared_ptr<CompileTicket>> needed;
    for (int index = 0; index < 3; ++index) {
        needed.push_back(service.Submit(Key(300 + index), CompilePriority::Needed, [&](CompileWorker&) {
            const int now = ++neededNow;
            int seen = neededPeak.load();
            while (now > seen && !neededPeak.compare_exchange_weak(seen, now)) {
            }
            ++arrived;
            while (arrived.load() < 3) std::this_thread::yield();
            --neededNow;
            return Product(1);
        }));
    }
    for (auto& ticket : needed) Check(ticket->Wait(5000), "needed jobs use every worker");
    Check(neededPeak == 3, "needed jobs run on all workers");
}

void TestFailures() {
    auto backend = std::make_shared<FakeBackend>();
    ShaderCompileService service(backend, {1, 1, 1 << 20});
    auto thrown = service.Submit(Key(1), CompilePriority::Needed, [](CompileWorker&) -> CompileOutcome { throw std::runtime_error("boom"); });
    auto failed = service.Submit(Key(2), CompilePriority::Needed, [](CompileWorker&) {
        CompileOutcome outcome;
        outcome.error = "bad source";
        outcome.product = std::make_shared<int>(1);
        return outcome;
    });
    auto empty = service.Submit(Key(3), CompilePriority::Needed, [](CompileWorker&) { return CompileOutcome{}; });
    Check(thrown->Wait(5000) && thrown->State() == CompileState::Failed && thrown->Error() == "boom", "an exception fails the ticket");
    Check(failed->Wait(5000) && failed->State() == CompileState::Failed && failed->Product() == nullptr, "a failed ticket has no product");
    Check(empty->Wait(5000) && empty->State() == CompileState::Failed, "an empty product fails the ticket");
    auto kept = service.Submit(Key(2), CompilePriority::Urgent, [](CompileWorker&) { return Product(1); });
    Check(kept == failed, "a failure is remembered until forgotten");
    service.Forget(Key(2));
    auto retried = service.Submit(Key(2), CompilePriority::Urgent, [](CompileWorker&) { return Product(5); });
    Check(retried != failed && retried->Wait(5000) && retried->Ready(), "a forgotten failure can be retried");
    Check(service.Stats().failed == 3, "failures are counted");
}

void TestRetention() {
    auto backend = std::make_shared<FakeBackend>();
    ShaderCompileService service(backend, {1, 1, 100});
    std::vector<std::shared_ptr<CompileTicket>> tickets;
    for (int index = 0; index < 5; ++index) {
        tickets.push_back(service.Submit(Key(500 + index), CompilePriority::Needed, [index](CompileWorker&) { return Product(index, 40); }));
        Check(tickets.back()->Wait(5000), "a result is produced");
    }
    const auto stats = service.Stats();
    Check(stats.retainedBytes <= 100 && stats.evicted == 3, "the least recently used results are evicted over budget");
    Check(service.Find(Key(500)) == nullptr && service.Find(Key(504)) != nullptr, "the newest result stays");
    Check(*tickets[0]->ProductAs<int>() == 0, "an evicted result stays valid for its holder");
}

class BrokenBackend final : public CompileBackend {
public:
    std::unique_ptr<CompileWorker> CreateWorker(std::uint32_t) override { throw std::runtime_error("no compiler"); }
};

void TestBrokenWorker() {
    ShaderCompileService service(std::make_shared<BrokenBackend>(), {1, 1, 1 << 20});
    auto ticket = service.Submit(Key(1), CompilePriority::Urgent, [](CompileWorker&) { return Product(1); });
    Check(ticket->Wait(5000) && ticket->State() == CompileState::Failed && ticket->Error() == "no compiler", "a worker that cannot be created fails its jobs instead of hanging them");
}

void TestShutdown() {
    auto backend = std::make_shared<FakeBackend>();
    auto service = std::make_unique<ShaderCompileService>(backend, CompileServiceConfig{1, 1, 1 << 20});
    Gate gate;
    auto blocker = service->Submit(Key(1), CompilePriority::Needed, [&](CompileWorker&) {
        gate.Wait();
        return Product(1);
    });
    while (service->Stats().running == 0) std::this_thread::yield();
    auto queued = service->Submit(Key(2), CompilePriority::Needed, [](CompileWorker&) { return Product(2); });
    std::thread stopper([&] { service->Shutdown(); });
    queued->Wait(5000);
    Check(queued->State() == CompileState::Cancelled, "queued work is cancelled at shutdown");
    gate.Open();
    stopper.join();
    Check(blocker->Ready(), "running work finishes at shutdown");
    auto late = service->Submit(Key(3), CompilePriority::Needed, [](CompileWorker&) { return Product(3); });
    Check(late->State() == CompileState::Cancelled, "a late submit is cancelled");
    Check(!queued->Wait(10) || queued->Done(), "waiting on a done ticket returns at once");
    Gate never;
    ShaderCompileService other(backend, {1, 1, 1 << 20});
    auto stuck = other.Submit(Key(9), CompilePriority::Needed, [&](CompileWorker&) {
        never.Wait();
        return Product(1);
    });
    Check(!stuck->Wait(20), "Wait times out on a running job");
    never.Open();
    Check(stuck->Wait(5000), "the job finishes after the gate opens");
}

std::filesystem::path TempRoot(const char* name) {
    auto path = std::filesystem::temp_directory_path() / (std::string("aps5_compile_test_") + name + "_" + std::to_string(std::rand()));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

void Touch(const std::filesystem::path& path, std::size_t bytes) {
    std::ofstream file(path, std::ios::binary);
    file << std::string(bytes, 'x');
}

void TestCacheDirectory() {
    const auto root = TempRoot("dir");
    CacheEnvironment environment{"25A1", "Apple A18 Pro GPU", 40, 1, 1};
    auto other = environment;
    other.osBuild = "25A2";
    Check(!(environment.Fingerprint() == other.Fingerprint()), "an OS build change moves the cache");
    auto changed = environment;
    changed.abiVersion = 2;
    Check(!(environment.Fingerprint() == changed.Fingerprint()), "an ABI change moves the cache");

    CompileCacheDirectory cache(root, environment, 300, 3);
    Check(cache.Shards().empty(), "a fresh cache has no shards");
    const auto first = cache.NewShardPath();
    Check(first.extension() == ".part" && first != cache.NewShardPath(), "shard paths are unique partials");
    Check(!cache.CommitShard(first), "a missing partial does not commit");
    Touch(first, 100);
    Check(cache.CommitShard(first) && !std::filesystem::exists(first) && cache.Shards().size() == 1, "a partial commits to a shard");
    const auto empty = cache.NewShardPath();
    Touch(empty, 0);
    Check(!cache.CommitShard(empty) && !std::filesystem::exists(empty), "an empty partial is dropped");

    for (int index = 0; index < 4; ++index) {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        const auto partial = cache.NewShardPath();
        Touch(partial, 100);
        Check(cache.CommitShard(partial), "shard commits");
    }
    Check(cache.Shards().size() == 3, "only the newest shards up to the cap are loaded");
    cache.Prune();
    Check(cache.Bytes() <= 300 && cache.Bytes() >= 200, "pruning drops the oldest shards over budget");
    Check(cache.Shards().size() == 3, "the newest shards survive");

    const auto shard = cache.Shards().front();
    cache.Quarantine(shard);
    Check(!std::filesystem::exists(shard) && std::filesystem::exists(shard.string() + ".bad"), "a bad shard is quarantined");

    const auto stale = cache.NewShardPath();
    Touch(stale, 10);
    std::filesystem::last_write_time(stale, std::filesystem::file_time_type::clock::now() - std::chrono::minutes(5));
    cache.Prune();
    Check(!std::filesystem::exists(stale), "a stale partial is removed");

    CompileCacheDirectory oldBuild(root, other, 300, 3);
    Touch(oldBuild.Directory() / "shard-1.mtl4a", 10);
    std::filesystem::last_write_time(oldBuild.Directory() / "shard-1.mtl4a", std::filesystem::file_time_type::clock::now() - std::chrono::hours(24 * 40));
    cache.Prune();
    Check(!std::filesystem::exists(oldBuild.Directory()), "an environment unused for a long time is removed");
    std::filesystem::remove_all(root);
}

void TestJournal() {
    const auto root = TempRoot("journal");
    const auto file = root / "warm.txt";
    {
        WarmJournal journal(file, 3);
        Check(journal.Load().empty(), "a missing journal is empty");
        journal.Record("a 1");
        journal.Record("b 2");
        journal.Record("a 1");
        journal.Record("c 3");
        journal.Record("d 4");
        Check(journal.Dirty(), "recording marks the journal dirty");
        Check(journal.Save() && !journal.Dirty(), "saving clears the dirty flag");
    }
    {
        WarmJournal journal(file, 3);
        const auto entries = journal.Load();
        Check(entries == std::vector<std::string>({"a 1", "c 3", "d 4"}), "the journal keeps the most recent distinct entries in order");
        journal.Record("d 4");
        Check(!journal.Dirty(), "re-recording the newest entry changes nothing");
        journal.Record("e 5\nbad");
        Check(!journal.Dirty(), "an entry with a newline is refused");
        journal.Record("e 5");
        Check(journal.Save(), "the journal saves again");
    }
    {
        WarmJournal journal(file, 3);
        const auto entries = journal.Load();
        Check(entries.size() == 3 && entries.back() == "e 5", "the newest entry is last after a reload");
    }
    std::filesystem::remove_all(root);
}

}

int main() {
    TestKeys();
    TestRunsAndDeduplicates();
    TestPriorityOrder();
    TestPrefetchLimit();
    TestFailures();
    TestRetention();
    TestBrokenWorker();
    TestShutdown();
    TestCacheDirectory();
    TestJournal();
    std::puts("shader compile service tests passed");
    return 0;
}
