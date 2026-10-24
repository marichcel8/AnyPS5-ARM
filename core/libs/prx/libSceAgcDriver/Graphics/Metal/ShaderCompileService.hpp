#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_SHADERCOMPILESERVICE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_SHADERCOMPILESERVICE_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/CompileKey.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace AgcDriver::Graphics::Metal {

enum class CompilePriority : std::uint8_t { Prefetch, Needed, Urgent };
enum class CompileState : std::uint8_t { Pending, Running, Ready, Failed, Cancelled };

struct CompileOutcome {
    std::shared_ptr<void> product;
    std::uint64_t costBytes = 0;
    std::string error;
};

class CompileWorker {
public:
    virtual ~CompileWorker() = default;
};

class CompileBackend {
public:
    virtual ~CompileBackend() = default;
    virtual std::unique_ptr<CompileWorker> CreateWorker(std::uint32_t index) = 0;
    virtual void Prepare(CompileWorker& worker, CompilePriority priority) {
        (void)worker;
        (void)priority;
    }
};

using CompileTask = std::function<CompileOutcome(CompileWorker&)>;

class CompileTicket {
public:
    [[nodiscard]] CompileState State() const { return state.load(std::memory_order_acquire); }
    [[nodiscard]] bool Ready() const { return State() == CompileState::Ready; }
    [[nodiscard]] bool Done() const { return State() >= CompileState::Ready; }
    [[nodiscard]] CompilePriority Priority() const { return static_cast<CompilePriority>(priority.load(std::memory_order_relaxed)); }
    [[nodiscard]] const CompileKey& Key() const { return key; }
    [[nodiscard]] const std::shared_ptr<void>& Product() const { return product; }
    [[nodiscard]] const std::string& Error() const { return error; }
    [[nodiscard]] std::uint64_t CompileNanos() const { return compileNanos; }
    [[nodiscard]] std::uint64_t QueueNanos() const { return queueNanos; }

    template <class T>
    [[nodiscard]] std::shared_ptr<T> ProductAs() const { return std::static_pointer_cast<T>(product); }

    bool Wait(std::uint32_t timeoutMs) const;

private:
    friend class ShaderCompileService;

    CompileKey key;
    std::atomic<CompileState> state{CompileState::Pending};
    std::atomic<std::uint8_t> priority{0};
    std::shared_ptr<void> product;
    std::string error;
    std::uint64_t compileNanos = 0;
    std::uint64_t queueNanos = 0;
    mutable std::mutex mutex;
    mutable std::condition_variable done;
};

struct CompileServiceConfig {
    std::uint32_t workers = 2;
    std::uint32_t prefetchWorkers = 1;
    std::uint64_t retainedBytes = 64ull << 20u;
};

struct CompileStats {
    std::uint64_t submitted = 0;
    std::uint64_t deduplicated = 0;
    std::uint64_t promoted = 0;
    std::uint64_t completed = 0;
    std::uint64_t failed = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t evicted = 0;
    std::uint64_t queued = 0;
    std::uint64_t running = 0;
    std::uint64_t retainedBytes = 0;
    std::uint64_t compileNanos = 0;
    std::uint64_t queueNanos = 0;
    std::uint64_t maxQueueNanos = 0;
};

class ShaderCompileService {
public:
    ShaderCompileService(std::shared_ptr<CompileBackend> backend, const CompileServiceConfig& config);
    ~ShaderCompileService();
    ShaderCompileService(const ShaderCompileService&) = delete;
    ShaderCompileService& operator=(const ShaderCompileService&) = delete;

    std::shared_ptr<CompileTicket> Submit(const CompileKey& key, CompilePriority priority, CompileTask task);
    std::shared_ptr<CompileTicket> Find(const CompileKey& key);
    void Forget(const CompileKey& key);
    bool WaitIdle(std::uint32_t timeoutMs);
    void Shutdown();
    CompileStats Stats() const;

private:
    struct Job {
        std::shared_ptr<CompileTicket> ticket;
        CompileTask task;
        std::atomic<bool> claimed{false};
        std::chrono::steady_clock::time_point enqueued;
        std::list<CompileKey>::iterator lruPosition;
        bool inLru = false;
        std::uint64_t cost = 0;
    };

    void WorkerMain(std::uint32_t index);
    std::shared_ptr<Job> PopLocked(CompilePriority& priority);
    void FinishLocked(const std::shared_ptr<Job>& job, CompileOutcome& outcome, std::uint64_t compileNanos, std::uint64_t queueNanos, CompilePriority ran);
    void EvictLocked();

    std::shared_ptr<CompileBackend> backend;
    CompileServiceConfig config;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::unordered_map<CompileKey, std::shared_ptr<Job>, CompileKeyHash> jobs;
    std::deque<std::shared_ptr<Job>> queues[3];
    std::list<CompileKey> lru;
    std::uint32_t runningPrefetch = 0;
    std::uint64_t pendingJobs = 0;
    std::uint64_t runningJobs = 0;
    bool stopping = false;
    CompileStats stats;
    std::vector<std::thread> threads;
};

}

#endif
