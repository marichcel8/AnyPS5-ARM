#include "prx/libSceAgcDriver/Graphics/Metal/ShaderCompileService.hpp"
#include <algorithm>
#include <exception>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr std::uint64_t FailedCost = 64;

std::uint64_t nanosSince(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

}

bool CompileTicket::Wait(std::uint32_t timeoutMs) const {
    std::unique_lock lock(mutex);
    done.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return State() >= CompileState::Ready; });
    return State() >= CompileState::Ready;
}

ShaderCompileService::ShaderCompileService(std::shared_ptr<CompileBackend> backend, const CompileServiceConfig& config) : backend(std::move(backend)), config(config) {
    this->config.workers = std::max<std::uint32_t>(this->config.workers, 1);
    this->config.prefetchWorkers = std::clamp<std::uint32_t>(this->config.prefetchWorkers, 1, this->config.workers);
    for (std::uint32_t index = 0; index < this->config.workers; ++index) threads.emplace_back([this, index] { WorkerMain(index); });
}

ShaderCompileService::~ShaderCompileService() {
    Shutdown();
}

std::shared_ptr<CompileTicket> ShaderCompileService::Submit(const CompileKey& key, CompilePriority priority, CompileTask task) {
    std::lock_guard lock(mutex);
    ++stats.submitted;
    const auto found = jobs.find(key);
    if (found != jobs.end()) {
        auto& job = found->second;
        ++stats.deduplicated;
        if (job->inLru) lru.splice(lru.begin(), lru, job->lruPosition);
        if (!job->claimed.load(std::memory_order_acquire) && static_cast<std::uint8_t>(priority) > job->ticket->priority.load(std::memory_order_relaxed)) {
            job->ticket->priority.store(static_cast<std::uint8_t>(priority), std::memory_order_relaxed);
            queues[static_cast<std::size_t>(priority)].push_back(job);
            ++stats.promoted;
            wake.notify_one();
        }
        return job->ticket;
    }
    auto job = std::make_shared<Job>();
    job->ticket = std::make_shared<CompileTicket>();
    job->ticket->key = key;
    job->ticket->priority.store(static_cast<std::uint8_t>(priority), std::memory_order_relaxed);
    if (stopping) {
        job->ticket->state.store(CompileState::Cancelled, std::memory_order_release);
        ++stats.cancelled;
        return job->ticket;
    }
    job->task = std::move(task);
    job->enqueued = std::chrono::steady_clock::now();
    jobs.emplace(key, job);
    queues[static_cast<std::size_t>(priority)].push_back(job);
    ++pendingJobs;
    wake.notify_one();
    return job->ticket;
}

std::shared_ptr<CompileTicket> ShaderCompileService::Find(const CompileKey& key) {
    std::lock_guard lock(mutex);
    const auto found = jobs.find(key);
    if (found == jobs.end()) return nullptr;
    if (found->second->inLru) lru.splice(lru.begin(), lru, found->second->lruPosition);
    return found->second->ticket;
}

void ShaderCompileService::Forget(const CompileKey& key) {
    std::lock_guard lock(mutex);
    const auto found = jobs.find(key);
    if (found == jobs.end() || !found->second->ticket->Done()) return;
    auto& job = found->second;
    if (job->inLru) {
        lru.erase(job->lruPosition);
        stats.retainedBytes -= job->cost;
    }
    jobs.erase(found);
}

bool ShaderCompileService::WaitIdle(std::uint32_t timeoutMs) {
    std::unique_lock lock(mutex);
    return idle.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return pendingJobs == 0 && runningJobs == 0; });
}

void ShaderCompileService::Shutdown() {
    std::vector<std::thread> joining;
    {
        std::lock_guard lock(mutex);
        if (threads.empty()) return;
        stopping = true;
        for (auto& queue : queues) {
            for (auto& job : queue) {
                if (job->claimed.exchange(true)) continue;
                {
                    std::lock_guard ticketLock(job->ticket->mutex);
                    job->ticket->state.store(CompileState::Cancelled, std::memory_order_release);
                }
                job->ticket->done.notify_all();
                job->task = {};
                --pendingJobs;
                ++stats.cancelled;
            }
            queue.clear();
        }
        joining = std::move(threads);
        threads.clear();
        wake.notify_all();
        idle.notify_all();
    }
    for (auto& thread : joining) thread.join();
}

CompileStats ShaderCompileService::Stats() const {
    std::lock_guard lock(mutex);
    auto copy = stats;
    copy.queued = pendingJobs;
    copy.running = runningJobs;
    return copy;
}

std::shared_ptr<ShaderCompileService::Job> ShaderCompileService::PopLocked(CompilePriority& priority) {
    for (int level = 2; level >= 0; --level) {
        if (level == 0 && runningPrefetch >= config.prefetchWorkers) continue;
        auto& queue = queues[level];
        while (!queue.empty()) {
            auto job = std::move(queue.front());
            queue.pop_front();
            if (job->claimed.exchange(true)) continue;
            priority = static_cast<CompilePriority>(job->ticket->priority.load(std::memory_order_relaxed));
            return job;
        }
    }
    return nullptr;
}

void ShaderCompileService::WorkerMain(std::uint32_t index) {
    std::unique_ptr<CompileWorker> worker;
    std::string workerError;
    try {
        worker = backend->CreateWorker(index);
    } catch (const std::exception& failure) {
        workerError = failure.what();
    } catch (...) {
        workerError = "compile worker creation failed";
    }
    if (worker == nullptr && workerError.empty()) workerError = "compile worker creation failed";
    std::unique_lock lock(mutex);
    for (;;) {
        std::shared_ptr<Job> job;
        CompilePriority priority = CompilePriority::Needed;
        while (!stopping) {
            job = PopLocked(priority);
            if (job != nullptr) break;
            wake.wait(lock);
        }
        if (job == nullptr) return;
        --pendingJobs;
        ++runningJobs;
        if (priority == CompilePriority::Prefetch) ++runningPrefetch;
        job->ticket->state.store(CompileState::Running, std::memory_order_release);
        const auto queueNanos = nanosSince(job->enqueued);
        auto task = std::move(job->task);
        job->task = {};
        lock.unlock();

        const auto started = std::chrono::steady_clock::now();
        CompileOutcome outcome;
        try {
            if (worker == nullptr) {
                outcome.error = workerError;
            } else {
                backend->Prepare(*worker, priority);
                outcome = task(*worker);
            }
        } catch (const std::exception& failure) {
            outcome = {};
            outcome.error = failure.what();
        } catch (...) {
            outcome = {};
            outcome.error = "compile task threw";
        }
        task = {};
        if (outcome.error.empty() && outcome.product == nullptr) outcome.error = "compile task returned no product";
        if (!outcome.error.empty()) outcome.product.reset();
        const auto compileNanos = nanosSince(started);

        lock.lock();
        FinishLocked(job, outcome, compileNanos, queueNanos, priority);
    }
}

void ShaderCompileService::FinishLocked(const std::shared_ptr<Job>& job, CompileOutcome& outcome, std::uint64_t compileNanos, std::uint64_t queueNanos, CompilePriority ran) {
    auto& ticket = *job->ticket;
    const bool ok = outcome.error.empty();
    {
        std::lock_guard ticketLock(ticket.mutex);
        ticket.product = std::move(outcome.product);
        ticket.error = std::move(outcome.error);
        ticket.compileNanos = compileNanos;
        ticket.queueNanos = queueNanos;
        ticket.state.store(ok ? CompileState::Ready : CompileState::Failed, std::memory_order_release);
    }
    ticket.done.notify_all();
    --runningJobs;
    if (ran == CompilePriority::Prefetch) --runningPrefetch;
    if (ok) ++stats.completed;
    else ++stats.failed;
    stats.compileNanos += compileNanos;
    stats.queueNanos += queueNanos;
    stats.maxQueueNanos = std::max(stats.maxQueueNanos, queueNanos);
    job->cost = ok ? std::max<std::uint64_t>(outcome.costBytes, 1) : FailedCost;
    lru.push_front(ticket.key);
    job->lruPosition = lru.begin();
    job->inLru = true;
    stats.retainedBytes += job->cost;
    EvictLocked();
    wake.notify_one();
    if (pendingJobs == 0 && runningJobs == 0) idle.notify_all();
}

void ShaderCompileService::EvictLocked() {
    while (stats.retainedBytes > config.retainedBytes && lru.size() > 1) {
        const auto key = lru.back();
        lru.pop_back();
        const auto found = jobs.find(key);
        if (found == jobs.end()) continue;
        stats.retainedBytes -= found->second->cost;
        jobs.erase(found);
        ++stats.evicted;
    }
}

}
