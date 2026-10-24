#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_SYNC_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_SYNC_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/LabelTable.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver::Graphics::Metal {

enum class LabelOutcome : std::uint8_t { Queued, Direct, Rejected };

struct Recording {
    void* commands = nullptr;
    QueueKind queue = QueueKind::Graphics;
    std::uint32_t slot = 0;
    std::vector<std::uint32_t> staging;
    std::vector<LabelRecord> labels;
    std::vector<std::uint64_t> labelOffsets;
    std::vector<std::byte> labelPayload;

    explicit operator bool() const { return commands != nullptr; }
};

struct SyncCounters {
    std::uint64_t recordings = 0;
    std::uint64_t submits = 0;
    std::uint64_t allocatorsCreated = 0;
    std::uint64_t allocatorsReused = 0;
    std::uint64_t labelsQueued = 0;
    std::uint64_t labelsDirect = 0;
    std::uint64_t labelsRejected = 0;
    std::uint64_t labelsEncoded = 0;
    std::uint64_t labelsDropped = 0;
    std::uint64_t labelDispatches = 0;
    std::uint64_t labelsStored = 0;
    std::uint64_t labelBytes = 0;
    std::uint64_t stagingChunks = 0;
    std::uint64_t stagingBytes = 0;
    std::uint64_t queueWaits = 0;
    std::uint64_t completionsInline = 0;
    std::uint64_t completionsDeferred = 0;
};

class Gate {
public:
    ~Gate();
    Gate(const Gate&) = delete;
    Gate& operator=(const Gate&) = delete;

    void Signal(std::uint64_t value);
    std::uint64_t Value() const;
    bool Wait(std::uint64_t value, std::uint32_t timeoutMs) const;
    void* Native() const;

    struct State;

private:
    friend class Sync;
    explicit Gate(std::unique_ptr<State> state);

    std::unique_ptr<State> state;
};

class Sync {
public:
    static constexpr std::size_t MaxLabelBytes = 65536;

    static std::unique_ptr<Sync> Create(Device& device, std::string* failure);
    ~Sync();
    Sync(const Sync&) = delete;
    Sync& operator=(const Sync&) = delete;

    Recording Begin(QueueKind queue, const char* label);
    std::uint64_t Submit(Recording* recordings, std::size_t count);
    void Discard(Recording& recording);

    LabelOutcome WriteLabel(QueueKind queue, std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, bool openWork);
    std::size_t EncodeLabels(Recording& recording);
    bool HasQueuedLabels(QueueKind queue) const;
    bool LabelPendingIn(std::uint64_t address, std::uint64_t bytes);
    bool WaitLabels(std::uint64_t address, std::uint64_t bytes, std::uint32_t timeoutMs);
    std::optional<LabelRecord> NewestLabel(std::uint64_t address, std::uint64_t bytes, std::uint64_t afterStamp);

    void WaitOnQueue(QueueKind waiter, QueueKind signaler, std::uint64_t serial);
    std::shared_ptr<Gate> NewGate(const char* label);
    void WaitOnGate(QueueKind waiter, const Gate& gate, std::uint64_t value);

    void OnComplete(QueueKind queue, std::uint64_t serial, std::function<void()> action);

    void SetGpuLabels(bool enabled);
    bool GpuLabels() const;
    void Trim(std::size_t keepIdle);
    SyncCounters Counters() const;

    struct State;

private:
    explicit Sync(std::unique_ptr<State> state);

    std::unique_ptr<State> state;
};

}

#endif
