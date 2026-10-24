#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_LABELTABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_LABELTABLE_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/Device.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

namespace AgcDriver::Graphics::Metal {

using SerialSet = std::array<std::uint64_t, QueueCount>;

struct LabelRecord {
    std::uint64_t address = 0;
    std::uint32_t bytes = 0;
    std::uint64_t stamp = 0;
    QueueKind queue = QueueKind::Graphics;
    std::uint64_t serial = 0;
};

class LabelTable {
public:
    void Add(const LabelRecord& record);
    void Settle(const SerialSet& completed);
    bool Overlaps(std::uint64_t address, std::uint64_t bytes) const;
    SerialSet SerialsOver(std::uint64_t address, std::uint64_t bytes) const;
    std::optional<LabelRecord> Newest(std::uint64_t address, std::uint64_t bytes, std::uint64_t afterStamp) const;
    std::size_t Size() const { return count.load(std::memory_order_acquire); }

private:
    mutable std::mutex mutex;
    std::vector<LabelRecord> records;
    std::atomic<std::size_t> count{0};
};

class CompletionActions {
public:
    using Action = std::function<void()>;

    bool Push(QueueKind queue, std::uint64_t serial, Action action);
    std::vector<Action> Take(QueueKind queue, std::uint64_t completed);
    std::size_t Size() const;
    void Clear();

private:
    mutable std::mutex mutex;
    std::array<std::multimap<std::uint64_t, Action>, QueueCount> pending;
};

}

#endif
