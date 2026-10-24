#include "prx/libSceAgcDriver/Graphics/Metal/LabelTable.hpp"
#include <algorithm>

namespace AgcDriver::Graphics::Metal {

namespace {

bool overlaps(const LabelRecord& record, std::uint64_t address, std::uint64_t bytes) {
    return record.address < address + bytes && address < record.address + record.bytes;
}

}

void LabelTable::Add(const LabelRecord& record) {
    std::lock_guard lock(mutex);
    records.push_back(record);
    count.store(records.size(), std::memory_order_release);
}

void LabelTable::Settle(const SerialSet& completed) {
    if (count.load(std::memory_order_acquire) == 0) return;
    std::lock_guard lock(mutex);
    std::erase_if(records, [&](const LabelRecord& record) { return record.serial <= completed[static_cast<std::size_t>(record.queue)]; });
    count.store(records.size(), std::memory_order_release);
}

bool LabelTable::Overlaps(std::uint64_t address, std::uint64_t bytes) const {
    if (count.load(std::memory_order_acquire) == 0 || bytes == 0) return false;
    std::lock_guard lock(mutex);
    return std::any_of(records.begin(), records.end(), [&](const LabelRecord& record) { return overlaps(record, address, bytes); });
}

SerialSet LabelTable::SerialsOver(std::uint64_t address, std::uint64_t bytes) const {
    SerialSet result{};
    if (count.load(std::memory_order_acquire) == 0 || bytes == 0) return result;
    std::lock_guard lock(mutex);
    for (const auto& record : records) {
        if (!overlaps(record, address, bytes)) continue;
        auto& slot = result[static_cast<std::size_t>(record.queue)];
        slot = std::max(slot, record.serial);
    }
    return result;
}

std::optional<LabelRecord> LabelTable::Newest(std::uint64_t address, std::uint64_t bytes, std::uint64_t afterStamp) const {
    if (count.load(std::memory_order_acquire) == 0 || bytes == 0) return std::nullopt;
    std::lock_guard lock(mutex);
    std::optional<LabelRecord> best;
    for (const auto& record : records) {
        if (record.stamp <= afterStamp || !overlaps(record, address, bytes)) continue;
        if (!best || record.stamp > best->stamp) best = record;
    }
    return best;
}

bool CompletionActions::Push(QueueKind queue, std::uint64_t serial, Action action) {
    std::lock_guard lock(mutex);
    auto& map = pending[static_cast<std::size_t>(queue)];
    const bool fresh = map.find(serial) == map.end();
    map.emplace(serial, std::move(action));
    return fresh;
}

std::vector<CompletionActions::Action> CompletionActions::Take(QueueKind queue, std::uint64_t completed) {
    std::vector<Action> ready;
    std::lock_guard lock(mutex);
    auto& map = pending[static_cast<std::size_t>(queue)];
    const auto end = map.upper_bound(completed);
    for (auto it = map.begin(); it != end; ++it) ready.push_back(std::move(it->second));
    map.erase(map.begin(), end);
    return ready;
}

std::size_t CompletionActions::Size() const {
    std::lock_guard lock(mutex);
    std::size_t total = 0;
    for (const auto& map : pending) total += map.size();
    return total;
}

void CompletionActions::Clear() {
    std::lock_guard lock(mutex);
    for (auto& map : pending) map.clear();
}

}
