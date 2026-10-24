#include "prx/libSceAgcDriver/Graphics/Metal/GuestWindows.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace AgcDriver::Graphics::Metal {

namespace {

std::uint64_t alignDown(std::uint64_t value, std::uint64_t alignment) {
    return value & ~(alignment - 1);
}

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

bool windowOrder(const Window& left, const Window& right) {
    if (left.span.base != right.span.base) return left.span.base < right.span.base;
    return left.ownerBegin < right.ownerBegin;
}

bool sameOwner(const Window& window, const GuestRange& range) {
    return window.ownerBegin == range.address && window.ownerEnd == range.address + range.bytes && window.writable == range.writable;
}

std::vector<GuestRange> sortedRanges(std::span<const GuestRange> ranges) {
    std::vector<GuestRange> sorted;
    sorted.reserve(ranges.size());
    for (const auto& range : ranges) {
        if (range.bytes != 0) sorted.push_back(range);
    }
    std::sort(sorted.begin(), sorted.end(), [](const GuestRange& left, const GuestRange& right) { return left.address < right.address; });
    return sorted;
}

const GuestRange* firstIntersecting(const std::vector<GuestRange>& sorted, std::uint64_t begin, std::uint64_t end) {
    auto it = std::partition_point(sorted.begin(), sorted.end(), [&](const GuestRange& range) { return range.address + range.bytes <= begin; });
    if (it != sorted.begin()) {
        const auto previous = std::prev(it);
        if (previous->address + previous->bytes > begin) return &*previous;
    }
    if (it != sorted.end() && it->address < end) return &*it;
    return nullptr;
}

const GuestRange* exactRange(const std::vector<GuestRange>& sorted, std::uint64_t begin) {
    const auto it = std::lower_bound(sorted.begin(), sorted.end(), begin, [](const GuestRange& range, std::uint64_t address) { return range.address < address; });
    if (it == sorted.end() || it->address != begin) return nullptr;
    return &*it;
}

}

HostSpan HostPagesCovering(std::uint64_t address, std::uint64_t bytes, std::uint64_t hostPageBytes) {
    if (hostPageBytes == 0 || (hostPageBytes & (hostPageBytes - 1)) != 0) throw std::invalid_argument("host page size must be a power of two");
    if (bytes == 0) return {alignDown(address, hostPageBytes), 0};
    if (address > std::numeric_limits<std::uint64_t>::max() - bytes - hostPageBytes) throw std::out_of_range("guest range wraps the address space");
    const auto base = alignDown(address, hostPageBytes);
    return {base, alignUp(address + bytes, hostPageBytes) - base};
}

std::uint64_t WindowStep(std::uint64_t maxWindowBytes, std::uint64_t hostPageBytes) {
    return std::max(hostPageBytes, alignDown(maxWindowBytes / 2, hostPageBytes));
}

std::vector<HostSpan> PlanWindows(HostSpan span, std::uint64_t maxWindowBytes, std::uint64_t hostPageBytes) {
    std::vector<HostSpan> plan;
    if (span.bytes == 0) return plan;
    if (maxWindowBytes < 2 * hostPageBytes) throw std::invalid_argument("window limit below two host pages");
    if (span.bytes <= maxWindowBytes) {
        plan.push_back(span);
        return plan;
    }
    const auto step = WindowStep(maxWindowBytes, hostPageBytes);
    const auto end = span.End();
    for (std::uint64_t start = span.base;; start += step) {
        const auto bytes = std::min<std::uint64_t>(2 * step, end - start);
        plan.push_back({start, bytes});
        if (start + bytes >= end) break;
    }
    return plan;
}

Resolved Resolve(const WindowSnapshot& snapshot, std::uint64_t address, std::uint64_t bytes) {
    const auto& windows = snapshot.windows;
    auto it = std::upper_bound(windows.begin(), windows.end(), address, [](std::uint64_t value, const Window& window) { return value < window.span.base; });
    for (int probe = 0; probe < 3 && it != windows.begin(); ++probe) {
        --it;
        if (it->span.Contains(address, bytes)) return {&*it, it->gpuAddress + (address - it->span.base)};
    }
    return {};
}

std::vector<ShaderRecompiler::BdaAbi::Range> BuildBdaTable(std::span<const Window> windows) {
    std::vector<const Window*> ordered;
    ordered.reserve(windows.size());
    for (const auto& window : windows) ordered.push_back(&window);
    std::sort(ordered.begin(), ordered.end(), [](const Window* left, const Window* right) {
        if (left->ownerBegin != right->ownerBegin) return left->ownerBegin < right->ownerBegin;
        return left->span.base < right->span.base;
    });
    std::vector<ShaderRecompiler::BdaAbi::Range> table;
    table.reserve(ordered.size());
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        const auto& window = *ordered[index];
        const bool last = index + 1 == ordered.size() || ordered[index + 1]->ownerBegin != window.ownerBegin;
        const auto begin = std::max(window.ownerBegin, window.span.base);
        const auto end = last ? window.ownerEnd : std::min(window.ownerEnd, ordered[index + 1]->span.base);
        if (begin >= end) continue;
        const auto permissions = ShaderRecompiler::BdaAbi::Read | (window.writable ? ShaderRecompiler::BdaAbi::Write : 0u);
        table.push_back({begin, end, window.gpuAddress + (begin - window.span.base), permissions, 0});
    }
    return table;
}

std::size_t BdaTableBytes(std::size_t entries) {
    return sizeof(ShaderRecompiler::BdaAbi::Header) + entries * sizeof(ShaderRecompiler::BdaAbi::Range);
}

void WriteBdaTable(std::span<std::byte> destination, std::span<const ShaderRecompiler::BdaAbi::Range> entries) {
    if (destination.size() < BdaTableBytes(entries.size())) throw std::length_error("BDA table destination too small");
    if (entries.size() > std::numeric_limits<std::uint32_t>::max()) throw std::length_error("BDA table range count overflow");
    const ShaderRecompiler::BdaAbi::Header header{ShaderRecompiler::BdaAbi::Version, static_cast<std::uint32_t>(entries.size()), sizeof(ShaderRecompiler::BdaAbi::Range), 0};
    std::memcpy(destination.data(), &header, sizeof(header));
    if (!entries.empty()) std::memcpy(destination.data() + sizeof(header), entries.data(), entries.size_bytes());
}

GuestWindows::GuestWindows(WindowBackend& backend, std::uint64_t hostPageBytes, std::uint64_t maxWindowBytes)
    : backend(backend), hostPage(hostPageBytes), maxWindow(alignDown(maxWindowBytes, hostPageBytes)) {
    if (hostPage < GuestPageBytes || (hostPage & (hostPage - 1)) != 0) throw std::invalid_argument("host page size must be a power of two of at least 4 KiB");
    if (maxWindow < 2 * hostPage) throw std::invalid_argument("window limit below two host pages");
    current.store(std::make_shared<const WindowSnapshot>(), std::memory_order_release);
}

GuestWindows::~GuestWindows() {
    Clear();
}

bool GuestWindows::wrapOwner(const GuestRange& range, std::vector<Window>& out) {
    const auto first = out.size();
    for (const auto& span : PlanWindows(HostPagesCovering(range.address, range.bytes, hostPage), maxWindow, hostPage)) {
        const auto wrapped = backend.Wrap(span, range.writable);
        if (wrapped.handle == nullptr) {
            for (auto index = first; index < out.size(); ++index) retire(out[index]);
            out.resize(first);
            ++counters.wrapFailures;
            return false;
        }
        out.push_back({span, range.address, range.address + range.bytes, wrapped.gpuAddress, wrapped.handle, range.writable, wrapped.copied});
        ++counters.windows;
        if (wrapped.copied) counters.copiedBytes += span.bytes;
        else counters.wrappedBytes += span.bytes;
    }
    return true;
}

void GuestWindows::retire(const Window& window) {
    backend.Retire({window.handle, window.gpuAddress, window.copied}, window.span);
    --counters.windows;
    if (window.copied) counters.copiedBytes -= window.span.bytes;
    else counters.wrappedBytes -= window.span.bytes;
}

void GuestWindows::publish(std::vector<Window> windows) {
    std::sort(windows.begin(), windows.end(), windowOrder);
    auto snapshot = std::make_shared<WindowSnapshot>();
    snapshot->serial = serial.load(std::memory_order_relaxed) + 1;
    snapshot->table = BuildBdaTable(windows);
    snapshot->windows = std::move(windows);
    const auto published = snapshot->serial;
    current.store(std::move(snapshot), std::memory_order_release);
    serial.store(published, std::memory_order_release);
    ++counters.publishes;
}

std::size_t GuestWindows::Apply(std::span<const GuestRange> map, std::span<const GuestRange> unmap) {
    const auto mapped = sortedRanges(map);
    const auto unmapped = sortedRanges(unmap);
    std::lock_guard lock(mutex);
    const auto snapshot = current.load(std::memory_order_acquire);
    std::vector<Window> windows;
    windows.reserve(snapshot->windows.size() + mapped.size());
    bool changed = false;
    for (const auto& window : snapshot->windows) {
        const bool unmappedOwner = firstIntersecting(unmapped, window.ownerBegin, window.ownerEnd) != nullptr;
        const auto* overlap = firstIntersecting(mapped, window.ownerBegin, window.ownerEnd);
        if (unmappedOwner || (overlap != nullptr && !sameOwner(window, *overlap))) {
            retire(window);
            changed = true;
            continue;
        }
        windows.push_back(window);
    }
    std::vector<std::uint64_t> owners;
    owners.reserve(windows.size());
    for (const auto& window : windows) owners.push_back(window.ownerBegin);
    std::sort(owners.begin(), owners.end());
    std::size_t failures = 0;
    for (const auto& range : mapped) {
        if (std::binary_search(owners.begin(), owners.end(), range.address)) continue;
        if (wrapOwner(range, windows)) changed = true;
        else ++failures;
    }
    if (changed) publish(std::move(windows));
    return failures;
}

bool GuestWindows::Map(const GuestRange& range) {
    return Apply(std::span(&range, 1), {}) == 0;
}

void GuestWindows::Unmap(std::uint64_t address, std::uint64_t bytes) {
    const GuestRange range{address, bytes, false};
    Apply({}, std::span(&range, 1));
}

void GuestWindows::Replace(std::span<const GuestRange> ranges) {
    const auto wanted = sortedRanges(ranges);
    std::lock_guard lock(mutex);
    const auto snapshot = current.load(std::memory_order_acquire);
    std::vector<Window> windows;
    windows.reserve(std::max(snapshot->windows.size(), wanted.size()));
    bool changed = false;
    for (const auto& window : snapshot->windows) {
        const auto* match = exactRange(wanted, window.ownerBegin);
        if (match == nullptr || !sameOwner(window, *match)) {
            retire(window);
            changed = true;
            continue;
        }
        windows.push_back(window);
    }
    std::vector<std::uint64_t> owners;
    owners.reserve(windows.size());
    for (const auto& window : windows) owners.push_back(window.ownerBegin);
    std::sort(owners.begin(), owners.end());
    for (const auto& range : wanted) {
        if (std::binary_search(owners.begin(), owners.end(), range.address)) continue;
        if (wrapOwner(range, windows)) changed = true;
    }
    if (changed) publish(std::move(windows));
}

void GuestWindows::Clear() {
    std::lock_guard lock(mutex);
    const auto snapshot = current.load(std::memory_order_acquire);
    if (snapshot->windows.empty()) return;
    for (const auto& window : snapshot->windows) retire(window);
    publish({});
}

WindowCounters GuestWindows::Counters() const {
    std::lock_guard lock(mutex);
    return counters;
}

}
