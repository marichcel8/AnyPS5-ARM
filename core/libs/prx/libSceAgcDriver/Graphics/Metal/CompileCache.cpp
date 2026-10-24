#include "prx/libSceAgcDriver/Graphics/Metal/CompileCache.hpp"
#include "ShaderCacheDirectory.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <system_error>
#include <unistd.h>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr std::string_view ShardSuffix = ".mtl4a";
constexpr std::string_view StagingDirectory = "tmp";
constexpr std::string_view BadSuffix = ".bad";
constexpr auto StalePartial = std::chrono::minutes(1);
constexpr auto StaleDirectory = std::chrono::hours(24 * 30);
constexpr std::size_t MaxBad = 4;

struct ShardFile {
    std::filesystem::path path;
    std::uint64_t bytes;
    std::filesystem::file_time_type modified;
};

bool endsWith(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<ShardFile> listFiles(const std::filesystem::path& directory, std::string_view suffix) {
    std::vector<ShardFile> files;
    std::error_code error;
    for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) {
        const auto name = it->path().filename().string();
        if (!endsWith(name, suffix)) continue;
        std::error_code stat;
        const auto bytes = it->file_size(stat);
        const auto modified = it->last_write_time(stat);
        if (stat) continue;
        files.push_back({it->path(), bytes, modified});
    }
    std::sort(files.begin(), files.end(), [](const ShardFile& a, const ShardFile& b) { return a.modified != b.modified ? a.modified > b.modified : a.path > b.path; });
    return files;
}

}

CompileKey CacheEnvironment::Fingerprint() const {
    return CompileKeyBuilder(0x4d544c43ull).AddString(osBuild).AddString(gpuName).Add(metalVersion).Add(abiVersion).Add(compilerVersion).Finish();
}

CompileCacheDirectory::CompileCacheDirectory(std::filesystem::path root, const CacheEnvironment& environment, std::uint64_t budgetBytes, std::uint32_t maxShards)
    : root(std::move(root)), directory(this->root / CompileKeyHex(environment.Fingerprint())), budgetBytes(budgetBytes), maxShards(std::max<std::uint32_t>(maxShards, 1)) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
}

std::vector<std::filesystem::path> CompileCacheDirectory::Shards() const {
    std::vector<std::filesystem::path> shards;
    for (const auto& file : listFiles(directory, ShardSuffix)) {
        if (shards.size() >= maxShards) break;
        if (file.bytes == 0) continue;
        shards.push_back(file.path);
    }
    return shards;
}

std::filesystem::path CompileCacheDirectory::NewShardPath() {
    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto name = "shard-" + std::to_string(stamp) + "-" + std::to_string(static_cast<long>(getpid())) + "-" + std::to_string(serial++);
    std::error_code error;
    std::filesystem::create_directories(directory / StagingDirectory, error);
    return directory / StagingDirectory / (name + std::string(ShardSuffix));
}

bool CompileCacheDirectory::CommitShard(const std::filesystem::path& partial) {
    if (partial.parent_path().filename() != StagingDirectory) return false;
    const auto target = directory / partial.filename();
    std::error_code error;
    const auto bytes = std::filesystem::file_size(partial, error);
    if (error || bytes == 0) {
        std::filesystem::remove(partial, error);
        return false;
    }
    std::filesystem::rename(partial, target, error);
    if (error) {
        std::filesystem::remove(partial, error);
        return false;
    }
    return true;
}

void CompileCacheDirectory::AbandonShard(const std::filesystem::path& partial) {
    std::error_code error;
    std::filesystem::remove(partial, error);
}

void CompileCacheDirectory::Quarantine(const std::filesystem::path& shard) {
    std::error_code error;
    auto bad = shard;
    bad += BadSuffix;
    std::filesystem::rename(shard, bad, error);
    if (error) std::filesystem::remove(shard, error);
}

void CompileCacheDirectory::Prune() {
    std::error_code error;
    const auto now = std::filesystem::file_time_type::clock::now();
    for (const auto& file : listFiles(directory / StagingDirectory, ShardSuffix)) {
        if (now - file.modified > StalePartial) std::filesystem::remove(file.path, error);
    }
    const auto bad = listFiles(directory, BadSuffix);
    for (std::size_t index = MaxBad; index < bad.size(); ++index) std::filesystem::remove(bad[index].path, error);
    auto shards = listFiles(directory, ShardSuffix);
    std::uint64_t total = 0;
    for (const auto& shard : shards) total += shard.bytes;
    for (std::size_t index = shards.size(); index > 1 && (total > budgetBytes || index > maxShards); --index) {
        const auto& oldest = shards[index - 1];
        std::filesystem::remove(oldest.path, error);
        total -= oldest.bytes;
    }
    for (std::filesystem::directory_iterator it(root, error), end; !error && it != end; it.increment(error)) {
        if (!it->is_directory() || it->path() == directory) continue;
        auto newest = std::filesystem::file_time_type::min();
        for (std::filesystem::directory_iterator inner(it->path(), error), innerEnd; !error && inner != innerEnd; inner.increment(error)) {
            std::error_code stat;
            const auto modified = inner->last_write_time(stat);
            if (!stat) newest = std::max(newest, modified);
        }
        error.clear();
        if (now - newest > StaleDirectory) std::filesystem::remove_all(it->path(), error);
    }
}

std::uint64_t CompileCacheDirectory::Bytes() const {
    std::uint64_t total = 0;
    for (const auto& shard : listFiles(directory, ShardSuffix)) total += shard.bytes;
    return total;
}

WarmJournal::WarmJournal(std::filesystem::path file, std::uint32_t maxEntries) : file(std::move(file)), maxEntries(std::max<std::uint32_t>(maxEntries, 1)) {}

std::vector<std::string> WarmJournal::Load() {
    std::lock_guard lock(mutex);
    if (!loaded) {
        loaded = true;
        std::ifstream stream(file);
        std::string line;
        std::vector<std::string> fromDisk;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;
            fromDisk.push_back(line);
        }
        for (const auto& recorded : entries) {
            fromDisk.erase(std::remove(fromDisk.begin(), fromDisk.end(), recorded), fromDisk.end());
            fromDisk.push_back(recorded);
        }
        entries = std::move(fromDisk);
        if (entries.size() > maxEntries) entries.erase(entries.begin(), entries.end() - maxEntries);
    }
    return entries;
}

void WarmJournal::Record(const std::string& line) {
    if (line.empty() || line.find('\n') != std::string::npos) return;
    std::lock_guard lock(mutex);
    if (!entries.empty() && entries.back() == line) return;
    entries.erase(std::remove(entries.begin(), entries.end(), line), entries.end());
    entries.push_back(line);
    if (entries.size() > maxEntries) entries.erase(entries.begin(), entries.end() - maxEntries);
    dirty = true;
}

bool WarmJournal::Save() {
    std::string text;
    {
        std::lock_guard lock(mutex);
        if (!dirty) return true;
        for (const auto& entry : entries) {
            text += entry;
            text += '\n';
        }
        dirty = false;
    }
    const auto* bytes = reinterpret_cast<const std::byte*>(text.data());
    const bool written = ShaderRecompiler::WriteFileAtomically(file, {bytes, text.size()});
    if (!written) {
        std::lock_guard lock(mutex);
        dirty = true;
    }
    return written;
}

bool WarmJournal::Dirty() const {
    std::lock_guard lock(mutex);
    return dirty;
}

}
