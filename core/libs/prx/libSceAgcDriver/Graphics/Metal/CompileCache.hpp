#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_COMPILECACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_COMPILECACHE_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/CompileKey.hpp"
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver::Graphics::Metal {

struct CacheEnvironment {
    std::string osBuild;
    std::string gpuName;
    std::uint32_t metalVersion = 0;
    std::uint32_t abiVersion = 0;
    std::uint32_t compilerVersion = 0;

    [[nodiscard]] CompileKey Fingerprint() const;
};

class CompileCacheDirectory {
public:
    CompileCacheDirectory(std::filesystem::path root, const CacheEnvironment& environment, std::uint64_t budgetBytes, std::uint32_t maxShards);

    [[nodiscard]] const std::filesystem::path& Directory() const { return directory; }
    [[nodiscard]] std::vector<std::filesystem::path> Shards() const;
    [[nodiscard]] std::filesystem::path NewShardPath();
    bool CommitShard(const std::filesystem::path& partial);
    void AbandonShard(const std::filesystem::path& partial);
    void Quarantine(const std::filesystem::path& shard);
    void Prune();
    [[nodiscard]] std::uint64_t Bytes() const;

private:
    std::filesystem::path root;
    std::filesystem::path directory;
    std::uint64_t budgetBytes;
    std::uint32_t maxShards;
    std::uint64_t serial = 0;
};

class WarmJournal {
public:
    WarmJournal(std::filesystem::path file, std::uint32_t maxEntries);

    std::vector<std::string> Load();
    void Record(const std::string& line);
    bool Save();
    [[nodiscard]] bool Dirty() const;

private:
    std::filesystem::path file;
    std::uint32_t maxEntries;
    mutable std::mutex mutex;
    std::vector<std::string> entries;
    bool loaded = false;
    bool dirty = false;
};

}

#endif
