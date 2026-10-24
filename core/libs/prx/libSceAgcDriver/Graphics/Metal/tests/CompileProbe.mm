#include "prx/libSceAgcDriver/Graphics/Metal/MetalCompiler.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace AgcDriver::Graphics::Metal;

int failures = 0;

void Check(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

double millisecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::string KernelSource(int variant, int depth) {
    std::string text = "#include <metal_stdlib>\nusing namespace metal;\nkernel void probe_main(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {\n    uint a = tid + " + std::to_string(variant) + "u;\n    uint b = tid * 2654435761u;\n";
    for (int step = 0; step < depth; ++step) {
        text += "    a = (a ^ (b >> " + std::to_string(step % 13 + 1) + ")) * " + std::to_string(2246822519u + step * 2u) + "u + (b << " + std::to_string(step % 7 + 1) + ");\n";
        text += "    b = rotate(b, a & 31u) ^ (a + " + std::to_string(step) + "u);\n";
        if (step % 16 == 15) text += "    if (a > n) { a -= n; b += a; }\n";
    }
    text += "    out[tid] = a ^ b;\n}\n";
    return text;
}

struct Source {
    std::shared_ptr<std::string> text;
    CompileKey key;
};

std::vector<Source> MakeSources(int count, int depth) {
    std::vector<Source> sources;
    for (int index = 0; index < count; ++index) {
        auto text = std::make_shared<std::string>(KernelSource(index, depth + (index % 4) * depth / 2));
        sources.push_back({text, CompileKeyBuilder(0x50524f42ull).AddString(*text).Finish()});
    }
    return sources;
}

ComputePipelineRequest Request(const Source& source, CompilePriority priority) {
    ComputePipelineRequest request;
    request.library.key = source.key;
    request.library.owner = source.text;
    request.library.text = *source.text;
    request.library.name = "probe";
    request.function = "probe_main";
    request.label = "probe";
    request.priority = priority;
    return request;
}

struct Phase {
    double submitMaxMs = 0;
    double loopMaxMs = 0;
    double totalMs = 0;
    int ready = 0;
};

Phase RunPhase(MetalCompiler& compiler, const std::vector<Source>& sources, CompilePriority priority) {
    Phase phase;
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<CompileTicket>> tickets;
    for (const auto& source : sources) {
        const auto before = std::chrono::steady_clock::now();
        tickets.push_back(compiler.Compute(Request(source, priority)));
        phase.submitMaxMs = std::max(phase.submitMaxMs, millisecondsSince(before));
    }
    for (;;) {
        const auto tick = std::chrono::steady_clock::now();
        bool done = true;
        for (const auto& ticket : tickets) done = done && ticket->Done();
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        phase.loopMaxMs = std::max(phase.loopMaxMs, millisecondsSince(tick));
        if (done) break;
        if (millisecondsSince(start) > 120000.0) break;
    }
    phase.totalMs = millisecondsSince(start);
    for (const auto& ticket : tickets) {
        if (ticket->Ready()) ++phase.ready;
        else std::fprintf(stderr, "ticket failed: %s\n", ticket->Error().c_str());
    }
    return phase;
}

void Print(const char* name, const Phase& phase, const CompilerReport& report, int count) {
    std::printf("%s: %d/%d ready in %.1f ms, max submit %.3f ms, max poll-loop gap %.3f ms | libraries %llu (%.1f ms total) pipelines %llu (%.1f ms total, %llu archive-fast) shards loaded %llu rejected %llu bytes %llu flushes %llu (%.1f ms)\n",
                name, phase.ready, count, phase.totalMs, phase.submitMaxMs, phase.loopMaxMs,
                static_cast<unsigned long long>(report.librariesBuilt), report.libraryNanos / 1.0e6,
                static_cast<unsigned long long>(report.pipelinesBuilt), report.pipelineNanos / 1.0e6, static_cast<unsigned long long>(report.pipelinesFromArchive),
                static_cast<unsigned long long>(report.archiveShardsLoaded), static_cast<unsigned long long>(report.archiveShardsRejected), static_cast<unsigned long long>(report.archiveBytes),
                static_cast<unsigned long long>(report.flushes), report.flushNanos / 1.0e6);
}

}

int main(int argc, char** argv) {
    @autoreleasepool {
        std::filesystem::path directory;
        int count = 24;
        int depth = 160;
        bool requireArchive = false;
        bool keep = false;
        for (int index = 1; index < argc; ++index) {
            if (std::strcmp(argv[index], "--dir") == 0 && index + 1 < argc) {
                directory = argv[++index];
                keep = true;
            } else if (std::strcmp(argv[index], "--count") == 0 && index + 1 < argc) {
                count = std::atoi(argv[++index]);
            } else if (std::strcmp(argv[index], "--depth") == 0 && index + 1 < argc) {
                depth = std::atoi(argv[++index]);
            } else if (std::strcmp(argv[index], "--require-archive") == 0) {
                requireArchive = true;
            }
        }
        if (directory.empty()) directory = std::filesystem::temp_directory_path() / ("aps5_compile_probe_" + std::to_string(std::rand()));
        if (!keep) std::filesystem::remove_all(directory);

        std::string failure;
        auto device = Device::Create(&failure);
        if (device == nullptr) {
            std::fprintf(stderr, "FAIL: %s\n", failure.c_str());
            return 1;
        }
        std::printf("device name=\"%s\" cache=%s count=%d depth=%d\n", device->Limits().name.c_str(), directory.c_str(), count, depth);
        const auto sources = MakeSources(count, depth);
        CompilerConfig config;
        config.cacheRoot = directory;

        double coldPipelineMs = 0;
        {
            auto compiler = MetalCompiler::Create(*device, config, &failure);
            if (compiler == nullptr) {
                std::fprintf(stderr, "FAIL: compiler: %s\n", failure.c_str());
                return 1;
            }
            const auto phase = RunPhase(*compiler, sources, CompilePriority::Needed);
            Check(phase.ready == count, "every cold compile finishes");
            Check(phase.submitMaxMs < 5.0, "a submit never blocks the caller");
            const auto report = compiler->Report();
            Print("cold", phase, report, count);
            coldPipelineMs = report.pipelineNanos / 1.0e6;

            const auto repeat = RunPhase(*compiler, sources, CompilePriority::Urgent);
            Check(repeat.ready == count && repeat.totalMs < 50.0, "a repeat request is served from the result cache");
            Check(compiler->Report().service.deduplicated >= static_cast<std::uint64_t>(count), "repeat requests deduplicate");

            Source broken{std::make_shared<std::string>("kernel void probe_main( {"), CompileKeyBuilder(1).Add(7).Finish()};
            auto ticket = compiler->Compute(Request(broken, CompilePriority::Urgent));
            Check(ticket->Wait(60000) && ticket->State() == CompileState::Failed && !ticket->Error().empty(), "broken source fails with a message");
            std::printf("broken source error: %.120s\n", ticket->Error().c_str());

            auto sampled = compiler->Compute(Request(sources[0], CompilePriority::Needed));
            Check(sampled->Ready(), "the first pipeline stays ready");
            const auto pipeline = sampled->ProductAs<CompiledPipeline>();
            Check(pipeline != nullptr && pipeline->maxTotalThreadsPerThreadgroup > 0 && pipeline->threadExecutionWidth > 0, "the pipeline reports its limits");
            std::printf("pipeline limits: maxThreads %u width %u staticTG %u\n", pipeline->maxTotalThreadsPerThreadgroup, pipeline->threadExecutionWidth, pipeline->staticThreadgroupMemoryBytes);
        }

        {
            auto compiler = MetalCompiler::Create(*device, config, &failure);
            if (compiler == nullptr) {
                std::fprintf(stderr, "FAIL: compiler (warm): %s\n", failure.c_str());
                return 1;
            }
            const auto phase = RunPhase(*compiler, sources, CompilePriority::Needed);
            Check(phase.ready == count, "every warm compile finishes");
            const auto report = compiler->Report();
            Print("warm", phase, report, count);
            const double warmPipelineMs = report.pipelineNanos / 1.0e6;
            std::printf("pipeline build time cold %.1f ms, warm %.1f ms (%.2fx), archive hits %llu of %llu\n", coldPipelineMs, warmPipelineMs, warmPipelineMs > 0 ? coldPipelineMs / warmPipelineMs : 0.0,
                        static_cast<unsigned long long>(report.pipelinesFromArchive), static_cast<unsigned long long>(report.pipelinesBuilt));
            Check(report.archiveShardsLoaded >= 1, "the second run loads the archive written by the first");
            if (requireArchive) Check(report.pipelinesFromArchive * 10 >= report.pipelinesBuilt * 9, "at least 90% of warm pipelines come from the archive");
        }

        if (!keep) std::filesystem::remove_all(directory);
        if (failures != 0) return 1;
        std::puts("compile probe passed");
        return 0;
    }
}
