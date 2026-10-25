#include "Elf.hpp"
#include "Lifter.hpp"

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/thread.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <malloc/malloc.h>
#include <pthread/qos.h>
#endif

namespace {

const llvm::Triple HostTriple("arm64-apple-macosx13.0.0");

std::size_t Setting(const char* name, std::size_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == 0) return fallback;
    return static_cast<std::size_t>(std::strtoull(text, nullptr, 10));
}

struct Options {
    std::size_t jobs = Setting("APS5_TRANSLATOR_JOBS", std::clamp(std::thread::hardware_concurrency(), 1u, 4u));
    std::size_t partition = Setting("APS5_TRANSLATOR_PARTITION", 4096);
    std::size_t huge = Setting("APS5_TRANSLATOR_HUGE", 2500);
    std::size_t giant = Setting("APS5_TRANSLATOR_GIANT", 2500);
    std::size_t large = Setting("APS5_TRANSLATOR_LARGE", 2000);
    std::size_t budget = Setting("APS5_TRANSLATOR_BUDGET", 16384);
    bool verbose = std::getenv("APS5_TRANSLATOR_STATS") != nullptr;
};

void Optimize(llvm::Module& module, llvm::TargetMachine& machine, const std::vector<llvm::Function*>& huge, const std::vector<llvm::Function*>& giant) {
    for (auto* function : huge) {
        function->addFnAttr(llvm::Attribute::OptimizeNone);
        function->addFnAttr(llvm::Attribute::NoInline);
    }
    for (auto* function : giant) {
        function->addFnAttr(llvm::Attribute::OptimizeNone);
        function->addFnAttr(llvm::Attribute::NoInline);
    }
    llvm::LoopAnalysisManager loops;
    llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager cgscc;
    llvm::ModuleAnalysisManager modules;
    llvm::PassInstrumentationCallbacks callbacks;
    llvm::StandardInstrumentations instrumentations(module.getContext(), false);
    instrumentations.registerCallbacks(callbacks, &modules);
    llvm::PassBuilder builder(&machine, llvm::PipelineTuningOptions(), std::nullopt, &callbacks);
    builder.registerModuleAnalyses(modules);
    builder.registerCGSCCAnalyses(cgscc);
    builder.registerFunctionAnalyses(functions);
    builder.registerLoopAnalyses(loops);
    builder.crossRegisterProxies(loops, functions, cgscc, modules);
    builder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2).run(module, modules);
    if (huge.empty() && giant.empty()) return;
    llvm::FunctionPassManager light;
    light.addPass(llvm::SROAPass(llvm::SROAOptions::PreserveCFG));
    light.addPass(llvm::EarlyCSEPass(false));
    light.addPass(llvm::SimplifyCFGPass());
    light.addPass(llvm::ADCEPass());
    for (auto* function : huge) {
        function->removeFnAttr(llvm::Attribute::OptimizeNone);
        functions.invalidate(*function, llvm::PreservedAnalyses::none());
        light.run(*function, functions);
    }
    for (auto* function : giant) {
        function->removeFnAttr(llvm::Attribute::OptimizeNone);
        functions.invalidate(*function, llvm::PreservedAnalyses::none());
        light.run(*function, functions);
        function->addFnAttr(llvm::Attribute::OptimizeNone);
    }
}

std::unique_ptr<llvm::TargetMachine> Machine() {
    std::string error;
    const auto* target = llvm::TargetRegistry::lookupTarget(HostTriple, error);
    if (target == nullptr) throw std::runtime_error(error);
    return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(HostTriple, "apple-m1", "", {}, llvm::Reloc::PIC_));
}

std::string Hash(const std::vector<llvm::StringRef>& parts) {
    llvm::SHA256 hash;
    for (const auto part : parts) {
        const std::uint64_t size = part.size();
        hash.update(llvm::ArrayRef<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(&size), sizeof(size)));
        hash.update(part);
    }
    const auto digest = hash.final();
    return llvm::toHex(llvm::ArrayRef<std::uint8_t>(digest.data(), 16), true);
}

std::string CacheDirectory(const std::string& elfKey) {
    const char* setting = std::getenv("APS5_TRANSLATOR_CACHE");
    if (setting != nullptr && std::string(setting) == "off") return {};
    llvm::SmallString<256> directory;
    if (setting != nullptr && *setting != 0) directory = setting;
    else if (!llvm::sys::path::cache_directory(directory)) return {};
    else llvm::sys::path::append(directory, "AnyPS5", "translator");
    llvm::sys::path::append(directory, elfKey);
    if (llvm::sys::fs::create_directories(directory)) return {};
    return std::string(directory);
}

std::string ToolKey(const char* argv0) {
    const auto path = llvm::sys::fs::getMainExecutable(argv0, reinterpret_cast<void*>(&Setting));
    llvm::sys::fs::file_status status;
    if (path.empty() || llvm::sys::fs::status(path, status)) return "unknown";
    return path + ":" + std::to_string(status.getSize()) + ":" + std::to_string(status.getLastModificationTime().time_since_epoch().count());
}

void Join(const std::vector<std::string>& objects, const std::string& output) {
    std::string linker;
    if (const char* setting = std::getenv("APS5_TRANSLATOR_LD"); setting != nullptr && *setting != 0) linker = setting;
    else if (auto found = llvm::sys::findProgramByName("ld")) linker = *found;
    else throw std::runtime_error("cannot find ld to join the partitions");
    std::vector<llvm::StringRef> arguments{linker, "-r", "-arch", "arm64", "-o", output};
    for (const auto& object : objects) arguments.emplace_back(object);
    std::string error;
    if (llvm::sys::ExecuteAndWait(linker, arguments, std::nullopt, {}, 0, 0, &error) != 0) throw std::runtime_error("ld -r failed: " + error);
}

std::string FunctionName(const Translator::Elf& elf, std::uint64_t entry) {
    return "g_" + llvm::utohexstr(entry - elf.ImageBegin(), true);
}

void Merge(Translator::Statistics& into, const Translator::Statistics& from) {
    into.instructions += from.instructions;
    for (const auto& [what, count] : from.unsupported) into.unsupported[what] += count;
}

void SaveStatistics(const Translator::Statistics& stats, const std::string& path) {
    std::error_code code;
    llvm::raw_fd_ostream output(path, code, llvm::sys::fs::OF_None);
    if (code) throw std::runtime_error("cannot write " + path + ": " + code.message());
    output << stats.instructions << "\n";
    for (const auto& [what, count] : stats.unsupported) output << count << " " << what << "\n";
}

Translator::Statistics LoadStatistics(const std::string& path) {
    Translator::Statistics stats;
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) return stats;
    llvm::SmallVector<llvm::StringRef> lines;
    (*buffer)->getBuffer().split(lines, '\n', -1, false);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto [number, what] = lines[i].split(' ');
        std::size_t count = 0;
        if (number.getAsInteger(10, count)) continue;
        if (i == 0) stats.instructions = count;
        else stats.unsupported[what.str()] += count;
    }
    return stats;
}

int EmitLlvm(const Translator::Elf& elf, Translator::Lifter& lifter, const Options& options, const char* path) {
    llvm::LLVMContext context;
    const auto partitions = lifter.Partitions(~std::size_t{0});
    Translator::Statistics stats;
    auto module = lifter.Lift(context, partitions.front(), true, false, options.huge, stats);
    if (llvm::verifyModule(*module, &llvm::errs())) {
        module->print(llvm::errs(), nullptr);
        std::fprintf(stderr, "lifted module does not verify\n");
        return 1;
    }
    auto machine = Machine();
    module->setTargetTriple(HostTriple);
    module->setDataLayout(machine->createDataLayout());
    std::vector<llvm::Function*> huge, giant;
    for (const auto entry : partitions.front()) {
        const auto size = lifter.FunctionSize(entry);
        auto* function = module->getFunction(FunctionName(elf, entry));
        if (size > options.giant) giant.push_back(function);
        else if (size > options.huge) huge.push_back(function);
        else if (size > options.large) function->addFnAttr(llvm::Attribute::NoInline);
    }
    Optimize(*module, *machine, huge, giant);
    std::error_code code;
    llvm::raw_fd_ostream output(path, code, llvm::sys::fs::OF_None);
    if (code) {
        std::fprintf(stderr, "cannot write %s: %s\n", path, code.message().c_str());
        return 1;
    }
    module->print(output, nullptr);
    return 0;
}

}

// aps5-translator <guest.elf> <output.o> [--emit-llvm]: lifts the guest's x86-64 code and writes an
// arm64 Mach-O object to link with the translator runtime and the host libraries.
int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <guest.elf> <output.o> [--emit-llvm]\n", argv[0]);
        return 2;
    }
    const bool emitLlvm = argc > 3 && std::string(argv[3]) == "--emit-llvm";
    const auto started = std::chrono::steady_clock::now();
    try {
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmPrinter();
        const Options options;
        const Translator::Elf elf(argv[1]);
        Translator::Lifter lifter(elf);
        lifter.Discover();
        if (emitLlvm) return EmitLlvm(elf, lifter, options, argv[2]);

        const auto partitions = lifter.Partitions(options.partition);
        const bool shared = partitions.size() > 1;
        const auto& bytes = elf.Bytes();
        const std::string settings = std::to_string(options.partition) + ":" + std::to_string(options.huge) + ":" + std::to_string(options.giant) + ":" +
                                     std::to_string(options.large) + ":" +
                                     (std::getenv("APS5_TRANSLATOR_ALL_FLAGS") != nullptr ? "flags" : "");
        const auto elfKey = Hash({"elf", llvm::StringRef(reinterpret_cast<const char*>(bytes.data()), bytes.size())});
        const auto runKey = Hash({ToolKey(argv[0]), elfKey, settings});
        auto cache = CacheDirectory(elfKey);
        llvm::SmallString<256> scratch;
        if (cache.empty()) {
            if (llvm::sys::fs::createUniqueDirectory("aps5-translator", scratch)) throw std::runtime_error("cannot create a work directory");
        }
        const std::string work = cache.empty() ? std::string(scratch) : cache;

        std::vector<std::string> objects(partitions.size());
        std::vector<std::size_t> sizes(partitions.size(), 0);
        std::vector<std::size_t> costs(partitions.size(), 0);
        std::size_t hugeCount = 0;
        for (std::size_t i = 0; i < partitions.size(); ++i) {
            std::string entries;
            for (const auto entry : partitions[i]) {
                sizes[i] += lifter.FunctionSize(entry);
                costs[i] += lifter.FunctionSize(entry) > options.giant ? lifter.FunctionSize(entry) / 8 : lifter.FunctionSize(entry);
                if (lifter.FunctionSize(entry) > options.huge) ++hugeCount;
                entries += llvm::utohexstr(entry, true) + ",";
            }
            llvm::SmallString<256> path(work);
            llvm::sys::path::append(path, Hash({runKey, std::to_string(i), entries}) + ".o");
            objects[i] = std::string(path);
        }
        std::vector<std::size_t> order(partitions.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return sizes[a] > sizes[b]; });

        if (options.verbose) {
            for (std::size_t i = 0; i < std::min<std::size_t>(5, order.size()); ++i)
                std::fprintf(stderr, "  largest partition %zu: %zu functions, %zu instructions\n", order[i], partitions[order[i]].size(), sizes[order[i]]);
        }
        Translator::Statistics total;
        std::mutex lock;
        std::size_t next = 0;
        std::atomic<std::size_t> cached{0};
        std::string failure;
        auto translate = [&](std::size_t index) {
            if (llvm::sys::fs::exists(objects[index])) {
                const auto stats = LoadStatistics(objects[index] + ".txt");
                const std::lock_guard<std::mutex> guard(lock);
                Merge(total, stats);
                ++cached;
                return;
            }
            const auto begun = std::chrono::steady_clock::now();
            llvm::LLVMContext context;
            Translator::Statistics stats;
            auto module = lifter.Lift(context, partitions[index], index == 0, shared, options.huge, stats);
            if (llvm::verifyModule(*module, &llvm::errs())) throw std::runtime_error("lifted partition " + std::to_string(index) + " does not verify");
            auto machine = Machine();
            module->setTargetTriple(HostTriple);
            module->setDataLayout(machine->createDataLayout());
            std::vector<llvm::Function*> huge, giant;
            for (const auto entry : partitions[index]) {
                auto* function = module->getFunction(FunctionName(elf, entry));
                const auto size = lifter.FunctionSize(entry);
                if (size > options.giant) giant.push_back(function);
                else if (size > options.huge) huge.push_back(function);
                else if (size > options.large) function->addFnAttr(llvm::Attribute::NoInline);
            }
            Optimize(*module, *machine, huge, giant);
            const auto partial = objects[index] + ".part";
            {
                std::error_code code;
                llvm::raw_fd_ostream output(partial, code, llvm::sys::fs::OF_None);
                if (code) throw std::runtime_error("cannot write " + partial + ": " + code.message());
                llvm::legacy::PassManager passes;
                if (machine->addPassesToEmitFile(passes, output, nullptr, llvm::CodeGenFileType::ObjectFile)) throw std::runtime_error("the target cannot emit an object file");
                passes.run(*module);
            }
            SaveStatistics(stats, objects[index] + ".txt");
            if (llvm::sys::fs::rename(partial, objects[index])) throw std::runtime_error("cannot write " + objects[index]);
            const std::lock_guard<std::mutex> guard(lock);
            Merge(total, stats);
            if (options.verbose) {
                const std::chrono::duration<double> took = std::chrono::steady_clock::now() - begun;
                std::fprintf(stderr, "  partition %zu: %zu functions, %zu instructions, %.1f s\n", index, partitions[index].size(), sizes[index], took.count());
            }
        };
        std::condition_variable released;
        std::size_t inFlight = 0;
        auto worker = [&]() {
#if defined(__APPLE__)
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
            for (;;) {
                std::size_t index = 0, cost = 0;
                {
                    std::unique_lock<std::mutex> guard(lock);
                    released.wait(guard, [&] {
                        return next >= order.size() || inFlight == 0 || inFlight + std::min(costs[order[next]], options.budget) <= options.budget;
                    });
                    if (next >= order.size()) return;
                    index = order[next++];
                    cost = std::min(costs[index], options.budget);
                    inFlight += cost;
                }
                auto release = llvm::make_scope_exit([&] {
#if defined(__APPLE__)
                    malloc_zone_pressure_relief(nullptr, 0);
#endif
                    {
                        const std::lock_guard<std::mutex> guard(lock);
                        inFlight -= cost;
                    }
                    released.notify_all();
                });
                try {
                    translate(index);
                } catch (const std::exception& exception) {
                    const std::lock_guard<std::mutex> guard(lock);
                    if (failure.empty()) failure = exception.what();
                    next = order.size();
                }
            }
        };
        const auto jobs = std::min(options.jobs, partitions.size());
        std::vector<llvm::thread> threads;
        for (std::size_t i = 0; i < jobs; ++i) threads.emplace_back(std::optional<unsigned>(256u << 20), worker);
        for (auto& thread : threads) thread.join();
        if (!failure.empty()) throw std::runtime_error(failure);

        if (objects.size() == 1) {
            if (llvm::sys::fs::copy_file(objects.front(), argv[2])) throw std::runtime_error(std::string("cannot write ") + argv[2]);
        } else {
            Join(objects, argv[2]);
        }
        if (cache.empty()) {
            llvm::sys::fs::remove_directories(work);
        } else {
            std::set<std::string> keep;
            for (const auto& object : objects) keep.insert({object, object + ".txt"});
            std::error_code code;
            for (llvm::sys::fs::directory_iterator it(work, code), end; it != end && !code; it.increment(code)) {
                if (keep.count(it->path()) == 0) llvm::sys::fs::remove(it->path());
            }
        }

        const auto& found = lifter.Stats();
        const std::chrono::duration<double> took = std::chrono::steady_clock::now() - started;
        std::fprintf(stderr, "%zu functions, %zu instructions lifted\n", found.functions, total.instructions);
        std::fprintf(stderr, "%zu partitions on %zu threads, %zu cached, %zu huge functions optimized lightly, %.1f s\n", partitions.size(), jobs, cached.load(), hugeCount,
                     took.count());
        for (const auto& [what, count] : total.unsupported) std::fprintf(stderr, "  unsupported: %s (%zu)\n", what.c_str(), count);
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "%s\n", exception.what());
        return 1;
    }
    return 0;
}
