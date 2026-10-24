#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_RENDERPASSPLAN_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_RENDERPASSPLAN_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/RenderState.hpp"
#include <array>
#include <cstdint>

namespace AgcDriver::Graphics::Metal {

enum class LoadAction : std::uint8_t { DontCare = 0, Load = 1, Clear = 2 };
enum class StoreAction : std::uint8_t { DontCare = 0, Store = 1, MultisampleResolve = 2, StoreAndMultisampleResolve = 3, Unknown = 4 };

struct AttachmentIntent {
    bool clear = false;
    bool contentsDefined = true;
    bool observed = true;
    bool resolve = false;
    bool transient = false;
};

struct AttachmentActions {
    LoadAction load = LoadAction::Load;
    StoreAction store = StoreAction::Store;
    bool deferred = false;
};

AttachmentActions ChooseActions(const AttachmentIntent& intent, bool deferredStores);
StoreAction DiscardedStore(const AttachmentIntent& intent);
StoreAction FinalStore(const AttachmentIntent& intent, bool discarded);

inline constexpr std::uint32_t PassAttachmentCount = MaxColorAttachments + 2u;
inline constexpr std::uint32_t DepthAttachmentIndex = MaxColorAttachments;
inline constexpr std::uint32_t StencilAttachmentIndex = MaxColorAttachments + 1u;

struct PassTargets {
    std::array<std::uint64_t, PassAttachmentCount> identities{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t layers = 1;
    std::uint32_t samples = 1;

    bool operator==(const PassTargets&) const = default;
};

enum class PassStep : std::uint8_t { Continue, Begin, Restart };

struct PassDecision {
    PassStep step = PassStep::Begin;
    std::uint32_t discardPrevious = 0;
};

struct PassCounters {
    std::uint64_t begun = 0;
    std::uint64_t continued = 0;
    std::uint64_t restarted = 0;
    std::uint64_t discarded = 0;
};

class PassTracker {
public:
    PassDecision Next(const PassTargets& targets, const std::array<AttachmentIntent, PassAttachmentCount>& intents);
    void End() { open = false; }
    [[nodiscard]] bool Open() const { return open; }
    [[nodiscard]] const PassTargets& Current() const { return current; }
    [[nodiscard]] const PassCounters& Counters() const { return counters; }

private:
    PassTargets current;
    PassCounters counters;
    bool open = false;
};

}

#endif
