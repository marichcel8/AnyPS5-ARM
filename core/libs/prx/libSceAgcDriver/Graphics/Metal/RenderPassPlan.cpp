#include "prx/libSceAgcDriver/Graphics/Metal/RenderPassPlan.hpp"

namespace AgcDriver::Graphics::Metal {

StoreAction DiscardedStore(const AttachmentIntent& intent) {
    return intent.resolve ? StoreAction::MultisampleResolve : StoreAction::DontCare;
}

StoreAction FinalStore(const AttachmentIntent& intent, bool discarded) {
    if (discarded || intent.transient || !intent.observed) return DiscardedStore(intent);
    return intent.resolve ? StoreAction::StoreAndMultisampleResolve : StoreAction::Store;
}

AttachmentActions ChooseActions(const AttachmentIntent& intent, bool deferredStores) {
    AttachmentActions actions;
    if (intent.clear) actions.load = LoadAction::Clear;
    else if (intent.contentsDefined && !intent.transient) actions.load = LoadAction::Load;
    else actions.load = LoadAction::DontCare;
    actions.store = FinalStore(intent, false);
    if (deferredStores && !intent.transient && intent.observed) {
        actions.store = StoreAction::Unknown;
        actions.deferred = true;
    }
    return actions;
}

PassDecision PassTracker::Next(const PassTargets& targets, const std::array<AttachmentIntent, PassAttachmentCount>& intents) {
    PassDecision decision;
    if (!open) {
        decision.step = PassStep::Begin;
        ++counters.begun;
    } else {
        bool clears = false;
        for (std::uint32_t index = 0; index < PassAttachmentCount; ++index) clears = clears || (targets.identities[index] != 0 && intents[index].clear);
        if (targets == current && !clears) {
            ++counters.continued;
            return {PassStep::Continue, 0};
        }
        decision.step = PassStep::Restart;
        ++counters.restarted;
        for (std::uint32_t previous = 0; previous < PassAttachmentCount; ++previous) {
            const auto identity = current.identities[previous];
            if (identity == 0) continue;
            for (std::uint32_t index = 0; index < PassAttachmentCount; ++index) {
                if (targets.identities[index] != identity || !intents[index].clear) continue;
                if (current.width > targets.width || current.height > targets.height || current.layers > targets.layers) continue;
                decision.discardPrevious |= 1u << previous;
                ++counters.discarded;
                break;
            }
        }
    }
    current = targets;
    open = true;
    return decision;
}

}
