#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_RENDERSTATETRANSLATE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_RENDERSTATETRANSLATE_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/RenderState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"

namespace AgcDriver::Graphics::Metal {

struct TranslatedState {
    RenderPipelineState pipeline;
    DepthStencilState depthStencil;
    DynamicState dynamic;
    PrimitiveType primitive = PrimitiveType::Triangle;
    bool cullsEverything = false;
    bool restartOnLists = false;
    const char* rejection = nullptr;
};

PixelFormat ColorTargetFormat(VkFormat format);
PixelFormat DepthTargetFormat(VkFormat format);
bool DepthTargetHasStencil(VkFormat format);
BlendFactor TranslateBlendFactor(VkBlendFactor factor);
Winding TranslateFrontFace(VkFrontFace face);
TranslatedState TranslateState(const State& state);

}

#endif
