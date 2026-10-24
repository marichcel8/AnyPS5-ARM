#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_POWERMONITOR_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_METAL_POWERMONITOR_HPP

#include "prx/libSceAgcDriver/Graphics/Metal/PowerPolicy.hpp"
#include <memory>

namespace AgcDriver::Graphics::Metal {

class PowerMonitor {
public:
    PowerMonitor();
    ~PowerMonitor();
    PowerMonitor(const PowerMonitor&) = delete;
    PowerMonitor& operator=(const PowerMonitor&) = delete;

    PowerInputs Sample(std::int64_t nowNs);

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
