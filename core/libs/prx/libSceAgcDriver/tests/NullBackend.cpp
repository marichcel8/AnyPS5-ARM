#include "prx/libSceAgcDriver/Graphics/Backend/Backend.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace AgcDriver;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}

int main() {
    try {
        check(Backend::kKind == Backend::Kind::Metal, "backend kind");
        check(std::strcmp(Backend::kName, "Metal") == 0, "backend name");
        VulkanDevice device;
        check(!device.Presentable(), "null device is not presentable");
        const std::array<std::byte, 8> label{};
        check(device.WriteLabelOnGpu(0x1000, label, 1, 0) == 3, "labels are written by the CPU");
        const std::array<std::uint32_t, 4> pattern{};
        check(!device.FillBuffer(0x1000, 16, pattern), "fills are done by the CPU");
        check(!Graphics::Recorder::LookupLabel(0x1000, 8, 0).has_value(), "no pending labels");
        check(!Graphics::StorageTexture::AnyPending(), "no pending storage textures");
        check(Graphics::Recorder::Active() == nullptr, "no active recorder");
        device.WaitIdle();
        std::puts("ok");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
