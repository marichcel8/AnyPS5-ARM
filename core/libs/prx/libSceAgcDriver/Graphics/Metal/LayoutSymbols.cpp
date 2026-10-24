#include "prx/libSceAgcDriver/Graphics/Metal/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

const Metal::FormatInfo& format(std::uint32_t guestFormat) {
    const auto* found = Metal::FindFormat(guestFormat);
    if (found == nullptr) throw std::runtime_error("AGC graphics: unsupported guest texture format " + std::to_string(guestFormat));
    return *found;
}

}

std::uint32_t BytesPerElement(std::uint32_t guestFormat) {
    return format(guestFormat).guestElementBytes;
}

bool IsBlockCompressed(std::uint32_t guestFormat) {
    return format(guestFormat).blockCompressed;
}

std::uint32_t BlockWidth(std::uint32_t guestFormat) {
    return IsBlockCompressed(guestFormat) ? 4u : 1u;
}

std::uint32_t BlockHeight(std::uint32_t guestFormat) {
    return IsBlockCompressed(guestFormat) ? 4u : 1u;
}

}
