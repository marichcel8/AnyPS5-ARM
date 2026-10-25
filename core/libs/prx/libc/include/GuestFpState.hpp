#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTFPSTATE_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTFPSTATE_HPP

#include <cstdint>

namespace LibcDetail {

struct GuestFpState {
    std::uint16_t control = 0x037f;
    std::uint16_t status = 0;
    std::uint32_t mxcsr = 0x1f80;
};

inline GuestFpState& GuestFp() {
    thread_local GuestFpState state;
    return state;
}

}

#endif
