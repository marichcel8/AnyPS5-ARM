#include "prx/libc/include/general/VabiMacros.hpp"
#include <array>
#include <cstdint>
#include <stdexcept>
#if defined(__x86_64__)
#include <xmmintrin.h>
#else
#include "prx/libc/include/GuestFpState.hpp"
#endif

namespace {

constexpr unsigned Exceptions = 0x3f;
constexpr unsigned Rounding = 0xc00;

#if defined(__x86_64__)
std::uint16_t ReadControl() {
    std::uint16_t control;
    asm volatile("fnstcw %0" : "=m"(control));
    return control;
}

void WriteControl(std::uint16_t control) {
    asm volatile("fldcw %0" : : "m"(control));
}

std::uint16_t ReadStatus() {
    std::uint16_t status;
    asm volatile("fnstsw %0" : "=am"(status));
    return status;
}

unsigned ReadCsr() {
    return _mm_getcsr();
}

void WriteCsr(unsigned csr) {
    _mm_setcsr(csr);
}

void ReplaceFlags(unsigned mask, unsigned flags) {
    std::array<std::uint32_t, 7> environment;
    asm volatile("fnstenv %0" : "=m"(environment));
    environment[1] = (environment[1] & ~mask) | (flags & mask);
    asm volatile("fldenv %0" : : "m"(environment));
    _mm_setcsr((_mm_getcsr() & ~mask) | (flags & mask));
}

void ClearFlags() {
    asm volatile("fnclex");
    _mm_setcsr(_mm_getcsr() & ~Exceptions);
}

void WaitForFlags() {
    asm volatile("fwait");
}
#else
std::uint16_t ReadControl() {
    return LibcDetail::GuestFp().control;
}

void WriteControl(std::uint16_t control) {
    LibcDetail::GuestFp().control = control;
}

std::uint16_t ReadStatus() {
    return LibcDetail::GuestFp().status;
}

unsigned ReadCsr() {
    return LibcDetail::GuestFp().mxcsr;
}

void WriteCsr(unsigned csr) {
    LibcDetail::GuestFp().mxcsr = csr;
}

void ReplaceFlags(unsigned mask, unsigned flags) {
    auto& state = LibcDetail::GuestFp();
    state.status = static_cast<std::uint16_t>((state.status & ~mask) | (flags & mask));
    state.mxcsr = (state.mxcsr & ~mask) | (flags & mask);
}

void ClearFlags() {
    auto& state = LibcDetail::GuestFp();
    state.status = static_cast<std::uint16_t>(state.status & ~Exceptions);
    state.mxcsr &= ~Exceptions;
}

void WaitForFlags() {
}
#endif

int ChangeTraps(int exceptions, bool enable) {
    const auto mask = static_cast<unsigned>(exceptions) & Exceptions;
    const auto control = ReadControl();
    const unsigned csr = ReadCsr();
    const auto changed = static_cast<std::uint16_t>(enable ? (control & ~mask) : (control | mask));
    WriteControl(changed);
    WriteCsr(enable ? (csr & ~(mask << 7)) : (csr | (mask << 7)));
    return static_cast<int>(~((csr >> 7) | control) | ~Exceptions);
}

}

extern "C" {

int APS5_VABI fegetround_nid_postfix() {
    return ReadControl() & Rounding;
}

int APS5_VABI fesetround_nid_postfix(int rounding) {
    const auto mode = static_cast<unsigned>(rounding);
    if ((mode & ~Rounding) != 0) return -1;
    const auto control = static_cast<std::uint16_t>((ReadControl() & ~Rounding) | mode);
    WriteControl(control);
    WriteCsr((ReadCsr() & ~(Rounding << 3)) | (mode << 3));
    return 0;
}

int APS5_VABI fetestexcept_nid_postfix(int exceptions) {
    return (ReadStatus() | ReadCsr()) & static_cast<unsigned>(exceptions) & Exceptions;
}

int APS5_VABI fegetexceptflag_nid_postfix(std::uint16_t* flags, int exceptions) {
    if (flags == nullptr) throw std::invalid_argument("fegetexceptflag: null flags");
    *flags = static_cast<std::uint16_t>(fetestexcept_nid_postfix(exceptions));
    return 0;
}

int* APS5_VABI __error_nid_postfix();

void APS5_VABI _Feraise_nid_postfix(int exceptions) {
    constexpr unsigned Invalid = 0x01, Ranges = 0x1c, Overflows = 0x18, Inexact = 0x20;
    constexpr int GuestEdom = 33, GuestErange = 34;
    auto raised = static_cast<unsigned>(exceptions);
    if ((raised & Overflows) != 0) raised |= Inexact;
    *__error_nid_postfix() = GuestEdom;
    ReplaceFlags(raised, raised);
    WaitForFlags();
    if ((raised & Invalid) != 0) *__error_nid_postfix() = GuestEdom;
    else if ((raised & Ranges) != 0) *__error_nid_postfix() = GuestErange;
}

int APS5_VABI fesetexceptflag_nid_postfix(const std::uint16_t* flags, int exceptions) {
    if (flags == nullptr) throw std::invalid_argument("fesetexceptflag: null flags");
    ReplaceFlags(static_cast<unsigned>(exceptions) & Exceptions, *flags);
    return 0;
}

int APS5_VABI feclearexcept_nid_postfix(int exceptions) {
    const auto mask = static_cast<unsigned>(exceptions) & Exceptions;
    if (mask == Exceptions) {
        ClearFlags();
    } else if (mask != 0) {
        ReplaceFlags(mask, 0);
    }
    return 0;
}


int APS5_VABI feenableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, true);
}

int APS5_VABI fedisableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, false);
}

int APS5_VABI __feenableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, true);
}

int APS5_VABI __fedisableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, false);
}

int APS5_VABI fegettrapenable_nid_postfix() {
    return ~ReadControl() & Exceptions;
}

int APS5_VABI fegetexcept_nid_postfix() {
    return fegettrapenable_nid_postfix();
}

int APS5_VABI fesettrapenable_nid_postfix(int exceptions) {
    const auto mask = static_cast<std::uint16_t>(exceptions);
    if (mask > Exceptions) return 1;
    WriteControl(static_cast<std::uint16_t>(ReadControl() & ~mask));
    WriteCsr(ReadCsr() & ~(static_cast<unsigned>(mask) << 7));
    return 0;
}

}
