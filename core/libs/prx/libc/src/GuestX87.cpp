#include "prx/libc/include/Api.hpp"
#include "prx/libc/include/General.hpp"

#if defined(__aarch64__)

#include <cstddef>
#include <type_traits>

// The guest passes long double the x87 way: arguments in memory on the stack, results in st(0)
// (st(0) and st(1) for a complex result), and long double* as pointers to 80-bit values. Natively
// the host's long double is a double in a vector register, so the translator's runtime calls these
// exports with converted values (tools/translator/runtime/Runtime.cpp, CallX87). Each signature has
// a letter for the result and then one per argument: L long double, P long double*, p const long
// double*, C complex long double, d double, f float, I 128-bit integer, i other integer or pointer,
// v nothing.
namespace {

struct X87Export {
    const void* function;
    const char* signature;
};

template<class T> constexpr char Letter() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_void_v<U>) return 'v';
    else if constexpr (std::is_same_v<U, long double>) return 'L';
    else if constexpr (std::is_same_v<U, long double*>) return 'P';
    else if constexpr (std::is_same_v<U, const long double*>) return 'p';
    else if constexpr (std::is_same_v<U, __complex__ long double>) return 'C';
    else if constexpr (std::is_same_v<U, double>) return 'd';
    else if constexpr (std::is_same_v<U, float>) return 'f';
    else if constexpr (std::is_same_v<U, __int128> || std::is_same_v<U, unsigned __int128>) return 'I';
    else {
        static_assert(std::is_integral_v<U> || std::is_pointer_v<U> || std::is_enum_v<U>, "an x87 export takes an argument the runtime cannot convert");
        return 'i';
    }
}

template<class F> struct Signature;
template<class R, class... A> struct Signature<R (*)(A...)> {
    static constexpr char text[] = {Letter<R>(), Letter<A>()..., 0};
};
template<class R, class... A> struct Signature<R (*)(A...) noexcept> : Signature<R (*)(A...)> {};

template<class F> constexpr X87Export Export(F function) {
    return {reinterpret_cast<const void*>(function), Signature<F>::text};
}

const X87Export Exports[] = {
    Export(&_LCosh_nid_postfix),
    Export(&_LDclass_nid_postfix),
    Export(&_LDivide_nid_postfix),
    Export(&_LDsign_nid_postfix),
    Export(&_LDtest_nid_postfix),
    Export(&_LLog_nid_postfix),
    Export(&_LRecip_nid_postfix),
    Export(&_LSin_nid_postfix),
    Export(&_LSinh_nid_postfix),
    Export(&_Stold_nid_postfix),
    Export(&_Stoldx_nid_postfix),
    Export(&__divxc3_nid_postfix),
    Export(&__fixunsxfdi_nid_postfix),
    Export(&__fixunsxfsi_nid_postfix),
    Export(&__fixunsxfti_nid_postfix),
    Export(&__fixxfdi_nid_postfix),
    Export(&__fixxfti_nid_postfix),
    Export(&__floatdixf_nid_postfix),
    Export(&__floattixf_nid_postfix),
    Export(&__floatundixf_nid_postfix),
    Export(&__floatuntixf_nid_postfix),
    Export(&__fpclassifyl_nid_postfix),
    Export(&__isfinitel_nid_postfix),
    Export(&__isinfl_nid_postfix),
    Export(&__isnanl_nid_postfix),
    Export(&__isnormall_nid_postfix),
    Export(&__mulxc3_nid_postfix),
    Export(&__powixf2_nid_postfix),
    Export(&__signbitl_nid_postfix),
    Export(&acoshl_nid_postfix),
    Export(&acosl_nid_postfix),
    Export(&asinhl_nid_postfix),
    Export(&asinl_nid_postfix),
    Export(&atan2l_nid_postfix),
    Export(&atanhl_nid_postfix),
    Export(&atanl_nid_postfix),
    Export(&cbrtl_nid_postfix),
    Export(&ceill_nid_postfix),
    Export(&copysignl_nid_postfix),
    Export(&coshl_nid_postfix),
    Export(&cosl_nid_postfix),
    Export(&erfcl_nid_postfix),
    Export(&erfl_nid_postfix),
    Export(&exp2l_nid_postfix),
    Export(&expl_nid_postfix),
    Export(&expm1l_nid_postfix),
    Export(&fabsl_nid_postfix),
    Export(&fdiml_nid_postfix),
    Export(&floorl_nid_postfix),
    Export(&fmal_nid_postfix),
    Export(&fmaxl_nid_postfix),
    Export(&fminl_nid_postfix),
    Export(&fmodl_nid_postfix),
    Export(&frexpl_nid_postfix),
    Export(&hypot3l_nid_postfix),
    Export(&hypotl_nid_postfix),
    Export(&ilogbl_nid_postfix),
    Export(&ldexpl_nid_postfix),
    Export(&lgammal_nid_postfix),
    Export(&llrintl_nid_postfix),
    Export(&llroundl_nid_postfix),
    Export(&log10l_nid_postfix),
    Export(&log1pl_nid_postfix),
    Export(&log2l_nid_postfix),
    Export(&logbl_nid_postfix),
    Export(&logl_nid_postfix),
    Export(&lrintl_nid_postfix),
    Export(&lroundl_nid_postfix),
    Export(&modfl_nid_postfix),
    Export(&nanl_nid_postfix),
    Export(&nearbyintl_nid_postfix),
    Export(&powl_nid_postfix),
    Export(&remainderl_nid_postfix),
    Export(&remquol_nid_postfix),
    Export(&rintl_nid_postfix),
    Export(&roundl_nid_postfix),
    Export(&scalblnl_nid_postfix),
    Export(&scalbnl_nid_postfix),
    Export(&sinhl_nid_postfix),
    Export(&sinl_nid_postfix),
    Export(&sqrtl_nid_postfix),
    Export(&strtold_nid_postfix),
    Export(&tanhl_nid_postfix),
    Export(&tanl_nid_postfix),
    Export(&tgammal_nid_postfix),
    Export(&truncl_nid_postfix),
    Export(&wcstold_nid_postfix),
};

}

extern "C" void Aps5X87Exports_nid_no_patch(const void** table, std::size_t* count) {
    *table = Exports;
    *count = sizeof(Exports) / sizeof(Exports[0]);
}

#endif
