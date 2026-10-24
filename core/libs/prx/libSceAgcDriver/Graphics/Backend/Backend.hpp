#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_BACKEND_BACKEND_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_BACKEND_BACKEND_HPP

namespace AgcDriver::Backend {

enum class Kind {
    Vulkan,
    Metal,
};

#if defined(ANYPS5_GPU_BACKEND_METAL)
inline constexpr Kind kKind = Kind::Metal;
inline constexpr const char* kName = "Metal";
inline constexpr bool kHostsNullDevice = true;
#elif defined(ANYPS5_GPU_BACKEND_VULKAN)
inline constexpr Kind kKind = Kind::Vulkan;
inline constexpr const char* kName = "Vulkan";
inline constexpr bool kHostsNullDevice = false;
#else
#error "ANYPS5_GPU_BACKEND_VULKAN or ANYPS5_GPU_BACKEND_METAL must be defined"
#endif

}

#endif
