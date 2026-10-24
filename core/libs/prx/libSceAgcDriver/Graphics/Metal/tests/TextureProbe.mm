#include "prx/libSceAgcDriver/Graphics/Metal/Textures.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureJobs.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;
using namespace AgcDriver::Graphics::Metal;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t MiB = 1ull << 20u;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

double Microseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

struct Context {
    Device& device;
    TextureManager& textures;
    id<MTLDevice> native;
};

template<typename Body>
double Run(Context& context, Body body) {
    @autoreleasepool {
        id<MTL4CommandAllocator> allocator = [context.native newCommandAllocator];
        id<MTL4CommandBuffer> commands = [context.native newCommandBuffer];
        [commands beginCommandBufferWithAllocator:allocator];
        id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        const auto begin = Clock::now();
        body(encoder);
        [encoder endEncoding];
        [commands endCommandBuffer];
        void* list[] = {(__bridge void*)commands};
        const auto serial = context.device.Submit(QueueKind::Compute, list, 1);
        context.textures.Seal(QueueKind::Compute, serial);
        Check(context.device.WaitSerial(QueueKind::Compute, serial, 60000), "the GPU finished in time");
        return Microseconds(begin);
    }
}

uint4 Pattern(std::uint32_t seed, std::uint32_t x, std::uint32_t y) {
    auto mix = [](std::uint32_t value) {
        value ^= value >> 16;
        value *= 0x7feb352dU;
        value ^= value >> 15;
        value *= 0x846ca68bU;
        value ^= value >> 16;
        return value;
    };
    const auto base = mix(seed * 0x9e3779b9U + x * 0x85ebca6bU + y * 0xc2b2ae35U);
    return {mix(base), mix(base + 1), mix(base + 2), mix(base + 3)};
}

struct Case {
    const char* name;
    TextureTileMode mode;
    std::uint32_t format;
    TextureDimension dimension;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t depth;
    std::uint32_t mips;
};

GuestTextureResource Describe(const Case& item) {
    GuestTextureResource resource{};
    resource.width = item.width;
    resource.height = item.height;
    resource.depthOrLastArray = item.depth - 1u;
    resource.mipCount = item.mips;
    resource.tileMode = item.mode;
    resource.dimension = item.dimension;
    resource.format = item.format;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    return resource;
}

std::uint32_t Seed(const MipJob& job, std::uint32_t format) {
    return job.level * 131u + job.layer * 17u + format;
}

std::vector<std::uint8_t> Expected(const FormatInfo& format, std::uint32_t seed, std::uint32_t x, std::uint32_t y) {
    auto value = Pattern(seed, x, y);
    const auto mask = format.guestElementBytes >= 4 ? 0xffffffffu : (1u << (format.guestElementBytes * 8)) - 1u;
    value.x &= mask;
    if (format.guestElementBytes < 8) value.y = 0;
    if (format.guestElementBytes < 16) value.z = value.w = 0;
    const auto texel = agc_transform(value, static_cast<std::uint32_t>(format.transform));
    std::vector<std::uint8_t> bytes(format.texelBytes);
    const std::uint32_t words[4] = {texel.x, texel.y, texel.z, texel.w};
    std::memcpy(bytes.data(), words, format.texelBytes);
    return bytes;
}

std::vector<std::uint8_t> GuestBytes(const FormatInfo& format, std::uint32_t seed, std::uint32_t x, std::uint32_t y) {
    auto value = Pattern(seed, x, y);
    const auto mask = format.guestElementBytes >= 4 ? 0xffffffffu : (1u << (format.guestElementBytes * 8)) - 1u;
    value.x &= mask;
    std::vector<std::uint8_t> bytes(format.guestElementBytes);
    const std::uint32_t words[4] = {value.x, value.y, value.z, value.w};
    std::memcpy(bytes.data(), words, format.guestElementBytes);
    return bytes;
}

struct Guest {
    void* memory = nullptr;
    std::uint64_t mapped = 0;
    std::uint64_t device = 0;
};

Guest MapGuest(Context& context, std::uint64_t guestBytes) {
    Guest guest;
    const auto page = context.device.Limits().hostPageBytes;
    guest.mapped = AlignUp(guestBytes, page) + page;
    guest.memory = mmap(nullptr, guest.mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    Check(guest.memory != MAP_FAILED, "mmap the guest surface");
    const auto address = reinterpret_cast<std::uint64_t>(guest.memory);
    Check(context.device.Guest().Map({address, guest.mapped, true}), "wrap the guest surface");
    const auto resolved = Resolve(*context.device.Guest().Snapshot(), address, guest.mapped);
    Check(static_cast<bool>(resolved) && !resolved.window->copied, "the guest surface resolves without a copy");
    guest.device = resolved.deviceAddress;
    return guest;
}

void UnmapGuest(Context& context, Guest& guest) {
    context.device.Guest().Unmap(reinterpret_cast<std::uint64_t>(guest.memory), guest.mapped);
    context.device.WaitIdle(10000);
    munmap(guest.memory, guest.mapped);
    guest = {};
}

void FillGuest(const GuestTextureResource& resource, const FormatInfo& format, const Guest& guest) {
    const auto hostPlan = PlanSurface(resource, reinterpret_cast<std::uint64_t>(guest.memory));
    std::set<std::uint64_t> used;
    for (const auto& job : hostPlan.jobs) {
        for (std::uint32_t y = 0; y < job.rows; ++y) {
            for (std::uint32_t x = 0; x < job.columns; ++x) {
                const auto address = job.params.tiled + agc_tiled_offset(job.params, x, y, format.guestElementBytes);
                Check(used.insert(address).second, "two probe elements share a guest address");
                const auto bytes = GuestBytes(format, Seed(job, resource.format), x, y);
                std::memcpy(reinterpret_cast<void*>(address), bytes.data(), bytes.size());
            }
        }
    }
}

struct Readback {
    NativeBuffer buffer;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint32_t> pitches;
};

Readback MakeReadback(Context& context, const SurfacePlan& plan, const FormatInfo& format) {
    Readback readback;
    std::uint64_t total = 0;
    for (const auto& job : plan.jobs) {
        const auto pitch = static_cast<std::uint32_t>(AlignUp(static_cast<std::uint64_t>(job.columns) * format.texelBytes, 16));
        readback.pitches.push_back(pitch);
        readback.offsets.push_back(total);
        total = AlignUp(total + static_cast<std::uint64_t>(pitch) * job.rows, 256);
    }
    readback.buffer = context.device.NewSharedBuffer(total, "probe readback");
    Check(static_cast<bool>(readback.buffer), "allocate the readback buffer");
    std::memset(readback.buffer.contents, 0xcd, total);
    return readback;
}

void EncodeReadback(id<MTL4ComputeCommandEncoder> encoder, const TextureHandle& texture, const TextureCreateInfo& info, const SurfacePlan& plan, const FormatInfo& format, const Readback& readback) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)readback.buffer.handle;
    id<MTLTexture> native = (__bridge id<MTLTexture>)texture.texture;
    const bool volume = info.kind == TextureKind::Tex3D;
    const auto block = format.blockCompressed ? 4u : 1u;
    for (std::size_t index = 0; index < plan.jobs.size(); ++index) {
        const auto& job = plan.jobs[index];
        const auto levelWidth = std::max(info.width >> job.level, 1u);
        const auto levelHeight = std::max(info.height >> job.level, 1u);
        const auto size = MTLSizeMake(std::min(job.columns * block, levelWidth), std::min(job.rows * block, levelHeight), 1);
        const auto origin = volume ? MTLOriginMake(0, 0, job.destinationDepth) : MTLOriginMake(0, 0, 0);
        [encoder copyFromTexture:native sourceSlice:volume ? 0 : job.destinationLayer sourceLevel:job.level sourceOrigin:origin sourceSize:size toBuffer:buffer destinationOffset:readback.offsets[index] destinationBytesPerRow:readback.pitches[index] destinationBytesPerImage:static_cast<NSUInteger>(readback.pitches[index]) * job.rows];
    }
}

void VerifyReadback(const SurfacePlan& plan, const FormatInfo& format, const GuestTextureResource& resource, const Readback& readback, const std::string& what) {
    const auto* base = static_cast<const std::uint8_t*>(readback.buffer.contents);
    std::size_t checked = 0;
    for (std::size_t index = 0; index < plan.jobs.size(); ++index) {
        const auto& job = plan.jobs[index];
        for (std::uint32_t y = 0; y < job.rows; ++y) {
            for (std::uint32_t x = 0; x < job.columns; ++x) {
                const auto expected = Expected(format, Seed(job, resource.format), x, y);
                const auto* actual = base + readback.offsets[index] + static_cast<std::uint64_t>(y) * readback.pitches[index] + static_cast<std::uint64_t>(x) * format.texelBytes;
                if (std::memcmp(expected.data(), actual, expected.size()) != 0) {
                    std::fprintf(stderr, "FAIL: %s: level %u layer %u element (%u,%u) differs\n", what.c_str(), job.level, job.layer, x, y);
                    std::exit(1);
                }
                ++checked;
            }
        }
    }
    (void)checked;
}

double RoundTrip(Context& context, const Case& item, UploadPath path, bool verifyWriteback) {
    const auto resource = Describe(item);
    const auto* format = FindFormat(resource.format);
    Check(format != nullptr, "known format");
    if (!context.textures.SupportsFormat(resource.format)) {
        std::printf("skip %s: format unsupported on this GPU\n", item.name);
        return 0;
    }
    const auto info = DescribeTexture(resource);
    std::string failure;
    auto texture = context.textures.Create(info, &failure);
    Check(static_cast<bool>(texture), std::string("create texture ") + item.name + ": " + failure);
    const auto planShape = PlanSurface(resource, 0);
    auto guest = MapGuest(context, planShape.geometry.guestBytes);
    FillGuest(resource, *format, guest);
    const auto plan = PlanSurface(resource, guest.device);
    auto readback = MakeReadback(context, plan, *format);
    const auto uploadUs = Run(context, [&](id<MTL4ComputeCommandEncoder> encoder) {
        context.textures.EncodeUpload((__bridge void*)encoder, texture, resource, guest.device, path);
        [encoder barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        EncodeReadback(encoder, texture, info, plan, *format, readback);
    });
    VerifyReadback(plan, *format, resource, readback, std::string(item.name) + (path == UploadPath::Direct ? " direct" : " staged"));
    if (verifyWriteback && format->transform == TextureTransform::None && !format->blockCompressed && info.kind != TextureKind::Tex1D && info.kind != TextureKind::Tex1DArray) {
        auto second = MapGuest(context, planShape.geometry.guestBytes);
        Run(context, [&](id<MTL4ComputeCommandEncoder> encoder) {
            context.textures.EncodeWriteback((__bridge void*)encoder, texture, resource, second.device);
        });
        const auto hostPlan = PlanSurface(resource, reinterpret_cast<std::uint64_t>(second.memory));
        for (const auto& job : hostPlan.jobs) {
            for (std::uint32_t y = 0; y < job.rows; ++y) {
                for (std::uint32_t x = 0; x < job.columns; ++x) {
                    const auto address = job.params.tiled + agc_tiled_offset(job.params, x, y, format->guestElementBytes);
                    const auto expected = GuestBytes(*format, Seed(job, resource.format), x, y);
                    if (std::memcmp(reinterpret_cast<const void*>(address), expected.data(), expected.size()) != 0) {
                        std::fprintf(stderr, "FAIL: %s writeback: level %u layer %u element (%u,%u) differs\n", item.name, job.level, job.layer, x, y);
                        std::exit(1);
                    }
                }
            }
        }
        UnmapGuest(context, second);
    }
    context.device.ReleaseBuffer(readback.buffer);
    UnmapGuest(context, guest);
    context.textures.Release(texture);
    context.device.WaitIdle(10000);
    return uploadUs;
}

const char* const SampleSource = R"(
#include <metal_stdlib>
using namespace metal;
struct Args {
    device float4* out;
    uint width;
};
kernel void sample_texel(constant Args& args [[buffer(0)]], texture2d<float, access::read> source [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width) return;
    args.out[gid.y * args.width + gid.x] = source.read(gid);
}
)";

struct SampleArgs {
    std::uint64_t out;
    std::uint32_t width;
    std::uint32_t pad;
};

void SwizzleThroughView(Context& context) {
    Case item{"swizzle rgba8", TextureTileMode::kD64KBX, 56, TextureDimension::k2D, 16, 8, 1, 1};
    const auto resource = Describe(item);
    const auto* format = FindFormat(56);
    const auto info = DescribeTexture(resource);
    std::string failure;
    auto texture = context.textures.Create(info, &failure);
    Check(static_cast<bool>(texture), "create the swizzle texture: " + failure);
    const auto shape = PlanSurface(resource, 0);
    auto guest = MapGuest(context, shape.geometry.guestBytes);
    FillGuest(resource, *format, guest);
    const auto plan = PlanSurface(resource, guest.device);
    Run(context, [&](id<MTL4ComputeCommandEncoder> encoder) {
        context.textures.EncodeUpload((__bridge void*)encoder, texture, resource, guest.device, UploadPath::Direct);
    });
    ViewInfo view;
    view.swizzle = ViewSwizzle(56, {7, 6, 5, 4});
    const auto viewHandle = context.textures.View(texture, view);
    Check(static_cast<bool>(viewHandle) && viewHandle.resourceId != 0 && viewHandle.resourceId != texture.resourceId, "a swizzled view has its own resource id");
    NSError* error = nil;
    id<MTLLibrary> library = [context.native newLibraryWithSource:[NSString stringWithUTF8String:SampleSource] options:nil error:&error];
    Check(library != nil, "compile the sample kernel");
    id<MTLComputePipelineState> pipeline = [context.native newComputePipelineStateWithFunction:[library newFunctionWithName:@"sample_texel"] error:&error];
    Check(pipeline != nil, "build the sample pipeline");
    MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = 1;
    descriptor.maxTextureBindCount = 1;
    id<MTL4ArgumentTable> table = [context.native newArgumentTableWithDescriptor:descriptor error:&error];
    Check(table != nil, "create the sample argument table");
    auto out = context.device.NewSharedBuffer(16 * 8 * sizeof(float) * 4, "probe samples");
    auto args = context.device.NewSharedBuffer(sizeof(SampleArgs), "probe sample args");
    *static_cast<SampleArgs*>(args.contents) = {out.gpuAddress, 16, 0};
    Run(context, [&](id<MTL4ComputeCommandEncoder> encoder) {
        [encoder setComputePipelineState:pipeline];
        [table setAddress:args.gpuAddress atIndex:0];
        id<MTLTexture> native = (__bridge id<MTLTexture>)viewHandle.texture;
        [table setTexture:native.gpuResourceID atIndex:0];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(16, 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    });
    const auto* samples = static_cast<const float*>(out.contents);
    for (std::uint32_t y = 0; y < 8; ++y) {
        for (std::uint32_t x = 0; x < 16; ++x) {
            const auto expected = GuestBytes(*format, Seed(plan.jobs[0], 56), x, y);
            const float want[4] = {expected[3] / 255.0f, expected[2] / 255.0f, expected[1] / 255.0f, expected[0] / 255.0f};
            for (int channel = 0; channel < 4; ++channel) {
                const float got = samples[(y * 16 + x) * 4 + channel];
                Check(std::abs(got - want[channel]) < 1.0f / 255.0f, "swizzled sample matches the guest texel");
            }
        }
    }
    context.device.ReleaseBuffer(out);
    context.device.ReleaseBuffer(args);
    UnmapGuest(context, guest);
    auto base = texture;
    context.textures.Release(base);
    context.device.WaitIdle(10000);
    std::puts("swizzle through view ok");
}

void Performance(Context& context) {
    Case item{"perf rgba8 4096", TextureTileMode::kD64KBX, 56, TextureDimension::k2D, 4096, 4096, 1, 1};
    const auto resource = Describe(item);
    const auto* format = FindFormat(56);
    const auto info = DescribeTexture(resource);
    std::string failure;
    auto texture = context.textures.Create(info, &failure);
    Check(static_cast<bool>(texture), "create the perf texture: " + failure);
    const auto shape = PlanSurface(resource, 0);
    auto guest = MapGuest(context, shape.geometry.guestBytes);
    std::memset(guest.memory, 0x5a, shape.geometry.guestBytes);
    (void)format;
    const auto before = static_cast<std::int64_t>(PhysFootprintBytes());
    for (const auto path : {UploadPath::Direct, UploadPath::Staged}) {
        double best = 1e30;
        for (int run = 0; run < 5; ++run) {
            const auto us = Run(context, [&](id<MTL4ComputeCommandEncoder> encoder) {
                context.textures.EncodeUpload((__bridge void*)encoder, texture, resource, guest.device, path);
            });
            best = std::min(best, us);
        }
        const auto after = static_cast<std::int64_t>(PhysFootprintBytes());
        std::printf("perf %s mib=%.1f best_us=%.0f gib_per_s=%.2f footprint_delta_mib=%.1f\n", path == UploadPath::Direct ? "direct" : "staged", static_cast<double>(shape.geometry.guestBytes) / MiB, best, static_cast<double>(shape.geometry.guestBytes) / (best * 1e-6) / (1024.0 * MiB), static_cast<double>(after - before) / MiB);
    }
    UnmapGuest(context, guest);
    context.textures.Release(texture);
    context.device.WaitIdle(10000);
}

}

int main() {
    @autoreleasepool {
        std::string failure;
        auto device = Device::Create(&failure);
        if (device == nullptr) {
            std::fprintf(stderr, "FAIL: %s\n", failure.c_str());
            return 1;
        }
        auto textures = TextureManager::Create(*device, &failure);
        if (textures == nullptr) {
            std::fprintf(stderr, "FAIL: texture manager: %s\n", failure.c_str());
            return 1;
        }
        Context context{*device, *textures, (__bridge id<MTLDevice>)device->NativeDevice()};
        std::printf("device name=\"%s\" bc=%d default_upload=%s\n", device->Limits().name.c_str(), textures->SupportsFormat(173) ? 1 : 0, textures->DefaultUploadPath() == UploadPath::Direct ? "direct" : "staged");
        const std::vector<Case> cases = {
            {"r8 linear", TextureTileMode::kLinear, 1, TextureDimension::k2D, 70, 33, 1, 4},
            {"r8 std256", TextureTileMode::kStandard256B, 1, TextureDimension::k2D, 64, 64, 1, 3},
            {"rg8 std4k", TextureTileMode::kStandard4KB, 14, TextureDimension::k2D, 100, 60, 1, 5},
            {"rgba8 std64k", TextureTileMode::kStandard64KB, 56, TextureDimension::k2D, 200, 150, 1, 6},
            {"rgba8 d4k", TextureTileMode::kD4KB, 56, TextureDimension::k2D, 100, 60, 1, 7},
            {"rgba8 d64kx", TextureTileMode::kD64KBX, 56, TextureDimension::k2D, 257, 129, 1, 8},
            {"rgba8 r64kx array", TextureTileMode::kR64KBX, 56, TextureDimension::k2DArray, 96, 80, 3, 4},
            {"rgba8 srgb s4kx", TextureTileMode::kS4KBX, 130, TextureDimension::k2D, 33, 20, 1, 4},
            {"r16f d64kt", TextureTileMode::kD64KBT, 13, TextureDimension::k2D, 64, 40, 1, 5},
            {"rgba16f d4kx", TextureTileMode::kD4KBX, 71, TextureDimension::k2D, 33, 20, 1, 4},
            {"rgba32f s64kx", TextureTileMode::kS64KBX, 77, TextureDimension::k2D, 64, 48, 1, 5},
            {"cube d64kx", TextureTileMode::kD64KBX, 56, TextureDimension::kCube, 32, 32, 6, 4},
            {"1d array", TextureTileMode::kLinear, 56, TextureDimension::k1DArray, 100, 1, 3, 3},
            {"r11g11b10 repack", TextureTileMode::kD64KBX, 43, TextureDimension::k2D, 64, 64, 1, 3},
            {"a2b10g10r10 snorm", TextureTileMode::kD64KBX, 51, TextureDimension::k2D, 64, 48, 1, 3},
            {"a2b10g10r10 sint", TextureTileMode::kD64KBX, 55, TextureDimension::k2D, 64, 48, 1, 3},
            {"rgb565", TextureTileMode::kD64KBX, 133, TextureDimension::k2D, 48, 40, 1, 3},
            {"volume s64kx r8", TextureTileMode::kS64KBX, 1, TextureDimension::k3D, 33, 20, 12, 4},
            {"volume s4kx rgba8", TextureTileMode::kS4KBX, 56, TextureDimension::k3D, 33, 20, 12, 4},
            {"volume z64kx rgba8", TextureTileMode::kZ64KBX, 56, TextureDimension::k3D, 40, 24, 20, 3},
            {"bc1", TextureTileMode::kD64KBX, 169, TextureDimension::k2D, 100, 60, 1, 5},
            {"bc3 array", TextureTileMode::kD64KBX, 173, TextureDimension::k2DArray, 64, 64, 2, 4},
            {"bc7", TextureTileMode::kS64KBX, 181, TextureDimension::k2D, 130, 66, 1, 5},
        };
        for (const auto& item : cases) {
            const auto direct = RoundTrip(context, item, UploadPath::Direct, true);
            const auto staged = RoundTrip(context, item, UploadPath::Staged, false);
            std::printf("case %s ok direct_us=%.0f staged_us=%.0f\n", item.name, direct, staged);
        }
        SwizzleThroughView(context);
        Performance(context);
        const auto stats = textures->Stats();
        std::printf("stats textures=%llu views=%llu bytes=%llu dispatches=%llu pipelines=%llu arena_chunks=%llu uploads=%llu writebacks=%llu\n", static_cast<unsigned long long>(stats.textures), static_cast<unsigned long long>(stats.views), static_cast<unsigned long long>(stats.bytes), static_cast<unsigned long long>(stats.detile.dispatches), static_cast<unsigned long long>(stats.detile.pipelines), static_cast<unsigned long long>(stats.detile.arenaChunks), static_cast<unsigned long long>(stats.uploads), static_cast<unsigned long long>(stats.writebacks));
        Check(stats.textures == 0 && stats.bytes == 0, "every probe texture was released");
        textures.reset();
        device->WaitIdle(10000);
        std::puts("metal texture probe: ok");
    }
    return 0;
}
