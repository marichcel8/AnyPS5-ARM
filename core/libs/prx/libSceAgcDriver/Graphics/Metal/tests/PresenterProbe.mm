#include "prx/libSceAgcDriver/Graphics/Metal/Presenter.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::Graphics::Metal;
using Clock = std::chrono::steady_clock;

void Check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

struct Pixel {
    std::uint8_t b, g, r, a;
};

id<MTLTexture> MakeTexture(Device& device, MTLPixelFormat format, std::uint32_t width, std::uint32_t height, MTLTextureUsage usage) {
    id<MTLDevice> mtl = (__bridge id<MTLDevice>)device.NativeDevice();
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = usage;
    id<MTLTexture> texture = [mtl newTextureWithDescriptor:descriptor];
    Check(texture != nil, "newTextureWithDescriptor");
    id<MTLResidencySet> set = (__bridge id<MTLResidencySet>)device.NativeResidencySet();
    [set addAllocation:texture];
    [set commit];
    [set requestResidency];
    return texture;
}

id<MTLTexture> SolidSource(Device& device, MTLPixelFormat format, std::uint32_t width, std::uint32_t height, const void* texel, std::size_t texelBytes) {
    id<MTLTexture> texture = MakeTexture(device, format, width, height, MTLTextureUsageShaderRead);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(width) * height * texelBytes);
    for (std::size_t index = 0; index < static_cast<std::size_t>(width) * height; ++index) std::memcpy(bytes.data() + index * texelBytes, texel, texelBytes);
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 withBytes:bytes.data() bytesPerRow:static_cast<NSUInteger>(width) * texelBytes];
    return texture;
}

Pixel ReadPixel(id<MTLTexture> texture, std::uint32_t x, std::uint32_t y) {
    Pixel pixel{};
    [texture getBytes:&pixel bytesPerRow:sizeof(pixel) fromRegion:MTLRegionMake2D(x, y, 1, 1) mipmapLevel:0];
    return pixel;
}

PresentSource Describe(id<MTLTexture> texture, SourceEncoding encoding) {
    PresentSource source;
    source.texture = (__bridge void*)texture;
    source.width = static_cast<std::uint32_t>(texture.width);
    source.height = static_cast<std::uint32_t>(texture.height);
    source.encoding = encoding;
    return source;
}

std::uint32_t PackPq(double nits) {
    const double m1 = 0.1593017578125;
    const double m2 = 78.84375;
    const double c1 = 0.8359375;
    const double c2 = 18.8515625;
    const double c3 = 18.6875;
    const double y = std::pow(nits / 10000.0, m1);
    const double e = std::pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
    const auto v = static_cast<std::uint32_t>(std::lround(e * 1023.0));
    return v | (v << 10) | (v << 20) | (3u << 30);
}

void CheckColorPaths(Device& device, std::string* failure) {
    auto presenter = Presenter::Create(device, nullptr, PresenterOptions{}, failure);
    Check(presenter != nullptr, "Presenter::Create without a layer");
    constexpr MTLTextureUsage TargetUsage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    const Pixel red{0, 0, 255, 255};
    id<MTLTexture> redSource = SolidSource(device, MTLPixelFormatBGRA8Unorm, 64, 36, &red, sizeof(red));
    id<MTLTexture> target = MakeTexture(device, MTLPixelFormatBGRA8Unorm, 64, 36, TargetUsage);
    Check(presenter->RenderTo(Describe(redSource, SourceEncoding::Unorm), (__bridge void*)target, false), "RenderTo same size");
    Pixel center = ReadPixel(target, 32, 18);
    Check(center.r == 255 && center.g == 0 && center.b == 0 && center.a == 255, "an unorm source passes through unchanged");

    id<MTLTexture> tall = MakeTexture(device, MTLPixelFormatBGRA8Unorm, 64, 100, TargetUsage);
    Check(presenter->RenderTo(Describe(redSource, SourceEncoding::Unorm), (__bridge void*)tall, false), "RenderTo letterbox");
    Pixel bar = ReadPixel(tall, 32, 2);
    Pixel body = ReadPixel(tall, 32, 50);
    Check(bar.r == 0 && bar.g == 0 && bar.b == 0, "the letterbox bar is black");
    Check(body.r == 255, "the picture sits in the middle");

    std::uint32_t pq[3] = {PackPq(0.0), PackPq(100.0), PackPq(1000.0)};
    std::uint8_t level[3] = {};
    for (int index = 0; index < 3; ++index) {
        id<MTLTexture> source = SolidSource(device, MTLPixelFormatRGB10A2Unorm, 64, 36, &pq[index], sizeof(std::uint32_t));
        Check(presenter->RenderTo(Describe(source, SourceEncoding::Pq), (__bridge void*)target, false), "RenderTo pq");
        level[index] = ReadPixel(target, 10, 10).g;
    }
    std::printf("pq 0/100/1000 nits -> %u/%u/%u\n", level[0], level[1], level[2]);
    Check(level[0] <= 1 && level[0] < level[1] && level[1] < level[2], "a PQ ramp is monotonic");

    const std::uint16_t half[4] = {0x3800, 0x3800, 0x3800, 0x3C00};
    id<MTLTexture> linear = SolidSource(device, MTLPixelFormatRGBA16Float, 64, 36, half, sizeof(half));
    Check(presenter->RenderTo(Describe(linear, SourceEncoding::LinearHalf), (__bridge void*)target, false), "RenderTo linear half");
    const Pixel mid = ReadPixel(target, 10, 10);
    std::printf("linear 0.5 -> %u (sRGB 188 expected)\n", mid.g);
    Check(mid.g >= 186 && mid.g <= 190, "linear 0.5 encodes to sRGB 188");
}

void CheckUpscale(Device& device, std::string* failure) {
    PresenterOptions options;
    options.upscale = UpscaleMode::On;
    auto presenter = Presenter::Create(device, nullptr, options, failure);
    Check(presenter != nullptr, "Presenter::Create for the upscaler");
    const Pixel green{0, 200, 0, 255};
    id<MTLTexture> source = SolidSource(device, MTLPixelFormatBGRA8Unorm, 640, 360, &green, sizeof(green));
    id<MTLTexture> target = MakeTexture(device, MTLPixelFormatBGRA8Unorm, 1280, 720, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
    Check(presenter->RenderTo(Describe(source, SourceEncoding::Unorm), (__bridge void*)target, true), "RenderTo upscaled");
    const auto stats = presenter->Stats();
    const Pixel center = ReadPixel(target, 640, 360);
    std::printf("metalfx engaged=%llu rebuilds=%llu center g=%u\n", static_cast<unsigned long long>(stats.upscaled), static_cast<unsigned long long>(stats.scalerRebuilds), center.g);
    Check(std::abs(static_cast<int>(center.g) - 200) <= 3 && center.r <= 3 && center.b <= 3, "a solid colour survives the upscaler");
}

double MeasurePacing(Device& device, CAMetalLayer* layer, double fps, int frames, std::string* failure) {
    PresenterOptions options;
    options.targetFps = fps;
    auto presenter = Presenter::Create(device, (__bridge void*)layer, options, failure);
    Check(presenter != nullptr, "Presenter::Create with a layer");
    presenter->SetDrawableSize(1280, 720);
    presenter->SetRefreshRate(60.0);
    const Pixel blue{255, 0, 0, 255};
    id<MTLTexture> source = SolidSource(device, MTLPixelFormatBGRA8Unorm, 640, 360, &blue, sizeof(blue));
    const auto description = Describe(source, SourceEncoding::Unorm);
    std::vector<double> intervals;
    Clock::time_point previous{};
    const auto footprint = PhysFootprintBytes();
    int presented = 0;
    for (int frame = 0; frame < frames; ++frame) {
        const auto result = presenter->Present(description);
        if (result.outcome != PresentOutcome::Presented) continue;
        ++presented;
        const auto now = Clock::now();
        if (previous != Clock::time_point{}) intervals.push_back(std::chrono::duration<double, std::milli>(now - previous).count());
        previous = now;
    }
    device.WaitIdle(5000);
    Check(presented > frames / 2, "most frames presented");
    std::sort(intervals.begin(), intervals.end());
    const double median = intervals[intervals.size() / 2];
    const auto stats = presenter->Stats();
    std::printf("target %.0f fps: median %.2f ms, p95 %.2f ms, effective %.1f Hz, paced %.1f ms total, encode %.3f ms/frame, footprint +%.1f MiB\n",
        fps, median, intervals[intervals.size() * 95 / 100], stats.effectiveHz, stats.pacedMs, stats.encodeMs / std::max<std::uint64_t>(1, stats.presented),
        (static_cast<double>(PhysFootprintBytes()) - static_cast<double>(footprint)) / 1048576.0);
    return median;
}

}

int main() {
    @autoreleasepool {
        std::string failure;
        auto device = Device::Create(&failure);
        if (device == nullptr) {
            std::fprintf(stderr, "no Metal 4 device: %s\n", failure.c_str());
            return 1;
        }
        CheckColorPaths(*device, &failure);
        CheckUpscale(*device, &failure);
        CAMetalLayer* layer = [CAMetalLayer layer];
        const double thirty = MeasurePacing(*device, layer, 30.0, 150, &failure);
        Check(thirty > 30.0 && thirty < 37.0, "30 fps mode paces to about 33.3 ms");
        MeasurePacing(*device, layer, 0.0, 150, &failure);
        std::puts("presenter ok");
    }
    return 0;
}
