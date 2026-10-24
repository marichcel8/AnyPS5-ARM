#include "prx/libSceAgcDriver/Graphics/Metal/TextureAddress.h"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/Metal/TextureJobs.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;
namespace M = AgcDriver::Graphics::Metal;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

struct Element {
    std::uint32_t mip;
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint64_t address;
    std::uint32_t pipeBankXor = 0;
};

struct Case {
    const char* name;
    TextureTileMode mode;
    std::uint32_t format;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t depth;
    std::uint32_t mips;
    bool volume;
    bool thick;
    std::vector<Element> elements;
};

const std::vector<Case>& cases() {
    static const std::vector<Case> list = {
    {"s4xTail1", TextureTileMode::kS4KBX, 1, 33, 20, 12, 6, true, true, std::vector<Element>{{0, 0, 0, 0, 0x3000, 0x0}, {0, 32, 19, 11, 0x843c, 0x0}, {0, 8, 9, 8, 0x3e08, 0x0}, {0, 15, 2, 8, 0x3163, 0x700}, {1, 0, 0, 0, 0x1000, 0x0}, {1, 15, 9, 5, 0x1acf, 0x0}, {1, 12, 4, 0, 0x1340, 0x0}, {1, 7, 2, 5, 0x17e7, 0x700}, {2, 0, 0, 0, 0x800, 0x0}, {2, 7, 4, 2, 0x953, 0x0}, {2, 2, 2, 0, 0x822, 0x0}, {2, 0, 1, 1, 0xf0c, 0x700}, {3, 0, 0, 0, 0x300, 0x0}, {3, 3, 1, 0, 0x30b, 0x0}, {3, 3, 1, 0, 0x30b, 0x0}, {3, 0, 0, 0, 0x400, 0x700}, {4, 0, 0, 0, 0x200, 0x0}, {4, 1, 0, 0, 0x201, 0x0}, {4, 0, 0, 0, 0x200, 0x0}, {4, 0, 0, 0, 0x500, 0x700}, {5, 0, 0, 0, 0x100, 0x0}, {5, 0, 0, 0, 0x100, 0x0}, {5, 0, 0, 0, 0x100, 0x0}, {5, 0, 0, 0, 0x600, 0x700}}},
    {"s4xTail2", TextureTileMode::kS4KBX, 12, 33, 20, 12, 6, true, true, std::vector<Element>{{0, 0, 0, 0, 0x6000, 0x0}, {0, 32, 19, 11, 0xf13c, 0x0}, {0, 8, 9, 8, 0x7408, 0x0}, {0, 15, 2, 8, 0x7962, 0x700}, {1, 0, 0, 0, 0x3000, 0x0}, {1, 15, 9, 5, 0x42ce, 0x0}, {1, 12, 4, 0, 0x4b00, 0x0}, {1, 7, 2, 5, 0x35e6, 0x700}, {2, 0, 0, 0, 0x1000, 0x0}, {2, 7, 4, 2, 0x1352, 0x0}, {2, 2, 2, 0, 0x1060, 0x0}, {2, 0, 1, 1, 0x170c, 0x700}, {3, 0, 0, 0, 0x800, 0x0}, {3, 3, 1, 0, 0x84a, 0x0}, {3, 3, 1, 0, 0x84a, 0x0}, {3, 0, 0, 0, 0xf00, 0x700}, {4, 0, 0, 0, 0x300, 0x0}, {4, 1, 0, 0, 0x302, 0x0}, {4, 0, 0, 0, 0x300, 0x0}, {4, 0, 0, 0, 0x400, 0x700}, {5, 0, 0, 0, 0x200, 0x0}, {5, 0, 0, 0, 0x200, 0x0}, {5, 0, 0, 0, 0x200, 0x0}, {5, 0, 0, 0, 0x500, 0x700}}},
    {"s4xTail4", TextureTileMode::kS4KBX, 56, 33, 20, 12, 6, true, true, std::vector<Element>{{0, 0, 0, 0, 0x6000, 0x0}, {0, 32, 19, 11, 0x1fdb8, 0x0}, {0, 8, 9, 8, 0x17808, 0x0}, {0, 15, 2, 8, 0x17564, 0x700}, {1, 0, 0, 0, 0x3000, 0x0}, {1, 15, 9, 5, 0x465c, 0x0}, {1, 12, 4, 0, 0x4b00, 0x0}, {1, 7, 2, 5, 0x3174, 0x700}, {2, 0, 0, 0, 0x1000, 0x0}, {2, 7, 4, 2, 0x13c4, 0x0}, {2, 2, 2, 0, 0x1060, 0x0}, {2, 0, 1, 1, 0x1718, 0x700}, {3, 0, 0, 0, 0x800, 0x0}, {3, 3, 1, 0, 0x84c, 0x0}, {3, 3, 1, 0, 0x84c, 0x0}, {3, 0, 0, 0, 0xf00, 0x700}, {4, 0, 0, 0, 0x300, 0x0}, {4, 1, 0, 0, 0x304, 0x0}, {4, 0, 0, 0, 0x300, 0x0}, {4, 0, 0, 0, 0x400, 0x700}, {5, 0, 0, 0, 0x200, 0x0}, {5, 0, 0, 0, 0x200, 0x0}, {5, 0, 0, 0, 0x200, 0x0}, {5, 0, 0, 0, 0x500, 0x700}}},
    {"s4xTail16", TextureTileMode::kS4KBX, 77, 33, 20, 12, 6, true, true, std::vector<Element>{{0, 0, 0, 0, 0x10000, 0x0}, {0, 32, 19, 11, 0x55bb0, 0x0}, {0, 8, 9, 8, 0x46820, 0x0}, {0, 15, 2, 8, 0x3e040, 0x700}, {1, 0, 0, 0, 0x6000, 0x0}, {1, 15, 9, 5, 0xee70, 0x0}, {1, 12, 4, 0, 0x9400, 0x0}, {1, 7, 2, 5, 0x7850, 0x700}, {2, 0, 0, 0, 0x3000, 0x0}, {2, 7, 4, 2, 0x42c0, 0x0}, {2, 2, 2, 0, 0x3300, 0x0}, {2, 0, 1, 1, 0x3730, 0x700}, {3, 0, 0, 0, 0x1000, 0x0}, {3, 3, 1, 0, 0x1260, 0x0}, {3, 3, 1, 0, 0x1260, 0x0}, {3, 0, 0, 0, 0x1700, 0x700}, {4, 0, 0, 0, 0x800, 0x0}, {4, 1, 0, 0, 0x840, 0x0}, {4, 0, 0, 0, 0x800, 0x0}, {4, 0, 0, 0, 0xf00, 0x700}, {5, 0, 0, 0, 0x300, 0x0}, {5, 0, 0, 0, 0x300, 0x0}, {5, 0, 0, 0, 0x300, 0x0}, {5, 0, 0, 0, 0x400, 0x700}}},
    {"chainInTail", TextureTileMode::kStandard4KB, 56, 8, 8, 8, 2, true, true, std::vector<Element>{{0, 0, 0, 0, 0x800}, {0, 7, 7, 7, 0xffc}, {0, 4, 2, 4, 0xe20}, {0, 2, 7, 0, 0x968}, {0, 7, 0, 7, 0xed4}, {1, 0, 0, 0, 0x300}, {1, 3, 3, 3, 0x3fc}, {1, 2, 1, 2, 0x3c8}, {1, 1, 3, 0, 0x32c}, {1, 3, 0, 3, 0x3d4}}},
    {"wideTail", TextureTileMode::kStandard64KB, 71, 40, 24, 20, 6, true, true, std::vector<Element>{{0, 0, 0, 0, 0x20000}, {0, 39, 23, 19, 0xb0bf8}, {0, 20, 8, 10, 0x2e280}, {0, 13, 23, 0, 0x41b28}, {0, 39, 0, 19, 0x902d8}, {1, 0, 0, 0, 0x10000}, {1, 19, 11, 9, 0x1e178}, {1, 10, 4, 5, 0x11c50}, {1, 6, 11, 0, 0x14360}, {1, 19, 0, 9, 0x1a058}, {2, 0, 0, 0, 0x8000}, {2, 9, 5, 4, 0x9c28}, {2, 5, 2, 2, 0x8388}, {2, 3, 5, 0, 0x8868}, {2, 9, 0, 4, 0x9408}, {3, 0, 0, 0, 0x4000}, {3, 4, 2, 1, 0x4310}, {3, 2, 1, 1, 0x4070}, {3, 1, 2, 0, 0x4108}, {3, 4, 0, 1, 0x4210}, {4, 0, 0, 0, 0x1000}, {4, 1, 0, 0, 0x1008}, {5, 0, 0, 0, 0xa00}}},
    {"xorDx8", TextureTileMode::kD64KBX, 1, 100, 60, 70, 7, true, true, std::vector<Element>{{0, 0, 0, 0, 0x20000}, {0, 99, 59, 69, 0x112b9f}, {0, 15, 52, 41, 0xa1347}, {0, 86, 43, 67, 0x11637e}, {0, 87, 59, 38, 0xb61fb}, {1, 0, 0, 0, 0x10000}, {1, 49, 29, 34, 0x76d29}, {1, 11, 28, 6, 0x126a3}, {1, 8, 27, 2, 0x12238}, {1, 30, 6, 17, 0x1c756}, {2, 0, 0, 0, 0x800}, {2, 24, 14, 16, 0xee10}, {2, 8, 3, 12, 0x1998}, {2, 15, 5, 9, 0x1d4f}, {2, 8, 8, 11, 0x3824}, {3, 0, 0, 0, 0x200}, {3, 11, 6, 7, 0x7b7}, {3, 7, 3, 6, 0x2fb}, {3, 3, 5, 4, 0x68b}, {3, 7, 2, 0, 0x253}, {4, 0, 0, 0, 0x4200}, {4, 5, 2, 3, 0x4275}, {4, 0, 2, 1, 0x4214}, {4, 2, 1, 1, 0x420e}, {4, 0, 2, 0, 0x4210}, {5, 0, 0, 0, 0x2000}, {5, 2, 0, 1, 0x2006}, {5, 1, 0, 0, 0x2001}, {6, 0, 0, 0, 0x2500}}},
    {"xorZx32", TextureTileMode::kZ64KBX, 56, 40, 24, 20, 4, true, false, std::vector<Element>{{0, 0, 0, 0, 0x8400}, {0, 39, 23, 19, 0x1382fc}, {0, 20, 8, 10, 0xab240}, {0, 13, 23, 0, 0x87ec}, {0, 39, 0, 19, 0x138054}, {1, 0, 0, 0, 0x4800}, {1, 19, 11, 9, 0x9723c}, {1, 10, 4, 5, 0x54390}, {1, 6, 11, 0, 0x5978}, {1, 19, 0, 9, 0x96314}, {2, 0, 0, 0, 0x800}, {2, 9, 5, 4, 0x40b8c}, {2, 5, 2, 2, 0x20c64}, {2, 3, 5, 0, 0x89c}, {2, 9, 0, 4, 0x40b04}, {3, 0, 0, 0, 0x400}, {3, 4, 2, 1, 0x10c60}, {3, 2, 1, 1, 0x10c18}, {3, 1, 2, 0, 0x424}, {3, 4, 0, 1, 0x10c40}}},
    {"xorZx64", TextureTileMode::kZ64KBX, 71, 33, 20, 12, 6, true, false, std::vector<Element>{{0, 0, 0, 0, 0x8400}, {0, 32, 19, 11, 0xbc350}, {0, 16, 6, 6, 0x69040}, {0, 11, 19, 0, 0xe778}, {0, 32, 0, 11, 0xb8100}, {1, 0, 0, 0, 0x400}, {1, 15, 9, 5, 0x52eb8}, {1, 8, 3, 3, 0x32950}, {1, 5, 9, 0, 0x598}, {1, 15, 0, 5, 0x52fa8}, {2, 0, 0, 0, 0x800}, {2, 7, 4, 2, 0x21ca8}, {2, 4, 1, 1, 0x10090}, {2, 2, 4, 0, 0x1820}, {2, 7, 0, 2, 0x20ca8}, {3, 0, 0, 0, 0x4200}, {3, 3, 1, 0, 0x4238}, {3, 2, 0, 0, 0x4220}, {3, 1, 1, 0, 0x4218}, {3, 3, 0, 0, 0x4228}, {4, 0, 0, 0, 0x200}, {4, 1, 0, 0, 0x208}, {4, 1, 0, 0, 0x208}, {4, 0, 0, 0, 0x200}, {4, 1, 0, 0, 0x208}, {5, 0, 0, 0, 0x2000}, {5, 0, 0, 0, 0x2000}, {5, 0, 0, 0, 0x2000}, {5, 0, 0, 0, 0x2000}, {5, 0, 0, 0, 0x2000}}},
    {"atD256B", TextureTileMode::kD256B, 1, 40, 20, 1, 3, false, false, std::vector<Element>{{0, 0, 0, 0, 0x300}, {0, 39, 19, 0, 0x81f}, {0, 20, 6, 0, 0x42c}, {0, 39, 0, 0, 0x507}, {0, 0, 19, 0, 0x618}, {1, 0, 0, 0, 0x100}, {1, 19, 9, 0, 0x293}, {1, 10, 3, 0, 0x15a}, {1, 19, 0, 0, 0x203}, {1, 0, 9, 0, 0x190}, {2, 0, 0, 0, 0x0}, {2, 9, 4, 0, 0x61}, {2, 5, 1, 0, 0x15}, {2, 9, 0, 0, 0x41}, {2, 0, 4, 0, 0x20}}},
    {"atD4KB", TextureTileMode::kD4KB, 71, 100, 60, 1, 7, false, false, std::vector<Element>{{0, 0, 0, 0, 0x6000}, {0, 99, 59, 0, 0x154b8}, {0, 50, 20, 0, 0xb920}, {0, 99, 0, 0, 0x9028}, {0, 0, 59, 0, 0x12490}, {1, 0, 0, 0, 0x2000}, {1, 49, 29, 0, 0x5d18}, {1, 25, 10, 0, 0x2e88}, {1, 49, 0, 0, 0x3808}, {1, 0, 29, 0, 0x4510}, {2, 0, 0, 0, 0x1000}, {2, 24, 14, 0, 0x1f80}, {2, 12, 5, 0, 0x1350}, {2, 24, 0, 0, 0x1a00}, {2, 0, 14, 0, 0x1580}, {3, 0, 0, 0, 0x800}, {3, 11, 6, 0, 0xba8}, {3, 6, 2, 0, 0x8e0}, {3, 11, 0, 0, 0xa28}, {3, 0, 6, 0, 0x980}, {4, 0, 0, 0, 0x600}, {4, 5, 2, 0, 0x6c8}, {4, 3, 1, 0, 0x638}, {4, 5, 0, 0, 0x648}, {4, 0, 2, 0, 0x680}, {5, 0, 0, 0, 0x500}, {5, 2, 0, 0, 0x520}, {5, 1, 0, 0, 0x508}, {5, 2, 0, 0, 0x520}, {5, 0, 0, 0, 0x500}, {6, 0, 0, 0, 0x400}, {6, 0, 0, 0, 0x400}, {6, 0, 0, 0, 0x400}, {6, 0, 0, 0, 0x400}, {6, 0, 0, 0, 0x400}}},
    {"atD64KB", TextureTileMode::kD64KB, 56, 200, 150, 1, 6, false, false, std::vector<Element>{{0, 0, 0, 0, 0x20000}, {0, 199, 149, 0, 0x584dc}, {0, 100, 50, 0, 0x2b4a0}, {0, 199, 0, 0, 0x3808c}, {0, 0, 149, 0, 0x40450}, {1, 0, 0, 0, 0x10000}, {1, 99, 74, 0, 0x1e12c}, {1, 50, 25, 0, 0x12d18}, {1, 99, 0, 0, 0x1a00c}, {1, 0, 74, 0, 0x14120}, {2, 0, 0, 0, 0x8000}, {2, 49, 36, 0, 0xb844}, {2, 25, 12, 0, 0x8b44}, {2, 49, 0, 0, 0xa804}, {2, 0, 36, 0, 0x9040}, {3, 0, 0, 0, 0x4000}, {3, 24, 17, 0, 0x4e10}, {3, 12, 6, 0, 0x42e0}, {3, 24, 0, 0, 0x4a00}, {3, 0, 17, 0, 0x4410}, {4, 0, 0, 0, 0x2000}, {4, 11, 8, 0, 0x230c}, {4, 6, 3, 0, 0x20b8}, {4, 11, 0, 0, 0x220c}, {4, 0, 8, 0, 0x2100}, {5, 0, 0, 0, 0x1000}, {5, 5, 3, 0, 0x10b4}, {5, 3, 1, 0, 0x101c}, {5, 5, 0, 0, 0x1084}, {5, 0, 3, 0, 0x1030}}},
    {"atS64KBT", TextureTileMode::kS64KBT, 56, 520, 260, 3, 4, false, false, std::vector<Element>{{0, 0, 0, 0, 0x90000}, {0, 519, 259, 0, 0x1700bc}, {0, 260, 86, 0, 0xb46e0}, {0, 519, 0, 0, 0xd008c}, {0, 0, 259, 0, 0x130030}, {0, 0, 0, 1, 0x210000}, {0, 519, 259, 1, 0x2f00bc}, {0, 260, 86, 1, 0x2346e0}, {0, 519, 0, 1, 0x25008c}, {0, 0, 259, 1, 0x2b0030}, {0, 0, 0, 2, 0x390000}, {0, 519, 259, 2, 0x4700bc}, {0, 260, 86, 2, 0x3b46e0}, {0, 519, 0, 2, 0x3d008c}, {0, 0, 259, 2, 0x430030}, {1, 0, 0, 0, 0x30000}, {1, 259, 129, 0, 0x8001c}, {1, 130, 43, 0, 0x41938}, {1, 259, 0, 0, 0x5000c}, {1, 0, 129, 0, 0x60010}, {1, 0, 0, 1, 0x1b0000}, {1, 259, 129, 1, 0x20001c}, {1, 130, 43, 1, 0x1c1938}, {1, 259, 0, 1, 0x1d000c}, {1, 0, 129, 1, 0x1e0010}, {1, 0, 0, 2, 0x330000}, {1, 259, 129, 2, 0x38001c}, {1, 130, 43, 2, 0x341938}, {1, 259, 0, 2, 0x35000c}, {1, 0, 129, 2, 0x360010}, {2, 0, 0, 0, 0x10000}, {2, 129, 64, 0, 0x24204}, {2, 65, 21, 0, 0x18554}, {2, 129, 0, 0, 0x20004}, {2, 0, 64, 0, 0x14200}, {2, 0, 0, 1, 0x190000}, {2, 129, 64, 1, 0x1a4204}, {2, 65, 21, 1, 0x198554}, {2, 129, 0, 1, 0x1a0004}, {2, 0, 64, 1, 0x194200}, {2, 0, 0, 2, 0x310000}, {2, 129, 64, 2, 0x324204}, {2, 65, 21, 2, 0x318554}, {2, 129, 0, 2, 0x320004}, {2, 0, 64, 2, 0x314200}, {3, 0, 0, 0, 0x0}, {3, 64, 31, 0, 0x8470}, {3, 32, 10, 0, 0x2520}, {3, 64, 0, 0, 0x8100}, {3, 0, 31, 0, 0x570}, {3, 0, 0, 1, 0x180000}, {3, 64, 31, 1, 0x188470}, {3, 32, 10, 1, 0x182520}, {3, 64, 0, 1, 0x188100}, {3, 0, 31, 1, 0x180570}, {3, 0, 0, 2, 0x300000}, {3, 64, 31, 2, 0x308470}, {3, 32, 10, 2, 0x302520}, {3, 64, 0, 2, 0x308100}, {3, 0, 31, 2, 0x300570}}},
    {"atS4KBX", TextureTileMode::kS4KBX, 1, 100, 70, 3, 6, false, false, std::vector<Element>{{0, 0, 0, 0, 0x2000}, {0, 99, 69, 0, 0x5453}, {0, 50, 23, 0, 0x2b72}, {0, 99, 0, 0, 0x3c03}, {0, 0, 69, 0, 0x4850}, {0, 0, 0, 1, 0x8800}, {0, 99, 69, 1, 0xbc53}, {0, 50, 23, 1, 0x8372}, {0, 99, 0, 1, 0x9403}, {0, 0, 69, 1, 0xa050}, {0, 0, 0, 2, 0xe400}, {0, 99, 69, 2, 0x11053}, {0, 50, 23, 2, 0xef72}, {0, 99, 0, 2, 0xf803}, {0, 0, 69, 2, 0x10c50}, {1, 0, 0, 0, 0x1000}, {1, 49, 34, 0, 0x1e21}, {1, 25, 11, 0, 0x12b9}, {1, 49, 0, 0, 0x1a01}, {1, 0, 34, 0, 0x1420}, {1, 0, 0, 1, 0x7800}, {1, 49, 34, 1, 0x7621}, {1, 25, 11, 1, 0x7ab9}, {1, 49, 0, 1, 0x7201}, {1, 0, 34, 1, 0x7c20}, {1, 0, 0, 2, 0xd400}, {1, 49, 34, 2, 0xda21}, {1, 25, 11, 2, 0xd6b9}, {1, 49, 0, 2, 0xde01}, {1, 0, 34, 2, 0xd020}, {2, 0, 0, 0, 0x800}, {2, 24, 16, 0, 0xb08}, {2, 12, 5, 0, 0x85c}, {2, 24, 0, 0, 0xa08}, {2, 0, 16, 0, 0x900}, {2, 0, 0, 1, 0x6000}, {2, 24, 16, 1, 0x6308}, {2, 12, 5, 1, 0x605c}, {2, 24, 0, 1, 0x6208}, {2, 0, 16, 1, 0x6100}, {2, 0, 0, 2, 0xcc00}, {2, 24, 16, 2, 0xcf08}, {2, 12, 5, 2, 0xcc5c}, {2, 24, 0, 2, 0xce08}, {2, 0, 16, 2, 0xcd00}, {3, 0, 0, 0, 0x600}, {3, 11, 7, 0, 0x67b}, {3, 6, 2, 0, 0x626}, {3, 11, 0, 0, 0x60b}, {3, 0, 7, 0, 0x670}, {3, 0, 0, 1, 0x6e00}, {3, 11, 7, 1, 0x6e7b}, {3, 6, 2, 1, 0x6e26}, {3, 11, 0, 1, 0x6e0b}, {3, 0, 7, 1, 0x6e70}, {3, 0, 0, 2, 0xc200}, {3, 11, 7, 2, 0xc27b}, {3, 6, 2, 2, 0xc226}, {3, 11, 0, 2, 0xc20b}, {3, 0, 7, 2, 0xc270}, {4, 0, 0, 0, 0x500}, {4, 5, 3, 0, 0x535}, {4, 3, 1, 0, 0x513}, {4, 5, 0, 0, 0x505}, {4, 0, 3, 0, 0x530}, {4, 0, 0, 1, 0x6d00}, {4, 5, 3, 1, 0x6d35}, {4, 3, 1, 1, 0x6d13}, {4, 5, 0, 1, 0x6d05}, {4, 0, 3, 1, 0x6d30}, {4, 0, 0, 2, 0xc100}, {4, 5, 3, 2, 0xc135}, {4, 3, 1, 2, 0xc113}, {4, 5, 0, 2, 0xc105}, {4, 0, 3, 2, 0xc130}, {5, 0, 0, 0, 0x400}, {5, 2, 1, 0, 0x412}, {5, 1, 0, 0, 0x401}, {5, 2, 0, 0, 0x402}, {5, 0, 1, 0, 0x410}, {5, 0, 0, 1, 0x6c00}, {5, 2, 1, 1, 0x6c12}, {5, 1, 0, 1, 0x6c01}, {5, 2, 0, 1, 0x6c02}, {5, 0, 1, 1, 0x6c10}, {5, 0, 0, 2, 0xc000}, {5, 2, 1, 2, 0xc012}, {5, 1, 0, 2, 0xc001}, {5, 2, 0, 2, 0xc002}, {5, 0, 1, 2, 0xc010}}},
    {"atD4KBX", TextureTileMode::kD4KBX, 56, 33, 20, 2, 4, false, false, std::vector<Element>{{0, 0, 0, 0, 0x2000}, {0, 32, 19, 0, 0x3030}, {0, 16, 6, 0, 0x2860}, {0, 32, 0, 0, 0x3400}, {0, 0, 19, 0, 0x2430}, {0, 0, 0, 1, 0x6800}, {0, 32, 19, 1, 0x7830}, {0, 16, 6, 1, 0x6060}, {0, 32, 0, 1, 0x7c00}, {0, 0, 19, 1, 0x6c30}, {1, 0, 0, 0, 0x1000}, {1, 15, 9, 0, 0x139c}, {1, 8, 3, 0, 0x1230}, {1, 15, 0, 0, 0x128c}, {1, 0, 9, 0, 0x1110}, {1, 0, 0, 1, 0x5800}, {1, 15, 9, 1, 0x5b9c}, {1, 8, 3, 1, 0x5a30}, {1, 15, 0, 1, 0x5a8c}, {1, 0, 9, 1, 0x5910}, {2, 0, 0, 0, 0x800}, {2, 7, 4, 0, 0x8cc}, {2, 4, 1, 0, 0x890}, {2, 7, 0, 0, 0x88c}, {2, 0, 4, 0, 0x840}, {2, 0, 0, 1, 0x4000}, {2, 7, 4, 1, 0x40cc}, {2, 4, 1, 1, 0x4090}, {2, 7, 0, 1, 0x408c}, {2, 0, 4, 1, 0x4040}, {3, 0, 0, 0, 0x600}, {3, 3, 1, 0, 0x61c}, {3, 2, 0, 0, 0x608}, {3, 3, 0, 0, 0x60c}, {3, 0, 1, 0, 0x610}, {3, 0, 0, 1, 0x4e00}, {3, 3, 1, 1, 0x4e1c}, {3, 2, 0, 1, 0x4e08}, {3, 3, 0, 1, 0x4e0c}, {3, 0, 1, 1, 0x4e10}}},
    };
    return list;
}

GuestTextureResource describe(const Case& item) {
    GuestTextureResource resource{};
    resource.width = item.width;
    resource.height = item.height;
    resource.depthOrLastArray = item.depth - 1u;
    resource.mipCount = item.mips;
    resource.tileMode = item.mode;
    resource.format = item.format;
    resource.dimension = item.volume ? TextureDimension::k3D : (item.depth > 1u ? TextureDimension::k2DArray : TextureDimension::k2D);
    return resource;
}

void addressesMatchAddrlib() {
    std::size_t checked = 0;
    for (const auto& item : cases()) {
        auto resource = describe(item);
        const auto geometry = DescribeSurface(resource);
        Require(geometry.thick == item.thick, std::string(item.name) + ": thick classification");
        const auto* format = M::FindFormat(item.format);
        Require(format != nullptr, std::string(item.name) + ": format missing");
        for (const auto& element : item.elements) {
            if (!geometry.HasLayer(element.mip, element.z)) continue;
            resource.pipeBankXor = element.pipeBankXor;
            const auto params = M::MakeDetileParams(resource, geometry, element.mip, element.z, 0);
            const auto offset = agc_tiled_offset(params, element.x, element.y, format->guestElementBytes);
            const auto address = params.tiled + offset;
            Require(address == element.address, std::string(item.name) + ": mip " + std::to_string(element.mip) + " (" + std::to_string(element.x) + "," + std::to_string(element.y) + "," + std::to_string(element.z) + ") maps to " + std::to_string(address) + " instead of addrlib's " + std::to_string(element.address));
            ++checked;
        }
    }
    std::printf("addrlib vectors ok elements=%zu\n", checked);
}

void blocksAreBijections() {
    std::size_t modes = 0;
    const TextureTileMode all[] = {TextureTileMode::kStandard256B, TextureTileMode::kStandard4KB, TextureTileMode::kStandard64KB, TextureTileMode::kZ64KBX, TextureTileMode::kS64KBX, TextureTileMode::kD64KBX, TextureTileMode::kR64KBX, TextureTileMode::kD256B, TextureTileMode::kD4KB, TextureTileMode::kD64KB, TextureTileMode::kS64KBT, TextureTileMode::kD64KBT, TextureTileMode::kS4KBX, TextureTileMode::kD4KBX};
    for (const auto mode : all) {
        for (std::uint32_t bytes = 1; bytes <= 16; bytes *= 2) {
            GuestTextureResource resource{};
            resource.width = 512;
            resource.height = 512;
            resource.mipCount = 1;
            resource.tileMode = mode;
            resource.dimension = TextureDimension::k2D;
            resource.format = bytes == 1 ? 1 : bytes == 2 ? 14 : bytes == 4 ? 56 : bytes == 8 ? 71 : 77;
            const auto geometry = DescribeSurface(resource);
            const auto params = M::MakeDetileParams(resource, geometry, 0, 0, 0);
            const auto block = ThinBlockLayout(mode, bytes);
            std::set<std::uint32_t> seen;
            for (std::uint32_t y = 0; y < block[2]; ++y) {
                for (std::uint32_t x = 0; x < block[1]; ++x) {
                    const auto offset = agc_tiled_offset(params, x, y, bytes);
                    Require(offset < block[0], "tile mode offset leaves its block");
                    Require(offset % bytes == 0, "tile mode offset is not element aligned");
                    Require(seen.insert(offset).second, "tile mode maps two elements to one offset");
                }
            }
            ++modes;
        }
    }
    std::printf("block bijections ok variants=%zu\n", modes);
}

void formatsMatchVulkan() {
    std::size_t compared = 0;
    for (std::uint32_t guest = 0; guest <= M::MaxGuestFormat + 4; ++guest) {
        bool vulkan = true;
        std::uint32_t bytes = 0;
        bool compressed = false;
        try {
            bytes = BytesPerElement(guest);
            compressed = IsBlockCompressed(guest);
        } catch (const std::exception&) {
            vulkan = false;
        }
        const auto* mine = M::FindFormat(guest);
        Require((mine != nullptr) == vulkan, "format " + std::to_string(guest) + " support differs from the Vulkan table");
        if (!vulkan) continue;
        Require(mine->guestElementBytes == bytes, "format " + std::to_string(guest) + " element size differs");
        Require(mine->blockCompressed == compressed, "format " + std::to_string(guest) + " block flag differs");
        Require(mine->srgb == IsSrgbTextureFormat(guest), "format " + std::to_string(guest) + " srgb flag differs");
        Require(M::IsRemappedFormat(guest) == IsConvertedTextureFormat(guest), "format " + std::to_string(guest) + " remap differs");
        Require((mine->transform == M::TextureTransform::Repack101111) == IsRepackedTextureFormat(guest), "format " + std::to_string(guest) + " repack differs");
        Require(mine->texelBytes >= mine->guestElementBytes, "format " + std::to_string(guest) + " texel smaller than element");
        Require(M::WriteViewFormat(mine->texelBytes) != M::PixelFormat::Invalid, "format " + std::to_string(guest) + " has no write view");
        Require(!mine->uploadable == (bytes == 12), "format " + std::to_string(guest) + " upload flag");
        ++compared;
    }
    std::printf("formats match vulkan table formats=%zu\n", compared);
}

M::TextureSwizzle fromVulkan(VkComponentSwizzle swizzle) {
    switch (swizzle) {
        case VK_COMPONENT_SWIZZLE_ZERO: return M::TextureSwizzle::Zero;
        case VK_COMPONENT_SWIZZLE_ONE: return M::TextureSwizzle::One;
        case VK_COMPONENT_SWIZZLE_R: return M::TextureSwizzle::Red;
        case VK_COMPONENT_SWIZZLE_G: return M::TextureSwizzle::Green;
        case VK_COMPONENT_SWIZZLE_B: return M::TextureSwizzle::Blue;
        case VK_COMPONENT_SWIZZLE_A: return M::TextureSwizzle::Alpha;
        default: break;
    }
    Require(false, "unexpected component swizzle");
    return M::TextureSwizzle::Zero;
}

VkComponentSwizzle toVulkan(std::uint8_t dstSel) {
    switch (dstSel) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 4: return VK_COMPONENT_SWIZZLE_R;
        case 5: return VK_COMPONENT_SWIZZLE_G;
        case 6: return VK_COMPONENT_SWIZZLE_B;
        default: return VK_COMPONENT_SWIZZLE_A;
    }
}

void swizzlesMatchVulkan() {
    const std::uint8_t selectors[] = {0, 1, 4, 5, 6, 7};
    for (std::uint32_t guest = 0; guest <= M::MaxGuestFormat; ++guest) {
        const auto* mine = M::FindFormat(guest);
        if (mine == nullptr) continue;
        for (const auto x : selectors) {
            for (const auto w : selectors) {
                const std::array<std::uint8_t, 4> sel{x, 5, 6, w};
                const auto result = M::ViewSwizzle(guest, sel);
                for (std::size_t index = 0; index < 4; ++index) {
                    const bool identity = M::IsRemappedFormat(guest) && mine->transform != M::TextureTransform::Repack101111;
                    auto expected = identity ? static_cast<M::TextureSwizzle>(static_cast<int>(M::TextureSwizzle::Red) + static_cast<int>(index)) : fromVulkan(TextureComponentChannel(guest, toVulkan(sel[index])));
                    if (!identity && expected >= M::TextureSwizzle::Red) expected = static_cast<M::TextureSwizzle>(static_cast<int>(M::TextureSwizzle::Red) + mine->layoutPermutation[static_cast<int>(expected) - static_cast<int>(M::TextureSwizzle::Red)]);
                    Require(result[index] == expected, "view swizzle of format " + std::to_string(guest) + " differs");
                }
            }
        }
    }
    const auto abgr = M::ViewSwizzle(136, {4, 5, 6, 7});
    Require(abgr[0] == M::TextureSwizzle::Alpha && abgr[1] == M::TextureSwizzle::Blue && abgr[2] == M::TextureSwizzle::Green && abgr[3] == M::TextureSwizzle::Red, "guest ARGB4 selects through the Metal ABGR4 layout");
    bool rejected = false;
    try {
        M::ViewSwizzle(56, {2, 5, 6, 7});
    } catch (const std::exception&) {
        rejected = true;
    }
    Require(rejected, "an invalid destination selector is rejected");
    std::puts("swizzles ok");
}

void transformsAreExact() {
    Require(agc_transform({0xdeadbeefu, 0, 0, 0}, AGC_TRANSFORM_REPACK101111).x == RepackTextureWord(0xdeadbeefu), "repack matches the Vulkan texel repack");
    std::mt19937 random(7);
    for (int index = 0; index < 4096; ++index) {
        const auto word = static_cast<std::uint32_t>(random());
        Require(agc_transform({word, 0, 0, 0}, AGC_TRANSFORM_REPACK101111).x == RepackTextureWord(word), "repack random word");
        const auto sint = agc_transform({word, 0, 0, 0}, AGC_TRANSFORM_EXPAND_SINT2101010);
        const auto snorm = agc_transform({word, 0, 0, 0}, AGC_TRANSFORM_EXPAND_SNORM2101010);
        const int fields[4] = {agc_sign_extend(word & 0x3ffu, 10), agc_sign_extend((word >> 10) & 0x3ffu, 10), agc_sign_extend((word >> 20) & 0x3ffu, 10), agc_sign_extend(word >> 30, 2)};
        const std::uint32_t packedSint[4] = {sint.x & 0xffffu, sint.x >> 16, sint.y & 0xffffu, sint.y >> 16};
        const std::uint32_t packedSnorm[4] = {snorm.x & 0xffffu, snorm.x >> 16, snorm.y & 0xffffu, snorm.y >> 16};
        for (int channel = 0; channel < 4; ++channel) {
            Require(static_cast<std::int16_t>(packedSint[channel]) == fields[channel], "sint expand sign extension");
            const double limit = channel == 3 ? 1.0 : 511.0;
            double expected = std::max(static_cast<double>(fields[channel]) / limit, -1.0);
            const auto want = static_cast<int>(expected >= 0 ? expected * 32767.0 + 0.5 : expected * 32767.0 - 0.5);
            Require(static_cast<std::int16_t>(packedSnorm[channel]) == want, "snorm expand rounding of channel " + std::to_string(channel) + " field " + std::to_string(fields[channel]));
        }
    }
    const auto edge = agc_transform({0x200u | (0x1ffu << 10) | (0x200u << 20) | (2u << 30), 0, 0, 0}, AGC_TRANSFORM_EXPAND_SNORM2101010);
    Require(static_cast<std::int16_t>(edge.x & 0xffffu) == -32767 && static_cast<std::int16_t>(edge.x >> 16) == 32767 && static_cast<std::int16_t>(edge.y & 0xffffu) == -32767 && static_cast<std::int16_t>(edge.y >> 16) == -32767, "snorm edge values clamp to plus and minus one");
    std::puts("transforms ok");
}

void elementsRoundTrip() {
    std::vector<std::uint8_t> bytes(64, 0);
    for (std::uint32_t size : {1u, 2u, 4u, 8u, 16u}) {
        std::fill(bytes.begin(), bytes.end(), 0);
        const AgcDetileParams unused{};
        (void)unused;
        const uint4 value{0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
        agc_store_element(bytes.data(), 16, size, value);
        const auto loaded = agc_load_element(bytes.data(), 16, size);
        const std::uint32_t mask = size >= 4 ? 0xffffffffu : (1u << (size * 8)) - 1u;
        Require(loaded.x == (value.x & mask), "element round trip word 0");
        if (size >= 8) Require(loaded.y == value.y, "element round trip word 1");
        if (size >= 16) Require(loaded.z == value.z && loaded.w == value.w, "element round trip words 2 and 3");
        for (std::uint32_t index = 0; index < 64; ++index) {
            if (index < 16 || index >= 16 + size) Require(bytes[index] == 0, "element store stays inside its element");
        }
    }
    std::puts("element load and store ok");
}

void plansCoverTheSurface() {
    GuestTextureResource resource{};
    resource.width = 100;
    resource.height = 60;
    resource.depthOrLastArray = 2;
    resource.mipCount = 5;
    resource.tileMode = TextureTileMode::kD64KBX;
    resource.dimension = TextureDimension::k2DArray;
    resource.format = 56;
    const auto plan = M::PlanSurface(resource, 0x10000);
    Require(plan.jobs.size() == 15, "one job per level and layer");
    Require(!plan.volume && plan.elementBytes == 4 && plan.texelBytes == 4, "plan element sizes");
    for (const auto& job : plan.jobs) {
        Require(job.params.destinationSlice == job.layer, "array layers land in their own slice");
        Require(job.columns == plan.geometry.mips[job.level].width && job.rows == plan.geometry.mips[job.level].height, "job grid is the mip");
    }
    resource.dimension = TextureDimension::k3D;
    resource.tileMode = TextureTileMode::kS64KBX;
    resource.format = 1;
    resource.mipCount = 3;
    const auto volume = M::PlanSurface(resource, 0);
    Require(volume.volume && volume.geometry.thick, "a thick volume is planned as a volume");
    for (const auto& job : volume.jobs) Require(job.params.family == AGC_FAMILY_EQUATION && job.params.destinationSlice == job.layer, "thick volume jobs use the equation family and their depth slice");
    resource.format = 72;
    bool rejected = false;
    try {
        M::PlanSurface(resource, 0);
    } catch (const std::exception&) {
        rejected = true;
    }
    Require(rejected, "a 96 bit format is rejected for detiling");
    std::puts("plans ok");
}

}

int main() {
    formatsMatchVulkan();
    swizzlesMatchVulkan();
    addressesMatchAddrlib();
    blocksAreBijections();
    transformsAreExact();
    elementsRoundTrip();
    plansCoverTheSurface();
    std::puts("metal texture tests: ok");
    return 0;
}
