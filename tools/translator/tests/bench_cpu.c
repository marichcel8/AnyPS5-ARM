typedef unsigned long long u64; typedef unsigned u32; typedef int i32;
int puts(const char*); void exit(int);

#ifndef ROUNDS
#define ROUNDS 40
#endif

static u64 seed = 0x9e3779b97f4a7c15ull;

static u64 next(void) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
}

static i32 values[1 << 16];

static void sort(i32* data, i32 low, i32 high) {
    while (low < high) {
        const i32 pivot = data[(low + high) / 2];
        i32 i = low, j = high;
        while (i <= j) {
            while (data[i] < pivot) ++i;
            while (data[j] > pivot) --j;
            if (i <= j) {
                const i32 t = data[i];
                data[i] = data[j];
                data[j] = t;
                ++i;
                --j;
            }
        }
        if (j - low < high - i) {
            sort(data, low, j);
            low = i;
        } else {
            sort(data, i, high);
            high = j;
        }
    }
}

static u64 sorting(void) {
    for (u32 i = 0; i < sizeof(values) / sizeof(values[0]); ++i) values[i] = (i32)(next() & 0x7fffffff);
    sort(values, 0, (i32)(sizeof(values) / sizeof(values[0])) - 1);
    u64 h = 0;
    for (u32 i = 0; i < sizeof(values) / sizeof(values[0]); i += 97) h = h * 31 + (u64)values[i];
    return h;
}

typedef struct { float x, y, z, w; } Vec;
static Vec vertices[4096];
static Vec transformed[4096];

static u64 transform(void) {
    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = (float)(next() % 1000) / 500.0f - 1.0f;
    for (u32 i = 0; i < 4096; ++i) vertices[i] = (Vec){(float)(i % 64), (float)(i / 64), (float)(i % 7), 1.0f};
    for (int pass = 0; pass < 16; ++pass) {
        for (u32 i = 0; i < 4096; ++i) {
            const Vec v = vertices[i];
            transformed[i].x = m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w;
            transformed[i].y = m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w;
            transformed[i].z = m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w;
            transformed[i].w = m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w;
        }
        for (u32 i = 0; i < 4096; ++i) vertices[i] = transformed[(i * 13) & 4095];
    }
    float sum = 0;
    for (u32 i = 0; i < 4096; ++i) sum += vertices[i].x * 0.001f + vertices[i].w * 0.0001f;
    union { float f; u32 u; } bits = {sum};
    return bits.u >> 8;
}

static i32 program[256];

static u64 interpret(void) {
    for (int i = 0; i < 256; ++i) program[i] = (i32)(next() % 8);
    i32 acc = 1, x = 7;
    u64 steps = 0;
    for (int round = 0; round < 2000; ++round) {
        for (int pc = 0; pc < 256; ++pc) {
            switch (program[pc]) {
            case 0: acc += x; break;
            case 1: acc ^= x << 3; break;
            case 2: x = acc - x; break;
            case 3: acc = acc * 3 + 1; break;
            case 4: if (acc & 1) x += 5; else x -= 3; break;
            case 5: acc >>= 1; break;
            case 6: x ^= acc; break;
            default: acc += pc; break;
            }
            ++steps;
        }
    }
    return (u64)(u32)acc * 1000003u + (u64)(u32)x + steps;
}

typedef struct Node { struct Node* left; struct Node* right; i32 key; i32 weight; } Node;
static Node nodes[1 << 14];

static u64 tree(void) {
    u32 count = 1;
    nodes[0] = (Node){0, 0, 1 << 30, 1};
    for (u32 i = 1; i < sizeof(nodes) / sizeof(nodes[0]); ++i) {
        const i32 key = (i32)(next() & 0x7fffffff);
        Node* at = &nodes[0];
        for (;;) {
            Node** link = key < at->key ? &at->left : &at->right;
            if (*link == 0) {
                nodes[count] = (Node){0, 0, key, (i32)(i & 15)};
                *link = &nodes[count++];
                break;
            }
            at = *link;
        }
    }
    u64 total = 0;
    for (int query = 0; query < 200000; ++query) {
        const i32 key = (i32)(next() & 0x7fffffff);
        const Node* at = &nodes[0];
        while (at != 0) {
            total += (u64)at->weight;
            at = key < at->key ? at->left : at->right;
        }
    }
    return total;
}

static unsigned char text[1 << 16];

static u64 strings(void) {
    for (u32 i = 0; i < sizeof(text); ++i) text[i] = (unsigned char)('a' + next() % 4);
    u64 found = 0;
    static const unsigned char needle[] = "abcab";
    for (int pass = 0; pass < 8; ++pass) {
        for (u32 i = 0; i + 5 <= sizeof(text); ++i) {
            u32 k = 0;
            while (k < 5 && text[i + k] == needle[k]) ++k;
            found += k == 5;
        }
    }
    return found;
}

static u64 mix(u64 h, u64 v) {
    h ^= v;
    h *= 0x100000001b3ull;
    return h ^ (h >> 29);
}

void _start(void* block) {
    (void)block;
    u64 h = 0xcbf29ce484222325ull;
    for (int round = 0; round < ROUNDS; ++round) {
        h = mix(h, sorting());
        h = mix(h, transform());
        h = mix(h, interpret());
        h = mix(h, tree());
        h = mix(h, strings());
    }
    char out[17];
    for (int i = 0; i < 16; ++i) out[i] = "0123456789abcdef"[(h >> (60 - 4 * i)) & 15];
    out[16] = 0;
    puts(out);
    exit(0);
}
