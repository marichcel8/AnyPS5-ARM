typedef unsigned long long u64; typedef long long i64; typedef unsigned u32;
int puts(const char*); void exit(int);

static int first = -1;

static void same(int id, i64 got, i64 want) {
    if (got != want && first < 0) first = id;
}

static int repe_cmpsb(const char* a, const char* b, u64 n, u64* left) {
    unsigned char below, equal;
    __asm__ volatile("cld\n\trepe cmpsb\n\tsetb %0\n\tsete %1" : "=q"(below), "=q"(equal), "+S"(a), "+D"(b), "+c"(n) : : "cc", "memory");
    *left = n;
    return equal ? 0 : below ? -1 : 1;
}

static u64 repne_scasb(const char* text, char wanted) {
    u64 count = ~0ull;
    __asm__ volatile("cld\n\trepne scasb" : "+D"(text), "+c"(count) : "a"(wanted) : "cc", "memory");
    return ~count - 1;
}

static u64 repe_cmpsq(const u64* a, const u64* b, u64 n) {
    __asm__ volatile("cld\n\trepe cmpsq" : "+S"(a), "+D"(b), "+c"(n) : : "cc", "memory");
    return n;
}

static u64 repe_scasd(const u32* p, u32 value, u64 n) {
    __asm__ volatile("cld\n\trepe scasl" : "+D"(p), "+c"(n) : "a"(value) : "cc", "memory");
    return n;
}

static u64 lods(const void* p) {
    u64 value = 0xffffffffffffffffull;
    __asm__ volatile("cld\n\tlodsb\n\tlodsl" : "+a"(value), "+S"(p) : : "memory");
    return value;
}

static u64 backwards_cmpsb(const char* a, const char* b) {
    __asm__ volatile("std\n\tcmpsb\n\tcld" : "+S"(a), "+D"(b) : : "cc", "memory");
    return (u64)a;
}

void _start(void* block) {
    (void)block;
    u64 left = 0;
    same(1, repe_cmpsb("hello world", "hello there", 11, &left), 1);
    same(2, left, 4);
    same(3, repe_cmpsb("abc", "abc", 3, &left), 0);
    same(4, left, 0);
    same(5, repe_cmpsb("abc", "abd", 3, &left), -1);
    same(6, repne_scasb("strlen test", 0), 11);
    static const u64 a[4] = {1, 2, 3, 4}, b[4] = {1, 2, 9, 4};
    same(7, repe_cmpsq(a, b, 4), 1);
    static const u32 words[5] = {7, 7, 7, 8, 7};
    same(8, repe_scasd(words, 7, 5), 1);
    static const unsigned char bytes[5] = {0x11, 0x44, 0x33, 0x22, 0x11};
    same(9, lods(bytes), 0x11223344);
    static const char text[2] = "ab";
    same(10, backwards_cmpsb(text + 1, text + 1), (u64)text);
    u32 eax, ebx, ecx, edx;
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0), "c"(0));
    same(11, ebx, 0x68747541);
    same(12, edx, 0x69746e65);
    same(13, ecx, 0x444d4163);
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1), "c"(0));
    same(14, (ecx >> 28) & 1, 1);
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    same(15, eax & 6, 6);
    if (first < 0) {
        puts("strings ok");
        exit(0);
    }
    char message[] = "strings fail 00";
    message[13] = (char)('0' + first / 10 % 10);
    message[14] = (char)('0' + first % 10);
    puts(message);
    exit(1);
}
