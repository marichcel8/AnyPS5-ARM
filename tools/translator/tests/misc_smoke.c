typedef unsigned long long u64; typedef long long i64; typedef unsigned u32; typedef unsigned short u16; typedef unsigned char u8;
int puts(const char*); void exit(int);

static int first = -1;

static void same(int id, i64 got, i64 want) {
    if (got != want && first < 0) first = id;
}

static void loops(void) {
    u64 count = 5, value = 0;
    __asm__ volatile("1:\n\tinc %0\n\tloop 1b" : "+r"(value), "+c"(count) : : "cc");
    same(1, (i64)value, 5);
    same(2, (i64)count, 0);
    count = 10;
    value = 0;
    __asm__ volatile("1:\n\tinc %0\n\tcmp $3, %0\n\tloopne 1b" : "+r"(value), "+c"(count) : : "cc");
    same(3, (i64)value, 3);
    same(4, (i64)count, 7);
    count = 10;
    value = 0;
    __asm__ volatile("1:\n\tinc %0\n\tcmp $4, %0\n\tloope 1b" : "+r"(value), "+c"(count) : : "cc");
    same(5, (i64)value, 1);
    same(6, (i64)count, 9);
}

static void rotates(void) {
    u8 al = 0x80, carry = 0;
    __asm__ volatile("stc\n\trclb $1, %0\n\tsetc %1" : "+q"(al), "=q"(carry) : : "cc");
    same(10, al, 0x01);
    same(11, carry, 1);
    al = 1;
    __asm__ volatile("clc\n\trcrb $1, %0\n\tsetc %1" : "+q"(al), "=q"(carry) : : "cc");
    same(12, al, 0);
    same(13, carry, 1);
    u64 wide = 0x8000000000000000ull;
    __asm__ volatile("stc\n\trclq $2, %0\n\tsetc %1" : "+r"(wide), "=q"(carry) : : "cc");
    same(14, (i64)wide, 3);
    same(15, carry, 0);
    u16 word = 0x1234;
    u8 shift = 17;
    __asm__ volatile("stc\n\trcrw %%cl, %0\n\tsetc %1" : "+r"(word), "=q"(carry) : "c"(shift) : "cc");
    same(16, word, 0x1234);
    same(17, carry, 1);
}

static void flags(void) {
    u64 ah = 0;
    __asm__ volatile("xor %%eax, %%eax\n\tmov $0, %%ecx\n\tcmp $1, %%ecx\n\tlahf\n\tshr $8, %%eax" : "=a"(ah) : : "rcx", "cc");
    same(20, (i64)(ah & 0xff), 0x97);
    u8 zero = 0, carry = 0;
    __asm__ volatile("mov $0x41, %%ah\n\tsahf\n\tsetz %0\n\tsetc %1" : "=q"(zero), "=q"(carry) : : "rax", "cc");
    same(21, zero, 1);
    same(22, carry, 1);
    u64 pushed = 0;
    __asm__ volatile("stc\n\tpushfq\n\tpop %0" : "=r"(pushed) : : "cc");
    same(23, (i64)(pushed & 0x203), 0x203);
    __asm__ volatile("push $0x242\n\tpopfq\n\tsetz %0\n\tsetc %1" : "=q"(zero), "=q"(carry) : : "cc");
    same(24, zero, 1);
    same(25, carry, 0);
}

static u64 jump(u64 selector) {
    u64 result;
    __asm__ volatile("xor %%ecx, %%ecx\n\tcmp $1, %%ecx\n\tcmp $2, %1\n\tja 9f\n\tmov %k1, %%ecx\n\tlea table%=(%%rip), %%rdx\n\t"
                     "movslq (%%rdx,%%rcx,4), %%rcx\n\tadd %%rdx, %%rcx\n\tjmp *%%rcx\n"
                     "0:\n\tmov $10, %0\n\tjmp 8f\n1:\n\tmov $11, %0\n\tjmp 8f\n2:\n\tmov $12, %0\n\tjmp 8f\n9:\n\tmov $99, %0\n8:\n"
                     "\t.section .rodata\n\t.p2align 2\ntable%=:\n\t.long 0b - table%=, 1b - table%=, 2b - table%=\n\t.previous"
                     : "=&r"(result) : "r"(selector) : "rcx", "rdx", "cc");
    return result;
}

static u64 bits(u64 first, u64 second) {
    u64 result;
    __asm__ volatile("tzcnt %k1, %%ecx\n\tlea one%=(%%rip), %%rdx\n\tmovslq (%%rdx,%%rcx,4), %%rcx\n\tadd %%rdx, %%rcx\n\tjmp *%%rcx\n"
                     "0:\n\tmov $1, %0\n\tjmp 5f\n1:\n\tmov $2, %0\n\tjmp 5f\n2:\n\tmov $3, %0\n5:\n"
                     "\ttzcnt %k2, %%ecx\n\tlea two%=(%%rip), %%rdx\n\tmovslq (%%rdx,%%rcx,4), %%rcx\n\tadd %%rdx, %%rcx\n\tjmp *%%rcx\n"
                     "6:\n\tadd $10, %0\n\tjmp 8f\n7:\n\tadd $20, %0\n\tjmp 8f\n3:\n\tadd $30, %0\n8:\n"
                     "\t.section .rodata\n\t.p2align 2\none%=:\n\t.long 0b - one%=, 1b - one%=, 2b - one%=\n"
                     "two%=:\n\t.long 6b - two%=, 7b - two%=, 3b - two%=\n\t.previous"
                     : "=&r"(result) : "r"(first), "r"(second) : "rcx", "rdx", "cc");
    return result;
}

static void tables(void) {
    static const char table[] = "abcdef";
    u64 al = 3;
    __asm__ volatile("xlatb" : "+a"(al) : "b"(table) : "memory");
    same(30, (i64)(al & 0xff), 'd');
    static volatile u64 selectors[2] = {2, 7};
    same(31, (i64)jump(selectors[0]), 12);
    same(32, (i64)jump(selectors[1]), 99);
    static volatile u64 masks[2] = {4, 2};
    same(33, (i64)bits(masks[0], masks[1]), 23);
}

static void frames(void) {
    u64 before, rsp, rbp, frame, outer;
    __asm__ volatile("mov %%rsp, %0\n\tenter $32, $0\n\tmov %%rsp, %1\n\tmov %%rbp, %2\n\tleave" : "=&r"(before), "=&r"(rsp), "=&r"(rbp) : : "rbp", "memory");
    same(40, (i64)(before - rsp), 40);
    same(41, (i64)(before - rbp), 8);
    static u64 buffer[2] = {0x1234, 0};
    __asm__ volatile("mov %%rbp, %%r8\n\tlea 8(%5), %%rbp\n\tmov %%rsp, %0\n\tenter $16, $2\n\tmov %%rsp, %1\n\tmov %%rbp, %2\n\tmov 16(%%rsp), %3\n\tmov 24(%%rsp), %4\n\tleave\n\tmov %%r8, %%rbp"
                     : "=&r"(before), "=&r"(rsp), "=&r"(rbp), "=&r"(frame), "=&r"(outer) : "r"(buffer) : "r8", "rbp", "memory");
    same(42, (i64)(before - rsp), 40);
    same(43, (i64)(before - rbp), 8);
    same(44, (i64)frame, (i64)(before - 8));
    same(45, (i64)outer, 0x1234);
}

static void halves(void) {
    static const float input[8] = {1.0f, -2.5f, 65504.0f, 0.1f, 0.5f, -0.0f, 3.0f, 1e-8f};
    u16 output[8];
    __asm__ volatile("vmovups %1, %%xmm0\n\tvcvtps2ph $0, %%xmm0, %0" : "=m"(*(u64*)output) : "m"(*(const float(*)[4])input) : "xmm0");
    same(50, output[0], 0x3c00);
    same(51, output[1], 0xc100);
    same(52, output[2], 0x7bff);
    same(53, output[3], 0x2e66);
    __asm__ volatile("vmovups %1, %%ymm0\n\tvcvtps2ph $4, %%ymm0, %%xmm1\n\tvmovdqu %%xmm1, %0\n\tvzeroupper" : "=m"(*(u16(*)[8])output) : "m"(*(const float(*)[8])input) : "xmm0", "xmm1");
    same(54, output[4], 0x3800);
    same(55, output[5], 0x8000);
    same(56, output[6], 0x4200);
    same(57, output[7], 0x0000);
    float back[8];
    __asm__ volatile("vcvtph2ps %1, %%ymm2\n\tvmovups %%ymm2, %0\n\tvzeroupper" : "=m"(*(float(*)[8])back) : "m"(*(const u16(*)[8])output) : "xmm2");
    same(58, back[1] == -2.5f, 1);
    same(59, back[2] == 65504.0f, 1);
    same(60, back[3] == 0.0999755859375f, 1);
    same(61, back[6] == 3.0f, 1);
}

static void environment(void) {
    u32 env[7];
    u16 word = 0x0c7f;
    __asm__ volatile("fninit\n\tfldcw %1\n\tfnstenv %0" : "=m"(*(u32(*)[7])env) : "m"(word));
    same(70, env[0] & 0xffff, 0x0c7f);
    word = 0x037f;
    u16 control = 0;
    __asm__ volatile("fldcw %1\n\tfldenv %2\n\tfnstcw %0" : "=m"(control) : "m"(word), "m"(*(const u32(*)[7])env));
    same(71, control, 0x0c7f);
    word = 0x037f;
    __asm__ volatile("fldcw %0" : : "m"(word));
}

static void simd(void) {
    static unsigned char bytes[32], threes[32];
    for (int i = 0; i < 32; ++i) {
        bytes[i] = (unsigned char)i;
        threes[i] = 3;
    }
    u64 sums[4];
    __asm__ volatile("movdqu %1, %%xmm0\n\tmovdqu %2, %%xmm1\n\tpsadbw %%xmm1, %%xmm0\n\tmovdqu %%xmm0, %0" : "=m"(*(u64(*)[2])sums) : "m"(*(u8(*)[16])bytes), "m"(*(u8(*)[16])threes) : "xmm0", "xmm1");
    same(80, (i64)sums[0], 16);
    same(81, (i64)sums[1], 68);
    __asm__ volatile("vmovdqu %1, %%ymm0\n\tvmovdqu %2, %%ymm1\n\tvpsadbw %%ymm1, %%ymm0, %%ymm2\n\tvmovdqu %%ymm2, %0\n\tvzeroupper" : "=m"(sums) : "m"(bytes), "m"(threes) : "xmm0", "xmm1", "xmm2");
    same(82, (i64)sums[0], 16);
    same(83, (i64)sums[3], 196);
    static const u8 unsignedBytes[16] = {200, 100, 255, 255};
    static const signed char signedBytes[16] = {-128, 127, 127, 127};
    short words[8];
    __asm__ volatile("vmovdqu %1, %%xmm0\n\tvpmaddubsw %2, %%xmm0, %%xmm1\n\tvmovdqu %%xmm1, %0" : "=m"(words) : "m"(unsignedBytes), "m"(signedBytes) : "xmm0", "xmm1");
    same(84, words[0], -12900);
    same(85, words[1], 32767);
    static const u16 candidates[8] = {9, 4, 7, 4, 10, 11, 12, 13};
    u32 position[4];
    __asm__ volatile("phminposuw %1, %%xmm0\n\tmovdqu %%xmm0, %0" : "=m"(position) : "m"(candidates) : "xmm0");
    same(86, position[0], 0x10004);
    same(87, position[1] | position[2] | position[3], 0);
    static const float left[4] = {1, 2, 3, 4}, right[4] = {10, 20, 30, 40};
    float out[4];
    __asm__ volatile("vmovups %1, %%xmm0\n\tvhaddps %2, %%xmm0, %%xmm1\n\tvmovups %%xmm1, %0" : "=m"(out) : "m"(left), "m"(right) : "xmm0", "xmm1");
    same(88, out[0] == 3 && out[1] == 7 && out[2] == 30 && out[3] == 70, 1);
    __asm__ volatile("vmovups %1, %%xmm0\n\tvaddsubps %2, %%xmm0, %%xmm1\n\tvmovups %%xmm1, %0" : "=m"(out) : "m"(left), "m"(right) : "xmm0", "xmm1");
    same(89, out[0] == -9 && out[1] == 22 && out[2] == -27 && out[3] == 44, 1);
    static const double doubles[2] = {1.5, -2.25};
    __asm__ volatile("vcvtpd2psx %1, %%xmm0\n\tvmovups %%xmm0, %0" : "=m"(out) : "m"(doubles) : "xmm0");
    same(90, out[0] == 1.5f && out[1] == -2.25f && out[2] == 0 && out[3] == 0, 1);
    static const float tenths[4] = {0.1f, -0.1f, 0.1f, -0.1f};
    u16 halves[4];
    __asm__ volatile("vmovups %1, %%xmm0\n\tvcvtps2ph $1, %%xmm0, %0" : "=m"(*(u64*)halves) : "m"(tenths) : "xmm0");
    same(91, halves[0], 0x2e66);
    same(92, halves[1], 0xae67);
    __asm__ volatile("vmovups %1, %%xmm0\n\tvcvtps2ph $2, %%xmm0, %0" : "=m"(*(u64*)halves) : "m"(tenths) : "xmm0");
    same(93, halves[0], 0x2e67);
    same(94, halves[1], 0xae66);
    __asm__ volatile("vmovups %1, %%xmm0\n\tvcvtps2ph $3, %%xmm0, %0" : "=m"(*(u64*)halves) : "m"(tenths) : "xmm0");
    same(95, halves[0], 0x2e66);
    same(96, halves[1], 0xae66);
    u32 control = 0x9fc0, stored = 0;
    __asm__ volatile("vldmxcsr %1\n\tvstmxcsr %0" : "=m"(stored) : "m"(control));
    same(97, stored, 0x9fc0);
    control = 0x1f80;
    __asm__ volatile("ldmxcsr %1\n\tstmxcsr %0" : "=m"(stored) : "m"(control));
    same(98, stored, 0x1f80);
    static _Alignas(32) u64 target[4];
    __asm__ volatile("vmovdqu %1, %%ymm0\n\tvmovntdq %%ymm0, %0\n\tsfence\n\tvzeroupper" : "=m"(target) : "m"(bytes) : "xmm0");
    same(99, (i64)target[3], 0x1f1e1d1c1b1a1918ll);
    u64 lanes[2] = {0x1234567890abcdefull, 0x1111111111111111ull}, fields[2], result[2];
    __asm__ volatile("movdqu %1, %%xmm0\n\textrq $8, $16, %%xmm0\n\tmovdqu %%xmm0, %0" : "=m"(result) : "m"(lanes) : "xmm0");
    same(34, (i64)result[0], 0xabcd);
    fields[0] = 4 | 4 << 8;
    __asm__ volatile("movdqu %1, %%xmm0\n\tmovdqu %2, %%xmm1\n\textrq %%xmm1, %%xmm0\n\tmovdqu %%xmm0, %0" : "=m"(result) : "m"(lanes), "m"(fields) : "xmm0", "xmm1");
    same(35, (i64)result[0], 0xe);
    u64 ones[2] = {~0ull, 0}, byte[2] = {0x5a, 0};
    __asm__ volatile("movdqu %1, %%xmm0\n\tmovdqu %2, %%xmm1\n\tinsertq $16, $8, %%xmm1, %%xmm0\n\tmovdqu %%xmm0, %0" : "=m"(result) : "m"(ones), "m"(byte) : "xmm0", "xmm1");
    same(36, (i64)result[0], (i64)0xffffffffff5affffull);
    u64 zero[2] = {0, 0};
    fields[0] = 3;
    fields[1] = 4 | 60 << 8;
    __asm__ volatile("movdqu %1, %%xmm0\n\tmovdqu %2, %%xmm1\n\tinsertq %%xmm1, %%xmm0\n\tmovdqu %%xmm0, %0" : "=m"(result) : "m"(zero), "m"(fields) : "xmm0", "xmm1");
    same(37, (i64)result[0], 0x3000000000000000ll);
}

void _start(void* block) {
    (void)block;
    loops();
    rotates();
    flags();
    tables();
    frames();
    halves();
    environment();
    simd();
    if (first < 0) {
        puts("misc ok");
        exit(0);
    }
    char message[] = "misc fail 00";
    message[10] = (char)('0' + first / 10 % 10);
    message[11] = (char)('0' + first % 10);
    puts(message);
    exit(1);
}
