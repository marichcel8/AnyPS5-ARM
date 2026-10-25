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

static void tables(void) {
    static const char table[] = "abcdef";
    u64 al = 3;
    __asm__ volatile("xlatb" : "+a"(al) : "b"(table) : "memory");
    same(30, (i64)(al & 0xff), 'd');
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

void _start(void* block) {
    (void)block;
    loops();
    rotates();
    flags();
    tables();
    frames();
    halves();
    environment();
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
