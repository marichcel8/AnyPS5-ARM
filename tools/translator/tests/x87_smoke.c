typedef unsigned long long u64; typedef long long i64; typedef unsigned short u16; typedef unsigned char u8;
int puts(const char*); void exit(int);

static int failures;
static int first = -1;

static void fail(int id) {
    if (first < 0) first = id;
    ++failures;
}

static void near(int id, double got, double want) {
    double difference = got - want;
    double scale = want < 0 ? -want : want;
    if (difference < 0) difference = -difference;
    if (scale < 1) scale = 1;
    if (!(difference <= 1e-12 * scale)) fail(id);
}

static void same(int id, i64 got, i64 want) {
    if (got != want) fail(id);
}

static u16 control(void) { u16 word; __asm__ volatile("fnstcw %0" : "=m"(word)); return word; }
static void set_control(u16 word) { __asm__ volatile("fldcw %0" : : "m"(word)); }
static double rndint(double x) { double r; __asm__ volatile("fldl %1\n\tfrndint\n\tfstpl %0" : "=m"(r) : "m"(x)); return r; }
static int fistp32(double x) { int r; __asm__ volatile("fldl %1\n\tfistpl %0" : "=m"(r) : "m"(x)); return r; }
static i64 fisttp64(double x) { i64 r; __asm__ volatile("fldl %1\n\tfisttpll %0" : "=m"(r) : "m"(x)); return r; }

#define UNARY(name, op) static double name(double x) { double r; __asm__ volatile("fldl %1\n\t" op "\n\tfstpl %0" : "=m"(r) : "m"(x)); return r; }
#define POPPING(name, op) static double name(double st1, double st0) { double r; __asm__ volatile("fldl %1\n\tfldl %2\n\t" op "\n\tfstpl %0" : "=m"(r) : "m"(st1), "m"(st0)); return r; }
#define KEEPING(name, op) static double name(double st1, double st0) { double r; __asm__ volatile("fldl %1\n\tfldl %2\n\t" op "\n\tfstpl %0\n\tfstp %%st(0)" : "=m"(r) : "m"(st1), "m"(st0)); return r; }
#define BELOW(name, op) static double name(double st1, double st0) { double r; __asm__ volatile("fldl %1\n\tfldl %2\n\t" op "\n\tfstp %%st(0)\n\tfstpl %0" : "=m"(r) : "m"(st1), "m"(st0)); return r; }
#define CONSTANT(name, op) static double name(void) { double r; __asm__ volatile(op "\n\tfstpl %0" : "=m"(r)); return r; }

UNARY(op_sqrt, "fsqrt") UNARY(op_sin, "fsin") UNARY(op_cos, "fcos") UNARY(op_f2xm1, "f2xm1") UNARY(op_chs, "fchs") UNARY(op_abs, "fabs")
POPPING(op_fpatan, "fpatan") POPPING(op_fyl2x, "fyl2x") POPPING(op_fyl2xp1, "fyl2xp1")
POPPING(op_faddp, ".byte 0xde, 0xc1") POPPING(op_fmulp, ".byte 0xde, 0xc9") POPPING(op_fsubp, ".byte 0xde, 0xe9") POPPING(op_fsubrp, ".byte 0xde, 0xe1")
POPPING(op_fdivp, ".byte 0xde, 0xf9") POPPING(op_fdivrp, ".byte 0xde, 0xf1")
KEEPING(op_fscale, "fscale") KEEPING(op_fprem, "fprem") KEEPING(op_fprem1, "fprem1") KEEPING(op_fsub0, ".byte 0xd8, 0xe1") KEEPING(op_fdivr0, ".byte 0xd8, 0xf9")
KEEPING(op_fxch, "fxch") BELOW(op_fsub1, ".byte 0xdc, 0xe9") BELOW(op_fdivr1, ".byte 0xdc, 0xf1") BELOW(op_fst1, "fst %%st(1)")
CONSTANT(op_fldpi, "fldpi") CONSTANT(op_fldl2e, "fldl2e") CONSTANT(op_fldln2, "fldln2") CONSTANT(op_fldlg2, "fldlg2") CONSTANT(op_fldl2t, "fldl2t")
CONSTANT(op_fld1, "fld1") CONSTANT(op_fldz, "fldz")

static double fld_st1(double a, double b) {
    double r;
    __asm__ volatile("fldl %1\n\tfldl %2\n\tfld %%st(1)\n\tfstpl %0\n\tfstp %%st(0)\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b));
    return r;
}

static double fiadd(double x, int m) { double r; __asm__ volatile("fldl %1\n\tfiaddl %2\n\tfstpl %0" : "=m"(r) : "m"(x), "m"(m)); return r; }
static double fisubr(double x, int m) { double r; __asm__ volatile("fldl %1\n\tfisubrl %2\n\tfstpl %0" : "=m"(r) : "m"(x), "m"(m)); return r; }
static double fidiv(double x, short m) { double r; __asm__ volatile("fldl %1\n\tfidivs %2\n\tfstpl %0" : "=m"(r) : "m"(x), "m"(m)); return r; }
static double fsubr64(double x, double m) { double r; __asm__ volatile("fldl %1\n\tfsubrl %2\n\tfstpl %0" : "=m"(r) : "m"(x), "m"(m)); return r; }
static double fmul32(double x, float m) { double r; __asm__ volatile("fldl %1\n\tfmuls %2\n\tfstpl %0" : "=m"(r) : "m"(x), "m"(m)); return r; }

static double ptan(double x) { double r; __asm__ volatile("fldl %1\n\tfptan\n\tfstp %%st(0)\n\tfstpl %0" : "=m"(r) : "m"(x)); return r; }
static void sincos(double x, double* s, double* c) { __asm__ volatile("fldl %2\n\tfsincos\n\tfstpl %1\n\tfstpl %0" : "=m"(*s), "=m"(*c) : "m"(x)); }
static void xtract(double x, double* significand, double* exponent) { __asm__ volatile("fldl %2\n\tfxtract\n\tfstpl %0\n\tfstpl %1" : "=m"(*significand), "=m"(*exponent) : "m"(x)); }

static u16 examine(double x) { u16 word; __asm__ volatile("fldl %1\n\tfxam\n\tfnstsw %%ax\n\tfstp %%st(0)" : "=a"(word) : "m"(x)); return word & 0x4700; }
static u16 compare(double a, double b) { u16 word; __asm__ volatile("fldl %2\n\tfldl %1\n\tfcompp\n\tfnstsw %%ax" : "=a"(word) : "m"(a), "m"(b)); return word & 0x4500; }
static u16 test_zero(double a) { u16 word; __asm__ volatile("fldl %1\n\tftst\n\tfnstsw %%ax\n\tfstp %%st(0)" : "=a"(word) : "m"(a)); return word & 0x4500; }
static u16 compare_m64(double a, double b) { u16 word; __asm__ volatile("fldl %1\n\tfcompl %2\n\tfnstsw %%ax" : "=a"(word) : "m"(a), "m"(b)); return word & 0x4500; }
static u8 below(double a, double b) { u8 r; __asm__ volatile("fldl %2\n\tfldl %1\n\tfcomip %%st(1), %%st\n\tfstp %%st(0)\n\tsetb %0" : "=q"(r) : "m"(a), "m"(b) : "cc"); return r; }
static u8 unordered(double a, double b) { u8 r; __asm__ volatile("fldl %2\n\tfldl %1\n\tfucomip %%st(1), %%st\n\tfstp %%st(0)\n\tsetp %0" : "=q"(r) : "m"(a), "m"(b) : "cc"); return r; }
static double select_below(double a, double b, double x, double y) {
    double r;
    __asm__ volatile("fldl %3\n\tfldl %4\n\tfldl %2\n\tfldl %1\n\tfcomip %%st(1), %%st\n\tfstp %%st(0)\n\tfcmovb %%st(1), %%st\n\tfstpl %0\n\tfstp %%st(0)"
                     : "=m"(r) : "m"(a), "m"(b), "m"(x), "m"(y) : "cc");
    return r;
}
static u16 status_after_push(void) { u16 word; __asm__ volatile("fld1\n\tfnstsw %0\n\tfstp %%st(0)" : "=m"(word)); return word & 0x3800; }

static void store80(double x, u8* bytes) { __asm__ volatile("fldl %1\n\tfstpt %0" : "=m"(*(long double*)bytes) : "m"(x)); }
static double load80(const u8* bytes) { double r; __asm__ volatile("fldt %1\n\tfstpl %0" : "=m"(r) : "m"(*(const long double*)bytes)); return r; }
static void storebcd(i64 x, u8* bytes) { __asm__ volatile("fildll %1\n\tfbstp %0" : "=m"(*(long double*)bytes) : "m"(x)); }
static double loadbcd(const u8* bytes) { double r; __asm__ volatile("fbld %1\n\tfstpl %0" : "=m"(r) : "m"(*(const long double*)bytes)); return r; }

static volatile long double la = 1.25L, lb = -3.5L, lc = 7.0L;
static volatile int vi = -17;
static volatile i64 vl = -1234567890123ll;
static volatile u64 vu = 0xf000000000000000ull;

static long double lmax(long double a, long double b) { return a > b ? a : b; }
static int lless(long double a, long double b) { return a < b; }

static void control_word(void) {
    same(1, control(), 0x037f);
    __asm__ volatile("wait\n\tfwait");
    static const double values[] = {2.5, -2.5, 2.7, -2.7};
    static const int expected[4][4] = {{2, -2, 3, -3}, {2, -3, 2, -3}, {3, -2, 3, -2}, {2, -2, 2, -2}};
    for (int mode = 0; mode < 4; ++mode) {
        set_control((u16)(0x037f | mode << 10));
        same(2, control(), 0x037f | mode << 10);
        for (int i = 0; i < 4; ++i) {
            same(3 + mode, (i64)rndint(values[i]), expected[mode][i]);
            same(7 + mode, fistp32(values[i]), expected[mode][i]);
        }
    }
    set_control(0x037f);
    same(11, fisttp64(2.7), 2);
    same(12, fisttp64(-2.7), -2);
    same(13, fistp32(3e10), (int)0x80000000u);
    set_control(0x0f7f);
    __asm__ volatile("fninit");
    same(14, control(), 0x037f);
    same(15, status_after_push(), 0x3800);
}

static void arithmetic(void) {
    near(20, op_faddp(10, 4), 14);
    near(21, op_fmulp(10, 4), 40);
    near(22, op_fsubp(10, 4), 6);
    near(23, op_fsubrp(10, 4), -6);
    near(24, op_fdivp(10, 4), 2.5);
    near(25, op_fdivrp(10, 4), 0.4);
    near(26, op_fsub0(10, 4), -6);
    near(27, op_fdivr0(10, 4), 2.5);
    near(28, op_fsub1(10, 4), 6);
    near(29, op_fdivr1(10, 4), 0.4);
    near(30, op_fxch(10, 4), 10);
    near(31, op_fst1(10, 4), 4);
    near(32, fld_st1(10, 4), 10);
    near(33, fiadd(2.5, -7), -4.5);
    near(34, fisubr(2.5, 10), 7.5);
    near(35, fidiv(10, 4), 2.5);
    near(36, fsubr64(2.5, 10), 7.5);
    near(37, fmul32(2.5, 0.5f), 1.25);
    near(38, op_chs(3), -3);
    near(39, op_abs(-3), 3);
    near(40, op_sqrt(2), 1.4142135623730951);
    near(41, op_fscale(4, 3), 48);
    near(42, op_fprem(2, 7.5), 1.5);
    near(43, op_fprem1(2, 7.5), -0.5);
    double significand, exponent;
    xtract(12, &significand, &exponent);
    near(44, significand, 1.5);
    near(45, exponent, 3);
}

static void transcendental(void) {
    near(50, op_sin(0.5), 0.479425538604203);
    near(51, op_cos(0.5), 0.8775825618903728);
    near(52, ptan(0.5), 0.5463024898437905);
    double s, c;
    sincos(0.5, &s, &c);
    near(53, s, 0.479425538604203);
    near(54, c, 0.8775825618903728);
    near(55, op_fpatan(1, 1), 0.7853981633974483);
    near(56, op_fyl2x(3, 8), 9);
    near(57, op_fyl2xp1(1, 1), 1);
    near(58, op_f2xm1(0.5), 0.41421356237309515);
    near(59, op_fldpi(), 3.141592653589793);
    near(60, op_fldl2e(), 1.4426950408889634);
    near(61, op_fldln2(), 0.6931471805599453);
    near(62, op_fldlg2(), 0.3010299956639812);
    near(63, op_fldl2t(), 3.321928094887362);
    near(64, op_fld1(), 1);
    near(65, op_fldz(), 0);
}

static void comparisons(void) {
    volatile double zero = 0.0;
    const double nan = zero / zero, inf = 1.0 / zero;
    same(70, examine(-0.0), 0x4200);
    same(71, examine(1.0), 0x0400);
    same(72, examine(-inf), 0x0700);
    same(73, examine(nan) & 0x4500, 0x0100);
    same(74, examine(4.9e-324), 0x0400);
    same(75, compare(1, 2), 0x0100);
    same(76, compare(2, 2), 0x4000);
    same(77, compare(3, 2), 0);
    same(78, compare(nan, 2), 0x4500);
    same(79, test_zero(-1), 0x0100);
    same(80, compare_m64(5, 5), 0x4000);
    same(81, below(1, 2), 1);
    same(82, below(2, 1), 0);
    same(83, unordered(nan, 1), 1);
    same(84, unordered(1, 1), 0);
    near(85, select_below(1, 2, 10, 20), 10);
    near(86, select_below(2, 1, 10, 20), 20);
}

static void memory_formats(void) {
    u8 bytes[10];
    store80(1.5, bytes);
    same(90, bytes[7], 0xc0);
    same(91, bytes[8] | bytes[9] << 8, 0x3fff);
    near(92, load80(bytes), 1.5);
    store80(-3.0, bytes);
    same(93, bytes[9], 0xc0);
    near(94, load80(bytes), -3.0);
    store80(1e-310, bytes);
    near(95, load80(bytes) * 1e300, 1e-10);
    storebcd(123456789012345678ll, bytes);
    same(96, bytes[0], 0x78);
    same(97, bytes[8], 0x12);
    same(98, bytes[9], 0);
    near(99, loadbcd(bytes), 123456789012345678.0);
    storebcd(-42, bytes);
    same(100, bytes[0], 0x42);
    same(101, bytes[9], 0x80);
    near(102, loadbcd(bytes), -42);
}

static void long_double(void) {
    near(110, (double)(la + lb), -2.25);
    near(111, (double)(la * lb), -4.375);
    near(112, (double)(lc / la), 5.6);
    near(113, (double)(la - lc), -5.75);
    near(114, (double)(long double)vi, -17);
    near(115, (double)(long double)vl, -1234567890123.0);
    near(116, (double)(long double)vu, 17293822569102704640.0);
    same(117, (int)lb, -3);
    same(118, (i64)(lc * 1e12L), 7000000000000ll);
    same(119, (u64)(lc * 2e18L), 14000000000000000000ull);
    near(120, (double)lmax(la, lb), 1.25);
    near(121, (double)lmax(lb, lc), 7);
    same(122, lless(la, lb), 0);
    same(123, lless(lb, la), 1);
    same(124, control(), 0x037f);
}

void _start(void* block) {
    (void)block;
    control_word();
    arithmetic();
    transcendental();
    comparisons();
    memory_formats();
    long_double();
    if (failures == 0) {
        puts("x87 ok");
        exit(0);
    }
    char text[] = "x87 fail 000";
    text[9] = (char)('0' + first / 100 % 10);
    text[10] = (char)('0' + first / 10 % 10);
    text[11] = (char)('0' + first % 10);
    puts(text);
    exit(1);
}
