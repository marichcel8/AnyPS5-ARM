typedef unsigned long long u64;
int puts(const char*); void exit(int);
int setjmp(void*) __attribute__((returns_twice));
void longjmp(void*, int) __attribute__((noreturn));

static int first = -1;
static u64 outer[12], inner[12], spare[12];
static volatile int depth_reached;

static void same(int id, long long got, long long want) {
    if (got != want && first < 0) first = id;
}

__attribute__((noinline)) static int descend(int levels, int value) {
    volatile int pad[8];
    pad[0] = levels;
    if (levels == 0) {
        depth_reached = 1;
        longjmp(outer, value);
    }
    int below = descend(levels - 1, value);
    return below + pad[0];
}

__attribute__((noinline)) static void jump_if(int condition, u64* buffer, int value) {
    if (condition) longjmp(buffer, value);
}

void _start(void* block) {
    (void)block;
    volatile int marker = 0;
    int result = setjmp(outer);
    if (result == 0) {
        marker = 1;
        same(1, descend(40, 7), -1);
    }
    same(2, result, 7);
    same(3, marker, 1);
    same(4, depth_reached, 1);

    volatile int passes = 0;
    int zero = setjmp(inner);
    if (passes == 0) {
        passes = 1;
        same(5, zero, 0);
        jump_if(1, inner, 0);
    }
    same(6, zero, 1);
    same(7, passes, 1);

    volatile long long sum = 0;
    for (int i = 0; i < 1000; ++i) {
        int r = setjmp(spare);
        if (r == 0) jump_if(1, spare, i + 1);
        sum += r;
    }
    same(8, sum, 500500);

    volatile int stage = 0;
    int first_return = setjmp(outer);
    if (first_return == 0) {
        stage = 1;
        int nested = setjmp(inner);
        if (nested == 0) {
            stage = 2;
            jump_if(1, inner, 3);
        }
        same(9, nested, 3);
        stage = 3;
        jump_if(1, outer, 5);
    }
    same(10, first_return, 5);
    same(11, stage, 3);

    if (first < 0) {
        puts("jumps ok");
        exit(0);
    }
    char message[] = "jumps fail 00";
    message[11] = (char)('0' + first / 10 % 10);
    message[12] = (char)('0' + first % 10);
    puts(message);
    exit(1);
}
