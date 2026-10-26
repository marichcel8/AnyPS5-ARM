extern "C" int puts(const char*);
extern "C" void exit(int);

struct Error {
    int code;
};

struct Guard {
    int* counter;
    ~Guard() { ++*counter; }
};

static int first = -1;

static void same(int id, long long got, long long want) {
    if (got != want && first < 0) first = id;
}

[[gnu::noinline]] static void thrower(int value) {
    if (value > 0) throw Error{value};
}

[[gnu::noinline]] static int middle(int value, int* cleaned) {
    Guard guard{cleaned};
    thrower(value);
    return 0;
}

[[gnu::noinline]] static int recurse(int depth, int* cleaned) {
    if (depth == 0) return middle(7, cleaned);
    Guard guard{cleaned};
    return recurse(depth - 1, cleaned) + 1;
}

[[gnu::noinline]] static int rethrown() {
    try {
        thrower(5);
    } catch (Error& error) {
        error.code += 1;
        throw;
    }
    return 0;
}

[[gnu::noinline]] static int caught_in_loop() {
    int sum = 0;
    for (int i = 1; i <= 100; ++i) {
        try {
            thrower(i);
        } catch (const Error& error) {
            sum += error.code;
        }
    }
    return sum;
}

extern "C" void _start(void* block) {
    (void)block;
    int cleaned = 0;
    int code = 0;
    try {
        recurse(3, &cleaned);
    } catch (const Error& error) {
        code = error.code;
    }
    same(1, code, 7);
    same(2, cleaned, 4);

    code = 0;
    try {
        rethrown();
    } catch (Error error) {
        code = error.code;
    }
    same(3, code, 6);

    code = 0;
    try {
        thrower(2);
    } catch (...) {
        code = 100;
    }
    same(4, code, 100);

    code = 0;
    try {
        throw Error{9};
    } catch (Error error) {
        code = error.code;
    }
    same(5, code, 9);

    same(6, caught_in_loop(), 5050);

    if (first < 0) {
        puts("exceptions ok");
        exit(0);
    }
    char message[] = "exceptions fail 00";
    message[16] = static_cast<char>('0' + first / 10 % 10);
    message[17] = static_cast<char>('0' + first % 10);
    puts(message);
    exit(1);
}
