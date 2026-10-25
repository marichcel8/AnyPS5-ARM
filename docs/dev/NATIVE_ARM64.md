# Native arm64 (no Rosetta): design and plan

Branch `native-arm64`. How to build and use it: [docs/native](../native/README.md). The goal is to run converted titles as arm64 processes, with no Rosetta. `main` keeps the Rosetta port ([MACOS_PORT.md](MACOS_PORT.md)).

## Where Rosetta sits today

- **Guest code:** the relinked guest x86-64 code calls host prx functions directly. dyld writes host addresses into the guest's own GOT slots, and both sides use the SysV calling convention (`APS5_VABI` is empty off Windows).
- **Host code:** host code calls guest code through plain function pointers:
  - the process entry (`Aps5StartGuest`)
  - module init/fini
  - threads (`scePthreadCreate`), Ult, Fiber
  - `atexit`, `__cxa_atexit`, thread destructors, TSD destructors, once
  - signals, C++ exception callbacks and the DWARF unwinder
  - qsort/bsearch, locale facets, guest malloc replacement
  - AvPlayer, Ngs2, Font, Agc command buffer callbacks
- **Rosetta's role:** it runs all of it, guest code and prx libraries alike, because a process has one architecture.

## Approach: translate guest code ahead of time

The relinker already decodes guest code (TLS scan, `--to-intel`). In the arm64 mode it translates every discovered guest function to arm64 at conversion time. The prx libraries build natively for arm64. A first `cmake -DCMAKE_OSX_ARCHITECTURES=arm64` build fails in only 6 of about 1250 compile steps:

- `setjmp`/`longjmp` asm
- the Fiber stack switch
- the guest TLS allocate asm
- TSC reads
- a few `__builtin_ia32_*` exports
- the 80-bit `long double` asserts

### Guest state

- **Registers:** translated code keeps the x86-64 architectural state in a per-thread `GuestState`: 16 GPRs, rip, flags, the 16 ymm registers, mxcsr, and the fs base.
- **Memory is shared:** guest memory is the host address space, so guest loads and stores are plain host loads and stores at the same addresses. No MMU is emulated, and guest pointers and host pointers are the same values.
- **Stack:** the guest stack is real memory. Guest `rsp` lives in `GuestState`, and translated functions use the native stack only for their own temporaries.

### Translation

- **Pipeline:** x86-64 is decoded with Capstone and lifted to LLVM IR, one LLVM function per guest function, operating on `GuestState`. LLVM optimizes it and emits arm64.
- **Flags:** flags are computed into `GuestState` fields; LLVM drops the dead ones.
- **SIMD:** SSE and AVX map to LLVM vector types, and 256-bit operations are lowered to two 128-bit NEON operations by LLVM.
- **x87:** the FPU stack is eight doubles in `GuestState` with a dynamic top, plus the control and status words (control word 0x037f at thread start). Loads, stores, arithmetic, comparisons (status word and `fcomi` flags), `fcmov`, rounding by the control word, transcendental instructions, BCD (`fbld`/`fbstp`) and 80-bit memory operands are lifted; 80-bit values are rounded to double precision, and tag words and exceptions are not modeled. `tests/x87_smoke.c` checks the set. `fnstenv`/`fldenv` keep the control and status words.
- **Function discovery:**
  - the ELF entry, `DT_INIT`/`INIT_ARRAY`
  - exported and local symbols
  - direct call targets
  - code addresses in relocations (`R_X86_64_RELATIVE` into executable segments)
  - recursive descent over basic blocks
- **Indirect jumps:** an indirect jump inside a function dispatches over that function's known block addresses.
- **Indirect calls:** they go through a runtime lookup of guest address to native function. An unknown target stops the process with its guest address.

### Calling across the boundary

- **Guest to host:** a call whose target is outside the guest code ranges is a host function. The bridge passes rdi, rsi, rdx, rcx, r8, r9 as x0–x5, guest stack arguments 7 and 8 as x6 and x7, and xmm0–7 as v0–v7. It returns x0/x1 into rax/rdx and v0/v1 into xmm0/xmm1. Both ABIs split integer and vector arguments into separate register files, so this covers every export with up to 8 integer and 8 vector arguments passed in registers. Variadic exports differ (Apple arm64 passes variadic arguments on the stack), and so do `va_list`, structure returns (rdi vs x8) and `long double`. Those exports need per-signature bridges; there are 26 variadic `APS5_VABI` exports.
- **Host to guest:** the guest image lives in a data segment, so a host function pointer to guest code faults when it is called. The runtime's SIGBUS/SIGSEGV handler looks the address up in the function table and resumes at `aps5_enter` with the translated function in x16, so the many callback sites (the process entry through `Aps5StartGuest`, thread entries, atexit, library callbacks) need no change. `aps5_enter` builds or reuses the thread's `GuestState` and TLS block, copies the AAPCS arguments into the SysV registers and the guest stack, pushes a return address, runs the translated function, and returns rax/rdx and xmm0/xmm1 as x0/x1 and q0/q1. Faults the redirect does not own go to the previous handler (the write watch chains the same way).
- **Variadic exports:** Apple arm64 passes variadic arguments on the stack, so a host `printf(fmt, ...)` cannot read the guest's registers. On arm64 the libc exports start a System V va_list over the running guest call instead (`aps5_guest_va_start` in the runtime, `GuestVarargs.hpp` in libc) and format with the same code the Windows build uses.
- **TLS:** `fs:` accesses read `GuestState.fsBase`, which is set per thread from the guest TLS allocator. The `gs:`/pthread-key stubs of the Rosetta mode are not used.

### Hard parts, in order of risk

1. **Non-local control flow:** guest `setjmp`/`longjmp` work (see below); C++ exception landing pads (a guest unwinder resuming mid-function), and the Fiber and Ult context switches. Translated functions nest native frames, so a guest transfer to an older frame must also unwind native frames. The plan is to resume through the thread's outermost guest entry, or to switch to a dispatch-loop model if that proves insufficient.
   `setjmp` and `longjmp` are recognized by address in `aps5_call` (`HostBridge.S`). `setjmp` saves the callee-saved registers, sp and return address of its translated caller in a per-thread table keyed by the guest `jmp_buf`, and stores the x86-64 registers in the buffer in the layout libc used under Rosetta. `longjmp` restores the x86-64 registers and the saved native ones, so the translated caller continues after its `setjmp` call; the native frames in between are discarded without running destructors. Registers a translated caller kept in native callee-saved registers have their value from `setjmp`; guest registers are reloaded from `GuestState` after every call, so only values outside it are affected.
2. **Memory ordering:** x86 is TSO and arm64 is weakly ordered. The hardware TSO mode of Apple silicon has no public interface. Guest loads and stores can be emitted as acquire and release (`LDAPR`/`STLR`); that costs throughput, so it is a translation option.
3. **Coverage:** every instruction the titles use must be lifted exactly. Each one gets differential tests against the same code running under Rosetta.
4. **Tooling at conversion time:** the converter would need LLVM and a linker, about 100 MB. A direct arm64 emitter is the alternative if that size is a problem.

## Building and running the translator

The translator builds on its own because it needs LLVM and Capstone (Homebrew `llvm` and `capstone`):

```sh
cmake -S tools/translator -B build-translator -G Ninja -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
ninja -C build-translator
python3 tools/translator/tests/run_native_fixtures.py build-translator build-arm64/core/libs/libs
```

`aps5-translator <eboot.elf> <guest.o>` writes one arm64 object with the lifted functions, the guest image and the function table. The image's relocated slots are symbolic, so the host linker binds the guest's imports to the NID exports of the arm64 prx libraries. Linking it with `libaps5-translator-runtime.a` and the needed `.prx` files gives a native executable.

### Large titles

Functions are found from the entry point, the symbols, the relocations and, when the ELF has a `PT_GNU_EH_FRAME`, every FDE of its unwind table. An FDE gives a function's exact range: exploration stays inside it, a jump to another function's start or out of the range is a tail call, and jump table entries outside it are dropped. Without this, functions ran on into their neighbours through tail jumps and misread jump tables; on Quake II the lifted code shrank from 5.1 to 3.1 million guest instructions.

A title's code is split into partitions of about 4k guest instructions. Each partition is lifted, optimized and compiled in its own LLVM context on its own thread, and `ld -r` joins the partition objects into the one `guest.o`. Partition objects are cached under `~/Library/Caches/AnyPS5/translator/<ELF hash>`, keyed by the translator binary, the ELF and the settings, so a repeated or interrupted run only compiles what is missing.

Three things keep very large guest functions from stalling LLVM:

- **Flag liveness.** A backward data flow over each function finds the arithmetic flags no later instruction reads; their stores (and the parity and adjust computations behind them) are not emitted. Flags are dead across calls, returns and jumps to other functions, as the System V ABI preserves none. `APS5_TRANSLATOR_ALL_FLAGS=1` keeps every flag store.
- **A small graph.** Only branch targets and instructions control reaches from elsewhere start a basic block. A jump through a bounded jump table switches over that table's targets only; every other indirect jump goes to one shared dispatch per function.
- **Guest registers in SSA.** In functions up to 2500 guest instructions, the lifter works on a copy of `GuestState` on the native stack and copies the fields the function uses back before each call, runtime call and return, and in again after a call returns. SROA turns the copy into SSA values, so alias analysis no longer sees one memory access per register use.
- **Optimization tiers.** Functions above 2000 guest instructions are not inlined. Above 2500 they keep `GuestState` in memory, skip the O2 pipeline for a few linear passes (SROA, EarlyCSE, SimplifyCFG, ADCE), and are compiled with fast instruction selection and register allocation, because the greedy allocator's live range splitting grows too fast on them.
- **Bounded memory.** Partitions run on threads with 256 MB stacks, because alias analysis recurses deeply on large phi webs. The threads together hold at most 3072 guest instructions per GB of RAM in flight (24576 on 8 GB), taken in order from the largest partition down. A function compiled with fast codegen counts an eighth of its size, because it needs far less memory than an optimized one, and a partition above the budget runs alone. On macOS the workers run at user-initiated QoS and freed heap pages go back to the system after each partition.

### Quick tier

`APS5_TRANSLATOR_TIER=quick` (the default of `package_native_app.py`) treats every function the profile does not show hot as cold: it gets the light passes, and a partition of only cold functions skips the O2 pipeline and is compiled at `-O0` (GlobalISel and the fast register allocator). Cold partitions compile about four times faster than optimized ones and give about three times the code. Quake II translates in 120 s instead of 342 s on four Linux cores. Each cold function counts its entries and loop iterations in a `{guest offset, count}` row of `__DATA,__aps5_prof`. The runtime of an executable translated this way adds the counts to the profile file every 30 seconds and at exit, one `<offset in hex> <count>` line per function. The next translation with the profile optimizes the functions with at least `APS5_TRANSLATOR_HOT` counts at O2 and without counters; only the partitions whose hot functions changed are compiled again. The profile only grows, so a function stays optimized once it was hot.

| Variable | Default | Meaning |
|---|---|---|
| `APS5_TRANSLATOR_TIER` | `full` | `quick` compiles cold functions quickly and profiles them |
| `APS5_TRANSLATOR_PROFILE` | `<cache>/profile.txt` | profile file of the quick tier, embedded in the executable |
| `APS5_TRANSLATOR_HOT` | 1000 | counts from which the quick tier optimizes a function |
| `APS5_TRANSLATOR_JOBS` | all CPU cores | threads; `max` also lifts the memory budget |
| `APS5_TRANSLATOR_PARTITION` | 4096 | guest instructions per partition |
| `APS5_TRANSLATOR_LARGE` / `_HUGE` / `_GIANT` | 2000 / 2500 / 2500 | no inlining / light pipeline and no register promotion / fast codegen, in guest instructions |
| `APS5_TRANSLATOR_BUDGET` | 3072 per GB of RAM | guest instructions compiled at once over all threads |
| `APS5_TRANSLATOR_CACHE` | `~/Library/Caches/AnyPS5/translator` | cache directory, `off` disables it |
| `APS5_TRANSLATOR_LD` | `ld` | linker that joins the partitions |
| `APS5_TRANSLATOR_VERIFY` | unset | verify each lifted partition, as the fixtures do |
| `APS5_TRANSLATOR_STATS` | unset | print the time of each partition and of each phase |

## Milestones

1. ✅ The prx libraries build for arm64. The x86-only parts throw until they are ported.
2. ✅ The smallest fixture (`import_fixture`: `puts` and `exit` through the GOT) runs as an arm64 process.
2b. ✅ DOOM (homebrew, `PPSA99666`) plays natively: launcher, attract demo and E1M1 at the game's 35 FPS, rendered through AGC, with no Rosetta in the process ([screenshot](../games/img/doom-native-e1m1.jpg)). What it took: SIMD/BMI lifting checked against Rosetta (`vector_smoke.c`), functions found through rip-relative code addresses and switch jump tables, guest TLS and image metadata (`__APS5DATA,__meta` for `dl_iterate_phdr`).
3. The C and C++ fixtures run: TLS, threads, modules, exceptions.
4. Breakout runs with video, pad and audio.
5. Dreaming Sarah runs.
