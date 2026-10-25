// aps5_host_bridge(GuestState* state, uint64_t target): calls a host (AAPCS64) function with the
// guest's SysV argument registers. rdi, rsi, rdx, rcx, r8, r9 become x0-x5, the guest stack
// arguments 7 and 8 become x6 and x7, and xmm0-7 become q0-q7. x0/x1 return into rax/rdx and
// q0/q1 into xmm0/xmm1. The guest return address is at [rsp], so stack arguments start at rsp + 8.
    .text
    .p2align 2
    .globl _aps5_host_bridge
_aps5_host_bridge:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x0
    mov x16, x1
    add x9, x19, #160
    ldr q0, [x9, #0]
    ldr q1, [x9, #32]
    ldr q2, [x9, #64]
    ldr q3, [x9, #96]
    ldr q4, [x9, #128]
    ldr q5, [x9, #160]
    ldr q6, [x9, #192]
    ldr q7, [x9, #224]
    ldr x10, [x19, #32]
    ldr x6, [x10, #8]
    ldr x7, [x10, #16]
    ldr x0, [x19, #56]
    ldr x1, [x19, #48]
    ldr x2, [x19, #16]
    ldr x3, [x19, #8]
    ldr x4, [x19, #64]
    ldr x5, [x19, #72]
    blr x16
    str x0, [x19, #0]
    str x1, [x19, #16]
    add x9, x19, #160
    str q0, [x9, #0]
    str q1, [x9, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// aps5_enter: a host (AAPCS64) call of a guest function, reached through the fault redirect with the
// translated function in x16. Saves x0-x7 and q0-q7, runs aps5_enter_guest(frame, x16) and returns
// its x0/x1 and q0/q1 to the host caller, whose return address is still in x30.
    .p2align 2
    .globl _aps5_enter
_aps5_enter:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    sub sp, sp, #192
    stp x0, x1, [sp, #0]
    stp x2, x3, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp q0, q1, [sp, #64]
    stp q2, q3, [sp, #96]
    stp q4, q5, [sp, #128]
    stp q6, q7, [sp, #160]
    mov x0, sp
    mov x1, x16
    bl _aps5_enter_guest
    ldp x0, x1, [sp, #0]
    ldp q0, q1, [sp, #64]
    add sp, sp, #192
    ldp x29, x30, [sp], #16
    ret

// aps5_call(GuestState* state, uint64_t target): every translated call or jump to an address the
// translator could not resolve. A call to the libc setjmp saves the registers of its translated
// caller, so that aps5_native_restore can later continue there as if this call returned again;
// everything else goes to aps5_call_impl.
    .p2align 2
    .globl _aps5_call
_aps5_call:
    adrp x2, _aps5_setjmp_target@PAGE
    ldr x2, [x2, _aps5_setjmp_target@PAGEOFF]
    cmp x1, x2
    b.eq Lsetjmp
    b _aps5_call_impl
Lsetjmp:
    stp x29, x30, [sp, #-16]!
    bl _aps5_prepare_jump
    ldp x29, x30, [sp], #16
    stp x19, x20, [x0, #0]
    stp x21, x22, [x0, #16]
    stp x23, x24, [x0, #32]
    stp x25, x26, [x0, #48]
    stp x27, x28, [x0, #64]
    stp x29, x30, [x0, #80]
    mov x2, sp
    str x2, [x0, #96]
    stp d8, d9, [x0, #104]
    stp d10, d11, [x0, #120]
    stp d12, d13, [x0, #136]
    stp d14, d15, [x0, #152]
    ret

// aps5_native_restore(const uint64_t* registers): continues after the aps5_call that saved them.
    .p2align 2
    .globl _aps5_native_restore
_aps5_native_restore:
    ldp x19, x20, [x0, #0]
    ldp x21, x22, [x0, #16]
    ldp x23, x24, [x0, #32]
    ldp x25, x26, [x0, #48]
    ldp x27, x28, [x0, #64]
    ldp x29, x30, [x0, #80]
    ldr x2, [x0, #96]
    mov sp, x2
    ldp d8, d9, [x0, #104]
    ldp d10, d11, [x0, #120]
    ldp d12, d13, [x0, #136]
    ldp d14, d15, [x0, #152]
    ret
