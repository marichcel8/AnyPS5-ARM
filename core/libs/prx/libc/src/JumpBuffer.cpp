#include "prx/libc/include/General.hpp"
#include "prx/libc/include/general/AsmMacros.hpp"

// setjmp/longjmp must capture the guest's own frame, so they are written directly in assembly with
// the guest (System V) calling convention. The saved state fits the guest's 96-byte jmp_buf:
// return address, rbx, rsp, rbp, r12-r15, MXCSR and the x87 control word.
#if defined(__x86_64__)
asm(".text\n"
    APS5_ASM_FUNCTION("setjmp_nid_postfix")
    "    mov (%rsp), %rax\n"
    "    mov %rax, 0(%rdi)\n"
    "    mov %rbx, 8(%rdi)\n"
    "    lea 8(%rsp), %rax\n"
    "    mov %rax, 16(%rdi)\n"
    "    mov %rbp, 24(%rdi)\n"
    "    mov %r12, 32(%rdi)\n"
    "    mov %r13, 40(%rdi)\n"
    "    mov %r14, 48(%rdi)\n"
    "    mov %r15, 56(%rdi)\n"
    "    stmxcsr 64(%rdi)\n"
    "    fnstcw 68(%rdi)\n"
    "    xor %eax, %eax\n"
    "    ret\n"
    APS5_ASM_FUNCTION("longjmp_nid_postfix")
#ifdef _WIN32
    "    mov %rdi, %rbx\n"
    "    mov %esi, %r12d\n"
    "    mov 16(%rdi), %rdi\n"
    "    and $-16, %rsp\n"
    "    call GuestStackSwitch_nid_no_patch\n"
    "    mov %rbx, %rdi\n"
    "    mov %r12d, %esi\n"
#endif
    "    mov %esi, %eax\n"
    "    test %eax, %eax\n"
    "    jnz 1f\n"
    "    inc %eax\n"
    "1:\n"
    "    mov 8(%rdi), %rbx\n"
    "    mov 16(%rdi), %rsp\n"
    "    mov 24(%rdi), %rbp\n"
    "    mov 32(%rdi), %r12\n"
    "    mov 40(%rdi), %r13\n"
    "    mov 48(%rdi), %r14\n"
    "    mov 56(%rdi), %r15\n"
    "    ldmxcsr 64(%rdi)\n"
    "    fldcw 68(%rdi)\n"
    "    jmp *0(%rdi)\n");
#elif defined(__aarch64__)
// Translated guest code reaches these two through aps5_call, which the translator runtime
// (tools/translator/runtime/Runtime.cpp) recognizes by address: setjmp has to run in its own frame
// and longjmp has to unwind the native frames. Any other caller has no guest state to save.
extern "C" {

int APS5_VABI setjmp_nid_postfix(void*) {
    NotImplemented_nid_no_patch("setjmp outside translated guest code");
}

void APS5_VABI longjmp_nid_postfix(void*, int) {
    NotImplemented_nid_no_patch("longjmp outside translated guest code");
}

}
#endif
