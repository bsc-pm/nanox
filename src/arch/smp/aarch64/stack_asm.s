.text
.globl switchStacks
.globl startHelper
.extern abort

// void switchStacks (oldWD, newWD, new_stack, helper_function)
//    x0 WD* oldWD
//    x1 WD* newWD
//    x2 void* new_stack
//    x3 void (*helper_function)( WD *oldWD, WD *newWD, void *arg)
.align 8
switchStacks:
    // Save callee-saved registers
    // AAPCS for AArch64 dictates that x19-x28 must be saved
    // x29 is saved because it is the FP
    // x30 is saved because it is the LR
    // FIXME: x17 and x18 may have to be saved
    stp x29, x30, [sp, #-16]!
    stp x27, x28, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    // If you uncomment this one adjust stack.cpp!
    //     stp x17, x18, [sp, #-16]!
    //
    // Floating registers.
    // AAPCS for AArch64 dictates that v8-v15 must be saved but only the lower
    // 64-bits are required (so we use d8-d15)
    stp d14, d15, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d8,  d9,  [sp, #-16]!

    mov x9, x2  // x9 <- new_stack
    mov x2, sp  // x2 <- old_stack
    mov sp, x9  // sp <- x9

    // Call to helper (oldWD, newWD, old_stack)
    blr x3

    // Restore callee-saved registers
    ldp d8,  d9,  [sp], #16
    ldp d10, d11, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d14, d15, [sp], #16
    // If you uncomment this one adjust stack.cpp!
    //     ldp x17, x18, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x29, x30, [sp], #16
    ret // equivalent to 'blr x30'

.align 8
startHelper:
    // Note: AAPCS for AArch64 states that x9 is a temporary register (i.e.
    // caller-saved). We do not need to save it here before a call because we
    // do not use it for anything else than the indirect call itself

    // Get argument and address of user function
    ldp x0, x9, [sp], #16
    // Call user function
    blr x9

    // Get argument and address of cleanup function
    ldp x0, x9, [sp], #16
    // Call cleanup function
    blr x9

    // We should not reach here
    b abort     // Abort if cleanup returns

