.text
.globl switchStacks
.globl startHelper
.extern abort

# void switchStacks (oldWD, newWD, new_stack, helper_function)
#    x10 WD* oldWD
#    x11 WD* newWD
#    x12 void* new_stack
#    x13 void (*helper_function)( WD *oldWD, WD *newWD, void *arg)
.align 8
switchStacks:
   # Callee-saved registers
   # GPR
   #   x8,   x9, x18, x19,
   #   x20, x21, x22, x23,
   #   x24, x25, x26, x27,
   # (The frame pointer is usually x8 and the base pointer is often x9)
   # FPR
   #   f8,   f9, f18, f19,
   #   f20, f21, f22, f23,
   #   f24, f25, f26, f27,
   # We also save x1 because it is the return address but this would render
   # the stack unaligned so we allocate +8 bytes more.
STACK_SIZE = 26*8

   # Grow stack
   addi sp, sp, - STACK_SIZE
   # Keep callee-saved registers
   # Note: x2 is the stack pointer
   sd x8, 0(x2)
   sd x9, 8(x2)
   sd x18, 16(x2)
   sd x19, 24(x2)
   sd x20, 32(x2)
   sd x21, 40(x2)
   sd x22, 48(x2)
   sd x23, 56(x2)
   sd x24, 64(x2)
   sd x25, 72(x2)
   sd x26, 80(x2)
   sd x27, 88(x2)
   fsd f8, 96(x2)
   fsd f9, 104(x2)
   fsd f18, 112(x2)
   fsd f19, 120(x2)
   fsd f20, 128(x2)
   fsd f21, 136(x2)
   fsd f22, 144(x2)
   fsd f23, 152(x2)
   fsd f24, 160(x2)
   fsd f25, 168(x2)
   fsd f26, 176(x2)
   fsd f27, 184(x2)
   sd  x1,  192(x2)

   mv x5, x12  # x5 <- new_stack
   mv x12, x2  # x12 <-  old_stack
   mv x2,  x5  # x2 <- x5 (x2 is the stack pointer)

   jalr x13    # Indirect call to helper_function

   # Restore callee-saved registers
   ld x8, 0(x2)
   ld x9, 8(x2)
   ld x18, 16(x2)
   ld x19, 24(x2)
   ld x20, 32(x2)
   ld x21, 40(x2)
   ld x22, 48(x2)
   ld x23, 56(x2)
   ld x24, 64(x2)
   ld x25, 72(x2)
   ld x26, 80(x2)
   ld x27, 88(x2)
   fld f8, 96(x2)
   fld f9, 104(x2)
   fld f18, 112(x2)
   fld f19, 120(x2)
   fld f20, 128(x2)
   fld f21, 136(x2)
   fld f22, 144(x2)
   fld f23, 152(x2)
   fld f24, 160(x2)
   fld f25, 168(x2)
   fld f26, 176(x2)
   fld f27, 184(x2)
   ld  x1,  192(x2)

   # Shrink stack
   addi sp, sp, STACK_SIZE

   ret

.align 8
startHelper:
    # x5 is a temporary register, so it should be fine to clobber it here.
    # Pop argument and address of user function
    ld x10, 0(x2)
    ld x5, 8(x2)
    addi sp, sp, 16
    # Call user function
    jalr x5

    # Pop argument and address of cleanup function
    ld x10, 0(x2)
    ld x5, 8(x2)
    addi sp, sp, 16
    # Call cleanup function
    jalr x5

    # We should not reach here
    call abort@plt     # Abort if cleanup returns

