/* -------------------------------------------------------------------
 * Note that some machines want labels to have leading underscores,
 * while others (e.g. System V) do not.  Thus, several labels appear
 * duplicated except for the leading underscore, e.g.
 *
 *   · qt_cswap:
 *   · _qt_cswap:
 *
 * Callee-save: %esi, %edi, %ebx, %ebp
 * Caller-save: %eax, %ecx
 * Can't tell: %edx (seems to work w/o saving it.)
 * -----------------------------------------------------------------*/
 
	.text
	.align 2
	.globl _switchStacks
	.globl switchStacks
	.type _switchStacks,@function
	.type switchStacks,@function

/* void *switchStacks (helper, arg0, arg1, new)
 *
 * On procedure entry, the helper is at 4(sp), args at 8(sp) and
 * 12(sp) and the new thread's sp at 16(sp).  It 'appears' that the
 * calling convention for the X86 requires the caller to save all
 * floating-point registers, this makes our life easy. 
 *
 * Halt the currently-running thread.  Save it's callee-save regs on
 * to the stack, 32 bytes.  Switch to the new stack (next == 16+32(sp))
 * and call the helper function (f == 4+32(sp)) with arguments: old sp
 * arg1 (8+32(sp)) and arg2 (12+32(sp)).  When the user function is
 * done, restore the new thread's state and return.
 *
 * The helper function (4(sp)) can return a void* that is returned
 * by the call to 'qt_blockk{,i}'.  Since we don't touch %eax in
 * between, we get that 'for free'. 
 */

switchStacks:
_switchStacks:
	pushl %ebp			/* Save callee-save, sp-=4. */
	pushl %esi			/* Save callee-save, sp-=4. */
	pushl %edi			/* Save callee-save, sp-=4. */
	pushl %ebx			/* Save callee-save, sp-=4. */
	movl %esp, %eax		        /* Remember old stack pointer. */
	movl 32(%esp), %esp	        /* Move to new thread. */
	pushl 28(%eax)		        /* Push arg 2. */
	pushl 24(%eax)		        /* Push arg 1. */
	pushl %eax			/* Push state pointer. */
	movl 20(%eax), %ebx	        /* Get function to call. */
	call *%ebx			/* Call f. */
	addl $12, %esp		        /* Pop args. */
	popl %ebx			/* Restore callee-save, sp+=4. */
	popl %edi			/* Restore callee-save, sp+=4. */
	popl %esi			/* Restore callee-save, sp+=4. */
	popl %ebp			/* Restore callee-save, sp+=4. */
	ret				/* Resume the stopped function. */
	hlt
.Lfe0:
	.size	_switchStacks,.Lfe0-_switchStacks
	.size	switchStacks,.Lfe0-switchStacks
