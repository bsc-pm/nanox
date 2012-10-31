 
	.text
	.align 2
	.globl _switchStacksMpi
	.globl switchStacksMpi
	.type _switchStacksMpi,@function
	.type switchStacksMpi,@function
	.globl _startHelperMpi
	.globl startHelperMpi
	.type _startHelperMpi,@function
	.type startHelperMpi,@function


/* void *switchStacksMpi (arg1, arg2, new sp, helper)
 *
 * %rdi = arg1
 * %rsi = arg2
 * %rdx = new sp
 * %rcx = helper
 */

switchStacksMpi:
_switchStacksMpi:
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx

        /* stack swap */
	movq    %rdx, %rax
	movq    %rsp, %rdx
	movq    %rax, %rsp
	
	/* arguments in %rdi=arg1, %rsi=arg2, %rdx=new sp */
	call	*%rcx
	
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq    %rbp
	
	ret
	hlt
.Lfe0:
	.size	switchStacksMpi, .Lfe0-switchStacksMpi
	.size	switchStacksMpi, .Lfe0-switchStacksMpi

startHelperMpi:
_startHelperMpi:
    popq   %rdi
	popq   %rax
	call   *%rax
    popq   %rdi
	popq   %rax
	call   *%rax
.Lfe1:
        .size startHelperMpi, .Lfe1-startHelperMpi
	.size _startHelperMpi, .Lfe1-_startHelperMpi

