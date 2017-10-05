.text
.globl switchStacks
.globl startHelper
/* void switchStacks (arg0, arg1, new, helper) */
.extern abort

switchStacks:

	/* Saves general purpose registers,
	      R12 is included to make them even number */
	push {r4-r12,lr}

	/* switch current sp to new */
	mov r4, r2
	mov r2, sp
	mov sp, r4

	/* arguments in r0=arg0, r1=arg1, r2=old sp */
	   blx r3

	/* Restores general purpose registers*/
	pop {r4-r12,pc}

startHelper:
	pop {r0,r4}  /*User Args (R0) to Call user Function (R4)*/
	blx r4       /*Jumps to user function*/
	pop {r0,r4}  /*Cleanup Args (R0) to Call cleanup function(R4)*/
	blx r4       /*Jumps to cleanup*/
	b abort     /*Aborts in case of cleanup returns*/
