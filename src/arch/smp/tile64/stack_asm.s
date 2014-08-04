/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/
/*      Port Tilera Tile64PRO by Artur Podobas (podobas@kth.se) under prof. Mats     */
/*      Brorsson (matsbror@kth.se)  and Vladimir Vlassov (vladv@kth.se) for the      */
/*      ENCORE project. February 2011 Royal Institute of Technology.                 */
/*************************************************************************************/

   .file	"tst.c"
   .text
   .align 8
   .global switchStacks
   .type switchStacks, @function
   .global startHelper
   .type startHelper, @function

switchStacks:
	sw sp,r52	
	addi sp,sp,-4
	sw sp,r51
	addi sp,sp,-4
	sw sp,r50
	addi sp,sp,-4
	sw sp,r49
	addi sp,sp,-4
	sw sp,r48
	addi sp,sp,-4
	sw sp,r47
	addi sp,sp,-4
	sw sp,r46
	addi sp,sp,-4
	sw sp,r45
	addi sp,sp,-4
	sw sp,r44
	addi sp,sp,-4
	sw sp,r43
	addi sp,sp,-4
	sw sp,r42
	addi sp,sp,-4
	sw sp,r41
	addi sp,sp,-4
	sw sp,r40
	addi sp,sp,-4
	sw sp,r39
	addi sp,sp,-4
	sw sp,r38
	addi sp,sp,-4
	sw sp,r37
	addi sp,sp,-4
	sw sp,r36
	addi sp,sp,-4
	sw sp,r35
	addi sp,sp,-4
	sw sp,r34
	addi sp,sp,-4
	sw sp,r33
	addi sp,sp,-4
	sw sp,r32
	addi sp,sp,-4
	sw sp,r31
	addi sp,sp,-4
	sw sp,r30
	addi sp,sp,-4
	sw sp,lr	/*Return Adress*/

	move r9,sp	/*Old stack pointer*/
	move sp,r2
	addi sp,sp,-4 

	/* helper (arg0, arg1 , oldstack)*/
	move r0,r0
	move r1,r1
	move r2,r9
	jalr	r3
		
	/*POP return adress*/
	addi sp,sp,4
	lw lr,sp	
	/*POP registers*/
	addi sp,sp,4
	lw r30,sp
	addi sp,sp,4
	lw r31,sp
	addi sp,sp,4
	lw r32,sp
	addi sp,sp,4
	lw r33,sp
	addi sp,sp,4
	lw r34,sp
	addi sp,sp,4
	lw r35,sp
	addi sp,sp,4
	lw r36,sp
	addi sp,sp,4
	lw r37,sp
	addi sp,sp,4
	lw r38,sp
	addi sp,sp,4
	lw r39,sp
	addi sp,sp,4
	lw r40,sp
	addi sp,sp,4
	lw r41,sp
	addi sp,sp,4
	lw r42,sp
	addi sp,sp,4
	lw r43,sp
	addi sp,sp,4
	lw r44,sp
	addi sp,sp,4
	lw r45,sp
	addi sp,sp,4
	lw r46,sp
	addi sp,sp,4
	lw r47,sp
	addi sp,sp,4
	lw r48,sp
	addi sp,sp,4
	lw r49,sp
	addi sp,sp,4
	lw r50,sp
	addi sp,sp,4
	lw r51,sp
	addi sp,sp,4
	lw r52,sp	/* Frame Pointer*/

	jrp lr

	.size	switchStacks, .-switchStacks

  
startHelper:
	addi sp,sp,-88
	lw lr,sp
	addi sp,sp,4
	lw r0,sp
	jalr lr

	addi sp,sp,4
	lw lr,sp
	addi sp,sp,4
	lw r0,sp
	addi sp,sp,4
	jalr lr
	.size	startHelper, .-startHelper

	.ident	"GCC: (GNU) 4.4.3"
