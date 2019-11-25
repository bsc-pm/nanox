/*************************************************************************************/
/*      Copyright 2019 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/


#include "smp_ult.hpp"
#include <iostream>

extern "C"
{
// low-level helper routine to start a new user-thread
   void startHelper ();
}

void * initContext( void *stack, size_t stackSize, void (*wrapperFunction)(nanos::WD&), nanos::WD *wd,
                    void *cleanup, void *cleanupArg )
{
   intptr_t * state = (intptr_t *) stack;
   state += (stackSize/sizeof(intptr_t)) - 1;

   // In RISC-V the stack must be aligned to 16-bytes
   unsigned int align = ( ( (uintptr_t) state ) & 15) / sizeof(intptr_t);
   state -= align;

   *state = ( intptr_t )cleanup;
   state--;
   *state = ( intptr_t )cleanupArg;
   state--;
   *state = ( intptr_t )wrapperFunction;
   state--;
   *state = ( intptr_t )wd;
   state--;

   // Padding to keep stack aligned.
   state--;

   *state = ( intptr_t )startHelper;

   state -= 24;

   return (void *) state;
}
