/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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

   //! Stack must be aligned to 8 byte boundary
   unsigned int align = ( ( (unsigned int) state ) & 7) / sizeof(intptr_t *);
   state -= align;

   // (A) Adjust to 8 bytes here. See (B) below
   state--;

   *state = ( intptr_t )cleanup;
   state--;
   *state = ( intptr_t )cleanupArg;
   state --;
   *state = ( intptr_t )wrapperFunction;
   state--;
   *state = ( intptr_t )wd;
   state--;
   *state = ( intptr_t )startHelper;

   // Make room for stored registers except lr (= the link register) that we
   // want to be popped by switchStacks so it returns to startHelper
   // (B) Note that this would leave the stack unaligned.
   // This is why in (A) above we compensated it
   state -= 9;

   return (void *) state;
}
