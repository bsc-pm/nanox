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

extern "C"
{
   //! \brief Low-level helper routine: start a new user-thread
   void startHelper ();
}

void * initContext ( void *stack, size_t stackSize, void (*wrapperFunction)(nanos::WD&), nanos::WD *wd,
                     void *cleanup, void *cleanupArg )
{
   //! In this architecture the stack grows down
   intptr_t * state = (intptr_t *) stack;
   state += (stackSize/sizeof(intptr_t)) - 1;

   *state = (intptr_t) cleanup;                  //!< Cleanup function
   state--;
   *state = (intptr_t) cleanupArg;               //!< Cleanup argument
   state --;
   *state = (intptr_t) wrapperFunction;          //!< Wrapper fucntion
   state--;
   *state = (intptr_t) wd;                       //!< Wrapper argument
   state--;
   *state = (intptr_t) startHelper;              //!< Start helper function (no argument)

   //! Skip first _state
   state -= 6;

   return (void *) state;
}
