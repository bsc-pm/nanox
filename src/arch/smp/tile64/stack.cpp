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

/*************************************************************************************/
/*      Port Tilera Tile64PRO by Artur Podobas (podobas@kth.se) under prof. Mats     */
/*      Brorsson (matsbror@kth.se)  and Vladimir Vlassov (vladv@kth.se) for the      */
/*      ENCORE project. February 2011 Royal Institute of Technology.                 */
/*************************************************************************************/

#include "smp_ult.hpp"

extern "C"
{
// low-level helper routine to start a new user-thread
   void startHelper();
}

void * initContext ( void *stack, size_t stackSize, void (*wrapperFunction)(nanos::WD&), nanos::WD *wd,
                     void *cleanup, void *cleanupArg )
{
   intptr_t * state = (intptr_t *) stack;

   state += (stackSize/sizeof(intptr_t)) - 96;

   state[0]  = ( intptr_t ) startHelper;
   state[1]  = ( intptr_t ) wrapperFunction; // r31
   state[2]  = ( intptr_t ) wd;
   state[3]  = ( intptr_t ) cleanup;
   state[4]  = ( intptr_t ) cleanupArg; // r34
   state[21] = ( intptr_t ) &state[0];

   return (void *) &state[0];
}

