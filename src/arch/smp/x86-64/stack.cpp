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

#include "smpdd.hpp"

using namespace nanos::ext;

extern "C"
{
// low-level helper routine to start a new user-thread
   void startHelper ();
}

void SMPDD::initStackDep ( void *userfuction, void *data, void *cleanup )
{
   _state = _stack;
   _state += _stackSize;

   *_state = ( intptr_t )cleanup;
   _state--;
   *_state = ( intptr_t )this;
   _state--;
   *_state = ( intptr_t )userfuction;
   _state --;
   *_state = ( intptr_t )data;
   _state--;
   *_state = ( intptr_t )startHelper;
   _state--;
   // skip first _state
   _state -= 5;
}
