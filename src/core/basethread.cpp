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

#include "basethread.hpp"
#include "processingelement.hpp"
#include "system.hpp"

using namespace nanos;

__thread BaseThread * nanos::myThread=0;

Atomic<int> BaseThread::idSeed = 0;

void BaseThread::run ()
{
   associate();
   run_dependent();
}

void BaseThread::associate ()
{
   started = true;
   myThread = this;

   if ( sys.getBinding() ) bind();

   threadWD.tieTo( *this );

   setCurrentWD( threadWD );
}

bool BaseThread::singleGuard ()
{
   // return getTeam()->singleGuard(++local_single); # doesn't work
   // probably by some gcc bug
   local_single++;
   return getTeam()->singleGuard( local_single );
}

