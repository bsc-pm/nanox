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
#include "synchronizedcondition.hpp"

using namespace nanos;

__thread BaseThread * nanos::myThread=0;

Atomic<int> BaseThread::_idSeed = 0;

void BaseThread::run ()
{
   associate();
   runDependent();
}

void BaseThread::associate ()
{
   _started = true;
   myThread = this;

   if ( sys.getBinding() ) bind();

   _threadWD.tieTo( *this );

   setCurrentWD( _threadWD );
}

bool BaseThread::singleGuard ()
{
   // return getTeam()->singleGuard(++localSingleCount); # doesn't work
   // probably because some gcc bug
   return getTeam()->singleGuard( getTeamData()->nextSingleGuard() );
}

void BaseThread::inlineWork (WorkDescriptor *wd)
{
   WD *oldwd = getCurrentWD();
   GenericSyncCond *syncCond = oldWD->getSyncCond();
   if ( syncCond != NULL ) {
      syncCond->unlock();
   }
   sys.getInstrumentor()->wdSwitch( oldwd, wd );
   setCurrentWD( *wd );
   inlineWorkDependent(*wd);
   wd->done();
   sys.getInstrumentor()->wdSwitch( wd, oldwd );
   myThread->setCurrentWD( *oldwd );
}

/*! \brief Performs the always-required operations when switching WD in a thread.
*/
void BaseThread::switchHelper( WD* oldWD, WD* newWD )
{
   GenericSyncCond *syncCond = oldWD->getSyncCond();
   if ( syncCond != NULL ) {
      oldWD->setBlocked();
      syncCond->unlock();
   } else {
      Scheduler::queue( *oldWD );
   }
   sys.getInstrumentor()->wdSwitch( oldWD, newWD );
   myThread->setCurrentWD( *newWD );
}

/*! \brief Performs the always-required operations when exit a WD in a thread.
*/
void BaseThread::exitHelper( WD* oldWD, WD* newWD )
{
   sys.getInstrumentor()->wdExit( oldWD, newWD );
   delete oldWD;
   myThread->setCurrentWD( *newWD );
}
 
/*
 * G++ optimizes TLS accesses by obtaining only once the address of the TLS variable
 * In this function this optimization does not work because the task can move from one thread to another
 * in different iterations and it will still be seeing the old TLS variable (creating havoc
 * and destruction and colorful runtime errors).
 * getMyThreadSafe ensures that the TLS variable is reloaded at least once per iteration while we still do some
 * reuse of the address (inside the iteration) so we're not forcing to go through TLS for each myThread access
 * It's important that the compiler doesn't inline it or the optimazer will cause the same wrong behavior anyway.
 */
BaseThread * nanos::getMyThreadSafe()
{
   return myThread;
}


