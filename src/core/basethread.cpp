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
#include "wddeque.hpp"

using namespace nanos;

__thread BaseThread * nanos::myThread = NULL;

void BaseThread::run ()
{
   _threadWD.tied().tieTo( *this );
   associate();
   initializeDependent();

   NANOS_INSTRUMENT ( sys.getInstrumentation()->threadStart ( *this ) );
   /* Notify that the thread has finished all its initialization and it's ready to run */
   if ( sys.getSynchronizedStart() ) sys.threadReady();
   runDependent();
   NANOS_INSTRUMENT ( sys.getInstrumentation()->threadFinish ( *this ) );
}

void BaseThread::addNextWD ( WD *next )
{
   if ( next != NULL ) {
      debug("Add next WD as: " << next << ":??" << " @ thread " << _id );
      _nextWDs.push_back( next );
   }
}

WD * BaseThread::getNextWD ()
{
   if ( !sys.getSchedulerConf().getSchedulerEnabled() ) return NULL;
   return _nextWDs.pop_front( this );
}

void BaseThread::associate ()
{
   _status.has_started = true;

   myThread = this;
   setCurrentWD( _threadWD );

   if ( sys.getBinding() ) bind();

   _threadWD.init();
   _threadWD.start(WD::IsNotAUserLevelThread);

   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( NULL, &_threadWD, false) );
}

bool BaseThread::singleGuard ()
{
   ThreadTeam *team = getTeam();
   if ( team == NULL ) return true;
   if ( _currentWD->isImplicit() == false ) return true;
   return team->singleGuard( _teamData->nextSingleGuard() );
}

bool BaseThread::enterSingleBarrierGuard ()
{
   ThreadTeam *team = getTeam();
   if ( team == NULL ) return true;
   return team->enterSingleBarrierGuard( _teamData->nextSingleBarrierGuard() );
}

void BaseThread::releaseSingleBarrierGuard ()
{
   getTeam()->releaseSingleBarrierGuard();
}

void BaseThread::waitSingleBarrierGuard ()
{
   getTeam()->waitSingleBarrierGuard( _teamData->currentSingleGuard() );
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
BaseThread * nanos::getMyThreadSafe() { return myThread; }

