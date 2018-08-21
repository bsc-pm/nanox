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

#include "basethread.hpp"
#include "processingelement.hpp"
#include "system.hpp"
#include "synchronizedcondition.hpp"
#include "wddeque.hpp"
#include "smpthread.hpp"

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

void BaseThread::finish ()
{
   lock();
   if ( _status.has_team ) {
      setLeaveTeam(true);
      leaveTeam();
   }
   unlock();
}

void BaseThread::addNextWD ( WD *next )
{
   if ( next != NULL ) {
      debug("Add next WD as: " << next << ":"<< next->getId() << " @ thread " << _id );
      _nextWDs.push_back( next );
   }

   sys.getThreadManager()->unblockThread(this);
}

WD * BaseThread::getNextWD ()
{
   if ( !sys.getSchedulerConf().getSchedulerEnabled() ) return NULL;
   WD * next = _nextWDs.pop_front( this );
   if ( next ) next->setReady();
   return next;
}

WDDeque &BaseThread::getNextWDQueue() {
   return _nextWDs;
}

void BaseThread::associate ( WD *wd )
{
   WD * current = wd? wd:&_threadWD;
   _status.has_started = true;

   myThread = this;
   setCurrentWD( *current );

   if ( sys.getSMPPlugin()->getBinding() ) bind();

   current->_mcontrol.preInit();
   current->_mcontrol.initialize( *runningOn() );
   current->init();
   current->start(WD::IsNotAUserLevelThread);

   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( NULL, current, false); )
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
BaseThread * nanos::getMyThreadSafe()
{
   return myThread;
}
void BaseThread::notifyOutlinedCompletionDependent( WD *completedWD ) {
   fatal0( "::notifyOutlinedCompletionDependent() not available for this thread type." );
}

int BaseThread::getCpuId() const {
   ensure( _parent != NULL, "Wrong call to getCpuId" );
   return _parent->getCpuId();
}

void BaseThread::leaveTeam()
{
   // It's allowed to make another thread leave the team as long as the target thread is blocked
   ensure( this == myThread || _status.is_waiting || _status.has_joined,
         "thread is not leaving team by itself" );

   if ( _teamData )
   {
      TeamData *td = _teamData;
      debug( "removing thread " << this << " with id " << toString<int>(getTeamId()) << " from " << _teamData->getTeam() );

      td->getTeam()->removeThread( getTeamId() );
      _teamData = _teamData->getParentTeamData();
      delete td;
   }
   _status.must_leave_team = false;
   _status.has_team = _teamData != NULL;
}

void BaseThread::setLeaveTeam( bool leave )
{
   _status.must_leave_team = leave;
   if ( leave ) {
      // Remove myself from the expected list,
      // either from my current team or from my next one
      ThreadTeam *team = getTeam() ? getTeam() : getNextTeam();
      if ( team ) team->removeExpectedThread( this );

      // Only if this thread is already waiting to avoid waking it up only for that
      if ( team && _status.is_waiting ) leaveTeam();
   }
}

void BaseThread::sleep()
{
   if ( !_status.must_sleep && canBlock() ) {
      _status.must_sleep = true;
      // Only abandon team if the PE is not active
      if ( sys.getSMPPlugin()->getBinding()
            && !runningOn()->isActive() ) {
         setLeaveTeam( true );
      }
   }
}

void BaseThread::tryWakeUp( ThreadTeam *team )
{
   if ( _status.is_waiting ) {
      // Thread is blocked. Set up team and wakeup
      if ( getTeam() == NULL ) {
         reserve();
         if ( team ) setNextTeam( team );
      }
      wakeup();
   } else {
      // Thread is running
      if ( _status.must_sleep ) {
         // Thread is only tagged. Fix flag.
         _status.must_sleep = false;
      }
      if ( getTeam() == NULL ) {
         // Thread is running but orphan
         reserve();
         setNextTeam( NULL );
         if ( team ) sys.acquireWorker( team, this, true, false, false );
      }
   }
   _status.must_leave_team = false;
   if ( team ) team->addExpectedThread(this);
}

unsigned int BaseThread::getOsId() const {
   return ( _parent != NULL ) ? _parent->getOsId() : _osId;
}
