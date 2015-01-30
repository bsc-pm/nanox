/**************************************************************************/
/*      Copyright 2010 Barcelona Supercomputing Center                    */
/*      Copyright 2009 Barcelona Supercomputing Center                    */
/*                                                                        */
/*      This file is part of the NANOS++ library.                         */
/*                                                                        */
/*      NANOS++ is free software: you can redistribute it and/or modify   */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or  */
/*      (at your option) any later version.                               */
/*                                                                        */
/*      NANOS++ is distributed in the hope that it will be useful,        */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the     */
/*      GNU Lesser General Public License for more details.               */
/*                                                                        */
/*      You should have received a copy of the GNU Lesser General Public License  */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.  */
/**************************************************************************/

#include "os.hpp"
#include "smpprocessor.hpp"
#include "pthread.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include "smp_ult.hpp"
#include "basethread.hpp"
#include "instrumentation.hpp"
//#include "clusterdevice_decl.hpp"
//#include "taskexecutionexception_decl.hpp"

using namespace nanos;
using namespace nanos::ext;

SMPThread & SMPThread::stackSize( size_t size )
{
   _pthread.setStackSize( size );
   return *this;
}

void SMPThread::runDependent ()
{
   WD &work = getThreadWD();
   setCurrentWD( work );

   SMPDD &dd = ( SMPDD & ) work.activateDevice( SMP );

   dd.execute( work );
}

void SMPThread::idle( bool debug )
{
   if ( sys.getNetwork()->getNumNodes() > 1 ) {
      sys.getNetwork()->poll(0);

      if ( !_pendingRequests.empty() ) {
         std::set<void *>::iterator it = _pendingRequests.begin();
         while ( it != _pendingRequests.end() ) {
            GetRequest *req = (GetRequest *) (*it);
            if ( req->isCompleted() ) {
               std::set<void *>::iterator toBeDeletedIt = it;
               it++;
               _pendingRequests.erase(toBeDeletedIt);
               req->clear();
               delete req;
            } else {
               it++;
            }
         }
      }
   }
}

void SMPThread::wait()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t cpuid_key = ID->getEventKey("cpuid"); )
   NANOS_INSTRUMENT ( nanos_event_value_t cpuid_value = (nanos_event_value_t) 0; )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &cpuid_key, &cpuid_value); )

   ThreadTeam *team = getTeam();

   /* Put WD's from local to global queue, leave team, and set flag
    * Locking pthread mutex assures us that the sleep flag is still set when we get here
    */
   lock();
   _pthread.mutexLock();

   if ( isSleeping() && getNextWDQueue().size()<=1 && canBlock() ) {

      if ( hasNextWD() ) {
         WD *next = getNextWD();
         next->untie();
         team->getSchedulePolicy().queue( this, *next );
      }
      fatal_cond( hasNextWD(), "Can't sleep a thread with more than 1 WD in its local queue" );

      if ( team != NULL ) leaveTeam();
      BaseThread::wait();

      unlock();

      NANOS_INSTRUMENT( InstrumentState state_stop(NANOS_STOPPED) );

      /* It is recommended to wait under a while loop to handle spurious wakeups
       * http://pubs.opengroup.org/onlinepubs/009695399/functions/pthread_cond_wait.html
       * But, for some reason this is causing deadlocks.
       */
      //while ( isSleeping() ) {
      _pthread.condWait();
      //}

      NANOS_INSTRUMENT( InstrumentState state_wake(NANOS_WAKINGUP) );
   //WORKAROUND for deadlock. Waiting for correctness checking
   //   lock();
      /* Set waiting status flag */
      BaseThread::resume();
   //   unlock();
      _pthread.mutexUnlock();

      /* Whether the thread should wait for the cpu to be free before doing some work */
      sys.getThreadManager()->waitForCpuAvailability();

      if ( isSleeping() ) wait();
      else {
         NANOS_INSTRUMENT ( if ( sys.getSMPPlugin()->getBinding() ) { cpuid_value = (nanos_event_value_t) getCpuId() + 1; } )
         NANOS_INSTRUMENT ( if ( !sys.getSMPPlugin()->getBinding() && sys.isCpuidEventEnabled() ) { cpuid_value = (nanos_event_value_t) sched_getcpu() + 1; } )
         NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &cpuid_key, &cpuid_value); )
      }
   }
   else {
      _pthread.mutexUnlock();
      unlock();
   }
}

void SMPThread::wakeup()
{
   _pthread.mutexLock();
   BaseThread::wakeup();
   if ( isWaiting() ) {
      _pthread.condSignal();
   }
   _pthread.mutexUnlock();
}

void SMPThread::sleep()
{
   _pthread.mutexLock();
   BaseThread::sleep();
   _pthread.mutexUnlock();
}

// This is executed in between switching stacks
void SMPThread::switchHelperDependent ( WD *oldWD, WD *newWD, void *oldState  )
{
   SMPDD & dd = ( SMPDD & )oldWD->getActiveDevice();
   dd.setState( (intptr_t *) oldState );
}

bool SMPThread::inlineWorkDependent ( WD &wd )
{
   // Now the WD will be inminently run
   wd.start(WD::IsNotAUserLevelThread);

   SMPDD &dd = ( SMPDD & )wd.getActiveDevice();

   NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code") );
   NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_RUNNING, key, val ) );

   //if ( sys.getNetwork()->getNodeNum() > 0 ) std::cerr << "Starting wd " << wd.getId() << std::endl;
   
   dd.execute( wd );

   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateAndBurst ( key, val ) );
   return true;
}

void SMPThread::switchTo ( WD *wd, SchedulerHelper *helper )
{
   // wd MUST have an active SMP Device when it gets here
   ensure( wd->hasActiveDevice(),"WD has no active SMP device" );
   SMPDD &dd = ( SMPDD & )wd->getActiveDevice();
   ensure( dd.hasStack(), "DD has no stack for ULT");

   ::switchStacks(
       ( void * ) getCurrentWD(),
       ( void * ) wd,
       ( void * ) dd.getState(),
       ( void * ) helper );
}

void SMPThread::exitTo ( WD *wd, SchedulerHelper *helper)
{
   // wd MUST have an active SMP Device when it gets here
   ensure( wd->hasActiveDevice(),"WD has no active SMP device" );
   SMPDD &dd = ( SMPDD & )wd->getActiveDevice();
   ensure( dd.hasStack(), "DD has no stack for ULT");

   //TODO: optimize... we don't really need to save a context in this case
   ::switchStacks(
      ( void * ) getCurrentWD(),
      ( void * ) wd,
      ( void * ) dd.getState(),
      ( void * ) helper );
}

int SMPThread::getCpuId() const {
   return _pthread.getCpuId();
}

SMPMultiThread::SMPMultiThread( WD &w, SMPProcessor *pe, unsigned int representingPEsCount, PE **representingPEs ) : SMPThread ( w, pe, pe ), _current( 0 ), _totalThreads( representingPEsCount ) {
   setCurrentWD( w );
   _threads.reserve( representingPEsCount );
   for ( unsigned int i = 0; i < representingPEsCount; i++ )
   {
      _threads[ i ] = &( representingPEs[ i ]->startWorker( this ) );
   }
}
