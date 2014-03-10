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

#include "os.hpp"
#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include "smp_ult.hpp"
#include "instrumentation.hpp"
#include "nanos-int.h"

using namespace nanos;
using namespace nanos::ext;

pthread_mutex_t SMPThread::_mutexWait = PTHREAD_MUTEX_INITIALIZER;

void * smp_bootthread ( void *arg )
{
   SMPThread *self = static_cast<SMPThread *>( arg );

   /* Set up the structure to specify task-recovery. */
   struct sigaction recovery_action;
   recovery_action.sa_sigaction = &taskExecutionHandler;
   sigemptyset(&recovery_action.sa_mask);
   recovery_action.sa_flags = SA_SIGINFO | SA_RESTART; // Important: resume system calls if interrupted by the signal.
   /* Program synchronous signals to use the default recovery handler.
    * Synchronous signals are: SIGILL, SIGTRAP, SIGBUS, SIGFPE, SIGSEGV, SIGSTKFLT (last one is no longer used)
    */
   assert(sigaction(SIGILL, &recovery_action, NULL) == 0);
   assert(sigaction(SIGTRAP, &recovery_action, NULL) == 0);
   assert(sigaction(SIGBUS, &recovery_action, NULL) == 0);
   assert(sigaction(SIGFPE, &recovery_action, NULL) == 0);
   assert(sigaction(SIGSEGV, &recovery_action, NULL) == 0);

   self->run();

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t cpuid_key = ID->getEventKey("cpuid"); )
   NANOS_INSTRUMENT ( nanos_event_value_t cpuid_value =  (nanos_event_value_t) 0; )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &cpuid_key, &cpuid_value); )


   self->BaseThread::finish();
   pthread_exit ( 0 );

   // We should never get here!
   return NULL;
}

// TODO: detect at configure
#ifndef PTHREAD_STACK_MIN
#define PTHREAD_STACK_MIN 16384
#endif

void SMPThread::start ()
{
   pthread_attr_t attr;
   pthread_attr_init(&attr);

   // user-defined stack size
   if ( _stackSize > 0 ) {
     if ( _stackSize < PTHREAD_STACK_MIN ) {
       warning("specified thread stack too small, adjusting it to minimum size");
       _stackSize = PTHREAD_STACK_MIN;
     }

     if (pthread_attr_setstacksize( &attr, _stackSize ) )
       warning("couldn't set pthread stack size stack");
   }

   if ( pthread_create( &_pth, &attr, smp_bootthread, this ) )
      fatal( "couldn't create thread" );

   if ( pthread_cond_init( &_condWait, NULL ) < 0 )
      fatal( "couldn't create pthread condition wait" );
}

void SMPThread::finish ()
{
   if ( pthread_cond_destroy( &_condWait ) < 0 )
      fatal( "couldn't destroy pthread condition wait" );
}

void SMPThread::runDependent ()
{
   WD &work = getThreadWD();
   setCurrentWD( work );

   SMPDD &dd = ( SMPDD & ) work.activateDevice( SMP );

   dd.getWorkFct()( work.getData() );
}

void SMPThread::join ()
{
   if ( pthread_join( _pth, NULL ) ) fatal("Thread cannot be joined");
   joined(); 
}

void SMPThread::bind( void )
{
   int cpu_id = getCpuId();

   cpu_set_t cpu_set;
   CPU_ZERO( &cpu_set );
   CPU_SET( cpu_id, &cpu_set );
   verbose( " Binding thread " << getId() << " to cpu " << cpu_id );
   OS::bindThread( &cpu_set );

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t cpuid_key = ID->getEventKey("cpuid"); )
   NANOS_INSTRUMENT ( nanos_event_value_t cpuid_value =  (nanos_event_value_t) getCpuId() + 1; )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &cpuid_key, &cpuid_value); )
}

void SMPThread::yield()
{
   if (sched_yield() != 0)
      warning("sched_yield call returned an error");
}

void SMPThread::wait()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t cpuid_key = ID->getEventKey("cpuid"); )
   NANOS_INSTRUMENT ( nanos_event_value_t cpuid_value = (nanos_event_value_t) 0; )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &cpuid_key, &cpuid_value); )

   lock();
   pthread_mutex_lock( &_mutexWait );

   ThreadTeam *team = getTeam();

   if ( hasNextWD() ) {
      WD *next = getNextWD();
      next->untie();
      team->getSchedulePolicy().queue( this, *next );
   }
   fatal_cond( hasNextWD(), "Can't sleep a thread with more than 1 WD in its local queue" );

   if ( team != NULL ) leaveTeam();

   if ( isSleeping() ) {
      BaseThread::wait();

      unlock();
      pthread_cond_wait( &_condWait, &_mutexWait );

      //! \note Then we call base thread wakeup, which just mark thread as active
      lock();
      BaseThread::resume();
      BaseThread::wakeup();
      unlock();
   } else {
      unlock();
   }

   pthread_mutex_unlock( &_mutexWait );

   NANOS_INSTRUMENT ( if ( sys.getBinding() ) { cpuid_value = (nanos_event_value_t) getCpuId() + 1; } )
   NANOS_INSTRUMENT ( if ( !sys.getBinding() && sys.isCpuidEventEnabled() ) { cpuid_value = (nanos_event_value_t) sched_getcpu() + 1; } )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &cpuid_key, &cpuid_value); )
}

void SMPThread::wakeup()
{
   //! \note This function has to be in free race condition environment or externally
   // protected, when called, with the thread common lock: lock() & unlock() functions.

   //! \note If thread is not marked as waiting, just ignore wakeup
   if ( !isSleeping() || !isWaiting() ) return;

   pthread_mutex_lock( &_mutexWait );
   pthread_cond_signal( &_condWait );
   pthread_mutex_unlock( &_mutexWait );
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
   ( dd.getWorkFct() )( wd.getData() );
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

void
taskExecutionHandler(int sig, siginfo_t* si, void* context) throw(task_execution_exception_t)
  {
    /*
     * In order to prevent the signal to be raised inside the handler, it is blocked inside it.
     * Because we are exiting the handler before it returns (via throwing an exception),
     * we must unblock the signal or it wont be catched again.
     *
     * It also works using SA_NODEFER.
     * Note: moved to the catch block instead
     *
     sigset_t x;
     sigemptyset(&x);
     sigaddset(&x, sig);
     pthread_sigmask(SIG_UNBLOCK, &x, NULL);
     */
    task_execution_exception_t ter = { sig, *si, *(ucontext_t*) context};

    throw ter; //throw value
    /*
     * Important note:
     * If the exception is thrown using new, then a pointer to the object will be passed. It must be catched with pointer. This is discouraged because it's allocated in the heap (it isn't reliable for out-of-memory SIGSEGVs where the stack is empty). In addition, the user is responsible for memory management in the catch clause.
     * catch(TaskExecutionError* e)
     * If otherwise is thrown without using new, a reference to the object is passed, and must be catched with reference.
     * catch(TaskExecutionError& e)
     * It can be catched without specifying the reference '&' in the argument. In that case, a copy of the data is performed.
     * catch(TaskExecutionError e)
     * http://stackoverflow.com/questions/9562053/do-the-default-catch-throw-statements-in-c-pass-by-value-or-reference
     */
  }
