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

#include "pthread.hpp"
#include "os.hpp"
#include "basethread_decl.hpp"
#include "instrumentation.hpp"
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>


// TODO: detect at configure
#ifndef PTHREAD_STACK_MIN
#define PTHREAD_STACK_MIN 16384
#endif


using namespace nanos;

void * os_bootthread ( void *arg )
{
   BaseThread *self = static_cast<BaseThread *>( arg );
#ifdef NANOS_RESILIENCY_ENABLED
   self->setupSignalHandlers();
#endif

   self->run();

   self->finish();
   pthread_exit ( 0 );

   // We should never get here!
   return NULL;
}

// Main thread does not get initializaed through start()
void PThread::initMain ()
{
   _pth = pthread_self();

   if ( pthread_cond_init( &_condWait, NULL ) < 0 )
      fatal( "couldn't create pthread condition wait" );

   if ( pthread_mutex_init(&_mutexWait, NULL) < 0 )
      fatal( "couldn't create pthread mutex wait" );
}

void PThread::start ( BaseThread * th )
{
   pthread_attr_t attr;
   pthread_attr_init( &attr );

   //! \note Checking user-defined stack size
   if ( _stackSize > 0 ) {
      if ( _stackSize < PTHREAD_STACK_MIN ) {
         warning("Specified thread stack size (" << _stackSize << " bytes) too small, adjusting to " << PTHREAD_STACK_MIN << " bytes");
         _stackSize = PTHREAD_STACK_MIN;
      }
      if (pthread_attr_setstacksize( &attr, _stackSize ) )
         warning( "Couldn't set pthread stack size stack" );
   }
 
   verbose( "Creating thread with " << _stackSize << " bytes of stack size" );

   if ( pthread_create( &_pth, &attr, os_bootthread, th ) )
      fatal( "Couldn't create thread" );

   if ( pthread_cond_init( &_condWait, NULL ) < 0 )
      fatal( "Couldn't create pthread condition wait" );

   if ( pthread_mutex_init(&_mutexWait, NULL) < 0 )
      fatal( "Couldn't create pthread mutex wait" );
}

void PThread::finish ()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t cpuid_key = ID->getEventKey("cpuid"); )
   NANOS_INSTRUMENT ( nanos_event_value_t cpuid_value =  (nanos_event_value_t) 0; )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &cpuid_key, &cpuid_value); )

   if ( pthread_mutex_destroy( &_mutexWait ) < 0 )
      fatal( "couldn't destroy pthread mutex wait" );

   if ( pthread_cond_destroy( &_condWait ) < 0 )
      fatal( "couldn't destroy pthread condition wait" );
}

void PThread::join ()
{
   if ( pthread_join( _pth, NULL ) )
      fatal( "Thread cannot be joined" );
}

void PThread::bind()
{
   const CpuSet& binding = _core->getBindingList();
   verbose( "Binding thread " << getMyThreadSafe()->getId() << " to cpus [" << binding << "]" );
   pthread_setaffinity_np( _pth, sizeof(cpu_set_t), binding.get_cpu_set_pointer() );

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t cpuid_key = ID->getEventKey("cpuid"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t numa_key = ID->getEventKey("thread-numa-node"); )
   
   NANOS_INSTRUMENT ( nanos_event_key_t keys[2]; )
   NANOS_INSTRUMENT ( keys[0] = cpuid_key )
   NANOS_INSTRUMENT ( keys[1] = numa_key )
   
   NANOS_INSTRUMENT ( nanos_event_value_t values[2]; )
   NANOS_INSTRUMENT ( values[0] = (nanos_event_value_t) _core->getBindingId() + 1; )
   NANOS_INSTRUMENT ( values[1] = (nanos_event_value_t) _core->getNumaNode() + 1; )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(2, keys, values); )
}

void PThread::yield()
{
   if ( sched_yield() != 0 )
      warning("sched_yield call returned an error");
}

void PThread::mutexLock()
{
   pthread_mutex_lock( &_mutexWait );
}

void PThread::mutexUnlock()
{
   pthread_mutex_unlock( &_mutexWait );
}

void PThread::condWait()
{
   pthread_cond_wait( &_condWait, &_mutexWait );
}

void PThread::condSignal()
{
   pthread_cond_signal( &_condWait );
}


#ifdef NANOS_RESILIENCY_ENABLED

void PThread::setupSignalHandlers ()
{
   /* Set up the structure to specify task-recovery. */
   struct sigaction recovery_action;
   recovery_action.sa_sigaction = &taskExecutionHandler;
   sigemptyset(&recovery_action.sa_mask);
   recovery_action.sa_flags = SA_SIGINFO // Provides context information to the handler.
                            | SA_RESTART; // Resume system calls interrupted by the signal.

   debug0("Resiliency: handling synchronous signals raised in tasks' context.");
   /* Program synchronous signals to use the default recovery handler.
    * Synchronous signals are: SIGILL, SIGTRAP, SIGBUS, SIGFPE, SIGSEGV, SIGSTKFLT (last one is no longer used)
    */
   fatal_cond0(sigaction(SIGILL, &recovery_action, NULL) != 0, "Signal setup (SIGILL) failed");
   fatal_cond0(sigaction(SIGTRAP, &recovery_action, NULL) != 0, "Signal setup (SIGTRAP) failed");
   fatal_cond0(sigaction(SIGBUS, &recovery_action, NULL) != 0, "Signal setup (SIGBUS) failed");
   fatal_cond0(sigaction(SIGFPE, &recovery_action, NULL) != 0, "Signal setup (SIGFPE) failed");
   fatal_cond0(sigaction(SIGSEGV, &recovery_action, NULL) != 0, "Signal setup (SIGSEGV) failed");

}

void taskExecutionHandler ( int sig, siginfo_t* si, void* context ) throw(TaskExecutionException)
{
   /*
    * In order to prevent the signal to be raised inside the handler,
    * the kernel blocks it until the handler returns.
    *
    * As we are exiting the handler before return (throwing an exception),
    * we must unblock the signal or that signal will not be available to catch
    * in the future (this is done in at the catch clause).
    */
   throw TaskExecutionException(getMyThreadSafe()->getCurrentWD(), *si, *(ucontext_t*)context);
}
#endif
