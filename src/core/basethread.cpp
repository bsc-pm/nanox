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
   WD * next = _nextWDs.pop_front( this );
   if ( next ) next->setReady();
   return next;
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

void BaseThread::setupSignalHandlers()
{
   /* Set up the structure to specify task-recovery. */
   struct sigaction recovery_action;
   recovery_action.sa_sigaction = &taskExecutionHandler;
   sigemptyset(&recovery_action.sa_mask);
   recovery_action.sa_flags = SA_SIGINFO | SA_RESTART; // Important: resume system calls if interrupted by the signal.
   /* Program synchronous signals to use the default recovery handler.
    * Synchronous signals are: SIGILL, SIGTRAP, SIGBUS, SIGFPE, SIGSEGV, SIGSTKFLT (last one is no longer used)
    */
   ensure(sigaction(SIGILL, &recovery_action, NULL) == 0, "Signal setup (SIGILL) failed");
   ensure(sigaction(SIGTRAP, &recovery_action, NULL) == 0, "Signal setup (SIGTRAP) failed");
   ensure(sigaction(SIGBUS, &recovery_action, NULL) == 0, "Signal setup (SIGBUS) failed");
   ensure(sigaction(SIGFPE, &recovery_action, NULL) == 0, "Signal setup (SIGFPE) failed");
   ensure(sigaction(SIGSEGV, &recovery_action, NULL) == 0, "Signal setup (SIGSEGV) failed");
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
