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

#include "smpdd.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "smp_ult.hpp"
#include "instrumentation.hpp"
#include "taskexecutionexception.hpp"
#include "smpdevice.hpp"
#include "schedule.hpp"
#include <string>

using namespace nanos;
using namespace nanos::ext;

//SMPDevice nanos::ext::SMP("SMP");

SMPDevice &nanos::ext::getSMPDevice() {
   return sys._getSMPDevice();
}

size_t SMPDD::_stackSize = 256*1024;

//! \brief Registers the Device's configuration options
//! \param reference to a configuration object.
//! \sa Config System
void SMPDD::prepareConfig ( Config &config )
{
   //! \note Get the stack size from system configuration
   size_t size = sys.getDeviceStackSize();
   if (size > 0) _stackSize = size;

   //! \note Get the stack size for this specific device
   config.registerConfigOption ( "smp-stack-size", NEW Config::SizeVar( _stackSize ), "Defines SMP::task stack size" );
   config.registerArgOption("smp-stack-size", "smp-stack-size");
}

void SMPDD::initStack ( WD *wd )
{
   _state = ::initContext(_stack, _stackSize, &workWrapper, wd, (void *) Scheduler::exit, 0);
}

void SMPDD::workWrapper ( WD &wd )
{
   SMPDD &dd = (SMPDD &) wd.getActiveDevice();
#ifdef NANOS_INSTRUMENTATION_ENABLED
   NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code") );
   NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
   NANOS_INSTRUMENT ( if ( wd.isRuntimeTask() ) { );
   NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseOpenStateEvent ( NANOS_RUNTIME ) );
   NANOS_INSTRUMENT ( } else { );
   NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_RUNNING, key, val ) );
   NANOS_INSTRUMENT ( } );
#endif

   dd.execute(wd);

#ifdef NANOS_INSTRUMENTATION_ENABLED
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateAndBurst ( key, val ) );
   NANOS_INSTRUMENT ( if ( wd.isRuntimeTask() ) { );
   NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( } else { );
   NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseCloseStateAndBurst ( key, val ) );
   NANOS_INSTRUMENT ( } );
#endif
}

void SMPDD::lazyInit ( WD &wd, bool isUserLevelThread, WD *previous )
{
   verbose0("Task " << wd.getId() << " initialization"); 
   if (isUserLevelThread) {
      if (previous == NULL) {
         _stack = (void *) NEW char[_stackSize];
         verbose0("   new stack created: " << _stackSize << " bytes");
      } else {
         verbose0("   reusing stacks");
         SMPDD &oldDD = (SMPDD &) previous->getActiveDevice();
         std::swap(_stack, oldDD._stack);
      }
      initStack(&wd);
   }
}

SMPDD * SMPDD::copyTo ( void *toAddr )
{
   SMPDD *dd = new (toAddr) SMPDD(*this);
   return dd;
}

void SMPDD::execute ( WD &wd ) throw ()
{
#ifdef NANOS_RESILIENCY_ENABLED
   bool retry = false;
   int num_tries = 0;
   if (wd.isInvalid() || (wd.getParent() != NULL && wd.getParent()->isInvalid())) {
      /*
       *  TODO Optimization?
       *  It could be better to skip this work if workdescriptor is flagged as invalid
       *  before allocating a new stack for the task and, perhaps,
       *  skip data copies of dependences.
       */
      wd.setInvalid(true);
      debug ( "Task " << wd.getId() << " is flagged as invalid.");
   } else {
      while (true) {
         try {
            // Call to the user function
            getWorkFct()( wd.getData() );
         } catch (TaskExecutionException& e) {
            /*
             * When a signal handler is executing, the delivery of the same signal
             * is blocked, and it does not become unblocked until the handler returns.
             * In this case, it will not become unblocked since the handler is exited
             * through an exception: it should be explicitly unblocked.
             */
            sigset_t sigs;
            sigemptyset(&sigs);
            sigaddset(&sigs, e.getSignal());
            pthread_sigmask(SIG_UNBLOCK, &sigs, NULL);

            if(!wd.setInvalid(true)) { // If the error isn't recoverable (i.e., no recoverable ancestor exists)
               message(e.what());
               // Unrecoverable error: terminate execution
               std::terminate();
            } else { // The error is recoverable. However, print a message for debugging purposes (do not make the error silent).
               debug( e.what() );
            }
         } catch (std::exception& e) {
            std::stringstream ss;
            ss << "Uncaught exception "
               << typeid(e).name()
               << ". Thrown in task "
               << wd.getId()
               << ". \n"
               << e.what();
            message(ss.str());
            // Unexpected error: terminate execution
            std::terminate();
         } catch (...) {
            message("Uncaught exception (unknown type). Thrown in task " << wd.getId() << ". ");
            // Unexpected error: terminate execution
            std::terminate();
         }
         // Only retry when ...
         retry = wd.isInvalid()// ... the execution failed,
         && wd.isRecoverable()// and the task is able to recover (pragma),
         && (wd.getParent() == NULL || !wd.getParent()->isInvalid())// and there is not an invalid parent.
         && num_tries < sys.getTaskMaxRetries();// This last condition avoids unbounded re-execution.

         if (!retry)
         break;

         // This is exceuted only on re-execution
         num_tries++;
         recover(wd);
      }
   }
#else
   // Workdescriptor execution
   getWorkFct()(wd.getData());
#endif
}

#ifdef NANOS_RESILIENCY_ENABLED
void SMPDD::recover( WD & wd ) {
   // Wait for successors to finish.
   wd.waitCompletion();

   // Do anything ...

   // Reset invalid state
   wd.setInvalid(false);
}
#endif
