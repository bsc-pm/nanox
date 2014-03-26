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
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "smp_ult.hpp"
#include "instrumentation.hpp"

using namespace nanos;
using namespace nanos::ext;

SMPDevice nanos::ext::SMP( "SMP" );

size_t SMPDD::_stackSize = 32*1024;

/*!
  \brief Registers the Device's configuration options
  \param reference to a configuration object.
  \sa Config System
*/
void SMPDD::prepareConfig( Config &config )
{
   /*!
      Get the stack size from system configuration
    */
   size_t size = sys.getDeviceStackSize(); 
   if ( size > 0 )
      _stackSize = size;

   /*!
      Get the stack size for this device
   */
   config.registerConfigOption ( "smp-stack-size", NEW Config::SizeVar( _stackSize ), "Defines SMP workdescriptor stack size" );
   config.registerArgOption ( "smp-stack-size", "smp-stack-size" );
   config.registerEnvOption ( "smp-stack-size", "NX_SMP_STACK_SIZE" );
}

void SMPDD::initStack ( WD *wd )
{
   _state = ::initContext( _stack, _stackSize, (void *)&workWrapper, wd,( void * )Scheduler::exit, 0 );
}

void SMPDD::workWrapper( WD &wd)
{
   SMPDD &dd = ( SMPDD & ) wd.getActiveDevice();
#ifdef NANOS_INSTRUMENTATION_ENABLED
   NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code") );
   NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_RUNNING, key, val ) );
#endif

   bool retry = false;
   int num_tries = 0;
   if (wd.isInvalid() || (wd.getParent() != NULL &&  wd.getParent()->isInvalid())){
       // TODO It is better to skip the work if workdescriptor is flagged as invalid before allocating a new stack for the task
       wd.setInvalid(true);
       debug ( "Task " << wd.getId() << " is flagged as invalid.");
   } else {
       while (true) {
           try {
              // Workdescriptor execution
              dd.getWorkFct()( wd.getData() );

           } catch (task_execution_exception_t& e) {
               // The execution error is catched. The signal has to be unblocked.
               sigset_t x;
               sigemptyset(&x);
               sigaddset(&x, e.signal);
               pthread_sigmask(SIG_UNBLOCK, &x, NULL);

               // Global recovery here (if this affects the global execution).
               debug( "Signal: " << strsignal(e.signal) << " while executing task " << wd.getId());
               wd.setInvalid(true);
           }
           // If we failed (invalid), we should only retry when ...
           retry = wd.isInvalid() // ... our execution failed,
                && wd.isRecoverable() // the task is told to recover,
                && (wd.getParent() == NULL || !wd.getParent()->isInvalid()) // if we have parent, he is not already invalid, and
                && num_tries < sys.getTaskMaxRetries(); // we have not exhausted all our trials

           if (!retry)
               break;

           // Local recovery here (inside recover() function)
           wd.recover();
           num_tries++;
       }
   }

#ifdef NANOS_INSTRUMENTATION_ENABLED
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateAndBurst ( key, val ) );
#endif
}

void SMPDD::lazyInit (WD &wd, bool isUserLevelThread, WD *previous)
{
   if (isUserLevelThread) {
     if ( previous == NULL )
       _stack = NEW intptr_t[_stackSize];
     else {
        SMPDD &oldDD = (SMPDD &) previous->getActiveDevice();

        std::swap(_stack,oldDD._stack);
     }
  
     initStack(&wd);
   }
}

SMPDD * SMPDD::copyTo ( void *toAddr )
{
   SMPDD *dd = new (toAddr) SMPDD(*this);
   return dd;
}

