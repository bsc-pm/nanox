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
// FIXME: (#131) This flag ENABLE_INSTRUMENTATION has to be managed through
//compilation in order to generate an instrumentation version
#define INSTRUMENTATION_ENABLED

#ifndef __NANOS_INSTRUMENTOR_H
#define __NANOS_INSTRUMENTOR_H
#include "instrumentor_ctx.hpp"
#include "basethread.hpp"

namespace nanos {

// forward decl
   class WorkDescriptor;

   typedef enum { IDLE_FUNCTION, RUNTIME, CREATE_WD, SUBMIT_WD, INLINE_WD, LOCK,
                  SINGLE_GUARD, BARRIER, SWITCH } nanos_event_id_t;
   typedef unsigned int nanos_event_value_t;

   typedef struct Event {
      nanos_event_id_t     id;
      nanos_event_value_t  value;
   } nanos_event_t;

   class Instrumentor {
      public:
         Instrumentor() {}
         virtual ~Instrumentor() {}

#ifdef INSTRUMENTATION_ENABLED

       // low-level instrumentation interface (pure virtual functions)

       virtual void initialize( void ) = 0;
       virtual void finalize( void ) = 0;
       virtual void changeStateEventList ( nanos_state_t state, unsigned int count,
                                           nanos_event_t *events ) = 0;
       virtual void addEventList ( unsigned int count, nanos_event_t *events ) = 0;

       // mid-level instrumentation interface (virtual functions)

       virtual void pushState ( nanos_state_t state )
       {
          InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
          instrContext.push(state);
          changeStateEventList ( state, 0, NULL );
       }
       virtual void popState( void )
       {
          InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
          /* Top is current state, so before we have to pop previous state
           * on top of the stack and then restore previous state */
          instrContext.pop(); 
          nanos_state_t state = instrContext.top();
          changeStateEventList ( state, 0, NULL );
       }
       virtual void pushStateEvent ( nanos_state_t state, nanos_event_t event)
       {
          InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
          instrContext.push(state);
          changeStateEventList ( state, 1, &event );
       }
       virtual void popStateEvent( nanos_event_t event )
       {
          InstrumentorContext &instrContext = myThread->getCurrentWD()->getInstrumentorContext();
          /* Top is current state, so before we have to pop previous state
           * on top of the stack and then restore previous state */
          instrContext.pop();
          nanos_state_t state = instrContext.top();
          changeStateEventList ( state, 1, &event );
       }
       virtual void addEvent( nanos_event_t event )
       {
          addEventList ( 1, &event );
       }

       // high-level instrumentation interface (virtual functions)

       virtual void enterIdle() { nanos_event_t e = {IDLE_FUNCTION,1}; pushStateEvent(IDLE,e); }
       virtual void leaveIdle() { nanos_event_t e = {IDLE_FUNCTION,0}; popStateEvent(e); }

       virtual void enterSingleGuard() { nanos_event_t e = {SINGLE_GUARD,1}; pushStateEvent(SYNCHRONIZATION,e); }
       virtual void leaveSingleGuard() { nanos_event_t e = {SINGLE_GUARD,0}; popStateEvent(e); }

       virtual void enterRuntime() { }
       virtual void leaveRuntime() { }
       virtual void enterCreateWD() { }
       virtual void leaveCreateWD() { }
       virtual void enterSubmitWD() { }
       virtual void leaveSubmitWD() { }
       virtual void enterInlineWD() { }
       virtual void leaveInlineWD() { }
       virtual void enterLock() { }
       virtual void leaveLock() { }
       virtual void enterBarrier() { }
       virtual void leaveBarrier() { }
       virtual void beforeContextSwitch() { }
       virtual void afterContextSwitch() { }

       virtual void wdSwitch( WorkDescriptor* oldWD, WorkDescriptor* newWD ) {}
       virtual void wdExit( WorkDescriptor* oldWD, WorkDescriptor* newWD ) {}
#if 0
       virtual void enterIdle() { nanos_event_t e = {IDLE_FUNCTION,1}; pushStateEvent(IDLE,e); }
       virtual void leaveIdle() { nanos_event_t e = {IDLE_FUNCTION,0}; popStateEvent(e); }
       virtual void enterRuntime() { nanos_event_t e = {RUNTIME,1}; pushStateEvent(OTHERS,e); }
       virtual void leaveRuntime() { nanos_event_t e = {RUNTIME,0}; popStateEvent(e); }
       virtual void enterCreateWD() { nanos_event_t e = {CREATE_WD,1}; pushStateEvent(FORK_JOIN,e); }
       virtual void leaveCreateWD() { nanos_event_t e = {CREATE_WD,0}; popStateEvent(e); }
       virtual void enterSubmitWD() { nanos_event_t e = {SUBMIT_WD,1}; pushStateEvent(FORK_JOIN,e); }
       virtual void leaveSubmitWD() { nanos_event_t e = {SUBMIT_WD,0}; popStateEvent(e); }
       virtual void enterInlineWD() { nanos_event_t e = {INLINE_WD,1}; pushStateEvent(FORK_JOIN,e); }
       virtual void leaveInlineWD() { nanos_event_t e = {INLINE_WD,0}; popStateEvent(e); }
       virtual void enterLock() { nanos_event_t e = {LOCK,1}; pushStateEvent(SYNCHRONIZATION,e); }
       virtual void leaveLock() { nanos_event_t e = {LOCK,0}; popStateEvent(e); }
       virtual void enterSingleGuard() { nanos_event_t e = {SINGLE_GUARD,1}; pushStateEvent(SYNCHRONIZATION,e); }
       virtual void leaveSingleGuard() { nanos_event_t e = {SINGLE_GUARD,0}; popStateEvent(e); }
       virtual void enterBarrier() { nanos_event_t e = {BARRIER,1}; pushStateEvent(SYNCHRONIZATION,e); }
       virtual void leaveBarrier() { nanos_event_t e = {BARRIER,0}; popStateEvent(e); }
       virtual void beforeContextSwitch() { nanos_event_t e = {SWITCH,1}; pushStateEvent(RUNNING,e); }
       virtual void afterContextSwitch() { nanos_event_t e = {SWITCH,0}; popStateEvent(e); }

       virtual void wdSwitch( WorkDescriptor* oldWD, WorkDescriptor* newWD ) {}
       virtual void wdExit( WorkDescriptor* oldWD, WorkDescriptor* newWD ) {}
#endif

#else

       // All functions here must be empty and  non-virtual so the compiler 
       // eliminates the instrumentation calls

       void initialize( void ) { }
       void finalize( void ) { }
       void enterCreateWD() { }
       void leaveCreateWD() { }

#endif

  };


}
#endif
