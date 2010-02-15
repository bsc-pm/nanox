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

#ifndef __NANOS_INSTRUMENTOR_H
#define __NANOS_INSTRUMENTOR_H
#include <stack>
#include "workdescriptor.hpp"

// FIXME: (#131) This flag has to be managed through compilation in order to generate an instrumentation version
#define ENABLE_INSTRUMENTATION

namespace nanos {

   typedef enum { IDLE, RUNNING, SYNCHRONIZATION, SCHEDULING, FORK_JOIN, OTHERS } nanos_state_t;
   typedef enum { IDLE_FUNCTION, RUNTIME, CREATE_WD, SUBMIT_WD, INLINE_WD, LOCK,
                  SINGLE_GUARD, BARRIER, SWITCH } nanos_event_id_t;
   typedef unsigned int nanos_event_value_t;

   class Instrumentor {
      protected:

         typedef struct Event {
            nanos_event_id_t     id;
            nanos_event_value_t  value;
         } nanos_event_t;

      public:
         Instrumentor() {}
         virtual ~Instrumentor() {}

#ifdef ENABLE_INSTRUMENTATION

       // low-level instrumentation interface (pure virtual functions)

       virtual void initialize( void ) = 0;
       virtual void finalize( void ) = 0;

       virtual void pushStateEventList ( nanos_state_t state, unsigned int count, nanos_event_t *events ) = 0;
       virtual void popStateEventList ( nanos_state_t state, unsigned int count, nanos_event_t *events ) = 0;
       virtual void addEventList ( unsigned int count, nanos_event_t *events ) = 0;

       // mid-level instrumentation interface (virtual functions)

       virtual void pushState ( nanos_state_t state ) { pushStateEventList ( state, 0, NULL ); }
       virtual void popState( nanos_state_t state ) { popStateEventList ( state, 0, NULL ); }
       virtual void pushStateEvent ( nanos_state_t state, nanos_event_t event) { pushStateEventList ( state, 1, &event ); }
       virtual void popStateEvent( nanos_state_t state, nanos_event_t event ) { popStateEventList ( state, 1, &event ); }
       virtual void addEvent( nanos_event_t event ) { addEventList ( 1, &event );}

       // high-level instrumentation interface (virtual functions)

       virtual void enterIdle() { nanos_event_t e = {IDLE_FUNCTION,1}; pushStateEvent(IDLE,e); }
       virtual void leaveIdle() { nanos_event_t e = {IDLE_FUNCTION,0}; popStateEvent(IDLE,e); }
       virtual void enterRuntime() { nanos_event_t e = {RUNTIME,1}; pushStateEvent(OTHERS,e); }
       virtual void leaveRuntime() { nanos_event_t e = {RUNTIME,0}; popStateEvent(OTHERS,e); }
       virtual void enterCreateWD() { nanos_event_t e = {CREATE_WD,1}; pushStateEvent(FORK_JOIN,e); }
       virtual void leaveCreateWD() { nanos_event_t e = {CREATE_WD,0}; popStateEvent(FORK_JOIN,e); }
       virtual void enterSubmitWD() { nanos_event_t e = {SUBMIT_WD,1}; pushStateEvent(FORK_JOIN,e); }
       virtual void leaveSubmitWD() { nanos_event_t e = {SUBMIT_WD,0}; popStateEvent(FORK_JOIN,e); }
       virtual void enterInlineWD() { nanos_event_t e = {INLINE_WD,1}; pushStateEvent(FORK_JOIN,e); }
       virtual void leaveInlineWD() { nanos_event_t e = {INLINE_WD,0}; popStateEvent(FORK_JOIN,e); }
       virtual void enterLock() { nanos_event_t e = {LOCK,1}; pushStateEvent(SYNCHRONIZATION,e); }
       virtual void leaveLock() { nanos_event_t e = {LOCK,0}; popStateEvent(SYNCHRONIZATION,e); }
       virtual void enterSingleGuard() { nanos_event_t e = {SINGLE_GUARD,1}; pushStateEvent(SYNCHRONIZATION,e); }
       virtual void leaveSingleGuard() { nanos_event_t e = {SINGLE_GUARD,0}; popStateEvent(SYNCHRONIZATION,e); }
       virtual void enterBarrier() { nanos_event_t e = {BARRIER,1}; pushStateEvent(SYNCHRONIZATION,e); }
       virtual void leaveBarrier() { nanos_event_t e = {BARRIER,0}; popStateEvent(SYNCHRONIZATION,e); }

       virtual void beforeContextSwitch() { nanos_event_t e = {SWITCH,1}; pushStateEvent(RUNNING,e); }
       virtual void afterContextSwitch() { nanos_event_t e = {SWITCH,0}; popStateEvent(RUNNING,e); }

       virtual void wdSwitch( WD* oldWD, WD* newWD ) {}
       virtual void wdExit( WD* oldWD, WD* newWD ) {}
#else

       // All functions here must be empty and  non-virtual so the compiler 
       // eliminates the instrumentation calls

       void initialize( void ) { }
       void finalize( void ) { }
       void enterCreateWD() { }
       void leaveCreateWD() { }

#endif

  };

   class InstrumentorContext {
#ifdef INSTRUMENTATION_ENABLED
      private:
         typedef std::stack<nanos_state_t> StateStack;
         StateStack _stateStack;
      public:
         void push ( nanos_state_t state ) { _stateStack.push(state); }
         nanos_state_t pop () { nanos_state_t state = _stateStack.top(); _stateStack.pop(); return state;}
         nanos_state_t top () { return _stateStack.top(); }
#endif
   };


}
#endif
