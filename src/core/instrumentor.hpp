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
// xteruel: FIXME: this flag has to be managed through compilation
#define ENABLE_INSTRUMENTATION

#define INSTRUMENTOR_MAX_STATES 10
#define INSTRUMENTOR_MAX_EVENTS 10

namespace nanos {

  class Instrumentor {
     protected:
       typedef unsigned int nanos_state_t;
       typedef unsigned int nanos_event_id_t;
       typedef unsigned int nanos_event_value_t;

       typedef struct Event {
          nanos_event_id_t       id;
          nanos_event_value_t    value;
       }nanos_event_t;

       typedef enum {IDLE, RUN, CREATE_WD} States;
       typedef enum { } Events;

       nanos_state_t     _states[INSTRUMENTOR_MAX_STATES];  /*<< state vector translator */
       nanos_event_id_t  _events[INSTRUMENTOR_MAX_EVENTS];  /*<< event id vector translator */

     public:
       Instrumentor() {}
       virtual ~Instrumentor() {}

#ifdef ENABLE_INSTRUMENTATION

#if 0
       void setStateItem ( nanos_state_t idx, nanos_state_t value ) { _states[idx] = value; }
       nanos_state_t getStateItem ( nanos_state_t idx ) { return _states[idx]; }

       void setEventItem ( nanos_event_id_t idx, nanos_event_id_t value ) { _events[idx] = value; }
       nanos_event_id_t getEventItem ( nanos_event_id_t idx ) { return _events[idx]; }
#endif

       // low-level instrumentation interface (pure virtual functions)

       virtual void pushStateEventList ( nanos_state_t state, int count, nanos_event_t *events ) = 0;
       virtual void popStateEventList ( int count, nanos_event_t *events ) = 0;
       virtual void addEventList ( int count, nanos_event_t *events ) = 0;

       // mid-level instrumentation interface (virtual functions)

       virtual void pushState ( nanos_state_t state ) { pushStateEventList ( state, 0, NULL ); }
       virtual void popState( void ) { popStateEventList ( 0, NULL ); }
       virtual void pushStateEvent ( nanos_state_t state, nanos_event_t event) { pushStateEventList ( state, 1, &event ); }
       virtual void popStateEvent( nanos_event_t event ) { popStateEventList ( 1, &event ); }
       virtual void addEvent( nanos_event_t event ) { addEventList ( 1, &event );}

       // high-level instrumentation interface (virtual functions)

       virtual void enterCreateWD() { pushState(_states[CREATE_WD]); }
       virtual void leaveCreateWD() { popState(); }
#if 0
       virtual void enterRuntime () { pushState(_states[RUNTIME]); }
       virtual void leaveRuntime () { popState(); }
       virtual void enterCPU () { pushState(_states[CPU]); }
       virtual void leaveCPU () { popState(); }
       virtual void enterBarrier() { pushState(_states[BARRIER]); }
       virtual void leaveBarrier() { popState(); }
       virtual void threadIdle() {}
       virtual void taskCompletation() {}
#endif

#else

       // All functions here must be empty and  non-virtual so the compiler 
       // eliminates the instrumentation calls

       void enterCreateWD() { }
       void leaveCreateWD() { }

#endif

  };
}
#endif
