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

#define INSTRUMENTOR_STATE_RUNTIME  0
#define INSTRUMENTOR_STATE_CPU      1
#define INSTRUMENTOR_STATE_BARRIER  2

namespace nanos {

  class Instrumentor {
     int   states[INSTRUMENTOR_MAX_STATES];  /*<< state vector translator */
     int   events[INSTRUMENTOR_MAX_EVENTS];  /*<< event vector translator */

     public:
       Instrumentor() {}
       virtual ~Instrumentor() {}

#ifdef ENABLE_INSTRUMENTATION

       // low-level instrumentation interface (pure virtual function)

       virtual void pushState( int state ) = 0;
       virtual void popState( void ) = 0;
       virtual void addEvent() = 0;
       virtual void addEventList() = 0;

       // high-level events

       virtual void enterRuntime ()
       {
          pushState(states[INSTRUMENTOR_STATE_RUNTIME]);
          
       }
       virtual void leaveRuntime ()
       {
          popState();
       }

       virtual void enterCPU () {}
       virtual void leaveCPU () {}

       virtual void threadIdle() {}

       virtual void taskCreation() {}
       virtual void taskCompletation() {}

       virtual void enterBarrier() {}
       virtual void leaveBarrier() {}

#else

       // All functions here must be empty and  non-virtual so the compiler 
       // eliminates the instrumentation calls

       void pushState () {}
       void popState () {}
       void addEvent () {}

       void enterRuntime () {} 
       void leaveRuntime () {}

       void enterCPU () {}
       void leaveCPU () {}

       void threadIdle() {}

       void taskCreation() {}
       void taskCompletation() {}

       void enterBarrier() {}
       void leaveBarrier() {}

#endif

  };
}
#endif
