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

namespace nanos
{

  class Instrumentor {
     public:
       Instrumentor() {}
       virtual ~Instrumentor() {}

#ifdef ENABLE_INSTRUMENTATION

       // low-level instrumentation interface

       virtual void pushState() {}
       virtual void popState() {}
       virtual void addEvent() {}

       // high-level events

       virtual void enterRuntime () {}
       virtual void leaveRuntime () {}



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
       void exitBarrier() {}


#endif

  };


}

#endif
