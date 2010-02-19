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
// compilation in order to generate an instrumentation version
#define INSTRUMENTATION_ENABLED

#ifndef __NANOS_INSTRUMENTOR_CTX_H
#define __NANOS_INSTRUMENTOR_CTX_H
#include <stack>

namespace nanos {

   typedef enum { ERROR, IDLE, RUNNING, SYNCHRONIZATION, SCHEDULING, FORK_JOIN, OTHERS } nanos_state_t;

   class InstrumentorContext {
#ifdef INSTRUMENTATION_ENABLED
      private:
         typedef std::stack<nanos_state_t> StateStack;
         StateStack       _stateStack;

         InstrumentorContext(const InstrumentorContext &);
      public:
         // constructors
         InstrumentorContext () :_stateStack() { _stateStack.push(ERROR); _stateStack.push(SCHEDULING); }
         ~InstrumentorContext() {}

         void push ( nanos_state_t state ) { _stateStack.push( state ); }
         void pop ( void ) { if ( !(_stateStack.empty()) ) _stateStack.pop(); }

         nanos_state_t top ( void )
         {
            if ( !(_stateStack.empty()) ) return _stateStack.top();
            else return ERROR;
         }

#endif
   };
}
#endif
