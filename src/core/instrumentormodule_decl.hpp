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
#ifndef __NANOS_INSTRUMENTOR_MODULE_DECL_H
#define __NANOS_INSTRUMENTOR_MODULE_DECL_H
#include "debug.hpp"
#include "nanos-int.h"
#include "instrumentor.hpp"
#include "system.hpp"

namespace nanos {

#ifdef NANOS_INSTRUMENTATION_ENABLED

   class InstrumentStateAndBurst {
      private:
         // FIXME: could _inst be static?
         Instrumentation        *_inst;
         nanos_event_key_t    _key;
	 bool		      _closed;
      public:
         InstrumentStateAndBurst ( const char* keydesc, const char *valdesc, nanos_event_state_value_t state ) : _closed(false)
         {
            _inst = sys.getInstrumentor();
            //if ( _inst == NULL ) _inst = sys.getInstrumentor();
            _key = _inst->getInstrumentorDictionary()->getEventKey(keydesc);
            nanos_event_value_t val = _inst->getInstrumentorDictionary()->getEventValue(keydesc,valdesc);
            _inst->raiseOpenStateAndBurst(state, _key, val);
         }
         void changeState ( nanos_event_state_value_t state ) 
         {
            _inst->raiseCloseStateEvent();
            _inst->raiseOpenStateEvent(state);
         }
	 void close() { _closed=true; _inst->raiseCloseStateAndBurst(_key);  }
         ~InstrumentStateAndBurst ( ) { if (!_closed) close(); }
   };

   class InstrumentState {
      private:
         Instrumentation        *_inst;
      public:
         InstrumentState ( nanos_event_state_value_t state ) 
         {
            _inst = sys.getInstrumentor();
            _inst->raiseOpenStateEvent( state );
         }
         void changeState ( nanos_event_state_value_t state ) 
         {
            _inst->raiseCloseStateEvent();
            _inst->raiseOpenStateEvent(state);
         }
         ~InstrumentState ( ) { _inst->raiseCloseStateEvent(); }
   };
#endif
}
#endif
