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
#include "system.hpp"

namespace nanos {

   class InstrumentorStateAndBurst {
      private:
         // FIXME: could _inst be static?
         Instrumentor        *_inst;
         nanos_event_key_t    _key;
      public:
         InstrumentorStateAndBurst ( const char* keydesc, const char *valdesc, nanos_event_state_value_t state )
         {
            _inst = sys.getInstrumentor();
            //if ( _inst == NULL ) _inst = sys.getInstrumentor();
            _key = _inst->getInstrumentorDictionary()->getEventKey(keydesc);
            nanos_event_value_t val = _inst->getInstrumentorDictionary()->getEventValue(keydesc,valdesc);
            _inst->raiseOpenStateAndBurst(state, _key, val);
         }

         ~InstrumentorStateAndBurst ( ) { _inst->raiseCloseStateAndBurst( _key ); }
   };

   class InstrumentorState {
      private:
         Instrumentor        *_inst;
      public:
         InstrumentorState ( nanos_event_state_value_t state ) 
         {
            _inst = sys.getInstrumentor();
            _inst->raiseOpenStateEvent( state );
         }
         ~InstrumentorState ( ) { _inst->raiseCloseStateEvent(); }
   };
}
#endif
