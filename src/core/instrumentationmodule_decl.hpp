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

#ifndef __NANOS_INSTRUMENTOR_MODULE_DECL_H
#define __NANOS_INSTRUMENTOR_MODULE_DECL_H
#include "debug.hpp"
#include "nanos-int.h"
#include "instrumentation.hpp"
#include "system_decl.hpp"

namespace nanos {

#ifdef NANOS_INSTRUMENTATION_ENABLED

/*!\class InstrumentStateAndBurst
 * \brief InstrumentStateAndBurst raises state and burst event when constructed and finalize them when destructed.
 */
   class InstrumentStateAndBurst {
      private:
         Instrumentation     &_inst;   /**< Instrumentation object*/
         nanos_event_key_t    _key;    /**< Key used in burst event */
         nanos_event_value_t  _val;
	 bool		      _closed; /**< Closed flag */
      private:
         /*! \brief InstrumentStateAndBurst default constructor (private)
          */
         InstrumentStateAndBurst ();
         /*! \brief InstrumentStateAndBurst copy constructor (private)
          */
         InstrumentStateAndBurst ( InstrumentStateAndBurst &isb );
         /*! \brief InstrumentStateAndBurst copy assignment operator (private)
          */
         InstrumentStateAndBurst& operator= ( InstrumentStateAndBurst &isb );
      public:
         /*! \brief InstrumentStateAndBurst constructor
          */
         InstrumentStateAndBurst ( const char* keydesc, const char *valdesc, nanos_event_state_value_t state )
            : _inst(*sys.getInstrumentation()), _key( _inst.getInstrumentationDictionary()->getEventKey(keydesc)), _val(0),
              _closed(false)
         {
            _val = _inst.getInstrumentationDictionary()->getEventValue(keydesc,valdesc);
            _inst.raiseOpenStateAndBurst(state, _key, _val);
         }
         /*! \brief InstrumentStateAndBurst constructor
          */
         InstrumentStateAndBurst ( const char* keydesc, nanos_event_value_t val, nanos_event_state_value_t state )
            : _inst(*sys.getInstrumentation()), _key( _inst.getInstrumentationDictionary()->getEventKey(keydesc)), _val(0),
              _closed(false)
         {
            _val = val;
            _inst.raiseOpenStateAndBurst(state, _key, _val);
         }
         /*! \brief InstrumentStateAndBurst destructor
          */
         ~InstrumentStateAndBurst ( ) { if (!_closed) close(); }
         /*! \brief Closes states and burst
          */
	 void close() { _closed=true; _inst.raiseCloseStateAndBurst(_key, _val);  }
   };

/*!\class InstrumentState
 * \brief InstrumentState raises a state event when constructed and finalize it when destructed.
 */
   class InstrumentState {
      private:
         Instrumentation     &_inst;    /**< Instrumentation object*/
         bool                 _internal;  /**< Internal flag */
         bool                 _closed;  /**< Closed flag */
      private:
         /*! \brief InstrumentState default constructor (private)
          */
         InstrumentState ();
         /*! \brief InstrumentState copy constructor (private)
          */
         InstrumentState ( InstrumentState &is );
         /*! \brief InstrumentState copy assignment operator (private)
          */
         InstrumentState& operator= ( InstrumentState &is );
      public:
         /*! \brief InstrumentState constructor
          */
         InstrumentState ( nanos_event_state_value_t state, bool internal = false )
            : _inst(*sys.getInstrumentation()), _internal(internal), _closed(false)
         {
            if ( _internal && !_inst.isInternalsEnabled() ) return;
            _inst.raiseOpenStateEvent( state );
         }
         /*! \brief InstrumentState destructor 
          */
         ~InstrumentState ( )
         {
            if ( (_internal && !_inst.isInternalsEnabled()) || _closed ) return;
            close();
         }
         /*! \brief Closes states
          */
	 void close()
    {
       _closed = true;
       if ( _internal && !_inst.isInternalsEnabled() ) return;
       _inst.raiseCloseStateEvent();
    }
   };

/*!\class InstrumentBurst
 * \brief InstrumentBurst raises a burst event when constructed and finalize it when destructed.
 */
   class InstrumentBurst {
      private:
         Instrumentation     &_inst;   /**< Instrumentation object*/
         nanos_event_key_t    _key;    /**< Key used in burst event */
         nanos_event_value_t  _val;
         bool                 _closed; /**< Closed flag */
      private:
         /*! \brief InstrumentBurst default constructor (private)
          */
         InstrumentBurst ();
         /*! \brief InstrumentBurst copy constructor (private)
          */
         InstrumentBurst ( InstrumentBurst &ib );
         /*! \brief InstrumentBurst copy assignment operator (private)
          */
         InstrumentBurst& operator= ( InstrumentBurst &ib );
      public:
         /*! \brief InstrumentBurst constructor
          */
         InstrumentBurst ( const char* keydesc, const char *valdesc )
            : _inst(*sys.getInstrumentation()), _key( _inst.getInstrumentationDictionary()->getEventKey(keydesc)), _val(0),
              _closed(false)
         {
            _val = _inst.getInstrumentationDictionary()->getEventValue(keydesc,valdesc);
            _inst.raiseOpenBurstEvent(_key, _val);
         }
         /*! \brief InstrumentBurst constructor
          */
         InstrumentBurst ( const char* keydesc, nanos_event_value_t val )
            : _inst(*sys.getInstrumentation()), _key( _inst.getInstrumentationDictionary()->getEventKey(keydesc)), _val(val),
              _closed(false)
         {
            _inst.raiseOpenBurstEvent(_key, _val);
         }
         /*! \brief InstrumentBurst destructor
          */
         ~InstrumentBurst ( ) { if (!_closed) close(); }
         /*! \brief Closes Burst event
          */
         void close() { _closed=true; _inst.raiseCloseBurstEvent(_key, _val);  }
   };
#endif

} // namespace nanos

#endif
