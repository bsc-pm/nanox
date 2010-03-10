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

#include "nanos.h"
#include "system.hpp"
#include "instrumentor_decl.hpp"

using namespace nanos;

nanos_err_t nanos_instrument_events ( unsigned int num_events, nanos_event_t events[] )
{
   try
   {
      Instrumentor::Event *e = (Instrumentor::Event *) alloca ( sizeof(Instrumentor::Event) * num_events ); 

      for (unsigned int i = 0; i < num_events; i++ ) {
         switch ( events[i].type ) {
            case STATE:
               sys.getInstrumentor()->createStateEvent ( e[i], events[i].info.state.value );
               break;
            case BURST_START:
               sys.getInstrumentor()->createBurstStart( e[i], events[i].info.burst.key, events[i].info.burst.value );
               break;
            case BURST_END:
               sys.getInstrumentor()->createBurstEnd( e[i], events[i].info.burst.key, events[i].info.burst.value );
               break;
            default:
               return NANOS_UNKNOWN_ERR;
               break;
         }
      }
      sys.getInstrumentor()->addEventList ( num_events, e);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_instrument_enter_state ( nanos_event_state_value_t state )
{
   try
   {
      Instrumentor::Event *e = (Instrumentor::Event *) alloca ( sizeof(Instrumentor::Event) ); 
      sys.getInstrumentor()->createStateEvent ( *e, state );
      sys.getInstrumentor()->addEventList ( 1, e);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_instrument_leave_state ( void )
{
   try
   {
      Instrumentor::Event *e = (Instrumentor::Event *) alloca ( sizeof(Instrumentor::Event) ); 
      sys.getInstrumentor()->returnPreviousStateEvent ( *e );
      sys.getInstrumentor()->addEventList ( 1, e);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_instrument_enter_burst( nanos_event_key_t key, nanos_event_value_t value )
{
   try
   {
      Instrumentor::Event *e = (Instrumentor::Event *) alloca ( sizeof(Instrumentor::Event) ); 
      sys.getInstrumentor()->createBurstStart ( *e, key, value );
      sys.getInstrumentor()->addEventList ( 1, e);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_instrument_leave_burst( nanos_event_key_t key, nanos_event_value_t value )
{
   try
   {
      Instrumentor::Event *e = (Instrumentor::Event *) alloca ( sizeof(Instrumentor::Event) ); 
      sys.getInstrumentor()->createBurstEnd ( *e, key, value );
      sys.getInstrumentor()->addEventList ( 1, e);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}
