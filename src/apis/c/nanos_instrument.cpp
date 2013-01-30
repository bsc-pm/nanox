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
#include "instrumentation.hpp"
#include <alloca.h>

#ifdef GPU_DEV
#include "gputhread_decl.hpp"
#endif

using namespace nanos;

NANOS_API_DEF(nanos_err_t, nanos_instrument_register_key, ( nanos_event_key_t *event_key, const char *key, const char *description, bool abort_when_registered ))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   try
   {
      *event_key = sys.getInstrumentation()->getInstrumentationDictionary()->registerEventKey(key, description, abort_when_registered);
   } catch ( nanos_err_t err) {
      return err;
   }
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_register_value, ( nanos_event_value_t *event_value, const char *key, const char *value, const char *description, bool abort_when_registered ))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   try
   {
      *event_value = sys.getInstrumentation()->getInstrumentationDictionary()->registerEventValue(key, value,  description, abort_when_registered);
   } catch ( nanos_err_t err) {
      return err;
   }
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_register_value_with_val, ( nanos_event_value_t val, const char *key, const char *value, const char *description, bool abort_when_registered ))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   try
   {
      sys.getInstrumentation()->getInstrumentationDictionary()->registerEventValue(key, value, val, description, abort_when_registered);
   } catch ( nanos_err_t err) {
      return err;
   }
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_get_key, (const char *key, nanos_event_key_t *event_key))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   try
   {
      *event_key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey(key);
   } catch ( nanos_err_t err) {
      return err;
   }
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_get_value, (const char *key, const char *value, nanos_event_value_t *event_value))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   try
   {
      *event_value = sys.getInstrumentation()->getInstrumentationDictionary()->getEventValue(key, value);
   } catch ( nanos_err_t err) {
      return err;
   }
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_events, ( unsigned int num_events, nanos_event_t events[] ))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   try
   {
      Instrumentation::Event *e = (Instrumentation::Event *) alloca ( sizeof(Instrumentation::Event) * num_events ); 

      for (unsigned int i = 0; i < num_events; i++ ) {
         switch ( events[i].type ) {
            case NANOS_STATE_START:
               sys.getInstrumentation()->createStateEvent( &e[i],(nanos_event_state_value_t ) events[i].value);
               break;
            case NANOS_STATE_END:
               sys.getInstrumentation()->returnPreviousStateEvent(&e[i]);
               break;
            case NANOS_BURST_START:
               sys.getInstrumentation()->createBurstEvent(&e[i],events[i].key,events[i].value);
               break;
            case NANOS_BURST_END:
               sys.getInstrumentation()->closeBurstEvent(&e[i],events[i].key);
               break;
            case NANOS_POINT:
               sys.getInstrumentation()->createPointEvent(&e[i],events[i].key,events[i].value );
               break;
            case NANOS_PTP_START:
#if 0
               sys.getInstrumentation()->createPtPStart(&e[i],events[i].info.ptp.domain,events[i].info.ptp.id,*events[i].info.ptp.keys[0],*events[i].info.ptp.values,events[i].info.ptp.values);
#endif
               return NANOS_UNKNOWN_ERR;
               break;
            case NANOS_PTP_END:
#if 0
               sys.getInstrumentation()->createPtPEnd(&e[i],events[i].info.ptp.domain,events[i].info.ptp.id,events[i].info.ptp.nkvs,events[i].info.ptp.keys,events[i].info.ptp.values);
#endif
               return NANOS_UNKNOWN_ERR;
               break;
            default:
               return NANOS_UNKNOWN_ERR;
         }
      }

      sys.getInstrumentation()->addEventList( num_events,e);
   } catch ( nanos_err_t err) {
      return err;
   }
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_close_user_fun_event, ( void ))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
#ifdef GPU_DEV
   try
   {
      ( ( ext::GPUThread *) myThread )->enableWDClosingEvents();
   } catch ( nanos_err_t err) {
      return err;
   }
#endif
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_enable,())
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   try
   {
      sys.getInstrumentation()->enable();
   } catch ( nanos_err_t err) {
      return err;
   }
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_disable,())
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   try
   {
      sys.getInstrumentation()->disable();
   } catch ( nanos_err_t err) {
      return err;
   }
#endif
   return NANOS_OK;
}

