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

/*! \file nanos_instrument.cpp
 *  \brief Nanos++ services related with the instrumentation
 */
#include "nanos.h"
#include "system.hpp"
#include "instrumentation.hpp"
#include <alloca.h>

#ifdef GPU_DEV
#include "gputhread_decl.hpp"
#endif

#ifdef OpenCL_DEV
#include "openclthread_decl.hpp"
#endif


/*!\defgroup capi_instrument Instrumentation services.
 * \ingroup capi
 * \page capi_instrumentation_page Instrumentation Services
 * \ingroup capi_instrument
 *
 * \section introduction Introduction
 *
 * In order to manage supported type of events Nanos++ offers several data structures. Most of these structures have an internal C++ object equivalent which allow to use them in a C environment.
 *
 * Nanos++ C API offers several services in order to call specific instrumentation methods. These services can be used to inject some instrumentation code from the final user side.
 *
 * - Register/get InstrumentationDictionary services
 * - Raising event services
 * - Enable/disable state event services
 */

/*! \addtogroup capi_instrument
 *  \{
 */

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
      if ( strncmp( "user-funct-name", key, 15 ) == 0 ) {
         nanos_event_value_t use_this_val = ( nanos_event_value_t ) ( event_value );
         sys.getInstrumentation()->getInstrumentationDictionary()->registerEventValue(key, value, use_this_val, description, abort_when_registered);
         *event_value = use_this_val;
      } else {
         *event_value = sys.getInstrumentation()->getInstrumentationDictionary()->registerEventValue(key, value,  description, abort_when_registered);
      }
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
               sys.getInstrumentation()->closeBurstEvent(&e[i],events[i].key,events[i].value);
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
#ifdef OpenCL_DEV
   try
   {
      ( ( ext::OpenCLThread * ) myThread )->enableWDClosingEvents();
   } catch ( nanos_err_t err ) {
      return err;
   }
#endif
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_raise_gpu_kernel_launch_event, ( void ))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
#ifdef GPU_DEV
   try
   {
      ( ( ext::GPUThread * ) myThread )->raiseKernelLaunchEvent();
   } catch ( nanos_err_t err ) {
      return err;
   }
#endif
#endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_close_gpu_kernel_launch_event, ( void ))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
#ifdef GPU_DEV
   try
   {
      ( ( ext::GPUThread * ) myThread )->closeKernelLaunchEvent();
   } catch ( nanos_err_t err ) {
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

NANOS_API_DEF(nanos_err_t, nanos_instrument_begin_burst,(nanos_string_t key, nanos_string_t key_descr, nanos_string_t value, nanos_string_t value_descr))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
    try
    {
        nanos_event_t e;
        nanos_instrument_register_key ( &e.key, (const char*) key,
                (const char*) key_descr, /* abort_when_registered */ false );

        nanos_instrument_register_value ( &e.value, (const char*) key,
                (const char*) value, (const char*) value_descr, /* abort_when_registered */ false);

        e.type = NANOS_BURST_START;
        nanos_instrument_events( 1, &e);
    } catch ( nanos_err_t err) {
        return err;
    }
 #endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_end_burst,(nanos_string_t key, nanos_string_t value))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
    try
    {
        nanos_event_t e;
        nanos_instrument_get_key( (const char*) key, &e.key);
        nanos_instrument_get_value( (const char*) key, (const char*) value, &e.value);
        e.type = NANOS_BURST_END;
        nanos_instrument_events( 1, &e);
    } catch ( nanos_err_t err) {
        return err;
    }
 #endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_begin_burst_with_val,(nanos_string_t key, nanos_string_t key_descr, nanos_event_value_t *val))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
    try
    {
        nanos_event_t e;
        nanos_instrument_register_key ( &e.key, (const char*) key,
                (const char*) key_descr, /* abort_when_registered */ false );

        e.value = *val;
        e.type = NANOS_BURST_START;
        nanos_instrument_events( 1, &e);
    } catch ( nanos_err_t err) {
        return err;
    }
 #endif
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_instrument_end_burst_with_val,(nanos_string_t key, nanos_event_value_t *val))
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
    try
    {
        nanos_event_t e;
        nanos_instrument_get_key( (const char*) key, &e.key);
        e.value = *val;
        e.type = NANOS_BURST_END;
        nanos_instrument_events( 1, &e);
    } catch ( nanos_err_t err) {
        return err;
    }
 #endif
   return NANOS_OK;
}
/*!
 * \}
 */ 
