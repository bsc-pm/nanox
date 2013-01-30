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
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "workdescriptor.hpp"
#include "smpdd.hpp"
#include "gpudd.hpp"
#include "plugin.hpp"
#include "instrumentation.hpp"
#include "instrumentationmodule_decl.hpp"

using namespace nanos;

// Internal definition for const
struct nanos_const_wd_definition_internal_t : nanos_const_wd_definition_tag 
{
   nanos_device_t devices[];
};

NANOS_API_DEF(void *, nanos_smp_factory, ( void *args ))
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   return ( void * )new ext::SMPDD( smp->outline );
}

NANOS_API_DEF(nanos_wd_t, nanos_current_wd, (void))
{
   nanos_wd_t cwd = myThread->getCurrentWD();

   return cwd;
}

NANOS_API_DEF(int, nanos_get_wd_id, ( nanos_wd_t wd ))
{
   WD *lwd = ( WD * )wd;
   int id = lwd->getId();

   return id;
}

/*! \brief Creates a new WorkDescriptor
 *
 *  \sa nanos::WorkDescriptor
 */
NANOS_API_DEF( nanos_err_t, nanos_create_wd_compact, ( nanos_wd_t *uwd, nanos_const_wd_definition_t *const_data_ext, nanos_wd_dyn_props_t *dyn_props,
                                                       size_t data_size, void ** data, nanos_wg_t uwg, nanos_copy_data_t **copies, nanos_region_dimension_internal_t **dimensions ) )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","*_create_wd",NANOS_CREATION) );

   nanos_const_wd_definition_internal_t *const_data = reinterpret_cast<nanos_const_wd_definition_internal_t*>(const_data_ext);

   try 
   {
      if ( ( &const_data->props == NULL  || ( &const_data->props != NULL  && !const_data->props.mandatory_creation ) ) && !sys.throttleTaskIn() ) {
         *uwd = 0;
         return NANOS_OK;
      }
      sys.createWD ( (WD **) uwd, const_data->num_devices, const_data->devices, data_size, const_data->data_alignment, (void **) data, (WG *) uwg, &const_data->props, dyn_props, const_data->num_copies, copies, const_data->num_dimensions, dimensions, NULL );

   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_set_translate_function, ( nanos_wd_t wd, nanos_translate_args_t translate_args ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","*_set_translate_function",NANOS_CREATION) );
   try 
   {
      WD *lwd = ( WD * ) wd;
      lwd->setTranslateArgs( translate_args );
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

/*! \brief Creates a new Sliced WorkDescriptor
 *
 *  \sa nanos::WorkDescriptor
 */
NANOS_API_DEF(nanos_err_t, nanos_create_sliced_wd, ( nanos_wd_t *uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size,
                                                     int outline_data_align, void ** outline_data, nanos_wg_t uwg, nanos_slicer_t slicer,
                                                     nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props, size_t num_copies,
                                                     nanos_copy_data_t **copies, size_t num_dimensions, nanos_region_dimension_internal_t **dimensions ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","*_create_wd",NANOS_CREATION) );

   try 
   {
      if ( ( props == NULL  || ( props != NULL  && !props->mandatory_creation ) ) && !sys.throttleTaskIn() ) {
         *uwd = 0;
         return NANOS_OK;
      }

      sys.createSlicedWD ( (WD **) uwd, num_devices, devices, outline_data_size, outline_data_align, outline_data, (WG *) uwg,
                           (Slicer *) slicer, props, dyn_props, num_copies, copies, num_dimensions, dimensions );

   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_submit, ( nanos_wd_t uwd, size_t num_data_accesses, nanos_data_access_t *data_accesses, nanos_team_t team ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","submit",NANOS_SCHEDULING) );

   try {
      ensure( uwd,"NULL WD received" );

      WD * wd = ( WD * ) uwd;

      if ( team != NULL ) {
         warning( "Submitting to another team not implemented yet" );
      }

      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

      NANOS_INSTRUMENT ( static nanos_event_key_t create_wd_id = ID->getEventKey("create-wd-id"); )
      NANOS_INSTRUMENT ( static nanos_event_key_t create_wd_ptr = ID->getEventKey("create-wd-ptr"); )
      NANOS_INSTRUMENT ( static nanos_event_key_t wd_num_deps = ID->getEventKey("wd-num-deps"); )
      NANOS_INSTRUMENT ( static nanos_event_key_t wd_deps_ptr = ID->getEventKey("wd-deps-ptr"); )

      NANOS_INSTRUMENT ( nanos_event_key_t Keys[4]; )
      NANOS_INSTRUMENT ( nanos_event_value_t Values[4]; )

      NANOS_INSTRUMENT ( Keys[0] = create_wd_id; )
      NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) wd->getId(); )

      NANOS_INSTRUMENT ( Keys[1] = create_wd_ptr; )
      NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) wd; )

      NANOS_INSTRUMENT ( Keys[2] = wd_num_deps; )
      NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) num_data_accesses; )

      NANOS_INSTRUMENT ( Keys[3] = wd_deps_ptr; );
      NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) data_accesses; )

      NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(4, Keys, Values); )

      NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) wd->getId(), 0, 0 );)

      if ( data_accesses != NULL ) {
         sys.submitWithDependencies( *wd, num_data_accesses, data_accesses );
         return NANOS_OK;
      }

      sys.submit( *wd );
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}


// data must be not null
NANOS_API_DEF( nanos_err_t, nanos_create_wd_and_run_compact, ( nanos_const_wd_definition_t *const_data_ext, nanos_wd_dyn_props_t *dyn_props, 
                                                               size_t data_size, void * data, size_t num_data_accesses, nanos_data_access_t *data_accesses,
                                                               nanos_copy_data_t *copies, nanos_region_dimension_internal_t *dimensions, nanos_translate_args_t translate_args ) )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","create_wd_and_run", NANOS_CREATION) );

   nanos_const_wd_definition_internal_t *const_data = reinterpret_cast<nanos_const_wd_definition_internal_t*>(const_data_ext);

   try {
      if ( const_data->num_devices > 1 ) warning( "Multiple devices not yet supported. Using first one" );

      // TODO: choose device
      
      WD wd( ( DD* ) const_data->devices[0].factory( const_data->devices[0].arg ), data_size, const_data->data_alignment, data, const_data->num_copies, copies );
      wd.setTranslateArgs( translate_args );
      
      // Set WD's socket
      wd.setSocket( sys.getCurrentSocket() );
      
      if ( wd.getSocket() >= sys.getNumSockets() )
         throw NANOS_INVALID_PARAM;

      // set properties
      if ( &const_data->props != NULL ) {
         if ( const_data->props.tied ) wd.tied();
         if ( dyn_props && dyn_props->tie_to ) {
            if (dyn_props->tie_to == myThread) {
               wd.tieTo( *( BaseThread * ) dyn_props->tie_to );
            } else {
               fatal ( "Tiedness violation" );
            }
            // Set priority
            wd.setPriority( dyn_props->priority );
         }
      }

      int pmDataSize = sys.getPMInterface().getInternalDataSize();
      char pmData[pmDataSize];
      if ( pmDataSize > 0 ) {
        wd.setInternalData(pmData);
      }

      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

      NANOS_INSTRUMENT ( static nanos_event_key_t create_wd_id = ID->getEventKey("create-wd-id"); )
      NANOS_INSTRUMENT ( static nanos_event_key_t create_wd_ptr = ID->getEventKey("create-wd-ptr"); )
      NANOS_INSTRUMENT ( static nanos_event_key_t wd_num_deps = ID->getEventKey("wd-num-deps"); )
      NANOS_INSTRUMENT ( static nanos_event_key_t wd_deps_ptr = ID->getEventKey("wd-deps-ptr"); )

      NANOS_INSTRUMENT ( nanos_event_key_t Keys[4]; )
      NANOS_INSTRUMENT ( nanos_event_value_t Values[4]; ) 

      NANOS_INSTRUMENT ( Keys[0] = create_wd_id; )
      NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) wd.getId(); )

      NANOS_INSTRUMENT ( Keys[1] = create_wd_ptr; )
      NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) &wd; )

      NANOS_INSTRUMENT ( Keys[2] = wd_num_deps; )
      NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) num_data_accesses; )

      NANOS_INSTRUMENT ( Keys[3] = wd_deps_ptr; );
      NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) data_accesses; )

      NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(4, Keys, Values); )

      NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent( NANOS_WD_DOMAIN, (nanos_event_id_t) wd.getId(), 0, 0 ); )

      if ( data_accesses != NULL ) {
         sys.waitOn( num_data_accesses, data_accesses );
      }

      NANOS_INSTRUMENT( InstrumentState inst1(NANOS_RUNTIME) );
      sys.inlineWork( wd );
      NANOS_INSTRUMENT( inst1.close() );

   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_set_internal_wd_data, ( nanos_wd_t wd, void *data ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","set_internal_wd_data",NANOS_RUNTIME) );

   try {
      WD *lwd = ( WD * ) wd;

      lwd->setInternalData( data );
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_get_internal_wd_data, ( nanos_wd_t wd, void **data ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_internal_wd_data",NANOS_RUNTIME) );

   try {
      WD *lwd = ( WD * ) wd;
      void *ldata;

      ldata = lwd->getInternalData();

      *data = ldata;
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_yield, ( void ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","yield",NANOS_SCHEDULING) );

   try {
      Scheduler::yield();

   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}


NANOS_API_DEF(nanos_err_t, nanos_slicer_get_specific_data, ( nanos_slicer_t slicer, void ** data ))
{                                                                                                                                                        
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_specific_data",NANOS_RUNTIME) );

   try {
      *data = ((Slicer *)slicer)->getSpecificData();
   } catch ( nanos_err_t e) { 
      return e;                                                                                                                          
   }                                                                                                                                                     
                                                                                                                                                         
   return NANOS_OK;                                                                                                                                      
}   

NANOS_API_DEF(unsigned int, nanos_get_wd_priority, ( nanos_wd_t wd ))
{
   WD *lwd = ( WD * )wd;
   return lwd->getPriority();
}

NANOS_API_DEF(nanos_err_t, nanos_get_num_ready_tasks, ( unsigned int *ready_tasks ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_num_ready_tasks",NANOS_RUNTIME) );
   try {
      *ready_tasks = (unsigned int) sys.getReadyNum();
   } catch ( nanos_err_t e) {
      return e;                                                                                                                          
   }
   return NANOS_OK;

}

NANOS_API_DEF(nanos_err_t, nanos_get_num_total_tasks, ( unsigned int *total_tasks ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_num_total_tasks",NANOS_RUNTIME) );
   try {
      *total_tasks = (unsigned int) sys.getTaskNum();
   } catch ( nanos_err_t e) {
      return e;                                                                                                                          
   }
   return NANOS_OK;

}

NANOS_API_DEF(nanos_err_t, nanos_get_num_nonready_tasks, ( unsigned int *nonready_tasks ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_num_nonready_tasks",NANOS_RUNTIME) );
   try {
      unsigned int ready = (unsigned int) sys.getReadyNum();
      unsigned int total = (unsigned int) sys.getTaskNum();
      *nonready_tasks = (total > ready)? total - ready : 0;
   } catch ( nanos_err_t e) {
      return e;                                                                                                                          
   }
   return NANOS_OK;

}

NANOS_API_DEF(nanos_err_t, nanos_get_num_running_tasks, ( unsigned int *running_tasks ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_num_running_tasks",NANOS_RUNTIME) );

   try {
      *running_tasks = sys.getRunningTasks();
   } catch ( nanos_err_t e ) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_get_num_blocked_tasks, ( unsigned int *blocked_tasks ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_num_blocked_tasks",NANOS_RUNTIME) );
   try {
      unsigned int ready = (unsigned int) sys.getReadyNum();
      unsigned int total = (unsigned int) sys.getTaskNum();
      unsigned int nonready = (total > ready)? total - ready : 0;
      unsigned int running = (unsigned int) sys.getRunningTasks();
      *blocked_tasks = ( nonready > running )? nonready - running : 0;
   } catch ( nanos_err_t e) {
      return e;
   }
   return NANOS_OK;

}
