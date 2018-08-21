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

//! \file nanos_wd.cpp
//! \brief Nanos++ services related with WorkDescriptor 

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

//! \defgroup capi_wd WorkDescriptor services.
//! \ingroup capi
//! \{

using namespace nanos;


/*! \brief Returns the WD of the current task.
 *
 */
NANOS_API_DEF(nanos_wd_t, nanos_current_wd, (void))
{
   nanos_wd_t cwd = myThread->getCurrentWD();

   return cwd;
}

/*! \brief Create a SMP DeviceData
 *  
 *  \param args Architecture (SMP) specific info
 *  \sa nanos_smp_args_t
 */
NANOS_API_DEF(void *, nanos_smp_factory, ( void *args ))
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   return ( void * ) NEW ext::SMPDD( smp->outline );
}

/*! \brief Returns the id of the specified WD.
 *
 *  \param wd is the WorkDescriptor
 */
NANOS_API_DEF(int, nanos_get_wd_id, ( nanos_wd_t wd ))
{
   WD *lwd = ( WD * )wd;
   int id = lwd->getId();
   if ( lwd->getHostId() != 0 ) {
      id = lwd->getHostId();
   }

   return id;
}

/*! \brief Returns the description of the specified WD.
 *
 *  \param [out] string description
 *  \param [in] wd is the WorkDescriptor
 */
NANOS_API_DEF(nanos_err_t, nanos_get_wd_description, ( const char **description, nanos_wd_t wd ))
{
   try 
   {
      WD *lwd = ( WD * )wd;
      *description = lwd->getDescription();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

/*! \brief Creates a new WorkDescriptor
 *
 *  \param uwd is WorkDescriptor to be initialized, if *uwd == 0 is allocated
 *  \param const_data_ext bundle of constant information (shared among all work descriptor instances)
 *  \param dyn_props bundle of dynamic properties (specific for a given work descriptor instance)
 *  \param data_size size of data environment
 *  \param data if *data == 0, a environment of data_size will be allocated, otherwise ignored
 *  \param uwg  if (wg != 0) the new WD is added to that WD
 *  \param copies  if num_copies > 0, list of nanos_copy_data_t elements that describe copy-in/out operations for the WD
 *  \param dimensions dimensions
 *  \sa nanos::WorkDescriptor
 */
NANOS_API_DEF( nanos_err_t, nanos_create_wd_compact, ( nanos_wd_t *uwd, nanos_const_wd_definition_t *const_data_ext, nanos_wd_dyn_props_t *dyn_props,
                                                       size_t data_size, void ** data, nanos_wg_t uwg, nanos_copy_data_t **copies, nanos_region_dimension_internal_t **dimensions ) )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","*_create_wd",NANOS_CREATION) );

   nanos_const_wd_definition_internal_t *const_data = reinterpret_cast<nanos_const_wd_definition_internal_t*>(const_data_ext);

   try 
   {
      if ( !const_data->props.mandatory_creation && !sys.throttleTaskIn() ) {
         *uwd = 0;
         return NANOS_OK;
      }
      sys.createWD ( (WD **) uwd, const_data->num_devices, const_data->devices, data_size, const_data->data_alignment,
                     (void **) data, (WD *) uwg, &const_data->props, dyn_props, const_data->num_copies, copies,
                     const_data->num_dimensions, dimensions, NULL, const_data->description, NULL );

   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

/*! \brief Set arguments ready for translation
 *
 *  \sa nanos::WorkDescriptor
 */
NANOS_API_DEF(nanos_err_t, nanos_set_translate_function, ( nanos_wd_t wd, nanos_translate_args_t translate_args ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","set_translate_function",NANOS_CREATION) );
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

      sys.createWD ( (WD **) uwd, num_devices, devices, outline_data_size, outline_data_align,
                     (void **) outline_data, (WD *) uwg, props, dyn_props, num_copies, copies,
                     num_dimensions, dimensions, NULL, "", (Slicer *) slicer );

   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

/*! \brief Submit a WorkDescriptor
 *
 *  \sa nanos::WorkDescriptor
 */
NANOS_API_DEF(nanos_err_t, nanos_submit, ( nanos_wd_t uwd, size_t num_data_accesses, nanos_data_access_t *data_accesses, nanos_team_t team ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","submit",NANOS_SCHEDULING) );

   try {
      ensure( uwd,"NULL WD received" );

      WD * wd = ( WD * ) uwd;

      if ( team != NULL ) {
         warning( "Submitting to another team not implemented yet" );
      }

      if ( sys.getVerboseCopies() ) {
         *myThread->_file << "Submitting WD " << wd->getId() << " " << (wd->getDescription() == NULL ? "n/a" : wd->getDescription()) << std::endl;
      }

      sys.setupWD( *wd, myThread->getCurrentWD() );

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

      if ( num_data_accesses != 0 && data_accesses != NULL ) {
         sys.submitWithDependencies( *wd, num_data_accesses, data_accesses );
         return NANOS_OK;
      }

      sys.submit( *wd );
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}


/*! \brief Creates a new WorkDescriptor and execute it inmediately
 *
 *  \param const_data_ext
 *  \param dyn_props
 *  \param data_size
 *  \param data (must not be null)
 *  \param num_data_accesses
 *  \param data_accesses
 *  \param copies
 *  \param dimensions
 *  \param translate_args
 *  \sa nanos::WorkDescriptor
 */
NANOS_API_DEF( nanos_err_t, nanos_create_wd_and_run_compact, ( nanos_const_wd_definition_t *const_data_ext, nanos_wd_dyn_props_t *dyn_props, 
                                                               size_t data_size, void * data, size_t num_data_accesses, nanos_data_access_t *data_accesses,
                                                               nanos_copy_data_t *copies, nanos_region_dimension_internal_t *dimensions, nanos_translate_args_t translate_args ) )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","create_wd_and_run", NANOS_CREATION) );

   nanos_const_wd_definition_internal_t *const_data = reinterpret_cast<nanos_const_wd_definition_internal_t*>(const_data_ext);

   try {
      if ( const_data->num_devices > 1 ) warning( "Multiple devices not yet supported. Using first one" );

      //! \todo if multiple devices we need to choose one of them
      
      WD wd( (DD*) const_data->devices[0].factory( const_data->devices[0].arg ), data_size, const_data->data_alignment,
             data, const_data->num_copies, copies, NULL, (char *) const_data->description);

      wd.setTranslateArgs( translate_args );
      wd.forceParent( myThread->getCurrentWD() );
      
      // Set WD's socket
      wd.setNUMANode( sys.getUserDefinedNUMANode() );

      wd.copyReductions (myThread->getCurrentWD() );

      if ( wd.getNUMANode() >= (int)sys.getNumNumaNodes() )
         throw NANOS_INVALID_PARAM;

      // set properties
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

      int pmDataSize = sys.getPMInterface().getInternalDataSize();
      char pmData[pmDataSize];
      if ( pmDataSize > 0 ) {
        sys.getPMInterface().initInternalData( pmData );
        wd.setInternalData(pmData, /* ownedByWD */ false);
      }

      int schedDataSize = sys.getDefaultSchedulePolicy()->getWDDataSize();
      char schedData[schedDataSize];
      if ( schedDataSize  > 0 ) {
         sys.getDefaultSchedulePolicy()->initWDData( schedData );
         wd.setSchedulerData( reinterpret_cast<ScheduleWDData*>( schedData ), /* ownedByWD */ false );
      }

      sys.setupWD( wd, myThread->getCurrentWD() );

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

/*! \brief Set internal WorkDescriptor's data
 *
 */
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

/*! \brief Get internal WorkDescriptor's data
 *
 */
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

/*! \brief Yields current thread (if other WorkDescriptor is ready to be executed)
 *
 */
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

/*! \brief Get Slicer specific data
 *
 */
NANOS_API_DEF(nanos_err_t, nanos_slicer_get_specific_data, ( nanos_slicer_t slicer, void ** data ))
{                                                                                                                                                        
   //! Why we are not instrumenting the next line
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_specific_data",NANOS_RUNTIME) );

   try {
      *data = ((Slicer *)slicer)->getSpecificData();
   } catch ( nanos_err_t e) { 
      return e;                                                                                                                          
   }                                                                                                                                                     
                                                                                                                                                         
   return NANOS_OK;                                                                                                                                      
}   

//! \brief Get WorkDescriptor's priority
NANOS_API_DEF(int, nanos_get_wd_priority, ( nanos_wd_t wd ))
{
   //! \note Why this is not instrumented?
   return ((WD*)wd)->getPriority();
}

//! \brief Set WorkDescriptor's priority
NANOS_API_DEF(void, nanos_set_wd_priority, ( nanos_wd_t wd, int p ))
{
   //! \note Why this is not instrumented?
   ((WD*)wd)->setPriority( p );
}

/*! \brief Get number of ready tasks
 *
 */
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

/*! \brief Get number of total tasks (ready, blocked and running)
 *
 */
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

/*! \brief Get number of non ready tasks (blocked and running)
 *
 */
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

/*! \brief Get number of running tasks
 *
 *  \param running_tasks
 */
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

/*! \brief Get number of blocked tasks
 *
 */
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


/*! \brief Sets copies of a wd
 *
 */
NANOS_API_DEF(nanos_err_t, nanos_set_copies, (nanos_wd_t wd, int num_copies, nanos_copy_data_t *copies))
{
    NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","set_copies",NANOS_RUNTIME) );
    try {
        WD *lwd = ( WD * )wd;
        lwd->setCopies(num_copies, copies);
    } catch ( nanos_err_t e) {
        return e;
    }

    return NANOS_OK;
}

/*! \brief Has current WD final attribute 
 *
 */
NANOS_API_DEF(nanos_err_t, nanos_in_final, ( bool *result ))
{
    NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","in_final",NANOS_RUNTIME) );
    try {
       *result = myThread->getCurrentWD()->isFinal();
    } catch ( nanos_err_t e) {
       return e;
    }
    return NANOS_OK;
}
/*! \brief Setting current WD final attribute
 *
 */
NANOS_API_DEF(nanos_err_t, nanos_set_final, ( bool value ))
{
    NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","set_final",NANOS_RUNTIME) );
    try {
       myThread->getCurrentWD()->setFinal( value );
    } catch ( nanos_err_t e) {
       return e;
    }
    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_set_create_local_tasks, ( bool value ))
{
    NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","set_create_local_tasks",NANOS_RUNTIME) );
    try {
       sys.setCreateLocalTasks( value );
    } catch ( nanos_err_t e) {
       return e;
    }
    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_switch_to_thread, ( unsigned int *thid ))
{
    // FIXME NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","switch_to_thread",NANOS_RUNTIME) );
    try {
       if ( myThread->getCurrentWD()->isTiedTo() ) {
          return NANOS_INVALID_REQUEST;
       }
       sys.switchToThread( *thid );
    } catch ( nanos_err_t e) {
       return e;
    }
    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_is_tied, ( bool *result ))
{
    // FIXME NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","is_tied",NANOS_RUNTIME) );
    try {
       *result = myThread->getCurrentWD()->isTiedTo() != NULL;
    } catch ( nanos_err_t e) {
       return e;
    }
    return NANOS_OK;
}

NANOS_API_DEF (nanos_err_t, nanos_task_fortran_array_reduction_register, ( void *orig, void *dep,
         size_t array_descriptor_size, void (*init)( void *, void * ), void (*reducer)( void *, void * ),
         void (*reducer_orig_var)( void *, void * ) ) )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","task_reduction_register",NANOS_RUNTIME) );
   try {
      myThread->getCurrentWD()->registerFortranArrayTaskReduction(
            orig, dep, array_descriptor_size, init, reducer, reducer_orig_var );
   } catch ( nanos_err_t e) {
      return e;
   }
   return NANOS_OK;
}


NANOS_API_DEF (nanos_err_t, nanos_task_reduction_register, ( void *orig, size_t size_target, size_t size_elem,
         void (*init)( void *, void * ), void (*reducer)( void *, void * ) ) )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","task_reduction_register",NANOS_RUNTIME) );
   try {
       myThread->getCurrentWD()->registerTaskReduction( orig, size_target, size_elem, init, reducer );
   } catch ( nanos_err_t e) {
      return e;
   }
   return NANOS_OK;
}

NANOS_API_DEF (nanos_err_t, nanos_task_reduction_get_thread_storage, ( void *orig, void **tpd ) )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","task_reduction_get_thread_storage",NANOS_RUNTIME) );
   try {
      *tpd = myThread->getCurrentWD()->getTaskReductionThreadStorage( orig, myThread->getTeamId() );
   } catch ( nanos_err_t e) {
      return e;
   }
   return NANOS_OK;
}
//! \}
