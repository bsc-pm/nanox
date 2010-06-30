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
#include "instrumentor.hpp"
#include "instrumentormodule_decl.hpp"

using namespace nanos;

// TODO: move to dependent part
const size_t nanos_smp_dd_size = sizeof(ext::SMPDD);

#ifdef GPU_DEV
const size_t nanos_gpu_dd_size = sizeof(ext::GPUDD);

void * nanos_gpu_factory( void *prealloc, void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   if ( prealloc != NULL )
   {
      return ( void * )new (prealloc) ext::GPUDD( smp->outline );
   }
   else
   {
      return ( void * ) new ext::GPUDD( smp->outline );
   }
}
#endif

void * nanos_smp_factory( void *prealloc, void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;

   if ( prealloc != NULL )
   {
      return ( void * )new (prealloc) ext::SMPDD( smp->outline );
   }
   else 
   {
      return ( void * )new ext::SMPDD( smp->outline );
   }
}

nanos_wd_t nanos_current_wd()
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","current_wd",RUNTIME) );

   nanos_wd_t cwd = myThread->getCurrentWD();

   return cwd;
}

int nanos_get_wd_id ( nanos_wd_t wd )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","get_wd_id",RUNTIME) );

   WD *lwd = ( WD * )wd;
   int id = lwd->getId();

   return id;
}

/*! \brief Creates a new WorkDescriptor
 *
 *  \sa nanos::WorkDescriptor
 */
nanos_err_t nanos_create_wd (  nanos_wd_t *uwd, size_t num_devices, nanos_device_t *devices, size_t data_size,
                               void ** data, nanos_wg_t uwg, nanos_wd_props_t *props, size_t num_copies, nanos_copy_data_t **copies )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","*_create_wd",RUNTIME) );

   try 
   {
      if ( ( props == NULL  || ( props != NULL  && !props->mandatory_creation ) ) && !sys.throttleTask() ) {
         *uwd = 0;
         return NANOS_OK;
      }
      sys.createWD ( (WD **) uwd, num_devices, devices, data_size, (void **) data, (WG *) uwg, props, num_copies, copies );

   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

/*! \brief Creates a new Sliced WorkDescriptor
 *
 *  \sa nanos::WorkDescriptor
 */
nanos_err_t nanos_create_sliced_wd ( nanos_wd_t *uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size,
                               void ** outline_data, nanos_wg_t uwg, nanos_slicer_t slicer, size_t slicer_data_size,
                               nanos_slicer_data_t * slicer_data, nanos_wd_props_t *props, size_t num_copies, nanos_copy_data_t **copies )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","*_create_wd",RUNTIME) );

   try 
   {
      if ( ( props == NULL  || ( props != NULL  && !props->mandatory_creation ) ) && !sys.throttleTask() ) {
         *uwd = 0;
         return NANOS_OK;
      }
      if ( slicer_data == NULL ) {
         return NANOS_UNKNOWN_ERR;
      }

      sys.createSlicedWD ( (WD **) uwd, num_devices, devices, outline_data_size, outline_data, (WG *) uwg,
                           (Slicer *) slicer, slicer_data_size, (SlicerData *&) *slicer_data, props, num_copies, copies );

   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_submit ( nanos_wd_t uwd, size_t num_deps, nanos_dependence_t *deps, nanos_team_t team )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","submit",RUNTIME) );

   try {
      ensure( uwd,"NULL WD received" );

      WD * wd = ( WD * ) uwd;

      if ( team != NULL ) {
         warning( "Submitting to another team not implemented yet" );
      }

      NANOS_INSTRUMENTOR ( static nanos_event_key_t throw_key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("wd-throw") );
      NANOS_INSTRUMENTOR ( static nanos_event_key_t dep_key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("nanos-dep") );
      NANOS_INSTRUMENTOR ( unsigned int nkvs = num_deps+1; );
      NANOS_INSTRUMENTOR ( nanos_event_key_t *Keys = new nanos_event_key_t(nkvs); );
      NANOS_INSTRUMENTOR ( nanos_event_value_t *Values = new nanos_event_value_t(nkvs); );
      NANOS_INSTRUMENTOR ( Keys[0] = (unsigned int) throw_key; );
      NANOS_INSTRUMENTOR ( Values[0] = (nanos_event_value_t) wd->getId(); );
      NANOS_INSTRUMENTOR ( for (unsigned int i = 1; i< nkvs; i++) {
          Keys[i] = (unsigned int) dep_key;
          Values[i] = (nanos_event_value_t) &deps[i-1];
      } );
      NANOS_INSTRUMENTOR( sys.getInstrumentor()->throwPointEventNkvs(nkvs, Keys, Values) ); 

      if ( deps != NULL ) {
         sys.submitWithDependencies( *wd, num_deps, deps );
         return NANOS_OK;
      }

      sys.submit( *wd );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}


// data must be not null
nanos_err_t nanos_create_wd_and_run ( size_t num_devices, nanos_device_t *devices, size_t data_size, void * data, 
                                      size_t num_deps, nanos_dependence_t *deps, nanos_wd_props_t *props,
                                      size_t num_copies, nanos_copy_data_t *copies )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","create_wd_and_run",RUNTIME) );

   try {
      if ( num_devices > 1 ) warning( "Multiple devices not yet supported. Using first one" );

      // TODO: choose device
      // pre-allocate device
      char chunk[devices[0].dd_size];

      WD wd( ( DD* ) devices[0].factory( chunk, devices[0].arg ), data_size, data, num_copies, copies );

      NANOS_INSTRUMENTOR ( static nanos_event_key_t throw_key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("wd-throw") );
      NANOS_INSTRUMENTOR ( static nanos_event_key_t dep_key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("nanos-dep") );
      NANOS_INSTRUMENTOR ( unsigned int nkvs = num_deps+1; );
      NANOS_INSTRUMENTOR ( nanos_event_key_t *Keys = new nanos_event_key_t(nkvs); );
      NANOS_INSTRUMENTOR ( nanos_event_value_t *Values = new nanos_event_value_t(nkvs); );
      NANOS_INSTRUMENTOR ( Keys[0] = (unsigned int) throw_key; );
      NANOS_INSTRUMENTOR ( Values[0] = (nanos_event_value_t) wd.getId(); );
      NANOS_INSTRUMENTOR ( for (unsigned int i = 1; i< nkvs; i++) {
          Keys[i] = (unsigned int) dep_key;
          Values[i] = (nanos_event_value_t) &deps[i-1];
      } );
      NANOS_INSTRUMENTOR( sys.getInstrumentor()->throwPointEventNkvs(nkvs, Keys, Values) ); 

      if ( deps != NULL ) {
         sys.waitOn( num_deps, deps );
      }

      sys.inlineWork( wd );

   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_set_internal_wd_data ( nanos_wd_t wd, void *data )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","set_internal_wd_data",RUNTIME) );

   try {
      WD *lwd = ( WD * ) wd;

      lwd->setInternalData( data );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_get_internal_wd_data ( nanos_wd_t wd, void **data )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","get_internal_wd_data",RUNTIME) );

   try {
      WD *lwd = ( WD * ) wd;
      void *ldata;

      ldata = lwd->getInternalData();

      *data = ldata;
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_yield ( void )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","yield",RUNTIME) );

   try {
      Scheduler::yield();

   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}


nanos_err_t nanos_slicer_get_specific_data ( nanos_slicer_t slicer, void ** data )
{                                                                                                                                                        
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","get_specific_data",RUNTIME) );

   try {
      *data = ((Slicer *)slicer)->getSpecificData();
   } catch ( ... ) {                                                                                                                                     
      return NANOS_UNKNOWN_ERR;                                                                                                                          
   }                                                                                                                                                     
                                                                                                                                                         
   return NANOS_OK;                                                                                                                                      
}   

