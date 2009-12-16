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

using namespace nanos;

// TODO: move to dependent part
const size_t nanos_smp_dd_size = sizeof(ext::SMPDD);

void * nanos_smp_factory( void *prealloc, void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;

   if ( prealloc != NULL )
      return ( void * )new (prealloc) ext::SMPDD( smp->outline );
   else 
      return ( void * )new ext::SMPDD( smp->outline );
}

nanos_wd_t nanos_current_wd()
{
   return myThread->getCurrentWD();
}

int nanos_get_wd_id ( nanos_wd_t wd )
{
   WD *lwd = ( WD * )wd;
   return lwd->getId();
}

nanos_slicer_t nanos_find_slicer ( char * slicer )
{
   try
   {
      return sys.getSlicer ( std::string(slicer) );

   } catch ( ... ) {
      return ( nanos_slicer_t ) NULL;
   }
}

/*! \brief Creates a new WorkDescriptor
 *
 *  \sa nanos::WorkDescriptor
 */
nanos_err_t nanos_create_wd (  nanos_wd_t *uwd, size_t num_devices, nanos_device_t *devices, size_t data_size,
                               void ** data, nanos_wg_t uwg, nanos_wd_props_t *props )
{
   try 
   {
      if ( ( props == NULL  || ( props != NULL  && !props->mandatory_creation ) ) && !sys.throttleTask() ) {
         *uwd = 0;
         return NANOS_OK;
      }
      sys.createWD ( (WD **) uwd, num_devices, devices, data_size, (void **) data, (WG *) uwg, props);

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
                               nanos_slicer_data_t * slicer_data, nanos_wd_props_t *props )
{
   try 
   {
      if ( ( props == NULL  || ( props != NULL  && !props->mandatory_creation ) ) && !sys.throttleTask() ) {
         *uwd = 0;
         return NANOS_OK;
      }
      sys.createSlicedWD ( (WD **) uwd, num_devices, devices, outline_data_size, outline_data, (WG *) uwg,
                           (Slicer *) slicer, slicer_data_size, (SlicerData **) slicer_data, props);

   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_submit ( nanos_wd_t uwd, size_t num_deps, nanos_dependence_t *deps, nanos_team_t team )
{
   try {
      ensure( uwd,"NULL WD received" );

      WD * wd = ( WD * ) uwd;

      if ( team != NULL ) {
         warning( "Submitting to another team not implemented yet" );
      }

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
                                      size_t num_deps, nanos_dependence_t *deps, nanos_wd_props_t *props )
{
   try {
      if ( num_devices > 1 ) warning( "Multiple devices not yet supported. Using first one" );

      // TODO: pre-allocate devices
      WD wd( ( DD* ) devices[0].factory( 0, devices[0].arg ), data_size, data );

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
