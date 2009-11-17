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


/*! \brief Creates a new WorkDescriptor
 *
 *  When it does a full allocation the layout is the following
 *
 *  +---------------+
 *  |     WD        |
 *  +---------------+
 *  |    data       |
 *  +---------------+
 *  |  dev_ptr[0]   |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |  dev_ptr[N]   |
 *  +---------------+
 *  |     DD0       |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |     DDN       |
 *  +---------------+
 *
 *  \sa nanos::WorkDescriptor
 */
nanos_err_t nanos_create_wd (  nanos_wd_t *uwd, size_t num_devices, nanos_device_t *devices, size_t data_size,
                               void ** data, nanos_wg_t uwg, nanos_wd_props_t *props )
{
   try {
      if ( ( props == NULL  ||
            ( props != NULL  && !props->mandatory_creation ) ) && !sys.throttleTask() ) {
         *uwd = 0;
         return NANOS_OK;
      }

      int dd_size = 0;
      for ( unsigned int i = 0; i < num_devices; i++ )
         dd_size += devices[i].dd_size;

      int size_to_allocate = ( ( *uwd == NULL ) ? sizeof( WD ) : 0 ) +
                             ( ( data != NULL && *data == NULL ) ? data_size : 0 ) +
                             sizeof( DD* ) * num_devices +
                             dd_size
                             ;

      char *chunk=0;
      char *start;

      if ( size_to_allocate )
         start = chunk = new char[size_to_allocate];

      // allocate WD
      if ( *uwd == NULL ) {
         *uwd = ( nanos_wd_t ) chunk;
         chunk += sizeof( WD );
      }

      // allocate WD data
      if ( data != NULL && *data == NULL ) {
         *data = chunk;
         chunk += data_size;
      }

      // allocate device pointers vector
      DD **dev_ptrs = ( DD ** ) chunk;
      chunk += sizeof( DD* ) * num_devices;

      // allocate device data
      for ( unsigned int i = 0 ; i < num_devices ; i ++ ) {
         dev_ptrs[i] = ( DD* ) devices[0].factory( chunk , devices[0].arg );
         chunk += devices[i].dd_size;
      }

      WD * wd =  new (*uwd) WD( num_devices, dev_ptrs, data != NULL ? *data : NULL );

      // add to workgroup
      if ( uwg != NULL ) {
         WG * wg = ( WG * )uwg;
         wg->addWork( *wd );
      }

      // set properties
      if ( props != NULL ) {
         if ( props->tied ) wd->tied();
         if ( props->tie_to ) wd->tieTo( *( BaseThread * )props->tie_to );
      }

   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_submit ( nanos_wd_t uwd, nanos_dependence_t *deps, nanos_team_t team )
{
   try {
      if ( deps != NULL ) {
         warning( "Dependence support not implemented yet" );
      }

      if ( team != NULL ) {
         warning( "Submitting to another team not implemented yet" );
      }

      ensure( uwd,"NULL WD received" );

      WD * wd = ( WD * ) uwd;
      sys.submit( *wd );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

// data must be not null
nanos_err_t nanos_create_wd_and_run ( size_t num_devices, nanos_device_t *devices, void * data,
                                      nanos_dependence_t *deps, nanos_wd_props_t *props )
{
   try {
      if ( num_devices > 1 ) warning( "Multiple devices not yet supported. Using first one" );

      if ( deps != NULL ) warning( "Dependence support not implemented yet" );

      // TODO: pre-allocate devices
      WD wd( ( DD* ) devices[0].factory( 0, devices[0].arg ), data );

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
