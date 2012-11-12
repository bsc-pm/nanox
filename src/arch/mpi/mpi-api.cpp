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
#include "nanos-mpi.h"
#include "system.hpp"
#include "basethread.hpp"
#include "mpidd.hpp"
#include "mpiprocessor.hpp"

using namespace nanos;

NANOS_API_DEF(void *, nanos_mpi_factory, ( void *args ))
{
   nanos_mpi_args_t *mpi = ( nanos_mpi_args_t * ) args;
   return ( void * ) new ext::MPIDD( mpi->outline );
}


NANOS_API_DEF(nanos_err_t, DEEP_Booster_alloc, ( MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm )) 
{
   try {
      sys.DEEP_Booster_alloc(comm, number_of_spawns, intercomm);
   } catch ( ... ) { 
      return NANOS_UNKNOWN_ERR;
   }   

   return NANOS_OK;
}
NANOS_API_DEF(nanos_err_t, setMpiFilename, ( char * new_name )) 
{
   try {
      sys.setMpiFilename(new_name);
   } catch ( ... ) { 
      return NANOS_UNKNOWN_ERR;
   }   

   return NANOS_OK;
}