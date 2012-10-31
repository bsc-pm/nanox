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

#ifndef _NANOS_MPI_ULT
#define _NANOS_MPI_ULT

#include <stddef.h>
#include <stdint.h>

extern "C"
{
// low-level routine to switch stacks
   void switchStacksMpi( void *,void *,void *,void * );
}

intptr_t * initContextMpi( intptr_t *stack, size_t stackSize, void *userFunction, void *userArg,
                       void *cleanup, void *cleanupArg );

#ifndef mpi_SUPPORTS_ULT

extern "C" {
   inline void switchStacksMpi( void *,void *,void *,void * ) {}
}
inline intptr_t * initContextMpi( intptr_t *stack, size_t stackSize, void *userFunction, void *userArg,
                       void *cleanup, void *cleanupArg ) { return 0; }
#endif

#endif

