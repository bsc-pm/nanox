/*************************************************************************************/
/*      Copyright 2010 Barcelona Supercomputing Center                               */
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

#include "nanos-fpga.h"
#include "basethread.hpp"
#include "fpgadd.hpp"
#include "debug.hpp"

#include "libxdma.h"

using namespace nanos;

NANOS_API_DEF( void *, nanos_fpga_factory, ( void *args ) )
{
   nanos_fpga_args_t *fpga = ( nanos_fpga_args_t * ) args;
   return ( void * ) NEW ext::FPGADD( fpga->outline, fpga->acc_num );
}

NANOS_API_DEF( void *, nanos_fpga_alloc_dma_mem, ( size_t len) )
{
    void *buffer;
    xdma_status status;
    status = xdmaAllocateKernelBuffer( &buffer, len );
    if ( status != XDMA_SUCCESS ) {
        warning( "Could not allocate memory in kernel space" );
        buffer = NULL;
    }

    return buffer;
}

NANOS_API_DEF( void, nanos_fpga_free_dma_mem, ( ) )
{
    xdma_status status;
    status = xdmaFreeKernelBuffers();
    if ( status != XDMA_SUCCESS ) {
        warning( "Could not free kernel memory" );
    }
}
