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

#ifndef _NANOS_FPGA_H
#define _NANOS_FPGA_H

#include "nanos-int.h"

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

    typedef struct {
        void (*outline) (void *);
        int acc_num;
    } nanos_fpga_args_t;


NANOS_API_DECL( void *, nanos_fpga_factory, ( void *args ) );
NANOS_API_DECL( void *, nanos_fpga_alloc_dma_mem, ( size_t len) );
NANOS_API_DECL( void, nanos_fpga_free_dma_mem, ( void * address ) );

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //_NANOS_FPGA_H
