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

#ifndef _NUMA_NANOS_H_
#define _NUMA_NANOS_H_

#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include "nanos.h"
#include "nanos-int.h"

#ifdef _MERCURIUM_
// define API version
#pragma nanos interface family(master) version(5000)
#endif

#ifdef __cplusplus

#define _Bool bool

extern "C" {
#endif

nanos_err_t nanos_get_addr ( uint64_t tag, nanos_sharing_t sharing, void **addr );

nanos_err_t nanos_copy_value ( void *dst, uint64_t tag, nanos_sharing_t sharing, size_t size );

#ifdef __cplusplus
}
#endif

#endif
