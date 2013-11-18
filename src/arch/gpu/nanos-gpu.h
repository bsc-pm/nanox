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

#ifndef _NANOS_GPU_H_
#define _NANOS_GPU_H_

#include <cuda_runtime.h>
#include <vector_types.h>

#include "nanos-int.h"

#ifdef NANOS_GPU_USE_CUDA32
#include <cublas_api.h>
#else
#include <cublas.h>
#include <cublas_v2.h>
#endif

/*! \page cuda_main CUDA Documentation
 *
 * The programmer needs to call cublasSetStream() function just before any call to the CUBLAS library.
 * If the function is not called CUBLAS library is not set to the appropriate stream and this makes
 * Nanos++ to incorrectly detect the end of the task: Nanox synchronizes with its execution stream,
 * but the CUBLAS call is sent to the default stream (NULL stream), so calling cudaStreamSynchronize()
 * returns immediately.
 */

#ifdef __cplusplus
extern "C" {
#endif

// gpu factory
NANOS_API_DECL(void *, nanos_gpu_factory,( void *args ));
#define NANOS_GPU_DESC( args ) { nanos_gpu_factory, &( args ) }

NANOS_API_DECL(cudaStream_t, nanos_get_kernel_execution_stream,());

NANOS_API_DECL(cublasHandle_t, nanos_get_cublas_handle,());

// Pinned memory
NANOS_API_DECL( void *, nanos_malloc_pinned_cuda, ( size_t size ) );
NANOS_API_DECL( void, nanos_free_pinned_cuda, ( void * address ) );

#ifdef __cplusplus
}
#endif

#endif
