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

#ifdef NANOS_GPU_USE_CUDA32
#include <cublas_api.h>
#else
#include <cublas.h>
#include <cublas_v2.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void * nanos_gpu_factory( void *prealloc , void *args );
extern const size_t nanos_gpu_dd_size;
#define NANOS_GPU_DESC( args ) { nanos_gpu_factory, nanos_gpu_dd_size, &( args ) }

cudaStream_t nanos_get_kernel_execution_stream();

cublasHandle_t nanos_get_cublas_handle();

// Commented out by now, as it does not compile
#if 0
   // gpu factory
NANOS_API_DECL(void *, nanos_gpu_factory,( void *prealloc ,void *args));
extern const size_t nanos_gpu_dd_size;
#define NANOS_GPU_DESC( args ) { nanos_gpu_factory, nanos_gpu_dd_size, &( args ) }

NANOS_API_DECL(cudaStream_t, nanos_get_kernel_execution_stream,());

NANOS_API_DECL(cublasHandle_t, nanos_get_cublas_handle,());
#endif

#ifdef __cplusplus
}
#endif

#endif
