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

#include "gpudevice.hpp"
#include "debug.hpp"
#include "basethread.hpp"

#include <cuda_runtime.h>

using namespace nanos;

void * GPUDevice::allocate( size_t size )
{
   void * address=0;
   cudaError_t err = cudaMalloc( (void **) &address, size );

   if (err == cudaSuccess)
      return address;

   fatal(cudaGetErrorString(err));
   return 0;
}

void GPUDevice::free( void *address )
{
   cudaError_t err = cudaFree( address );

   if (err != cudaSuccess) {
      fatal(cudaGetErrorString(err));
   }
}

void GPUDevice::copyIn( void *localDst, uint64_t remoteSrc, size_t size )
{
   // Copy from host memory to device memory
   cudaError_t err = cudaMemcpy( localDst, (void *) remoteSrc, size, cudaMemcpyHostToDevice );

   if (err != cudaSuccess) {
      fatal(cudaGetErrorString(err));
   }

}

void GPUDevice::copyOut( uint64_t remoteDst, void *localSrc, size_t size )
{
   // Copy from device memory to host memory
   cudaError_t err = cudaMemcpy( (void *) remoteDst, localSrc, size, cudaMemcpyDeviceToHost );

   if (err != cudaSuccess) {
      fatal(cudaGetErrorString(err));
   }
}

