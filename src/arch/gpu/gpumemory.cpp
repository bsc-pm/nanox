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

// CUDA
#include <cuda_runtime.h>

#include "gpumemory.hpp"


#include <iostream>

using namespace nanos;

void * GPUMemory::allocate( size_t size )
{
   std::cout << "allocating device memory with cudaMalloc" << std::endl;
   void * address;
   cudaMalloc( (void **) &address, size );
   return address;
}

void GPUMemory::free( void *address )
{
   std::cout << "freeing device memory with cudaFree" << std::endl;
   cudaFree( address );
}

void GPUMemory::copyIn( void *localDst, uint64_t remoteSrc, size_t size )
{
   std::cout << "copying data from host to device memory with cudaMemcpy" << std::endl;
   // Copy from host memory to device memory
   cudaMemcpy( localDst, (void *) remoteSrc, size, cudaMemcpyHostToDevice );
}

void GPUMemory::copyOut( uint64_t remoteDst, void *localSrc, size_t size )
{
   std::cout << "copying data from device to host memory with cudaMemcpy" << std::endl;
   // Copy from device memory to host memory
   cudaMemcpy( (void *) remoteDst, localSrc, size, cudaMemcpyDeviceToHost );
}

