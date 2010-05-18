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
#include "basethread.hpp"
#include "debug.hpp"
#if ASYNC
#include <string.h>
#endif
#if PINNED_OS
#include <sys/mman.h>
#endif

#include <cuda_runtime.h>

using namespace nanos;

#if ASYNC | PINNED_CUDA | WC
std::map< void *, uint64_t > GPUDevice::_pinnedMemory;
#endif

void * GPUDevice::allocate( size_t size )
{
   cudaError_t err = cudaSuccess;
   void * address = 0;

#if ASYNC
   uint64_t pinned = 0;
   err = cudaMallocHost( ( void ** ) &pinned, size );

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }
#endif

#if PINNED_CUDA
   uint64_t pinned = 0;
   err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocMapped );

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }

   err = cudaHostGetDevicePointer( ( void ** ) &address, ( void * ) pinned, 0 );

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }
#endif

#if WC
   // Find out more about this method --> it's not clear how to use it
   uint64_t pinned = 0;
   err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocMapped | cudaHostAllocWriteCombined );

   if ( err != cudaSuccess ) {
      std::cout << "[GPUDevice::allocate] error @ cudaHostAlloc :: size = " << size
            << "; pinned = " << pinned << "; address = " << address << std::endl;
      fatal( cudaGetErrorString( err ) );
   }

   err = cudaHostGetDevicePointer( ( void ** ) &address, ( void * ) pinned, 0 );

   if ( err != cudaSuccess ) {
      std::cout << "[GPUDevice::allocate] error @ cudaHostGetDevicePointer :: size = " << size
            << "; pinned = " << pinned << "; address = " << address << std::endl;
      fatal( cudaGetErrorString( err ) );
   }
#endif

#if NORMAL | ASYNC | PINNED_OS
   err = cudaMalloc( ( void ** ) &address, size );

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }
#endif

#if ASYNC | PINNED_CUDA | WC
   _pinnedMemory[address] = pinned;
#endif

   return address;

}

void GPUDevice::free( void *address )
{
   cudaError_t err = cudaSuccess;

#if NORMAL | ASYNC | PINNED_OS
   err = cudaFree( address );

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }

#endif

#if ASYNC | PINNED_CUDA | WC
   if ( _pinnedMemory.count( address ) > 1 ) {
      err = cudaFreeHost( ( void * ) _pinnedMemory[address] );

      _pinnedMemory.erase( address );
   }

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }
#endif
}

void GPUDevice::copyIn( void *localDst, uint64_t remoteSrc, size_t size )
{
   // Copy from host memory to device memory

   cudaError_t err = cudaSuccess;

#if ASYNC
   // Workaround to perform asynchronous copies
   memcpy( ( void * ) _pinnedMemory[localDst], ( void * ) remoteSrc, size );
   err = cudaMemcpyAsync( localDst, ( void * ) _pinnedMemory[localDst], size, cudaMemcpyHostToDevice, 0 );
#endif

#if PINNED_OS

   /*int error = */mlock( ( void * ) remoteSrc, size );

   //if (error != 0) std::cout << "ERROR in mlock()" << std::endl;
   //else std::cout << "mlock() SUCCEEDED" << std::endl;

   err = cudaMemcpyAsync( localDst, ( void * ) remoteSrc, size, cudaMemcpyHostToDevice, 0 );
#endif

#if NORMAL
   err = cudaMemcpy( localDst, ( void * ) remoteSrc, size, cudaMemcpyHostToDevice );

#endif

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }

}

void GPUDevice::copyOut( uint64_t remoteDst, void *localSrc, size_t size )
{
   // Copy from device memory to host memory

#if 0
   cudaError_t err = cudaMemcpyAsync( ( void * ) _pinnedMemory[localSrc], localSrc, size, cudaMemcpyDeviceToHost, 0 );

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }

   // Workaround to perform asynchronous copies
   memcpy( ( void * ) remoteDst, ( void * ) _pinnedMemory[localSrc], size );

#endif

#if PINNED_CUDA | WC
   // No need to copy back
#else
   cudaError_t err = cudaMemcpy( ( void * ) remoteDst, localSrc, size, cudaMemcpyDeviceToHost );

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }
#endif

#if PINNED_OS
   munlock( ( void * ) remoteDst, size );
#endif
}

void GPUDevice::copyLocal( void *dst, void *src, size_t size )
{
   // Copy from device memory to device memory

   cudaError_t err = cudaSuccess;

#if NORMAL
   err = cudaMemcpy( ( void *) dst, src, size, cudaMemcpyDeviceToDevice );

#else
   err = cudaMemcpyAsync( ( void *) dst, src, size, cudaMemcpyDeviceToDevice, 0 );
#endif

   if ( err != cudaSuccess ) {
      fatal( cudaGetErrorString( err ) );
   }
}
