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
#include "gpuprocessor.hpp"
//#if ASYNC
#include <string.h>
//#endif
//#if PINNED_OS
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <sys/resource.h>
//#endif


#include <cuda_runtime.h>

using namespace nanos;


transfer_mode GPUDevice::_transferMode = ASYNC;


//#if PINNED_OS
unsigned int GPUDevice::_rlimit;

void GPUDevice::getMemoryLockLimit()
{
   if ( _transferMode == PINNED_OS ) {
      struct rlimit rlim;
      int error = getrlimit( RLIMIT_MEMLOCK, &rlim );
      if ( error == 0 ) {
         _rlimit = rlim.rlim_cur >> 1;
      } else {
         _rlimit = 0;
      }
   }
}
//#endif

void * GPUDevice::allocate( size_t size )
{
   cudaError_t err = cudaSuccess;
   void * address = 0;
   uint64_t pinned = 0;

//#if ASYNC
   if ( _transferMode == ASYNC ) {
      err = cudaMallocHost( ( void ** ) &pinned, size );
      //err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocPortable);
      if ( err != cudaSuccess ) {
         std::string what = "Trying to allocate host memory with cudaMallocHost(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }
//#endif

//#if PINNED_CUDA
   if ( _transferMode == PINNED_CUDA ) {
      err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocMapped );

      if ( err != cudaSuccess ) {
         std::string what = "Trying to allocate host memory with cudaHostAlloc(): ";
         fatal( what + cudaGetErrorString( err ) );
      }

      err = cudaHostGetDevicePointer( ( void ** ) &address, ( void * ) pinned, 0 );

      if ( err != cudaSuccess ) {
         std::string what = "cudaHostGetDevicePointer(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }
//#endif

//#if WC
   if ( _transferMode == WC ) {
      // Find out more about this method --> it's not clear how to use it
      err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocMapped | cudaHostAllocWriteCombined );

      if ( err != cudaSuccess ) {
         std::string what = "Trying to allocate host memory with cudaHostAlloc(): ";
         fatal( what + cudaGetErrorString( err ) );
      }

      err = cudaHostGetDevicePointer( ( void ** ) &address, ( void * ) pinned, 0 );

      if ( err != cudaSuccess ) {
         std::string what = "cudaHostGetDevicePointer(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }
//#endif

//#if NORMAL | ASYNC | PINNED_OS
   if ( _transferMode == NORMAL || _transferMode == ASYNC || _transferMode == PINNED_OS) {
      err = cudaMalloc( ( void ** ) &address, size );

      if ( err != cudaSuccess ) {
         std::string what = "Trying to allocate host memory with cudaMalloc(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }
//#endif

//#if ASYNC | PINNED_CUDA | WC
   if ( _transferMode == ASYNC || _transferMode == PINNED_CUDA || _transferMode == WC) {
      ((nanos::ext::GPUProcessor *) myThread->runningOn())->setPinnedAddress( address, pinned );
   }
//#endif

   return address;

}

void GPUDevice::free( void *address )
{
   cudaError_t err = cudaSuccess;

//#if NORMAL | ASYNC | PINNED_OS
   if ( _transferMode == NORMAL || _transferMode == ASYNC || _transferMode == PINNED_OS) {
      err = cudaFree( address );

      if ( err != cudaSuccess ) {
         std::string what = "Trying to free device memory with cudaFree(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }
//#endif

//#if ASYNC | PINNED_CUDA | WC
   if ( _transferMode == ASYNC || _transferMode == PINNED_CUDA || _transferMode == WC) {
      uint64_t pinned = ((nanos::ext::GPUProcessor *) myThread->runningOn())->getPinnedAddress( address );
      if ( pinned != 0 ) {
         err = cudaFreeHost( ( void * ) pinned );
         ((nanos::ext::GPUProcessor *) myThread->runningOn())->removePinnedAddress( address );
      }

      if ( err != cudaSuccess ) {
         std::string what = "Trying to free host memory with cudaFreeHost(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }
//#endif
}

void GPUDevice::copyIn( void *localDst, uint64_t remoteSrc, size_t size )
{
   // Copy from host memory to device memory

   cudaError_t err = cudaSuccess;

//#if ASYNC
   if ( _transferMode == ASYNC ) {
      // Workaround to perform asynchronous copies
      uint64_t pinned = ((nanos::ext::GPUProcessor *) myThread->runningOn())->getPinnedAddress( localDst );
      memcpy( ( void * ) pinned, ( void * ) remoteSrc, size );
      err = cudaMemcpyAsync(
               localDst,
               ( void * ) pinned,
               size,
               cudaMemcpyHostToDevice,
               ((nanos::ext::GPUProcessor *) myThread->runningOn())->getTransferInfo()->getTransferStream()
            );

      if ( err != cudaSuccess ) {
         std::cout << "localDst = " << localDst << "; pinnedMemory = 0x"
               << pinned << "; size = " << size
               << "; stream = " << ((nanos::ext::GPUProcessor *) myThread->runningOn())->getTransferInfo()->getTransferStream()
               << std::endl;
         std::string what = "Trying to copy data from host to device with cudaMemcpy*(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }
//#endif

//#if PINNED_OS
   if ( _transferMode == PINNED_OS ) {
      unsigned int auxAddress = remoteSrc;
      int error = 0;

      for ( int bytesLeft = size; bytesLeft > 0;  ) {
         bool touch = * ((bool *) remoteSrc);
         std::cout << "aux@ = " << auxAddress << "; min(rlim,Bl) = " << std::min( _rlimit, (unsigned int) bytesLeft )
         << "; error = " << error << "; size = " << size << "; Bl = " << bytesLeft << " " << touch << std::endl;
         error += mlock( ( void * ) auxAddress, std::min( _rlimit, (unsigned int) bytesLeft ) );
         auxAddress += std::min( _rlimit, (unsigned int) bytesLeft );
         bytesLeft -= _rlimit;
         if (bytesLeft <= 0 ) std::cout << "+++++++++++++++++++++++++++++" << std::endl;
      }

      //int error = mlock( ( void * ) remoteSrc, size );

      if (error != 0) {
         std::cout << "ERROR in mlock() ::: " << errno << ": " << strerror( errno ) ;
         if ( errno == ENOMEM ) std::cout << " --> ENOMEM" << std::endl;
         else if ( errno == EAGAIN ) std::cout << " --> EAGAIN" << std::endl;
         else if ( errno == EINVAL ) std::cout << " --> EINVAL" << std::endl;
         else if ( errno == ENOMEM ) std::cout << " --> ENOMEM" << std::endl;
         else if ( errno == EPERM ) std::cout << " --> EPERM" << std::endl;
         else std::cout << " --> ??" << std::endl;
      }

      std::cout << "-------------------------------------------" << std::endl;

      err = cudaMemcpyAsync( localDst, ( void * ) remoteSrc, size, cudaMemcpyHostToDevice, 0 );
   }
//#endif

//#if NORMAL
   if ( _transferMode == NORMAL) {
      err = cudaMemcpy( localDst, ( void * ) remoteSrc, size, cudaMemcpyHostToDevice );
   }
//#endif

   if ( err != cudaSuccess ) {
      std::string what = "Trying to copy data from host to device with cudaMemcpy*(): ";
      fatal( what + cudaGetErrorString( err ) );
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

//#if PINNED_CUDA | WC
   // No need to copy back
//#else
   if ( _transferMode != PINNED_CUDA && _transferMode != WC ) {
#if 0
      cudaError_t err = cudaSuccess;
      if ( _transferMode == ASYNC ) {
         err = cudaMemcpyAsync(
                  ( void * ) _pinnedMemory[localSrc],
                  localSrc,
                  size,
                  cudaMemcpyDeviceToHost,
                  ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getTransferInfo()->getTransferStream()
               );
         err = cudaMemcpyAsync(
                  ( void * ) remoteDst,
                  ( void * ) _pinnedMemory[localSrc],
                  size,
                  cudaMemcpyHostToHost,
                  ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getTransferInfo()->getTransferStream()
               );
      }
      else {
         err = cudaMemcpy( ( void * ) remoteDst, localSrc, size, cudaMemcpyDeviceToHost );
      }
#endif

      cudaError_t err = cudaMemcpy( ( void * ) remoteDst, localSrc, size, cudaMemcpyDeviceToHost );

      if ( err != cudaSuccess ) {
         std::string what = "Trying to copy data from device to host with cudaMemcpy*(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }
//#endif

//#if PINNED_OS
   if ( _transferMode == PINNED_OS ) {
      munlock( ( void * ) remoteDst, size );
   }
//#endif
}

void GPUDevice::copyLocal( void *dst, void *src, size_t size )
{
   // Copy from device memory to device memory

   cudaError_t err = cudaSuccess;

//#if NORMAL
   if (_transferMode == NORMAL ) {
      err = cudaMemcpy( ( void *) dst, src, size, cudaMemcpyDeviceToDevice );
   }
//#else
   else {
      err = cudaMemcpyAsync( ( void *) dst, src, size, cudaMemcpyDeviceToDevice, 0 );
   }
//#endif

   if ( err != cudaSuccess ) {
      std::string what = "Trying to copy data from device to device with cudaMemcpy*(): ";
      fatal( what + cudaGetErrorString( err ) );
   }
}
