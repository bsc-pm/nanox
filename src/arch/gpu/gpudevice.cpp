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
#include <string.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <sys/resource.h>


#include <cuda_runtime.h>

using namespace nanos;


transfer_mode GPUDevice::_transferMode = ASYNC;


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

void * GPUDevice::allocateWholeMemory( size_t &size )
{
   void * address = 0;
   float percentage = 1.0;

   cudaError_t err = cudaMalloc( ( void ** ) &address, size * percentage );

   while ( ( err == cudaErrorMemoryValueTooLarge || err == cudaErrorMemoryAllocation )
         && percentage > 0.5 )
   {
      percentage -= 0.05;
      err = cudaMalloc( ( void ** ) &address, size * percentage );
   }

   if ( err != cudaSuccess ) {
      std::stringstream sizeStr;
      sizeStr << size;
      std::string what = "Trying to allocate "
            + sizeStr.str()
            + " bytes of device memory with cudaMalloc(): ";
      fatal( what + cudaGetErrorString( err ) );
   }

   size = size * percentage;

   return address;
}

void GPUDevice::freeWholeMemory( void * address )
{
   cudaError_t err = cudaFree( address );

   if ( err != cudaSuccess ) {
      std::stringstream addrStr;
      addrStr << address;
      std::string what = "Trying to free device memory at "
                         + addrStr.str()
                         + " with cudaFree(): ";
      fatal( what + cudaGetErrorString( err ) );
   }
}


void * GPUDevice::allocate( size_t size, ProcessingElement *pe )
{
   cudaError_t err = cudaSuccess;
   void * address = 0;
   uint64_t pinned = 0;

   if ( _transferMode == ASYNC ) {
      err = cudaMallocHost( ( void ** ) &pinned, size );

      // Use cudaHostAllocPortable flag in order to allocate host memory that must be
      // accessed from more than one GPU
      //err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocPortable);

      if ( err != cudaSuccess ) {
         std::stringstream sizeStr;
         sizeStr << size;
         std::string what = "Trying to allocate "
                            + sizeStr.str()
                            + " bytes of host memory with cudaMallocHost(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }

   if ( _transferMode == PINNED_CUDA ) {
      err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocMapped );

      if ( err != cudaSuccess ) {
         std::stringstream sizeStr;
         sizeStr << size;
         std::string what = "Trying to allocate "
                            + sizeStr.str()
                            + " bytes of host memory with cudaHostAlloc(): ";
         fatal( what + cudaGetErrorString( err ) );
      }

      err = cudaHostGetDevicePointer( ( void ** ) &address, ( void * ) pinned, 0 );

      if ( err != cudaSuccess ) {
         std::string what = "cudaHostGetDevicePointer(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }

   if ( _transferMode == WC ) {
      // Find out more about this method --> it's not clear how to use it
      err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocMapped | cudaHostAllocWriteCombined );

      if ( err != cudaSuccess ) {
         std::stringstream sizeStr;
         sizeStr << size;
         std::string what = "Trying to allocate "
                            + sizeStr.str()
                            + " bytes of host memory with cudaHostAlloc(): ";
         fatal( what + cudaGetErrorString( err ) );
      }

      err = cudaHostGetDevicePointer( ( void ** ) &address, ( void * ) pinned, 0 );

      if ( err != cudaSuccess ) {
         std::string what = "cudaHostGetDevicePointer(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }

   if ( _transferMode == NORMAL || _transferMode == ASYNC || _transferMode == PINNED_OS) {
      address = ( ( nanos::ext::GPUProcessor * ) pe )->allocate( size );
#if 0
      err = cudaMalloc( ( void ** ) &address, size );

      if ( err != cudaSuccess ) {
         std::stringstream sizeStr;
         sizeStr << size;
         std::string what = "Trying to allocate "
                            + sizeStr.str()
                            + " bytes of device memory with cudaMalloc(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
#endif
   }

   if ( _transferMode == ASYNC || _transferMode == PINNED_CUDA || _transferMode == WC) {
      ( ( nanos::ext::GPUProcessor * ) pe )->setPinnedAddress( address, pinned );
   }

   return address;

}

void GPUDevice::free( void *address, ProcessingElement *pe )
{
   cudaError_t err = cudaSuccess;

   if ( _transferMode == NORMAL || _transferMode == ASYNC || _transferMode == PINNED_OS) {

      // Check there are no pending copies to execute before we free the memory
      // (and if there are, execute them)
      ( ( nanos::ext::GPUProcessor * ) pe )->getOutTransferList()->checkAddressForMemoryTransfer( address );

      ( ( nanos::ext::GPUProcessor * ) pe )->free( address );

#if 0
      err = cudaFree( address );

      if ( err != cudaSuccess ) {
         std::stringstream addrStr;
         addrStr << address;
         std::string what = "Trying to free device memory at "
                            + addrStr.str()
                            + " with cudaFree(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
#endif

   }

   if ( _transferMode == ASYNC || _transferMode == PINNED_CUDA || _transferMode == WC) {
      uint64_t pinned = ((nanos::ext::GPUProcessor *) pe )->getPinnedAddress( address );
      if ( pinned != 0 ) {
         err = cudaFreeHost( ( void * ) pinned );
         ((nanos::ext::GPUProcessor *) pe )->removePinnedAddress( address );
      }

      if ( err != cudaSuccess ) {
         std::stringstream addrStr;
         addrStr << address;
         std::string what = "Trying to free host memory at "
                            + addrStr.str()
                            + " with cudaFreeHost(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }
}

bool GPUDevice::copyIn( void *localDst, uint64_t remoteSrc, size_t size, ProcessingElement *pe )
{
   // Copy from host memory to device memory

   ( ( nanos::ext::GPUProcessor * ) pe )->transferInput( size );

   cudaError_t err = cudaSuccess;

   if ( _transferMode == ASYNC ) {
      ( ( nanos::ext::GPUProcessor * ) pe )->getInTransferList()->addMemoryTransfer( remoteSrc );
      // Workaround to perform asynchronous copies
      uint64_t pinned = ((nanos::ext::GPUProcessor *) pe )->getPinnedAddress( localDst );
      memcpy( ( void * ) pinned, ( void * ) remoteSrc, size );

      err = cudaMemcpyAsync(
               localDst,
               ( void * ) pinned,
               size,
               cudaMemcpyHostToDevice,
               ((nanos::ext::GPUProcessor *) pe )->getGPUProcessorInfo()->getInTransferStream()
            );
   }

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

   if ( _transferMode == NORMAL) {
      err = cudaMemcpy( localDst, ( void * ) remoteSrc, size, cudaMemcpyHostToDevice );
      ( ( nanos::ext::GPUProcessor * ) pe )->synchronize( remoteSrc );
   }

   if ( err != cudaSuccess ) {
      std::stringstream sizeStr;
      sizeStr << size;
      std::stringstream srcStr;
      srcStr << remoteSrc;
      std::stringstream dstStr;
      dstStr << localDst;
      std::string what = "Trying to copy "
                         + sizeStr.str()
                         + " bytes of data from host ("
                         + srcStr.str()
                         + ") to device ("
                         + dstStr.str()
                         + ") with cudaMemcpy*(): ";
      fatal( what + cudaGetErrorString( err ) );
   }

   if ( _transferMode == NORMAL ) {
      return true;
   }

   return false;
}

bool GPUDevice::copyOut( uint64_t remoteDst, void *localSrc, size_t size, ProcessingElement *pe )
{
   // Copy from device memory to host memory

   ( ( nanos::ext::GPUProcessor * ) pe )->transferOutput( size );

   // No need to copy back for PINNED_CUDA or WC
   if ( _transferMode != PINNED_CUDA && _transferMode != WC ) {
#if 0
      cudaError_t err = cudaSuccess;
      if ( _transferMode == ASYNC ) {
         err = cudaMemcpyAsync(
                  ( void * ) _pinnedMemory[localSrc],
                  localSrc,
                  size,
                  cudaMemcpyDeviceToHost,
                  ( ( nanos::ext::GPUProcessor * ) pe )->getTransferInfo()->getOutTransferStream()
               );
         err = cudaMemcpyAsync(
                  ( void * ) remoteDst,
                  ( void * ) _pinnedMemory[localSrc],
                  size,
                  cudaMemcpyHostToHost,
                  ( ( nanos::ext::GPUProcessor * ) pe )->getTransferInfo()->getOutTransferStream()
               );
      }
      else {
         err = cudaMemcpy( ( void * ) remoteDst, localSrc, size, cudaMemcpyDeviceToHost );
      }
#endif
      cudaError_t err = cudaSuccess;

      if ( _transferMode == ASYNC ) {
         ( ( nanos::ext::GPUProcessor * ) pe )->getOutTransferList()->addMemoryTransfer( ( void * ) remoteDst, localSrc, size );
      }
      else {
         err = cudaMemcpy( ( void * ) remoteDst, localSrc, size, cudaMemcpyDeviceToHost );
         ( ( nanos::ext::GPUProcessor * ) pe )->synchronize( remoteDst );
      }

      if ( err != cudaSuccess ) {
         std::stringstream sizeStr;
         sizeStr << size;
         std::stringstream srcStr;
         srcStr << localSrc;
         std::stringstream dstStr;
         dstStr << remoteDst;
         std::string what = "Trying to copy "
                            + sizeStr.str()
                            + " bytes of data from device ("
                            + srcStr.str()
                            + ") to host ("
                            + dstStr.str()
                            + ") with cudaMemcpy*(): ";
         fatal( what + cudaGetErrorString( err ) );
      }
   }

   if ( _transferMode == PINNED_OS ) {
      munlock( ( void * ) remoteDst, size );
   }

   if ( _transferMode == NORMAL ) {
      return true;
   }

   return false;
}

void GPUDevice::copyLocal( void *dst, void *src, size_t size, ProcessingElement *pe )
{
   // Copy from device memory to device memory

   cudaError_t err = cudaSuccess;

   if (_transferMode == NORMAL ) {
      err = cudaMemcpy( ( void *) dst, src, size, cudaMemcpyDeviceToDevice );
   }
   else {
      err = cudaMemcpyAsync(
               ( void *) dst,
               src,
               size,
               cudaMemcpyDeviceToDevice,
               ((nanos::ext::GPUProcessor *) pe )->getGPUProcessorInfo()->getInTransferStream()
            );

   }

   if ( err != cudaSuccess ) {
      std::stringstream sizeStr;
      sizeStr << size;
      std::stringstream srcStr;
      srcStr << src;
      std::stringstream dstStr;
      dstStr << dst;
      std::string what = "Trying to copy "
                         + sizeStr.str()
                         + " bytes of data from device ("
                         + srcStr.str()
                         + ") to device ("
                         + dstStr.str()
                         + ") with cudaMemcpy*(): ";
      fatal( what + cudaGetErrorString( err ) );
   }
}

void GPUDevice::syncTransfer( uint64_t hostAddress, ProcessingElement *pe)
{
   // syncTransfer() is used to ensure that somebody will update the data
   // related to 'hostAddress' of main memory at some time
   // since we use copy back, this is always ensured
}

void GPUDevice::copyOutAsyncToBuffer ( void * dst, void * src, size_t size )
{
   cudaError_t err = cudaMemcpyAsync(
            ( void * ) ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getPinnedAddress(src),
            src,
            size,
            cudaMemcpyDeviceToHost,
            ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getOutTransferStream()
         );

   if ( err != cudaSuccess ) {
      std::stringstream sizeStr;
      sizeStr << size;
      std::stringstream srcStr;
      srcStr << src;
      std::stringstream dstStr;
      dstStr << dst;
      std::string what = "Trying to copy "
                         + sizeStr.str()
                         + " bytes of data from device ("
                         + srcStr.str()
                         + ") to host ("
                         + dstStr.str()
                         + ") with cudaMemcpy*(): ";
      fatal( what + cudaGetErrorString( err ) );
   }

}

void GPUDevice::copyOutAsyncWait ()
{
   cudaStreamSynchronize( ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getOutTransferStream() );
}

void GPUDevice::copyOutAsyncToHost ( void * dst, void * src, size_t size )
{
   memcpy( dst, ( void * ) ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getPinnedAddress(src), size );
}

void GPUDevice::copyOutSyncToHost ( void * dst, void * src, size_t size )
{
   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost );

   if ( err != cudaSuccess ) {
      std::stringstream sizeStr;
      sizeStr << size;
      std::stringstream srcStr;
      srcStr << src;
      std::stringstream dstStr;
      dstStr << dst;
      std::string what = "Trying to copy "
                         + sizeStr.str()
                         + " bytes of data from device ("
                         + srcStr.str()
                         + ") to host ("
                         + dstStr.str()
                         + ") with cudaMemcpy*(): ";
      fatal( what + cudaGetErrorString( err ) );
   }
}

