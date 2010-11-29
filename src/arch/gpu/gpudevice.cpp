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


transfer_mode GPUDevice::_transferMode = NANOS_GPU_TRANSFER_ASYNC;


unsigned int GPUDevice::_rlimit;

void GPUDevice::getMemoryLockLimit()
{
   if ( _transferMode == NANOS_GPU_TRANSFER_PINNED_OS ) {
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

   cudaError_t err = cudaMalloc( ( void ** ) &address, ( size_t ) ( size * percentage ) );

   while ( ( err == cudaErrorMemoryValueTooLarge || err == cudaErrorMemoryAllocation )
         && percentage > 0.5 )
   {
      percentage -= 0.05;
      err = cudaMalloc( ( void ** ) &address, ( size_t ) ( size * percentage ) );
   }

   size = ( size_t ) ( size * percentage );

   fatal_cond( err != cudaSuccess, "Trying to allocate " +  toString<size_t>( size ) +
         + " bytes of device memory with cudaMalloc(): " +  cudaGetErrorString( err ) );

   return address;
}

void GPUDevice::freeWholeMemory( void * address )
{
   cudaError_t err = cudaFree( address );

   fatal_cond( err != cudaSuccess, "Trying to free device memory at " + toString<void *>( address ) +
         + " with cudaFree(): " + cudaGetErrorString( err ) );
}


void * GPUDevice::allocate( size_t size, ProcessingElement *pe )
{
   cudaError_t err = cudaSuccess;
   void * address = 0;
   uint64_t pinned = 0;

   if ( _transferMode == NANOS_GPU_TRANSFER_NORMAL || _transferMode == NANOS_GPU_TRANSFER_ASYNC || _transferMode == NANOS_GPU_TRANSFER_PINNED_OS) {
      address = ( ( nanos::ext::GPUProcessor * ) pe )->allocate( size );

      if ( address == NULL ) return NULL;
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


   if ( _transferMode == NANOS_GPU_TRANSFER_ASYNC ) {
      err = cudaMallocHost( ( void ** ) &pinned, size );

      // Use cudaHostAllocPortable flag in order to allocate host memory that must be
      // accessed from more than one GPU
      //err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocPortable);

      fatal_cond( err != cudaSuccess, "Trying to allocate " + toString<size_t>( size )
            + " bytes of host memory with cudaMallocHost(): " + cudaGetErrorString( err ) );
   }

   if ( _transferMode == NANOS_GPU_TRANSFER_PINNED_CUDA ) {
      err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocMapped );

      fatal_cond( err != cudaSuccess, "Trying to allocate " + toString<size_t>( size )
            + " bytes of host memory with cudaHostAlloc(): " + cudaGetErrorString( err ) );

      err = cudaHostGetDevicePointer( ( void ** ) &address, ( void * ) pinned, 0 );

      fatal_cond( err != cudaSuccess, "Error in cudaHostGetDevicePointer("
            + toString<uint64_t>( pinned ) + "): " + cudaGetErrorString( err ) );
   }

   if ( _transferMode == NANOS_GPU_TRANSFER_WC ) {
      // Find out more about this method --> it's not clear how to use it
      err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocMapped | cudaHostAllocWriteCombined );

      fatal_cond( err != cudaSuccess, "Trying to allocate " + toString<size_t>( size )
            + " bytes of host memory with cudaHostAlloc(): " + cudaGetErrorString( err ) );

      err = cudaHostGetDevicePointer( ( void ** ) &address, ( void * ) pinned, 0 );

      fatal_cond( err != cudaSuccess, "Error in cudaHostGetDevicePointer("
            + toString<uint64_t>( pinned ) + "): " + cudaGetErrorString( err ) );
   }

   if ( _transferMode == NANOS_GPU_TRANSFER_ASYNC || _transferMode == NANOS_GPU_TRANSFER_PINNED_CUDA || _transferMode == NANOS_GPU_TRANSFER_WC) {
      ( ( nanos::ext::GPUProcessor * ) pe )->setPinnedAddress( address, pinned );
   }

   return address;
}

void GPUDevice::free( void *address, ProcessingElement *pe )
{
   cudaError_t err = cudaSuccess;

   if ( _transferMode == NANOS_GPU_TRANSFER_NORMAL || _transferMode == NANOS_GPU_TRANSFER_ASYNC || _transferMode == NANOS_GPU_TRANSFER_PINNED_OS) {

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

   if ( _transferMode == NANOS_GPU_TRANSFER_ASYNC || _transferMode == NANOS_GPU_TRANSFER_PINNED_CUDA || _transferMode == NANOS_GPU_TRANSFER_WC) {
      uint64_t pinned = ( ( nanos::ext::GPUProcessor * ) pe )->getPinnedAddress( address );
      if ( pinned != 0 ) {
         err = cudaFreeHost( ( void * ) pinned );
         ((nanos::ext::GPUProcessor *) pe )->removePinnedAddress( address );
      }

      fatal_cond( err != cudaSuccess, "Trying to free host memory at " + toString<void *>( address )
            + " with cudaFreeHost(): " + cudaGetErrorString( err ) );
   }
}

bool GPUDevice::copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe )
{
   // Copy from host memory to device memory

   ( ( nanos::ext::GPUProcessor * ) pe )->transferInput( size );

   cudaError_t err = cudaSuccess;

   if ( _transferMode == NANOS_GPU_TRANSFER_ASYNC ) {
      ( ( nanos::ext::GPUProcessor * ) pe )->getInTransferList()->addMemoryTransfer( remoteSrc );
      // Workaround to perform asynchronous copies
      uint64_t pinned = ( ( nanos::ext::GPUProcessor * ) pe )->getPinnedAddress( localDst );
      SMPDevice::copyLocal( ( void * ) pinned, ( void * ) remoteSrc.getTag(), size, NULL );

      err = cudaMemcpyAsync(
               localDst,
               ( void * ) pinned,
               size,
               cudaMemcpyHostToDevice,
               ((nanos::ext::GPUProcessor *) pe )->getGPUProcessorInfo()->getInTransferStream()
            );
   }

   if ( _transferMode == NANOS_GPU_TRANSFER_PINNED_OS ) {
      unsigned int auxAddress = remoteSrc.getTag();
      int error = 0;

      for ( int bytesLeft = size; bytesLeft > 0;  ) {
         bool touch = * ((bool *) remoteSrc.getTag());
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

      err = cudaMemcpyAsync( localDst, ( void * ) remoteSrc.getTag(), size, cudaMemcpyHostToDevice, 0 );
   }

   if ( _transferMode == NANOS_GPU_TRANSFER_NORMAL) {
      err = cudaMemcpy( localDst, ( void * ) remoteSrc.getTag(), size, cudaMemcpyHostToDevice );
   }

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from host (" + toString<uint64_t>( remoteSrc.getTag() ) + ") to device ("
         + toString<void *>( localDst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );

   if ( _transferMode == NANOS_GPU_TRANSFER_NORMAL ) {
      return true;
   }

   return false;
}

bool GPUDevice::copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe )
{
   // Copy from device memory to host memory

   ( ( nanos::ext::GPUProcessor * ) pe )->transferOutput( size );

   // No need to copy back for NANOS_GPU_TRANSFER_PINNED_CUDA or NANOS_GPU_TRANSFER_WC
   if ( _transferMode != NANOS_GPU_TRANSFER_PINNED_CUDA && _transferMode != NANOS_GPU_TRANSFER_WC ) {
#if 0
      cudaError_t err = cudaSuccess;
      if ( _transferMode == NANOS_GPU_TRANSFER_ASYNC ) {
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

      if ( _transferMode == NANOS_GPU_TRANSFER_ASYNC ) {
         ( ( nanos::ext::GPUProcessor * ) pe )->getOutTransferList()->addMemoryTransfer( remoteDst, localSrc, size );
      }
      else {
         err = cudaMemcpy( ( void * ) remoteDst.getTag(), localSrc, size, cudaMemcpyDeviceToHost );
      }

      fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
            + " bytes of data from device (" + toString<void *>( localSrc ) + ") to host ("
            + toString<uint64_t>( remoteDst.getTag() ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
   }

   if ( _transferMode == NANOS_GPU_TRANSFER_PINNED_OS ) {
      munlock( ( void * ) remoteDst.getTag(), size );
   }

   if ( _transferMode == NANOS_GPU_TRANSFER_NORMAL ) {
      return true;
   }

   return false;
}

void GPUDevice::copyLocal( void *dst, void *src, size_t size, ProcessingElement *pe )
{
   // Copy from device memory to device memory

   cudaError_t err = cudaSuccess;

   if (_transferMode == NANOS_GPU_TRANSFER_NORMAL ) {
      err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToDevice );
   }
   else {
      err = cudaMemcpyAsync(
               dst,
               src,
               size,
               cudaMemcpyDeviceToDevice,
               ((nanos::ext::GPUProcessor *) pe )->getGPUProcessorInfo()->getLocalTransferStream()
            );

   }

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from device (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::syncTransfer( uint64_t hostAddress, ProcessingElement *pe)
{
   // syncTransfer() is used to ensure that somebody will update the data
   // related to 'hostAddress' of main memory at some time
   // since we use copy back, this is always ensured

   // Anyway, we can help the system and tell that somebody is waiting for it
   ( ( nanos::ext::GPUProcessor * ) pe )->getOutTransferList()->requestTransfer( (void * ) hostAddress );
}

void * GPUDevice::realloc( void * address, size_t size, size_t ceSize, ProcessingElement *pe )
{
   free( address, pe );
   return allocate( size, pe );
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

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyOutAsyncWait ()
{
   cudaStreamSynchronize( ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getOutTransferStream() );
}

void GPUDevice::copyOutAsyncToHost ( void * dst, void * src, size_t size )
{
   SMPDevice::copyLocal( dst, ( void * ) ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getPinnedAddress(src), size, NULL );
}

void GPUDevice::copyOutSyncToHost ( void * dst, void * src, size_t size )
{
   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost );

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

