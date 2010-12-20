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
#include <sys/resource.h>

#include <cuda_runtime.h>

using namespace nanos;


unsigned int GPUDevice::_rlimit;


void GPUDevice::getMemoryLockLimit()
{
   if ( nanos::ext::GPUConfig::getTransferMode() == nanos::ext::NANOS_GPU_TRANSFER_PINNED_OS ) {
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

uint64_t GPUDevice::allocateIntermediateBuffer( void * deviceAddress, size_t size, ProcessingElement *pe )
{
   uint64_t pinned;
   cudaError_t err = cudaMallocHost( ( void ** ) &pinned, size );

   // Note: Use cudaHostAllocPortable flag in order to allocate host memory that must be
   // accessed from more than one GPU
   //err = cudaHostAlloc( ( void ** ) &pinned, size, cudaHostAllocPortable);

   fatal_cond( err != cudaSuccess, "Trying to allocate " + toString<size_t>( size )
         + " bytes of host memory with cudaMallocHost(): " + cudaGetErrorString( err ) );

   ( ( nanos::ext::GPUProcessor * ) pe )->setPinnedAddress( deviceAddress, pinned );

   return pinned;
}

void GPUDevice::freeIntermediateBuffer( uint64_t pinnedAddress, void * deviceAddress, ProcessingElement *pe )
{
   cudaError_t err = cudaFreeHost( ( void * ) pinnedAddress );
   ( ( nanos::ext::GPUProcessor * ) pe )->removePinnedAddress( deviceAddress );

   fatal_cond( err != cudaSuccess, "Trying to free host memory at " + toString<uint64_t>( pinnedAddress ) +
         + " with cudaFreeHost(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyLocal( void *dst, void *src, size_t size, ProcessingElement *pe )
{
   // Copy from device memory to device memory
   cudaError_t err = cudaSuccess;

   if ( ( ( nanos::ext::GPUProcessor * ) pe )->getGPUProcessorInfo()->getLocalTransferStream() != 0 ) {
      err = cudaMemcpyAsync(
               dst,
               src,
               size,
               cudaMemcpyDeviceToDevice,
               ((nanos::ext::GPUProcessor *) pe )->getGPUProcessorInfo()->getLocalTransferStream()
            );
    }
    else {
       err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToDevice );
   }

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from device (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInSyncToDevice ( void * dst, void * src, size_t size )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferInput( size );

   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice );

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from host (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInAsyncToBuffer( void * dst, void * src, size_t size )
{
   SMPDevice::copyLocal(
            ( void * ) ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getPinnedAddress( dst ),
            src,
            size,
            NULL
         );
}

void GPUDevice::copyInAsyncToDevice( void * dst, void * src, size_t size )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferInput( size );

   cudaError_t err = cudaMemcpyAsync(
            dst,
            ( void * ) ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getPinnedAddress( dst ),
            size,
            cudaMemcpyHostToDevice,
            ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getOutTransferStream()
         );

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from host (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInAsyncWait()
{
   cudaStreamSynchronize( ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getInTransferStream() );
}

void GPUDevice::copyOutSyncToHost ( void * dst, void * src, size_t size )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferOutput( size );

   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost );

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyOutAsyncToBuffer ( void * dst, void * src, size_t size )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferOutput( size );

   cudaError_t err = cudaMemcpyAsync(
            ( void * ) ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getPinnedAddress( src ),
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
   SMPDevice::copyLocal(
            dst,
            ( void * ) ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getPinnedAddress( src ),
            size,
            NULL
         );
}
