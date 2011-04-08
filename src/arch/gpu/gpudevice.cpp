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

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MALLOC_EVENT );
   cudaError_t err = cudaMalloc( ( void ** ) &address, ( size_t ) ( size * percentage ) );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   while ( ( err == cudaErrorMemoryValueTooLarge || err == cudaErrorMemoryAllocation )
         && percentage > 0.5 )
   {
      percentage -= 0.05;
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MALLOC_EVENT );
      err = cudaMalloc( ( void ** ) &address, ( size_t ) ( size * percentage ) );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
   }

   size = ( size_t ) ( size * percentage );

   fatal_cond( err != cudaSuccess, "Trying to allocate " +  toString<size_t>( size ) +
         + " bytes of device memory with cudaMalloc(): " +  cudaGetErrorString( err ) );

   // Reset CUDA errors that may have occurred during this memory allocation
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_GET_LAST_ERROR_EVENT );
   err = cudaGetLastError();
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   return address;
}

void GPUDevice::freeWholeMemory( void * address )
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_FREE_EVENT );
   cudaError_t err = cudaFree( address );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to free device memory at " + toString<void *>( address ) +
         + " with cudaFree(): " + cudaGetErrorString( err ) );
}

void * GPUDevice::allocatePinnedMemory( size_t size )
{
   void * address = 0;

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MALLOC_HOST_EVENT );
   cudaError_t err = cudaMallocHost( &address, size );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to allocate " +  toString<size_t>( size ) +
         + " bytes of host memory with cudaMallocHost(): " +  cudaGetErrorString( err ) );

   return address;
}

void GPUDevice::freePinnedMemory( void * address )
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_FREE_HOST_EVENT );
   cudaError_t err = cudaFreeHost( address );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to free host memory at " + toString<void *>( address ) +
         + " with cudaFreeHost(): " + cudaGetErrorString( err ) );
}

#if 0
uint64_t GPUDevice::allocateIntermediateBuffer( void * deviceAddress, size_t size, ProcessingElement *pe )
{
   uint64_t pinned;
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MALLOC_HOST_EVENT );
   cudaError_t err = cudaMallocHost( ( void ** ) &pinned, size );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

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
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_FREE_HOST_EVENT );
   cudaError_t err = cudaFreeHost( ( void * ) pinnedAddress );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
   ( ( nanos::ext::GPUProcessor * ) pe )->removePinnedAddress( deviceAddress );

   fatal_cond( err != cudaSuccess, "Trying to free host memory at " + toString<uint64_t>( pinnedAddress ) +
         + " with cudaFreeHost(): " + cudaGetErrorString( err ) );
}
#endif

void GPUDevice::copyLocal( void *dst, void *src, size_t size, ProcessingElement *pe )
{
   // Copy from device memory to device memory
   cudaError_t err = cudaSuccess;

   if ( ( ( nanos::ext::GPUProcessor * ) pe )->getGPUProcessorInfo()->getLocalTransferStream() != 0 ) {
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_DEVICE_EVENT );
      err = cudaMemcpyAsync(
               dst,
               src,
               size,
               cudaMemcpyDeviceToDevice,
               ((nanos::ext::GPUProcessor *) pe )->getGPUProcessorInfo()->getLocalTransferStream()
            );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
    }
    else {
       NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MEMCOPY_TO_DEVICE_EVENT );
       err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToDevice );
       NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
   }

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from device (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInSyncToDevice ( void * dst, void * src, size_t size )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferInput( size );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MEMCOPY_TO_DEVICE_EVENT );
   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from host (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInAsyncToBuffer( void * dst, void * src, size_t size )
{
   //nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();

#if 0
   // Workaround to perform asynchronous copies
   uint64_t pinned = myPE->getPinnedAddress( dst );
   if ( pinned == 0 )
      pinned = ( uint64_t ) allocateIntermediateBuffer( dst, size, myPE );
#elif 0
   void * pinned = ( void * ) myPE->getPinnedAddress( dst );
   if ( pinned == 0 ) {
      pinned = myPE->allocatePinnedMemory( size );
      myPE->setPinnedAddress( dst, ( uint64_t ) pinned );
   }
#endif

   SMPDevice::copyLocal( dst, src, size, NULL );
   //SMPDevice::copyLocal( ( void * ) myPE->getPinnedAddress( dst ), src, size, NULL );
}

void GPUDevice::copyInAsyncToDevice( void * dst, void * src, size_t size )
{
   nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();

   myPE->transferInput( size );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_DEVICE_EVENT );
   cudaError_t err = cudaMemcpyAsync(
            dst,
            src,
            size,
            cudaMemcpyHostToDevice,
            myPE->getGPUProcessorInfo()->getOutTransferStream()
         );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from host (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInAsyncWait()
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_INPUT_STREAM_SYNC_EVENT );
   cudaStreamSynchronize( ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getInTransferStream() );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

void GPUDevice::copyOutSyncToHost ( void * dst, void * src, size_t size )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferOutput( size );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MEMCOPY_TO_HOST_EVENT );
   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyOutAsyncToBuffer ( void * dst, void * src, size_t size )
{
   nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();
   myPE->transferOutput( size );

#if 0
   // Workaround to perform asynchronous copies
   uint64_t pinned = myPE->getPinnedAddress( src );
   if ( pinned == 0 )
      pinned = ( uint64_t ) allocateIntermediateBuffer( src, size, myPE );
#elif 0
   void * pinned = ( void * ) myPE->getPinnedAddress( src );
   if ( pinned == 0 ) {
      pinned = myPE->allocatePinnedMemory( size );
      myPE->setPinnedAddress( src, ( uint64_t ) pinned );
   }
#endif

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_HOST_EVENT );
   cudaError_t err = cudaMemcpyAsync(
            dst,
            //( void * ) myPE->getPinnedAddress( src ),
            src,
            size,
            cudaMemcpyDeviceToHost,
            myPE->getGPUProcessorInfo()->getOutTransferStream()
         );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + toString<size_t>( size )
         + " bytes of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyOutAsyncWait ()
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_OUTPUT_STREAM_SYNC_EVENT );
   cudaStreamSynchronize( ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getOutTransferStream() );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

void GPUDevice::copyOutAsyncToHost ( void * dst, void * src, size_t size )
{
   //nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();
   //void * pinned = ( void * ) myPE->getPinnedAddress( src );

   SMPDevice::copyLocal( dst, src, size, NULL );

   //myPE->removePinnedAddress( src );
   //myPE->freePinnedMemory( pinned );
}
