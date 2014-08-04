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
#include "gpuutils.hpp"
#include "gpucallback.hpp"
#include "gpuconfig.hpp"
#include "gpumemoryspace_decl.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "deviceops.hpp"
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

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MALLOC_EVENT );
   cudaError_t err = cudaMalloc( ( void ** ) &address, ( size_t ) ( size * percentage ) );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   while ( ( err == cudaErrorMemoryValueTooLarge || err == cudaErrorMemoryAllocation )
         && percentage > 0.5 )
   {
      percentage -= 0.05;
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MALLOC_EVENT );
      err = cudaMalloc( ( void ** ) &address, ( size_t ) ( size * percentage ) );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
   }

   size = ( size_t ) ( size * percentage );

   if ( err != cudaSuccess ) {
      size_t free, total;
      cudaMemGetInfo( &free, &total );

      fatal_cond( size > free, "Trying to allocate " + ext::GPUUtils::bytesToHumanReadable( size )
            + " of device memory with cudaMalloc() when available memory is only "
            + ext::GPUUtils::bytesToHumanReadable( free ) );

      fatal_cond( err != cudaSuccess, "Trying to allocate " +  ext::GPUUtils::bytesToHumanReadable( size )
            + " of device memory with cudaMalloc(): " +  cudaGetErrorString( err ) );
   }

   // Reset CUDA errors that may have occurred during this memory allocation
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_GET_LAST_ERROR_EVENT );
   err = cudaGetLastError();
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   return address;
}

void GPUDevice::freeWholeMemory( void * address )
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_FREE_EVENT );
   cudaError_t err = cudaFree( address );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to free device memory at " + toString<void *>( address ) +
         + " with cudaFree(): " + cudaGetErrorString( err ) );
}

void * GPUDevice::allocatePinnedMemory( size_t size )
{
   void * address = NULL;

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MALLOC_HOST_EVENT );
   cudaError_t err = cudaMallocHost( &address, size );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   if ( err == cudaErrorMemoryAllocation) {
      // Out of memory error, try a workaround: first, allocate and then register the memory as pinned
      address = std::malloc( size );
      if ( address != NULL ) {
         err = cudaHostRegister( address, size, cudaHostRegisterPortable );

         if ( err != cudaSuccess ) {
            warning( "Memory allocation succeeded, but could not register the address as pinned memory with cudaHostRegister(): "
                  << cudaGetErrorString( err ) );
            err = cudaSuccess; // To avoid the fatal_cond below
         }
      }
   }

   ensure( address != NULL, "cudaMallocHost() returned a NULL pointer while trying to allocate "
         + ext::GPUUtils::bytesToHumanReadable( size ) + ". Error returned by CUDA is: " + cudaGetErrorString( err ) );

   fatal_cond( err != cudaSuccess, "Trying to allocate " +  ext::GPUUtils::bytesToHumanReadable( size ) +
         + " of host memory with cudaMallocHost(): " +  cudaGetErrorString( err ) );

   return address;
}

void GPUDevice::freePinnedMemory( void * address )
{
   // There are 2 possible ways of getting pinned memory:
   // 1) Calling cudaMallocHost() or cudaHostAlloc()
   // 2) Allocating memory + calling cudaHostRegister()
   // As we don't know which method we used, we need to control the errors returned by
   // CUDA and act properly.

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_FREE_HOST_EVENT );
   cudaError_t err = cudaFreeHost( address );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   if ( err == cudaSuccess ) return;

   // err value should then be cudaErrorInvalidValue
   err = cudaHostUnregister( address );

   if ( err == cudaSuccess || err == cudaErrorHostMemoryNotRegistered ) {
      std::free( address );
      err = cudaSuccess;
   }

   fatal_cond( err != cudaSuccess, "Trying to free host memory at " + toString<void *>( address ) +
         + " with cudaFreeHost(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInSyncToDevice ( void * dst, void * src, size_t size )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferInput( size );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_TO_DEVICE_EVENT );
   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( size )
         + " of data from host (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInAsyncToBuffer( void * dst, void * src, size_t size )
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_MEMCOPY_EVENT );
   ::memcpy( dst, src, size );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

void GPUDevice::copyInAsyncToDevice( void * dst, void * src, size_t size )
{
   nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();

   myPE->transferInput( size );

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cudaEvent_t evt1;
   cudaEventCreate( &evt1, 0 );
   cudaEventRecord( evt1, myPE->getGPUProcessorInfo()->getInTransferStream() );

   cudaStreamWaitEvent( myPE->getGPUProcessorInfo()->getTracingInputStream(), evt1, 0 );

   nanos::ext::GPUCallbackData * cbd = NEW nanos::ext::GPUCallbackData( ( nanos::ext::GPUThread * ) myThread, size );

   cudaStreamAddCallback( myPE->getGPUProcessorInfo()->getTracingInputStream(), nanos::ext::beforeAsyncInputCallback, ( void * ) cbd, 0 );
#endif

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_DEVICE_EVENT );
   cudaError_t err = cudaMemcpyAsync(
            dst,
            src,
            size,
            cudaMemcpyHostToDevice,
            myPE->getGPUProcessorInfo()->getInTransferStream()
         );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cudaEvent_t evt2;
   cudaEventCreate( &evt2, 0 );
   cudaEventRecord( evt2, myPE->getGPUProcessorInfo()->getInTransferStream() );

   cudaStreamWaitEvent( myPE->getGPUProcessorInfo()->getTracingInputStream(), evt2, 0 );

   nanos::ext::GPUCallbackData * cbd2 = NEW nanos::ext::GPUCallbackData( ( nanos::ext::GPUThread * ) myThread, size );

   cudaStreamAddCallback( myPE->getGPUProcessorInfo()->getTracingInputStream(), nanos::ext::afterAsyncInputCallback, ( void * ) cbd2, 0 );
#endif

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( size )
         + " of data from host (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInAsyncWait()
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_INPUT_STREAM_SYNC_EVENT );
   cudaStreamSynchronize( ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getInTransferStream() );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

void GPUDevice::copyOutSyncToHost ( void * dst, void * src, size_t size )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferOutput( size );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_TO_HOST_EVENT );
   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( size )
         + " of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyOutAsyncToBuffer ( void * dst, void * src, size_t size )
{
   nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();
   myPE->transferOutput( size );

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cudaEvent_t evt1;
   cudaEventCreate( &evt1, 0 );
   cudaEventRecord( evt1, myPE->getGPUProcessorInfo()->getOutTransferStream() );

   cudaStreamWaitEvent( myPE->getGPUProcessorInfo()->getTracingOutputStream(), evt1, 0 );

   nanos::ext::GPUCallbackData * cbd = NEW nanos::ext::GPUCallbackData( ( nanos::ext::GPUThread * ) myThread, size );

   cudaStreamAddCallback( myPE->getGPUProcessorInfo()->getTracingOutputStream(), nanos::ext::beforeAsyncOutputCallback, ( void * ) cbd, 0 );
#endif

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_HOST_EVENT );
   cudaError_t err = cudaMemcpyAsync(
            dst,
            src,
            size,
            cudaMemcpyDeviceToHost,
            myPE->getGPUProcessorInfo()->getOutTransferStream()
         );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cudaEvent_t evt2;
   cudaEventCreate( &evt2, 0 );
   cudaEventRecord( evt2, myPE->getGPUProcessorInfo()->getOutTransferStream() );

   cudaStreamWaitEvent( myPE->getGPUProcessorInfo()->getTracingOutputStream(), evt2, 0 );

   nanos::ext::GPUCallbackData * cbd2 = NEW nanos::ext::GPUCallbackData( ( nanos::ext::GPUThread * ) myThread, size );

   cudaStreamAddCallback( myPE->getGPUProcessorInfo()->getTracingOutputStream(), nanos::ext::afterAsyncOutputCallback, ( void * ) cbd2, 0 );
#endif

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( size )
         + " of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyOutAsyncWait ()
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_OUTPUT_STREAM_SYNC_EVENT );
   cudaStreamSynchronize( ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getOutTransferStream() );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

void GPUDevice::copyOutAsyncToHost ( void * dst, void * src, size_t size )
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_MEMCOPY_EVENT );
   ::memcpy( dst, src, size );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

void * GPUDevice::memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WorkDescriptor const &wd, unsigned int copyIdx )
{
   void * address = NULL;

   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   SimpleAllocator *allocator = gpuMemData->getAllocator();
   allocator->lock();
   address = allocator->allocate( NANOS_ALIGNED_MEMORY_OFFSET( 0, size, gpuMemData->getAlignment() ) );
   allocator->unlock();
   return address;
}

void GPUDevice::memFree( uint64_t addr, SeparateMemoryAddressSpace &mem )
{
   // Check there are no pending copies to execute before we free the memory (and if there are, execute them)
   
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   gpuMemData->getGPU()->getOutTransferList()->checkAddressForMemoryTransfer( (void*) addr );
   SimpleAllocator *allocator = gpuMemData->getAllocator();
   allocator->lock();
   allocator->free( (void*)addr );
   allocator->unlock();
}

void GPUDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops,
      Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const
{
   CopyDescriptor cd( hostAddr );
   cd._ops = ops;
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   ops->addOp();
   ( myThread->runningOn() == gpu ) ? isMycopyIn( ( void * ) devAddr, cd, len, mem, gpu ) : isNotMycopyIn( ( void * ) devAddr, cd, len, mem, gpu );
}

void GPUDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops,
      Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const
{
   CopyDescriptor cd( hostAddr );
   cd._ops = ops;
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   ops->addOp();
   ( myThread->runningOn() == gpu ) ? isMycopyOut( cd, (void *) devAddr, len, mem, gpu ) : isNotMycopyOut( cd, (void *) devAddr, len, mem, gpu );
}

bool GPUDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest,
      SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const
{
   CopyDescriptor cd( 0xdeaddead );
   cd._ops = ops;
   cd._functor = f;

   ext::GPUMemorySpace *gpuMemDataOrig = ( ext::GPUMemorySpace * ) memOrig.getSpecificData();
   ext::GPUMemorySpace *gpuMemDataDest = ( ext::GPUMemorySpace * ) memDest.getSpecificData();
   ext::GPUProcessor *gpuOrig = gpuMemDataOrig->getGPU();
   ext::GPUProcessor *gpuDest = gpuMemDataDest->getGPU();

   ops->addOp();

   gpuOrig->transferDevice( len );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_ASYNC_EVENT );
   cudaError_t err = cudaMemcpyPeerAsync( ( void * ) devDestAddr, gpuDest->getDeviceId(), ( void * ) devOrigAddr, gpuOrig->getDeviceId(),
         len, gpuDest->getGPUProcessorInfo()->getInTransferStream() );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( len )
   + " of data from device #" + toString<int>( gpuOrig->getDeviceId() ) + " (" + toString<uint64_t>( devOrigAddr )
   + ") to device #" + toString<int>( gpuDest->getDeviceId() ) + " ("
   + toString<uint64_t>( devDestAddr ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );

   gpuDest->getInTransferList()->addMemoryTransfer( cd );

   return false;
}

void GPUDevice::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace const &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) {
   std::cerr << __FUNCTION__ << ": unimplemented" << std::endl;
}

void GPUDevice::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace const &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) {
   std::cerr << __FUNCTION__ << ": unimplemented" << std::endl;
}

bool GPUDevice::_copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace const &memDest, SeparateMemoryAddressSpace const &memOrig, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const {
   std::cerr << __FUNCTION__ << ": unimplemented" << std::endl;
   return false;
}
std::size_t GPUDevice::getMemCapacity( SeparateMemoryAddressSpace const &mem ) const {
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   return gpu->getMaxMemoryAvailable();
}

void GPUDevice::_getFreeMemoryChunksList( SeparateMemoryAddressSpace const &mem, SimpleAllocator::ChunkList &list ) const {
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   SimpleAllocator *allocator = gpuMemData->getAllocator();
   allocator->getFreeChunksList( list );
}

void GPUDevice::_canAllocate( SeparateMemoryAddressSpace const &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) const {
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   SimpleAllocator *allocator = gpuMemData->getAllocator();
   allocator->canAllocate( sizes, numChunks, remainingSizes );
}
