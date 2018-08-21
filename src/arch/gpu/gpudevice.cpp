/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
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

void * GPUDevice::allocatePinnedMemory2( size_t size )
{
   void * address = 0;

   cudaError_t err = cudaHostAlloc( &address, size, cudaHostAllocDefault );

   fatal_cond( err != cudaSuccess, "Trying to allocate " +  toString<size_t>( size ) +
         + " bytes of host memory with cudaHostAlloc(): " +  cudaGetErrorString( err ) );

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

void GPUDevice::copyInSyncToDevice ( void * dst, void * src, size_t len, size_t count, size_t ld )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferInput( len * count );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_TO_DEVICE_EVENT );
   cudaError_t err = ( count == 1 ) ?
      cudaMemcpy( dst, src, len, cudaMemcpyHostToDevice ) :
      cudaMemcpy2D( dst, ld, src, ld, len, count, cudaMemcpyHostToDevice );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( len * count )
         + " of data from host (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInAsyncToBuffer( void * dst, void * src, size_t len, size_t count, size_t ld )
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_MEMCOPY_EVENT );
   for ( std::size_t icount = 0; icount < count; icount += 1 ) {
      ::memcpy( ((char *)dst) + len * icount , ((char *)src) + ld * icount, len );
   }
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

void GPUDevice::copyInAsyncToDevice( void * dst, void * src, size_t len, size_t count, size_t ld )
{
   nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();

   myPE->transferInput( len * count );

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cudaEvent_t evt1;
   cudaEventCreate( &evt1, 0 );
   cudaEventRecord( evt1, myPE->getGPUProcessorInfo()->getInTransferStream() );

   cudaStreamWaitEvent( myPE->getGPUProcessorInfo()->getTracingInputStream(), evt1, 0 );

   nanos::ext::GPUCallbackData * cbd = NEW nanos::ext::GPUCallbackData( ( nanos::ext::GPUThread * ) myThread, len * count );

   cudaStreamAddCallback( myPE->getGPUProcessorInfo()->getTracingInputStream(), nanos::ext::beforeAsyncInputCallback, ( void * ) cbd, 0 );
#endif

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_DEVICE_EVENT );
   cudaError_t err = ( count == 1 ) ?
      cudaMemcpyAsync(
            dst,
            src,
            len,
            cudaMemcpyHostToDevice,
            myPE->getGPUProcessorInfo()->getInTransferStream()
      ) :
      cudaMemcpy2DAsync(
            dst,
            ld,
            src,
            len,
            len,
            count,
            cudaMemcpyHostToDevice,
            myPE->getGPUProcessorInfo()->getInTransferStream()
      );

   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cudaEvent_t evt2;
   cudaEventCreate( &evt2, 0 );
   cudaEventRecord( evt2, myPE->getGPUProcessorInfo()->getInTransferStream() );

   cudaStreamWaitEvent( myPE->getGPUProcessorInfo()->getTracingInputStream(), evt2, 0 );

   nanos::ext::GPUCallbackData * cbd2 = NEW nanos::ext::GPUCallbackData( ( nanos::ext::GPUThread * ) myThread, len * count );

   cudaStreamAddCallback( myPE->getGPUProcessorInfo()->getTracingInputStream(), nanos::ext::afterAsyncInputCallback, ( void * ) cbd2, 0 );
#endif

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( len * count )
         + " of data from host (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInAsyncWait()
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_INPUT_STREAM_SYNC_EVENT );
   cudaStreamSynchronize( ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getInTransferStream() );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

void GPUDevice::copyOutSyncToHost ( void * dst, void * src, size_t len, size_t count, size_t ld )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferOutput( len*count );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_TO_HOST_EVENT );
   cudaError_t err = ( count == 1 ) ? 
      cudaMemcpy( dst, src, len, cudaMemcpyDeviceToHost ) :
      cudaMemcpy2D( dst, ld, src, ld, len, count, cudaMemcpyDeviceToHost );

   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( len*count )
         + " of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyOutAsyncToBuffer ( void * dst, void * src, size_t len, size_t count, size_t ld )
{
   nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();
   myPE->transferOutput( len * count );

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cudaEvent_t evt1;
   cudaEventCreate( &evt1, 0 );
   cudaEventRecord( evt1, myPE->getGPUProcessorInfo()->getOutTransferStream() );

   cudaStreamWaitEvent( myPE->getGPUProcessorInfo()->getTracingOutputStream(), evt1, 0 );

   nanos::ext::GPUCallbackData * cbd = NEW nanos::ext::GPUCallbackData( ( nanos::ext::GPUThread * ) myThread, len * count );

   cudaStreamAddCallback( myPE->getGPUProcessorInfo()->getTracingOutputStream(), nanos::ext::beforeAsyncOutputCallback, ( void * ) cbd, 0 );
#endif

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_HOST_EVENT );
   cudaError_t err = ( count == 1 ) ?
      cudaMemcpyAsync(
            dst,
            src,
            len,
            cudaMemcpyDeviceToHost,
            myPE->getGPUProcessorInfo()->getOutTransferStream()
         ) :
      cudaMemcpy2DAsync(
            dst,
            len,
            src,
            ld,
            len,
            count,
            cudaMemcpyDeviceToHost,
            myPE->getGPUProcessorInfo()->getOutTransferStream()
         );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cudaEvent_t evt2;
   cudaEventCreate( &evt2, 0 );
   cudaEventRecord( evt2, myPE->getGPUProcessorInfo()->getOutTransferStream() );

   cudaStreamWaitEvent( myPE->getGPUProcessorInfo()->getTracingOutputStream(), evt2, 0 );

   nanos::ext::GPUCallbackData * cbd2 = NEW nanos::ext::GPUCallbackData( ( nanos::ext::GPUThread * ) myThread, len * count );

   cudaStreamAddCallback( myPE->getGPUProcessorInfo()->getTracingOutputStream(), nanos::ext::afterAsyncOutputCallback, ( void * ) cbd2, 0 );
#endif

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( len * count )
         + " of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyOutAsyncWait ()
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_OUTPUT_STREAM_SYNC_EVENT );
   cudaStreamSynchronize( ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getOutTransferStream() );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

void GPUDevice::copyOutAsyncToHost ( void * dst, void * src, size_t len, size_t count, size_t ld )
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_MEMCOPY_EVENT );
   for ( std::size_t icount = 0; icount < count; icount += 1 ) {
      ::memcpy( ((char *)dst) + ld * icount, ((char *)src) + len * icount, len );
   }
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

//void *GPUDevice::memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem ) {
//   void *mem = allocate( size, &pe );
//   //std::cerr << "GPU memAllocate( " << size << " )  returns: " << mem << std::endl;
//   return mem;
//}
void * GPUDevice::memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WD const *wd, unsigned int copyIdx )
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
      WD const *wd, void *hostObject, reg_t hostRegionId )
{
   CopyDescriptor cd( hostAddr );
   cd._ops = ops;
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   ops->addOp();
   ( myThread->runningOn() == gpu ) ? isMycopyIn( ( void * ) devAddr, cd, len, 1, 0, mem, gpu ) : isNotMycopyIn( ( void * ) devAddr, cd, len, 1, 0, mem, gpu );
}

void GPUDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops,
      WD const *wd, void *hostObject, reg_t hostRegionId ) 
{
   CopyDescriptor cd( hostAddr );
   cd._ops = ops;
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   ops->addOp();
   ( myThread->runningOn() == gpu ) ? isMycopyOut( cd, (void *) devAddr, len, 1, 0, mem, gpu ) : isNotMycopyOut( cd, (void *) devAddr, len, 1, 0, mem, gpu );
}

bool GPUDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest,
      SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) 
{
   CopyDescriptor cd( 0xdeaddead );
   cd._ops = ops;

   ext::GPUMemorySpace *gpuMemDataOrig = ( ext::GPUMemorySpace * ) memOrig.getSpecificData();
   ext::GPUMemorySpace *gpuMemDataDest = ( ext::GPUMemorySpace * ) memDest.getSpecificData();
   ext::GPUProcessor *gpuOrig = gpuMemDataOrig->getGPU();
   ext::GPUProcessor *gpuDest = gpuMemDataDest->getGPU();



   nanos::ext::GPUThread * thread = ( nanos::ext::GPUThread * ) gpuDest->getFirstThread();

   GenericEvent * evt = thread->createPreRunEvent( thread->getCurrentWD() );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " copy dev2dev: " + toString<uint64_t>( devDestAddr ) );
#endif
   evt->setCreated();

   Action * action = new_action( ( ActionMemFunPtr0<DeviceOps>::MemFunPtr0 ) &DeviceOps::completeOp, cd._ops );
   evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:DeviceOps::completeOp" );
#endif

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

   evt->setPending();

   thread->addEvent( evt );

   return true;
}

void GPUDevice::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {
   CopyDescriptor cd( hostAddr );
   cd._ops = ops;
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   ops->addOp();
   ( myThread->runningOn() == gpu ) ? isMycopyIn( ( void * ) devAddr, cd, len, count, ld, mem, gpu ) : isNotMycopyIn( ( void * ) devAddr, cd, len, count, ld, mem, gpu );
}

void GPUDevice::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {
   CopyDescriptor cd( hostAddr );
   cd._ops = ops;
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   ops->addOp();
   ( myThread->runningOn() == gpu ) ? isMycopyOut( cd, (void *) devAddr, len, count, ld, mem, gpu ) : isNotMycopyOut( cd, (void *) devAddr, len, count, ld, mem, gpu );
}

bool GPUDevice::_copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {
   CopyDescriptor cd( 0xdeaddead );
   cd._ops = ops;

   ext::GPUMemorySpace *gpuMemDataOrig = ( ext::GPUMemorySpace * ) memOrig.getSpecificData();
   ext::GPUMemorySpace *gpuMemDataDest = ( ext::GPUMemorySpace * ) memDest.getSpecificData();
   ext::GPUProcessor *gpuOrig = gpuMemDataOrig->getGPU();
   ext::GPUProcessor *gpuDest = gpuMemDataDest->getGPU();



   nanos::ext::GPUThread * thread = ( nanos::ext::GPUThread * ) gpuDest->getFirstThread();

   GenericEvent * evt = thread->createPreRunEvent( thread->getCurrentWD() );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " copy dev2dev: " + toString<uint64_t>( devDestAddr ) );
#endif
   evt->setCreated();

   Action * action = new_action( ( ActionMemFunPtr0<DeviceOps>::MemFunPtr0 ) &DeviceOps::completeOp, cd._ops );
   evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:DeviceOps::completeOp" );
#endif

   ops->addOp();

   gpuOrig->transferDevice( len );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_MEMCOPY_ASYNC_EVENT );

   cudaMemcpy3DPeerParms myParms = {0};
   myParms.dstDevice = gpuDest->getDeviceId();
   myParms.dstArray = NULL;
   myParms.dstPos = make_cudaPos( 0, 0, 0 );
   myParms.dstPtr.ptr = (void *) devDestAddr;
   myParms.dstPtr.pitch = ld;
   myParms.dstPtr.xsize = ld;
   myParms.dstPtr.ysize = count;

   myParms.extent = make_cudaExtent( len, count, 1 );

   myParms.srcArray = NULL;
   myParms.srcDevice = gpuOrig->getDeviceId();
   myParms.srcPos = make_cudaPos( 0, 0, 0 );
   myParms.srcPtr.ptr = (void *) devOrigAddr;
   myParms.srcPtr.pitch = ld;
   myParms.srcPtr.xsize = ld;
   myParms.srcPtr.ysize = count;

   cudaError_t err = cudaMemcpy3DPeerAsync( &myParms, gpuDest->getGPUProcessorInfo()->getInTransferStream() );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::GPUUtils::bytesToHumanReadable( len )
   + " of data from device #" + toString<int>( gpuOrig->getDeviceId() ) + " (" + toString<uint64_t>( devOrigAddr )
   + ") to device #" + toString<int>( gpuDest->getDeviceId() ) + " ("
   + toString<uint64_t>( devDestAddr ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );

   evt->setPending();

   thread->addEvent( evt );

   return true;
}
std::size_t GPUDevice::getMemCapacity( SeparateMemoryAddressSpace &mem ) {
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   return gpu->getMaxMemoryAvailable();
}

void GPUDevice::_getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list ) {
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   SimpleAllocator *allocator = gpuMemData->getAllocator();
   allocator->getFreeChunksList( list );
}

void GPUDevice::_canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) {
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   SimpleAllocator *allocator = gpuMemData->getAllocator();
   allocator->canAllocate( sizes, numChunks, remainingSizes );
}
