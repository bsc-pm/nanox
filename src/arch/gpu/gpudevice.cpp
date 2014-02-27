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

   if ( err != cudaSuccess ) {
      size_t free, total;
      cudaMemGetInfo( &free, &total );

      fatal_cond( size > free, "Trying to allocate " + ext::bytesToHumanReadable( size )
            + " of device memory with cudaMalloc() when available memory is only "
            + ext::bytesToHumanReadable( free ) );

      fatal_cond( err != cudaSuccess, "Trying to allocate " +  ext::bytesToHumanReadable( size )
            + " of device memory with cudaMalloc(): " +  cudaGetErrorString( err ) );
   }

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

   ensure( address != NULL, "cudaMallocHost() returned a NULL pointer while trying to allocate "
         + ext::bytesToHumanReadable( size ) + ". Error returned by CUDA is: " + cudaGetErrorString( err ) );

   fatal_cond( err != cudaSuccess, "Trying to allocate " +  ext::bytesToHumanReadable( size ) +
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
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_FREE_HOST_EVENT );
   cudaError_t err = cudaFreeHost( address );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to free host memory at " + toString<void *>( address ) +
         + " with cudaFreeHost(): " + cudaGetErrorString( err ) );
}

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

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::bytesToHumanReadable( size )
         + " of data from device (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInSyncToDevice ( void * dst, void * src, size_t size )
{
   ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->transferInput( size );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MEMCOPY_TO_DEVICE_EVENT );
   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice );
   //std::cerr <<"cudaMemCpy " << dst << " " << src << " "<<size <<std::endl;
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::bytesToHumanReadable( size )
         + " of data from host (" + toString<void *>( src ) + ") to device ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyInAsyncToBuffer( void * dst, void * src, size_t size )
{
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_MEMCOPY_EVENT );
   SMPDevice::copyLocal( dst, src, size, NULL );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
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
            myPE->getGPUProcessorInfo()->getInTransferStream()
         );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::bytesToHumanReadable( size )
         + " of data from host (" + toString<void *>( src ) + ") to device ("
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
   //std::cerr <<"cudaMemCpy " << dst << " " << src << " "<<size <<std::endl;
   cudaError_t err = cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::bytesToHumanReadable( size )
         + " of data from device (" + toString<void *>( src ) + ") to host ("
         + toString<void *>( dst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );
}

void GPUDevice::copyOutAsyncToBuffer ( void * dst, void * src, size_t size )
{
   nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();
   myPE->transferOutput( size );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_HOST_EVENT );
   cudaError_t err = cudaMemcpyAsync(
            dst,
            src,
            size,
            cudaMemcpyDeviceToHost,
            myPE->getGPUProcessorInfo()->getOutTransferStream()
         );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::bytesToHumanReadable( size )
         + " of data from device (" + toString<void *>( src ) + ") to host ("
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
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_MEMCOPY_EVENT );
   SMPDevice::copyLocal( dst, src, size, NULL );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
}

bool GPUDevice::copyDevToDev( void * addrDst, CopyDescriptor &dstCd, void * addrSrc, std::size_t size, ProcessingElement *peDst, ProcessingElement *peSrc )
{
#ifndef NANOS_GPU_USE_CUDA32
//   fatal_cond( peDst->getDeviceType().getName() != peSrc->getDeviceType().getName(),
//         "Do not know how to copy between different devices: from " +  peSrc->getDeviceType().getName()
//         + " to " + peDst->getDeviceType().getName() );

   nanos::ext::GPUProcessor * gpuDst = ( nanos::ext::GPUProcessor * ) peDst;
   nanos::ext::GPUProcessor * gpuSrc = ( nanos::ext::GPUProcessor * ) peSrc;

   gpuSrc->transferDevice( size );

   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_MEMCOPY_ASYNC_EVENT );
   cudaError_t err = cudaMemcpyPeerAsync( addrDst, gpuDst->getDeviceId(), addrSrc, gpuSrc->getDeviceId(), size,
         gpuDst->getGPUProcessorInfo()->getInTransferStream() );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   fatal_cond( err != cudaSuccess, "Trying to copy " + ext::bytesToHumanReadable( size )
         + " of data from device #" + toString<int>( gpuSrc->getDeviceId() ) + " (" + toString<void *>( addrSrc )
         + ") to device #" + toString<int>( gpuDst->getDeviceId() ) + " ("
         + toString<void *>( addrDst ) + ") with cudaMemcpy*(): " + cudaGetErrorString( err ) );

   // If input's stream is NULL, return true, as the transfer won't be overlapped
   if ( gpuDst->getGPUProcessorInfo()->getInTransferStream() == 0 ) return true;

   // Otherwise, keep track of the transfer and return false
   gpuDst->getInTransferList()->addMemoryTransfer( dstCd );

   return false;

#endif
   return true;
}

//void *GPUDevice::memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem ) const {
//   void *mem = allocate( size, &pe );
//   //std::cerr << "GPU memAllocate( " << size << " )  returns: " << mem << std::endl;
//   return mem;
//}

void * GPUDevice::memAllocate( size_t size, SeparateMemoryAddressSpace &mem, uint64_t targetHostAddr) const
{
   void * address = NULL;

   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   SimpleAllocator *allocator = gpuMemData->getAllocator();
   allocator->lock();
   address = allocator->allocate( NANOS_ALIGNED_MEMORY_OFFSET( 0, size, gpuMemData->getAlignment() ) );
   allocator->unlock();
   return address;
}

void GPUDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const {
   CopyDescriptor cd( hostAddr ); cd._ops = ops;
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   ops->addOp();
   //bool done = copyIn( (void *) devAddr, cd, len, &pe );
   bool done = ( myThread->runningOn() == gpu ) ? isMycopyIn2( (void *) devAddr, cd, len, mem, gpu ) : isNotMycopyIn2( (void *) devAddr, cd, len, mem, gpu );
   if ( done ) ops->completeOp();
}

void GPUDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const {
   CopyDescriptor cd( hostAddr ); cd._ops = ops;
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   ops->addOp();
   //bool done = copyOut( cd, (void *) devAddr, len, &pe );
   bool done = ( myThread->runningOn() == gpu ) ? isMycopyOut2( cd, (void *) devAddr, len, mem, gpu ) : isNotMycopyOut2( cd, (void *) devAddr, len, mem, gpu );
   if ( done ) ops->completeOp();
}

bool GPUDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, Functor *f, WD const &wd, void *hostObject, reg_t hostRegionId ) const {
   CopyDescriptor cd( 0xdeaddead ); cd._ops = ops; cd._functor = f;
   ext::GPUMemorySpace *gpuMemDataOrig = ( ext::GPUMemorySpace * ) memOrig.getSpecificData();
   ext::GPUMemorySpace *gpuMemDataDest = ( ext::GPUMemorySpace * ) memDest.getSpecificData();
   ext::GPUProcessor *gpuOrig = gpuMemDataOrig->getGPU();
   ext::GPUProcessor *gpuDest = gpuMemDataDest->getGPU();
   ops->addOp();
   bool done = copyDevToDev( (void *) devDestAddr, cd, (void *) devOrigAddr, len, gpuDest, gpuOrig );
   if ( done ) {
      ops->completeOp(); 
      if ( f ) {
         (*f)(); 
      }
   }
   return true;
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
std::size_t GPUDevice::getMemCapacity( SeparateMemoryAddressSpace &mem ) const {
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   ext::GPUProcessor *gpu = gpuMemData->getGPU();
   return gpu->getMaxMemoryAvailable();
}

void GPUDevice::_getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list ) const {
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   SimpleAllocator *allocator = gpuMemData->getAllocator();
   allocator->getFreeChunksList( list );
}

void GPUDevice::_canAllocate( SeparateMemoryAddressSpace const &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) const {
   ext::GPUMemorySpace *gpuMemData = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   SimpleAllocator *allocator = gpuMemData->getAllocator();
   allocator->canAllocate( sizes, numChunks, remainingSizes );
}
