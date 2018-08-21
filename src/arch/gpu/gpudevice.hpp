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

#ifndef _GPU_DEVICE
#define _GPU_DEVICE

#include "basethread.hpp"
#include "gpudevice_decl.hpp"
#include "gpuprocessor.hpp"
#include "gpumemoryspace_decl.hpp"

namespace nanos {

GPUDevice::GPUDevice ( const char *n ) : Device ( n )
{
   getMemoryLockLimit();
}

GPUDevice::~GPUDevice() {}

//void GPUDevice::free( void *address, ProcessingElement *pe )
//{
//   // Check there are no pending copies to execute before we free the memory (and if there are, execute them)
//   ( ( nanos::ext::GPUProcessor * ) pe )->getOutTransferList()->checkAddressForMemoryTransfer( address );
//   ( ( nanos::ext::GPUProcessor * ) pe )->free( address );
//}


void GPUDevice::isNotMycopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t len, size_t count, size_t ld, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu )
{
   gpu->getInTransferList()->addMemoryTransfer( remoteSrc, localDst, len, count, ld );
}


void GPUDevice::isMycopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t len, size_t count, size_t ld, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu )
{
   // Copy from host memory to device memory
   nanos::ext::GPUThread * thread = ( nanos::ext::GPUThread * ) gpu->getFirstThread();

   GenericEvent * evt = thread->createPreRunEvent( thread->getCurrentWD() );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " copy input: " + toString<uint64_t>( remoteSrc.getTag() ) );
#endif
   evt->setCreated();

   Action * action = new_action( ( ActionMemFunPtr0<DeviceOps>::MemFunPtr0 ) &DeviceOps::completeOp, remoteSrc._ops );
   evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:DeviceOps::completeOp" );
#endif

   ext::GPUMemorySpace *gpuMem = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   // Check for synchronous or asynchronous mode
   if ( gpu->getGPUProcessorInfo()->getInTransferStream() != 0 ) {
      void * pinned = ( sys.getPinnedAllocatorCUDA().isPinned( ( void * ) remoteSrc.getTag(), len * count ) ) ?
            ( void * ) remoteSrc.getTag() :
            gpuMem->allocateInputPinnedMemory( len * count );

      // allocateInputPinnedMemory() can return NULL, so we have to check the pointer to pinned memory
      pinned = pinned ? pinned : ( void * ) remoteSrc.getTag();

      if ( pinned != ( void * ) remoteSrc.getTag() ) {
         copyInAsyncToBuffer( pinned, ( void * ) remoteSrc.getTag(), len, count, ld );
      }

      copyInAsyncToDevice( localDst, pinned, len, count, ld );

   } else {
      copyInSyncToDevice( localDst, ( void * ) remoteSrc.getTag(), len, count, ld );
   }

   evt->setPending();

   thread->addEvent( evt );
}


void GPUDevice::isNotMycopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t len, size_t count, size_t ld, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu )
{
   //ext::GPUMemorySpace *gpuMem = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   //std::cerr << __FUNCTION__ << " from: "<< (void *)remoteDst.getTag() << " to: "<< localSrc <<std::endl;
   gpu->getOutTransferList()->addMemoryTransfer( remoteDst, localSrc, len, count, ld );
   // Mark the copy as requested, because the thread invoking this function needs the data
   syncTransfer( remoteDst.getTag(), mem, gpu );
}

void GPUDevice::isMycopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t len, size_t count, size_t ld, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu )
{
   // Copy from device memory to host memory
   // Check for synchronous or asynchronous mode
   if ( gpu->getGPUProcessorInfo()->getOutTransferStream() != 0 ) {
      gpu->getOutTransferList()->addMemoryTransfer( remoteDst, localSrc, len, count, ld );
   } else {
      nanos::ext::GPUThread * thread = ( nanos::ext::GPUThread * ) gpu->getFirstThread();

      GenericEvent * evt = thread->createPostRunEvent( thread->getCurrentWD() );
#ifdef NANOS_GENERICEVENT_DEBUG
      evt->setDescription( evt->getDescription() + " copy output: " + toString<uint64_t>( remoteDst.getTag() ) );
#endif
      evt->setCreated();

      Action * action = new_action( ( ActionMemFunPtr0<DeviceOps>::MemFunPtr0 ) &DeviceOps::completeOp, remoteDst._ops );
      evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
      evt->setDescription( evt->getDescription() + " action:DeviceOps::completeOp" );
#endif

      copyOutSyncToHost( ( void * ) remoteDst.getTag(), localSrc, len, count, ld );

      evt->setPending();

      thread->addEvent( evt );
   }
}


void GPUDevice::syncTransfer( uint64_t hostAddress, SeparateMemoryAddressSpace &mem, ext::GPUProcessor *gpu ) const
{
   //ext::GPUMemorySpace *gpuMem = ( ext::GPUMemorySpace * ) mem.getSpecificData();
   // syncTransfer() is used to ensure that somebody will update the data related to
   // 'hostAddress' of main memory at some time (when using copy back, this is always ensured)

   // Anyway, we can help the system and tell that somebody is waiting for it
   gpu->getOutTransferList()->requestTransfer( ( void * ) hostAddress );
}

} // namespace nanos

#endif
