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


#include "gpumemorytransfer.hpp"
#include "gpuevent.hpp"
#include "gpuprocessor.hpp"
#include "gpumemoryspace_decl.hpp"
#include "deviceops.hpp"
#include "instrumentationmodule_decl.hpp"


using namespace nanos;
using namespace nanos::ext;


void GPUMemoryTransferOutList::removeMemoryTransfer ()
{
   if ( !_pendingTransfersAsync.empty() ) {
      bool found = false;
      for ( std::list<GPUMemoryTransfer *>::iterator it = _pendingTransfersAsync.begin();
            it != _pendingTransfersAsync.end(); it++ ) {
         _lock.acquire();
         if ( ( *it )->_requested ) {
            found = true;
            GPUMemoryTransfer * mt ( *it );
            it = _pendingTransfersAsync.erase( it );
            _lock.release();
            removeMemoryTransfer( mt );
            break;
         }
         _lock.release();
     }

     //NOTE: Following code is suspicious and probably can be removed
      if ( !found ) {
         _lock.acquire();
         ensure( _pendingTransfersAsync.begin() != _pendingTransfersAsync.end(),
            "Bad assumption in GPUMemoryTransferOutList::removeMemoryTransfer" );
         GPUMemoryTransfer * mt ( *_pendingTransfersAsync.begin() );
         _pendingTransfersAsync.erase( _pendingTransfersAsync.begin() );
         _lock.release();
         removeMemoryTransfer( mt );
      }
   }
}

void GPUMemoryTransferOutList::checkAddressForMemoryTransfer ( void * address )
{
   for ( std::list<GPUMemoryTransfer *>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ ) {
      _lock.acquire();
      if ( ( *it )->_hostAddress.getTag() == ( uint64_t ) address ) {
         GPUMemoryTransfer * mt ( *it );
         it = _pendingTransfersAsync.erase( it );
         _lock.release();
         removeMemoryTransfer( mt );
         _lock.acquire();
      }
      _lock.release();
   }
}

void GPUMemoryTransferOutList::requestTransfer( void * address )
{
   _lock.acquire();
   for ( std::list<GPUMemoryTransfer *>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end(); it++ ) {
      if ( ( *it )->_hostAddress.getTag() == ( uint64_t ) address ) {
         ( *it )->_requested = true;
      }
   }
   _lock.release();
}


void GPUMemoryTransferOutSyncList::removeMemoryTransfer ( GPUMemoryTransfer * mt )
{
   GPUDevice::copyOutSyncToHost( ( void * ) mt->_hostAddress.getTag(), mt->_deviceAddress, mt->_len, mt->_count, mt->_ld );
   //( ( GPUProcessor * ) myThread->runningOn() )->synchronize( mt._hostAddress );
   mt->completeTransfer();
}

void GPUMemoryTransferOutSyncList::clearRequestedMemoryTransfers ()
{
   _lock.acquire();
   for ( std::list<GPUMemoryTransfer *>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ )
   {
      if ( ( *it )->_requested ) {
         GPUMemoryTransfer * mt ( *it );
         it = _pendingTransfersAsync.erase( it );
         removeMemoryTransfer( mt );
      }
   }
   _lock.release();
}

void GPUMemoryTransferOutSyncList::executeMemoryTransfers ()
{
   while ( !_pendingTransfersAsync.empty() ) {
      _lock.acquire();
      GPUMemoryTransfer * mt ( *_pendingTransfersAsync.begin() );
      _pendingTransfersAsync.erase( _pendingTransfersAsync.begin() );
      _lock.release();

      removeMemoryTransfer( mt );
   }
}


void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( GPUMemoryTransfer * mt )
{
   GPUThread * thread = ( GPUThread * ) myThread;

   // Even there is only one copy, we must do it asynchronously, as we may be doing something else
   // No need to copy data to the intermediate pinned buffer if it's already pinned
   void * pinned = ( sys.getPinnedAllocatorCUDA().isPinned( ( void * ) mt->_hostAddress.getTag(), mt->_len * mt->_count ) ) ?
         ( void * ) mt->_hostAddress.getTag() :
         ( ( nanos::ext::GPUProcessor * ) thread->runningOn() )->getGPUMemory().allocateOutputPinnedMemory( mt->_len * mt->_count );

   // allocateOutputPinnedMemory() can return NULL, so we have to check the pointer to pinned memory
   pinned = pinned ? pinned : ( void * ) mt->_hostAddress.getTag();

   GenericEvent * evt = thread->createPostRunEvent( thread->getCurrentWD() );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " remMemTxAsync DtH"
         + toString<uint64_t>( ( uint64_t ) mt->_deviceAddress )
         + "->" + toString<uint64_t>( mt->_hostAddress.getTag() )
         + toString<size_t>( mt->_ld * mt->_count ) );
#endif

   GPUDevice::copyOutAsyncToBuffer( pinned, mt->_deviceAddress, mt->_len, mt->_count, mt->_ld );

   evt->setPending();

   if ( pinned != ( void * ) mt->_hostAddress.getTag() ) {
      Action * action = new_action( ( ActionFunPtr5<void *, void *, size_t, size_t, size_t>::FunPtr5 ) &GPUDevice::copyOutAsyncToHost,
            ( void * ) mt->_hostAddress.getTag(), pinned, mt->_len, mt->_count, mt->_ld );
      evt->addNextAction( action );
   }

   Action * action = new_action( ( ActionPtrMemFunPtr0<GPUMemoryTransfer>::PtrMemFunPtr0 ) &GPUMemoryTransfer::completeTransfer, mt );
   evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:GPUMemTx::completeTx()" );
#endif

   thread->addEvent( evt );
}

void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( CopyDescriptor &hostAddress )
{
   for ( std::list<GPUMemoryTransfer *>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ )
   {
      if ( ( *it )->_hostAddress.getTag() == hostAddress.getTag() ) {
         _lock.acquire();
         GPUMemoryTransfer * mt ( *it );
         it = _pendingTransfersAsync.erase( it );
         _lock.release();
         removeMemoryTransfer( mt );
      }
   }
}

void GPUMemoryTransferOutAsyncList::executeRequestedMemoryTransfers ()
{
   std::list<GPUMemoryTransfer *> itemsToRemove;
   _lock.acquire();
   for ( std::list<GPUMemoryTransfer *>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ )
   {
      if ( ( *it )->_requested ) {
         itemsToRemove.push_back(*it);
         it = _pendingTransfersAsync.erase( it );
      }
   }
   _lock.release();
   executeMemoryTransfers( itemsToRemove );
}

void GPUMemoryTransferOutAsyncList::executeMemoryTransfers ( std::list<GPUMemoryTransfer *> &pendingTransfersAsync )
{
   while ( !_pendingTransfersAsync.empty() ) {
      _lock.acquire();
      GPUMemoryTransfer * mt ( *_pendingTransfersAsync.begin() );
      _pendingTransfersAsync.erase( _pendingTransfersAsync.begin() );
      _lock.release();

      removeMemoryTransfer( mt );
   }
}

void GPUMemoryTransferInAsyncList::removeMemoryTransfer ( GPUMemoryTransfer * mt )
{
   GPUThread * thread = ( GPUThread * ) myThread;

   // No need to copy data to the intermediate pinned buffer if it's already pinned
   void * pinned = ( sys.getPinnedAllocatorCUDA().isPinned( ( void * ) mt->_hostAddress.getTag(), mt->_len * mt->_count ) ) ?
         ( void * ) mt->_hostAddress.getTag() :
         ( ( nanos::ext::GPUProcessor * ) thread->runningOn() )->getGPUMemory().allocateInputPinnedMemory( mt->_len * mt->_count );

   if ( pinned != ( void * ) mt->_hostAddress.getTag() ) {
      GPUDevice::copyInAsyncToBuffer( pinned, ( void * ) mt->_hostAddress.getTag(), mt->_len, mt->_count, mt->_ld );
   }

   GenericEvent * evt = thread->createPreRunEvent( thread->getCurrentWD() );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " remMemTxAsync HtD"
         + toString<uint64_t>( mt->_hostAddress.getTag() ) + "(" + toString<uint64_t>( ( uint64_t ) pinned ) + ")"
         + "->" + toString<uint64_t>( ( uint64_t ) mt->_deviceAddress )
         + toString<size_t>( mt->_len * mt->_count ) );
#endif

   GPUDevice::copyInAsyncToDevice( mt->_deviceAddress, pinned, mt->_len, mt->_count, mt->_ld );

   evt->setPending();

   Action * action = new_action( ( ActionPtrMemFunPtr0<GPUMemoryTransfer>::PtrMemFunPtr0 ) &GPUMemoryTransfer::completeTransfer, mt );
   evt->addNextAction( action );
#ifdef NANOS_GENERICEVENT_DEBUG
   evt->setDescription( evt->getDescription() + " action:GPUMemTx::completeTx()" );
#endif

   thread->addEvent( evt );
}

void GPUMemoryTransferInAsyncList::executeMemoryTransfers ()
{
   while ( !_requestedTransfers.empty() ) {
      _lock.acquire();
      GPUMemoryTransfer * mt ( *_requestedTransfers.begin() );
      _requestedTransfers.erase( _requestedTransfers.begin() );
      _lock.release();

      removeMemoryTransfer( mt );
   }
}
