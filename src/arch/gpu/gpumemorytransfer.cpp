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


#include "gpumemorytransfer.hpp"
#include "gpuprocessor.hpp"


using namespace nanos;
using namespace nanos::ext;


nanos::ext::GPUMemoryTransferOutList::~GPUMemoryTransferOutList()
{
   ensure( _pendingTransfersAsync.empty(),
         "Attempting to delete the output pending transfers list with already "
         + toString<size_t>( _pendingTransfersAsync.size() ) + " pending transfers to perform" );
}

void GPUMemoryTransferOutList::removeMemoryTransfer ()
{
   if ( !_pendingTransfersAsync.empty() ) {
      bool found = false;
      for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
            it != _pendingTransfersAsync.end(); it++ ) {
         _lock.acquire();
         if ( it->_requested ) {
            found = true;
            GPUMemoryTransfer mt ( *it );
            it = _pendingTransfersAsync.erase( it );
            _lock.release();
            removeMemoryTransfer( mt );
            break;
         }
         _lock.release();
     }

      if ( !found ) {
         _lock.acquire();
         GPUMemoryTransfer mt ( *_pendingTransfersAsync.begin() );
         _pendingTransfersAsync.erase( _pendingTransfersAsync.begin() );
         _lock.release();
         removeMemoryTransfer( mt );
      }
   }
}

void GPUMemoryTransferOutList::checkAddressForMemoryTransfer ( void * address )
{
   for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ ) {
      _lock.acquire();
      if ( it->_hostAddress.getTag() == ( uint64_t ) address ) {
         GPUMemoryTransfer mt ( *it );
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
   for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end(); it++ ) {
      if ( it->_hostAddress.getTag() == ( uint64_t ) address ) {
         it->_requested = true;
      }
   }
   _lock.release();
}


void GPUMemoryTransferOutSyncList::removeMemoryTransfer ( GPUMemoryTransfer &mt )
{
   GPUDevice::copyOutSyncToHost( ( void * ) mt._hostAddress.getTag(), mt._deviceAddress, mt._size );
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( mt._hostAddress );
}

void GPUMemoryTransferOutSyncList::clearRequestedMemoryTransfers ()
{
   _lock.acquire();
   for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ )
   {
      if ( it->_requested ) {
         GPUMemoryTransfer mt ( *it );
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
      GPUMemoryTransfer mt ( *_pendingTransfersAsync.begin() );
      _pendingTransfersAsync.erase( _pendingTransfersAsync.begin() );
      _lock.release();

      removeMemoryTransfer( mt );
   }
}


void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( GPUMemoryTransfer &mt )
{

   // Even there is only one copy, we must do it asynchronously, as we may be doing something else
   // No need to copy data to the intermediate pinned buffer if it's already pinned
   void * pinned = ( sys.getPinnedAllocatorCUDA().isPinned( ( void * ) mt._hostAddress.getTag(), mt._size ) ) ?
         ( void * ) mt._hostAddress.getTag() :
         ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->allocateOutputPinnedMemory( mt._size );

   // allocateOutputPinnedMemory() can return NULL, so we have to check the pointer to pinned memory
   pinned = pinned ? pinned : ( void * ) mt._hostAddress.getTag();

   GPUDevice::copyOutAsyncToBuffer( pinned, mt._deviceAddress, mt._size );
   GPUDevice::copyOutAsyncWait();

   if ( pinned != ( void * ) mt._hostAddress.getTag() ) {
      GPUDevice::copyOutAsyncToHost( ( void * ) mt._hostAddress.getTag(), pinned, mt._size );
   }

   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( mt._hostAddress );
}

void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( CopyDescriptor &hostAddress )
{
   for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ )
   {
      if ( it->_hostAddress.getTag() == hostAddress.getTag() ) {
         _lock.acquire();
         GPUMemoryTransfer mt ( *it );
         it = _pendingTransfersAsync.erase( it );
         _lock.release();
         removeMemoryTransfer( mt );
      }
   }
}

void GPUMemoryTransferOutAsyncList::executeRequestedMemoryTransfers ()
{
   std::list<GPUMemoryTransfer> itemsToRemove;
   _lock.acquire();
   for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ )
   {
      if ( it->_requested ) {
         itemsToRemove.push_back(*it);
         it = _pendingTransfersAsync.erase( it );
      }
   }
   _lock.release();
   executeMemoryTransfers( itemsToRemove );
}

void GPUMemoryTransferOutAsyncList::executeMemoryTransfers ( std::list<GPUMemoryTransfer> &pendingTransfersAsync )
{
   if ( !pendingTransfersAsync.empty() ) {

      nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();

      // First copy
      std::list<GPUMemoryTransfer>::iterator it1 = pendingTransfersAsync.begin();

      _lock.acquire();
      while( it1 != pendingTransfersAsync.end() && !it1->_requested ) {
         it1++;
      }
      if ( it1 == pendingTransfersAsync.end() ) it1 = pendingTransfersAsync.begin();

      GPUMemoryTransfer mt1 ( *it1 );
      it1 = pendingTransfersAsync.erase( it1 );
      _lock.release();

      bool isPinned1 = sys.getPinnedAllocatorCUDA().isPinned( ( void * ) mt1._hostAddress.getTag(), mt1._size );

      void * pinned1 = ( isPinned1 ) ?
            ( void * ) mt1._hostAddress.getTag() :
            myPE->allocateOutputPinnedMemory( mt1._size );

      GPUDevice::copyOutAsyncToBuffer( pinned1, mt1._deviceAddress, mt1._size );

      while ( pendingTransfersAsync.size() > 1) {
         // First copy
         GPUDevice::copyOutAsyncWait();

         // Second copy
         // Check if there is another GPUMemoryTransfer requested
         _lock.acquire();
         std::list<GPUMemoryTransfer>::iterator it2 = pendingTransfersAsync.begin();
         while( !it2->_requested && it2 != pendingTransfersAsync.end() ) {
            it2++;
         }
         // If no requested transfer is found, take the first transfer
         if ( it2 == pendingTransfersAsync.end() ) {
            it2 = pendingTransfersAsync.begin();
         }

         GPUMemoryTransfer mt2 ( *it2 );
         it2 = pendingTransfersAsync.erase( it2 );
         _lock.release();

         bool isPinned2 = sys.getPinnedAllocatorCUDA().isPinned( ( void * ) mt2._hostAddress.getTag(), mt2._size );

         void * pinned2 = ( isPinned2 ) ?
               ( void * ) mt2._hostAddress.getTag() :
               myPE->allocateOutputPinnedMemory( mt2._size );

         GPUDevice::copyOutAsyncToBuffer( pinned2, mt2._deviceAddress, mt2._size );

         // First copy: if user memory isn't pinned, copy data to the original address
         if ( !isPinned1 ) {
            GPUDevice::copyOutAsyncToHost( ( void * ) mt1._hostAddress.getTag(), pinned1, mt1._size );
         }

         // Synchronize first copy
         myPE->synchronize( mt1._hostAddress );

         // Update second copy to be first copy at next iteration
         mt1 = mt2;
         isPinned1 = isPinned2;
         pinned1 = pinned2;
      }

      GPUDevice::copyOutAsyncWait();

      // If user memory isn't pinned, copy data to the original address
      if ( !isPinned1 ) {
         GPUDevice::copyOutAsyncToHost( ( void * ) mt1._hostAddress.getTag(), pinned1, mt1._size );
      }

      // Synchronize copy
      myPE->synchronize( mt1._hostAddress );

      myPE->freeOutputPinnedMemory();
   }
}


nanos::ext::GPUMemoryTransferInAsyncList::~GPUMemoryTransferInAsyncList()
{
   ensure( _pendingTransfersAsync.empty(),
         "Attempting to delete the input pending transfers list with already "
         + toString<size_t>( _pendingTransfersAsync.size() ) + " pending transfers to perform" );
}

void GPUMemoryTransferInAsyncList::clearMemoryTransfers()
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( _pendingTransfersAsync );

   _pendingTransfersAsync.clear();
}

void GPUMemoryTransferInAsyncList::removeMemoryTransfer ( GPUMemoryTransfer &mt )
{
   // No need to copy data to the intermediate pinned buffer if it's already pinned
   void * pinned = ( sys.getPinnedAllocatorCUDA().isPinned( ( void * ) mt._hostAddress.getTag(), mt._size ) ) ?
         ( void * ) mt._hostAddress.getTag() :
         ( ( nanos::ext::GPUProcessor * ) myThread->runningOn() )->allocateInputPinnedMemory( mt._size );

   if ( pinned != ( void * ) mt._hostAddress.getTag() ) {
      GPUDevice::copyInAsyncToBuffer( pinned, ( void * ) mt._hostAddress.getTag(), mt._size );
   }

   GPUDevice::copyInAsyncToDevice( mt._deviceAddress, pinned, mt._size );
}

void GPUMemoryTransferInAsyncList::executeMemoryTransfers ()
{
   while ( !_requestedTransfers.empty() ) {
      _lock.acquire();
      GPUMemoryTransfer mt ( *_requestedTransfers.begin() );
      _requestedTransfers.erase( _requestedTransfers.begin() );
      _lock.release();

      removeMemoryTransfer( mt );
   }
}
