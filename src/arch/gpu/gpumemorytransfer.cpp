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
#include "instrumentationmodule_decl.hpp"


using namespace nanos;
using namespace nanos::ext;


void GPUMemoryTransferOutSyncList::removeMemoryTransfer ( GPUMemoryTransfer &mt )
{
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER_DEVICE_OUT ) );
   NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, mt._size ) );

   GPUDevice::copyOutSyncToHost( ( void * ) mt._hostAddress.getTag(), mt._deviceAddress, mt._size );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );

   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( mt._hostAddress );
}


void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( GPUMemoryTransfer &mt )
{
   nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();
   void * pinned = myPE->allocateOutputPinnedMemory( mt._size );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER_DEVICE_OUT ) );
   NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, mt._size ) );

   // Even there is only one copy, we must do it asynchronously, as we may be doing something else
   GPUDevice::copyOutAsyncToBuffer( pinned, mt._deviceAddress, mt._size );
   GPUDevice::copyOutAsyncWait();
   GPUDevice::copyOutAsyncToHost( ( void * ) mt._hostAddress.getTag(), pinned, mt._size );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );

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
         _pendingTransfersAsync.erase( it );
         _lock.release();
         removeMemoryTransfer( mt );
      }
   }
}

void GPUMemoryTransferOutAsyncList::executeMemoryTransfers ()
{
   if ( !_pendingTransfersAsync.empty() ) {

      nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();

      // First copy
      std::list<GPUMemoryTransfer>::iterator it1 = _pendingTransfersAsync.begin();

      _lock.acquire();
      while( it1 != _pendingTransfersAsync.end() && !it1->_requested ) {
         it1++;
      }
      if ( it1 == _pendingTransfersAsync.end() ) it1 = _pendingTransfersAsync.begin();

      GPUMemoryTransfer mt1 ( *it1 );
      _pendingTransfersAsync.erase( it1 );
      _lock.release();

      void * pinned1 = myPE->allocateOutputPinnedMemory( mt1._size );

      NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER_DEVICE_OUT ) );
      NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, mt1._size ) );

      GPUDevice::copyOutAsyncToBuffer( pinned1, mt1._deviceAddress, mt1._size );

      while ( _pendingTransfersAsync.size() > 1) {
         // First copy
         GPUDevice::copyOutAsyncWait();

         // Second copy
         // Check if there is another GPUMemoryTransfer requested
         _lock.acquire();
         std::list<GPUMemoryTransfer>::iterator it2 = _pendingTransfersAsync.begin();
         while( !it2->_requested && it2 != _pendingTransfersAsync.end() ) {
            it2++;
         }
         // If no requested transfer is found, take the first transfer
         if ( it2 == _pendingTransfersAsync.end() ) {
            it2 = _pendingTransfersAsync.begin();
         }

         GPUMemoryTransfer mt2 ( *it2 );
         _pendingTransfersAsync.erase( it2 );
         _lock.release();

         void * pinned2 = myPE->allocateOutputPinnedMemory( mt2._size );

         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, mt2._size ) );
         GPUDevice::copyOutAsyncToBuffer( pinned2, mt2._deviceAddress, mt2._size );

         // First copy
         GPUDevice::copyOutAsyncToHost( ( void * ) mt1._hostAddress.getTag(), pinned1, mt1._size );

         // Synchronize first copy
         myPE->synchronize( mt1._hostAddress );

         // Update second copy to be first copy at next iteration
         mt1 = mt2;
         pinned1 = pinned2;
      }

      GPUDevice::copyOutAsyncWait();
      GPUDevice::copyOutAsyncToHost( ( void * ) mt1._hostAddress.getTag(), pinned1, mt1._size );

      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );

      // Synchronize copy
      myPE->synchronize( mt1._hostAddress );

      myPE->freeOutputPinnedMemory();
   }
}


void GPUMemoryTransferInAsyncList::clearMemoryTransfers()
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( _pendingTransfersAsync );

   _pendingTransfersAsync.clear();
}

void GPUMemoryTransferInAsyncList::removeMemoryTransfer ( GPUMemoryTransfer &mt )
{
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER_DEVICE_IN ) );
   NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-in") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, mt._size ) );

   void *pinned = ( nanos::ext::GPUProcessor * ) myThread->runningOn()->allocateInputPinnedMemory( mt._size );

   GPUDevice::copyInAsyncToBuffer( pinned, ( void * ) mt._hostAddress.getTag(), mt._size );
   GPUDevice::copyInAsyncToDevice( mt._deviceAddress, pinned, mt._size );
   //GPUDevice::copyInSyncToDevice( mt._deviceAddress, ( void * ) mt._hostAddress.getTag(), mt._size );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );

#if 0
   GPUProcessor * myPE = ( GPUProcessor * ) myThread->runningOn();
   void * pinned = ( void * ) myPE->getPinnedAddress( it->_deviceAddress );
   myPE->removePinnedAddress( it->_deviceAddress );
   myPE->freePinnedMemory( pinned );
#endif
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
