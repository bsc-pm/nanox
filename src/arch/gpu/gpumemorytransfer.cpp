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


GPUMemoryTransfer::~GPUMemoryTransfer()
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( _hostAddress );
}


void GPUMemoryTransferOutSyncList::removeMemoryTransfer ( std::list<GPUMemoryTransfer>::iterator it )
{
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER_DEVICE_OUT ) );
   NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, it->_size ) );

   GPUDevice::copyOutSyncToHost( ( void * ) it->_hostAddress.getTag(), it->_deviceAddress, it->_size );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );

   _lock.acquire();
   _pendingTransfersAsync.erase( it );
   _lock.release();
}


void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( std::list<GPUMemoryTransfer>::iterator it )
{
   nanos::ext::GPUProcessor * myPE = ( nanos::ext::GPUProcessor * ) myThread->runningOn();
   void * pinned = myPE->allocateOutputPinnedMemory( it->_size );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER_DEVICE_OUT ) );
   NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, it->_size ) );

   // Even there is only one copy, we must do it asynchronously, as we may be doing something else
   GPUDevice::copyOutAsyncToBuffer( pinned, it->_deviceAddress, it->_size );
   GPUDevice::copyOutAsyncWait();
   GPUDevice::copyOutAsyncToHost( ( void * ) it->_hostAddress.getTag(), pinned, it->_size );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );

   _lock.acquire();
   _pendingTransfersAsync.erase( it );
   _lock.release();

}

void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( CopyDescriptor &hostAddress )
{
   for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ )
   {
      if ( it->_hostAddress.getTag() == hostAddress.getTag() ) {
         removeMemoryTransfer( it );
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
      _lock.release();

      void * pinned1 = myPE->allocateOutputPinnedMemory( it1->_size );

      NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER_DEVICE_OUT ) );
      NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, it1->_size ) );

      //GPUDevice::copyOutAsyncToBuffer( ( void * ) it1->_hostAddress.getTag(), it1->_deviceAddress, it1->_size );
      GPUDevice::copyOutAsyncToBuffer( pinned1, it1->_deviceAddress, it1->_size );

      while ( _pendingTransfersAsync.size() > 1) {
         // First copy
         GPUDevice::copyOutAsyncWait();

         // Second copy
         // Check if there is another GPUMemoryTransfer requested
         _lock.acquire();
         std::list<GPUMemoryTransfer>::iterator it2 = _pendingTransfersAsync.begin();
         while( !it2->_requested && it2 != _pendingTransfersAsync.end() ) {
            it2++;
            if ( it1 == it2 && it2 != _pendingTransfersAsync.end() ) {
               it2++;
            }
         }
         // If no requested transfer is found, take the first transfer that
         // has not been taken by it1
         if ( it2 == _pendingTransfersAsync.end() ) {
            it2 = _pendingTransfersAsync.begin();
         }

         // Check it2 != it1; as we are at the beginning of the list, we can safely do it2++
         // because at least, there are two elements in the list
         if ( it2 == it1 ) it2++;
         _lock.release();

         void * pinned2 = myPE->allocateOutputPinnedMemory( it2->_size );

         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, it2->_size ) );
         GPUDevice::copyOutAsyncToBuffer( pinned2, it2->_deviceAddress, it2->_size );

         // First copy
         GPUDevice::copyOutAsyncToHost( ( void * ) it1->_hostAddress.getTag(), pinned1, it1->_size );

         // Remove first copy from the list
         _lock.acquire();
         _pendingTransfersAsync.erase( it1 );

         // Update second copy to be first copy at next iteration
         it1 = it2;
         _lock.release();
         pinned1 = pinned2;
      }

      GPUDevice::copyOutAsyncWait();
      GPUDevice::copyOutAsyncToHost( ( void * ) it1->_hostAddress.getTag(), pinned1, it1->_size );

      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );

      // Remove copy from the list
      _lock.acquire();
      _pendingTransfersAsync.erase( it1 );
      _lock.release();

      myPE->freeOutputPinnedMemory();
   }
}


void GPUMemoryTransferInAsyncList::clearMemoryTransfers()
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( _pendingTransfersAsync );

   _pendingTransfersAsync.clear();
}

void GPUMemoryTransferInAsyncList::removeMemoryTransfer ( std::list<GPUMemoryTransfer>::iterator it )
{
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER_DEVICE_IN ) );
   NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-in") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, it->_size ) );

   GPUDevice::copyInSyncToDevice( it->_deviceAddress, ( void * ) it->_hostAddress.getTag(), it->_size );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );

#if 0
   GPUProcessor * myPE = ( GPUProcessor * ) myThread->runningOn();
   void * pinned = ( void * ) myPE->getPinnedAddress( it->_deviceAddress );
   myPE->removePinnedAddress( it->_deviceAddress );
   myPE->freePinnedMemory( pinned );
#endif

   _lock.acquire();
   _requestedTransfers.erase( it );
   _lock.release();
}

void GPUMemoryTransferInAsyncList::executeMemoryTransfers ()
{
   while ( !_requestedTransfers.empty() ) {
      removeMemoryTransfer( _requestedTransfers.begin() );
   }
}
