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
#include "gpudevice.hpp"
#include "gpuprocessor.hpp"
#include "instrumentationmodule_decl.hpp"


using namespace nanos;
using namespace nanos::ext;


GPUMemoryTransfer::~GPUMemoryTransfer()
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( _hostAddress );
}


void GPUMemoryTransferOutSyncList::addMemoryTransfer ( CopyDescriptor &hostAddress, void * deviceAddress, size_t size )
{
   GPUDevice::copyOutSyncToHost( (void *)hostAddress.getTag(), deviceAddress, size );
}


void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( std::list<memTxPtr>::iterator it )
{
   GPUMemoryTransfer * copy = it->get();

   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER, key, copy->_size ) );

   // Even there is only one copy, we must do it asynchronously, as we may be doing something else
   GPUDevice::copyOutAsyncToBuffer( ( void * ) copy->_hostAddress.getTag(), copy->_deviceAddress, copy->_size );
   GPUDevice::copyOutAsyncWait();
   GPUDevice::copyOutAsyncToHost( ( void * ) copy->_hostAddress.getTag(), copy->_deviceAddress, copy->_size );

   _lock.acquire();
   _pendingTransfersAsync.erase( it );
   _lock.release();

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
}

void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( CopyDescriptor &hostAddress )
{
   for ( std::list<memTxPtr>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ )
   {
      if ( it->get()->_hostAddress.getTag() == hostAddress.getTag() ) {
         removeMemoryTransfer( it );
      }
   }
}

void GPUMemoryTransferOutAsyncList::executeMemoryTransfers ()
{
   if ( !_pendingTransfersAsync.empty() ) {
      // First copy
      std::list<memTxPtr>::iterator it1 = _pendingTransfersAsync.begin();

      while( it1 != _pendingTransfersAsync.end() && !( ( GPUMemoryTransfer * ) it1->get() )->_requested ) {
         it1++;
      }
      if ( it1 == _pendingTransfersAsync.end() ) it1 = _pendingTransfersAsync.begin();

      GPUMemoryTransfer * copy1 = it1->get();

      NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER ) );
      NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );

      GPUDevice::copyOutAsyncToBuffer( ( void * ) copy1->_hostAddress.getTag(), copy1->_deviceAddress, copy1->_size );

      while ( _pendingTransfersAsync.size() > 1) {
         // First copy
         GPUDevice::copyOutAsyncWait();

         // Second copy
         // Check if there is another GPUMemoryTransfer requested
         std::list<memTxPtr>::iterator it2 = _pendingTransfersAsync.begin();
         while( !( ( GPUMemoryTransfer ) *it2->get() )._requested && it2 != _pendingTransfersAsync.end() ) {
            it2++;
            if ( it1 == it2 && it2 != _pendingTransfersAsync.end() ) {
               it2++;
            }
         }
         // If no requested transfer is found, take the first transfer that
         // has not been taken by copy1
         if ( it2 == _pendingTransfersAsync.end() ) {
            it2 = _pendingTransfersAsync.begin();
         }


         GPUMemoryTransfer * copy2 = it2->get();
         GPUDevice::copyOutAsyncToBuffer( ( void * ) copy2->_hostAddress.getTag(), copy2->_deviceAddress, copy2->_size );

         // First copy
         GPUDevice::copyOutAsyncToHost( ( void * ) copy1->_hostAddress.getTag(), copy1->_deviceAddress, copy1->_size );

         // Remove first copy from the list
         _lock.acquire();
         _pendingTransfersAsync.pop_front();
         _lock.release();

         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, copy1->_size ) );

         // Update second copy to be first copy at next iteration
         it1 = it2;
         copy1 = it1->get();
      }

      GPUDevice::copyOutAsyncWait();
      GPUDevice::copyOutAsyncToHost( ( void * ) copy1->_hostAddress.getTag(), copy1->_deviceAddress, copy1->_size );

      // Remove copy from the list
      _lock.acquire();
      _pendingTransfersAsync.pop_front();
      _lock.release();

      NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, copy1->_size ) );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );
   }
#if 0
   while ( !_pendingTransfersAsync.empty() ) {
      GPUMemoryTransfer & copy = *_pendingTransfersAsync.begin();

      NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-copy-out") );
      NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( MEM_TRANSFER, key, copy._size ) );

      GPUDevice::copyOutAsyncToBuffer( copy._dst, copy._src, copy._size );
      GPUDevice::copyOutAsyncWait();
      GPUDevice::copyOutAsyncToHost( copy._dst, copy._src, copy._size );

      // Finish DO of the copy and remove it from the list
      finishMemoryTransfer( _pendingTransfersAsync.begin() );

      NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
   }
#endif
}


void GPUMemoryTransferInAsyncList::clearMemoryTransfers()
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( _pendingTransfersAsync );
   _pendingTransfersAsync.clear();
}
