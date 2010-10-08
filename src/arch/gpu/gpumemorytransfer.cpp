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


void GPUMemoryTransferOutSyncList::addMemoryTransfer ( void * dest, void * source, size_t size )
{
   GPUDevice::copyOutSyncToHost( dest, source, size );
}

void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( std::vector<GPUMemoryTransfer>::iterator it )
{
   GPUMemoryTransfer copy = *it;

   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER, key, copy._size ) );

   GPUDevice::copyOutAsyncToBuffer( copy._dst, copy._src, copy._size );
   GPUDevice::copyOutAsyncWait();
   GPUDevice::copyOutAsyncToHost( copy._dst, copy._src, copy._size );

   finishMemoryTransfer( it );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
}

void GPUMemoryTransferOutAsyncList::removeMemoryTransfer ( void * dstAddress )
{
   for ( std::vector<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
         it != _pendingTransfersAsync.end();
         it++ )
   {
      if ( it->_dst == dstAddress ) {
         removeMemoryTransfer( it );
      }
   }
}

void GPUMemoryTransferOutAsyncList::executeMemoryTransfers ()
{
   if ( !_pendingTransfersAsync.empty() ) {
      // First copy
      GPUMemoryTransfer & copy1 = _pendingTransfersAsync[0];

      NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER ) );
      NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );

      GPUDevice::copyOutAsyncToBuffer( copy1._dst, copy1._src, copy1._size );

      while ( _pendingTransfersAsync.size() > 1) {
         // First copy
         GPUDevice::copyOutAsyncWait();

         // Second copy
         GPUMemoryTransfer & copy2 = _pendingTransfersAsync[1];
         GPUDevice::copyOutAsyncToBuffer( copy2._dst, copy2._src, copy2._size );

         // First copy
         GPUDevice::copyOutAsyncToHost( copy1._dst, copy1._src, copy1._size );

         // Finish DO of first copy and remove it from the list
         finishMemoryTransfer( _pendingTransfersAsync.begin() );

         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, copy1._size ) );

         // Update second copy to be first copy at next iteration
         copy1 = _pendingTransfersAsync[0];
      }

      GPUDevice::copyOutAsyncWait();
      GPUDevice::copyOutAsyncToHost( copy1._dst, copy1._src, copy1._size );

      // Finish DO of the copy and remove it from the list
      finishMemoryTransfer( _pendingTransfersAsync.begin() );

      NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, copy1._size ) );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );
   }
#if 0
   while ( !_pendingTransfersAsync.empty() ) {
      PendingCopy & copy = *_pendingTransfersAsync.begin();

      NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-copy-out") );
      NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( MEM_TRANSFER, key, copy._size ) );

      GPUDevice::copyOutAsyncToBuffer( copy._dst, copy._src, copy._size );
      GPUDevice::copyOutAsyncWait();
      GPUDevice::copyOutAsyncToHost( copy._dst, copy._src, copy._size );

      // Finish DO of the copy and remove it from the list
      finishPendingCopy( _pendingTransfersAsync.begin() );

      NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
   }
#endif
}

void GPUMemoryTransferOutAsyncList::finishMemoryTransfer( std::vector<GPUMemoryTransfer>::iterator it )
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( ( uint64_t ) it->_dst );

   if ( it->_do != NULL) {
      it->_do->finished();
   }
   it->done();
   _pendingTransfersAsync.erase( it );
}



void GPUMemoryTransferInAsyncList::clearMemoryTransfers()
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( _pendingTransfersAsync );

   _pendingTransfersAsync.clear();
}
