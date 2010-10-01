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

#include "gputhread.hpp"
#include "gpuprocessor.hpp"
#include "instrumentationmodule_decl.hpp"
#include "schedule.hpp"
#include "system.hpp"

#include <cuda_runtime.h>

using namespace nanos;
using namespace nanos::ext;


void GPUThread::runDependent ()
{
   WD &work = getThreadWD();
   setCurrentWD( work );
   setNextWD( (WD *) 0 );

   cudaError_t err = cudaSetDevice( _gpuDevice );
   if ( err != cudaSuccess )
      warning( "Couldn't set the GPU device for the thread: " << cudaGetErrorString( err ) );

   if ( GPUDevice::getTransferMode() == nanos::PINNED_CUDA || GPUDevice::getTransferMode() == nanos::WC ) {
      err = cudaSetDeviceFlags( cudaDeviceMapHost | cudaDeviceBlockingSync );
      if ( err != cudaSuccess )
         warning( "Couldn't set the GPU device flags: " << cudaGetErrorString( err ) );
   }
   else {
      err = cudaSetDeviceFlags( cudaDeviceBlockingSync );
      if ( err != cudaSuccess )
         warning( "Couldn't set the GPU device flags:" << cudaGetErrorString( err ) );
   }

   if ( GPUDevice::getTransferMode() != nanos::NORMAL ) {
      ((GPUProcessor *) myThread->runningOn())->getGPUProcessorInfo()->init();
   }

   if ( ( (GPUProcessor *) myThread->runningOn())->getGPUProcessorInfo()->getOutTransferStream() ) {
      // If overlapping outputs is defined, create the list
      _pendingCopiesOut = new PendingCopiesOutAsyncList();
   }
   else {
      // Else, create a 'fake list' which copies outputs synchronously
      _pendingCopiesOut = new PendingCopiesOutSyncList();
   }

   // Create a list of inputs that have been ordered to copy but the operation is still not completed
   _pendingCopiesIn = new PendingCopiesInAsyncList();

   // Avoid the so slow first data allocation and transfer to device
   //bool b = true;
   //bool * b_d = ( bool * ) GPUDevice::allocate( sizeof( b ) );
   //GPUDevice::copyIn( ( void * ) b_d, ( uint64_t ) &b, sizeof( b ) );
   //GPUDevice::free( b_d );

   // Clear copy-in list, just in case last operations filled it
   _pendingCopiesIn->reset();


   SMPDD &dd = ( SMPDD & ) work.activateDevice( SMP );

   dd.getWorkFct()( work.getData() );

   ( ( GPUProcessor * ) myThread->runningOn() )->freeWholeMemory();

}

void GPUThread::inlineWorkDependent ( WD &wd )
{
   GPUDD &dd = ( GPUDD & )wd.getActiveDevice();

   if ( GPUDevice::getTransferMode() != nanos::NORMAL ) {
      // Wait for the input transfer stream to finish
      cudaStreamSynchronize( ( (GPUProcessor *) myThread->runningOn() )->getGPUProcessorInfo()->getInTransferStream() );
      // Erase the wait input list and synchronize it with cache
      _pendingCopiesIn->clearPendingCopies();
   }

   // We should ask the cache to wait for wd inputs: waitInputs(), but as we have
   // just waited for them, we skip this step
#if 0
   CopyData *copies = wd.getCopies();
   for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
      CopyData & cd = copies[i];
      uint64_t tag = (uint64_t) cd.isPrivate() ? ((uint64_t) wd.getData() + (unsigned long)cd.getAddress()) : cd.getAddress();
      if ( cd.isInput() ) {
         ( (GPUProcessor *) myThread->runningOn() )->waitInput( tag );
      }
   }
#endif

   NANOS_INSTRUMENT ( InstrumentStateAndBurst inst1( "user-code", wd.getId(), NANOS_RUNNING ) );
   ( dd.getWorkFct() )( wd.getData() );

   if ( GPUDevice::getTransferMode() != nanos::NORMAL ) {
      NANOS_INSTRUMENT ( InstrumentSubState inst2( NANOS_RUNTIME ) );
      // Get next task in order to prefetch data to device memory
      WD *next = Scheduler::prefetch( ( nanos::BaseThread * ) this, wd );

      setNextWD( next );
      if ( next != 0 ) {
         next->init(false);
      }

      // Copy out results from tasks executed previously
      _pendingCopiesOut->executePendingCopies();
   }

   // Wait for the GPU kernel to finish
   cudaThreadSynchronize();
}

void GPUThread::yield()
{
   executePendingCopies();
}



void GPUThread::PendingCopiesOutAsyncList::removePendingCopy ( std::vector<PendingCopy>::iterator it )
{
   PendingCopy copy = *it;

   NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER, key, copy._size ) );

   GPUDevice::copyOutAsyncToBuffer( copy._dst, copy._src, copy._size );
   GPUDevice::copyOutAsyncWait();
   GPUDevice::copyOutAsyncToHost( copy._dst, copy._src, copy._size );

   finishPendingCopy( it );

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
}

void GPUThread::PendingCopiesOutAsyncList::executePendingCopies ()
{
   if ( !_pendingCopiesAsync.empty() ) {
      // First copy
      PendingCopy & copy1 = _pendingCopiesAsync[0];

      NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent( NANOS_MEM_TRANSFER ) );
      NANOS_INSTRUMENT( nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-out") );

      GPUDevice::copyOutAsyncToBuffer( copy1._dst, copy1._src, copy1._size );

      while ( _pendingCopiesAsync.size() > 1) {
         // First copy
         GPUDevice::copyOutAsyncWait();

         // Second copy
         PendingCopy & copy2 = _pendingCopiesAsync[1];
         GPUDevice::copyOutAsyncToBuffer( copy2._dst, copy2._src, copy2._size );

         // First copy
         GPUDevice::copyOutAsyncToHost( copy1._dst, copy1._src, copy1._size );

         // Finish DO of first copy and remove it from the list
         finishPendingCopy( _pendingCopiesAsync.begin() );

         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, copy1._size ) );

         // Update second copy to be first copy at next iteration
         copy1 = _pendingCopiesAsync[0];
      }

      GPUDevice::copyOutAsyncWait();
      GPUDevice::copyOutAsyncToHost( copy1._dst, copy1._src, copy1._size );

      // Finish DO of the copy and remove it from the list
      finishPendingCopy( _pendingCopiesAsync.begin() );

      NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvent( key, copy1._size ) );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateEvent() );
   }
#if 0
   while ( !_pendingCopiesAsync.empty() ) {
      PendingCopy & copy = *_pendingCopiesAsync.begin();

      NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("cache-copy-out") );
      NANOS_INSTRUMENT( sys.getInstrumentor()->raiseOpenStateAndBurst( MEM_TRANSFER, key, copy._size ) );

      GPUDevice::copyOutAsyncToBuffer( copy._dst, copy._src, copy._size );
      GPUDevice::copyOutAsyncWait();
      GPUDevice::copyOutAsyncToHost( copy._dst, copy._src, copy._size );

      // Finish DO of the copy and remove it from the list
      finishPendingCopy( _pendingCopiesAsync.begin() );

      NANOS_INSTRUMENT( sys.getInstrumentor()->raiseCloseStateAndBurst( key ) );
   }
#endif
}

void GPUThread::PendingCopiesOutAsyncList::finishPendingCopy( std::vector<PendingCopy>::iterator it )
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( ( uint64_t ) it->_dst );

   if ( it->_do != NULL) {
      it->_do->finished();
   }
   it->done();
   _pendingCopiesAsync.erase( it );
}



void GPUThread::PendingCopiesInAsyncList::clearPendingCopies()
{
   ( ( GPUProcessor * ) myThread->runningOn() )->synchronize( _pendingCopiesAsync );

   _pendingCopiesAsync.clear();
}

