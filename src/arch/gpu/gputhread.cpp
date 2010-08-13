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
#include "instrumentormodule_decl.hpp"
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

   // Avoid the so slow first data allocation and transfer to device
   bool b = true;
   bool * b_d = ( bool * ) GPUDevice::allocate( sizeof( b ) );
   GPUDevice::copyIn( ( void * ) b_d, ( uint64_t ) &b, sizeof( b ) );
   GPUDevice::free( b_d );

   SMPDD &dd = ( SMPDD & ) work.activateDevice( SMP );

   dd.getWorkFct()( work.getData() );
}

void GPUThread::inlineWorkDependent ( WD &wd )
{
   GPUDD &dd = ( GPUDD & )wd.getActiveDevice();

   NANOS_INSTRUMENT ( InstrumentStateAndBurst inst1( "user-code", wd.getId(), RUNNING ) );
   NANOS_INSTRUMENT ( InstrumentSubState inst2( RUNTIME ) );

   if ( GPUDevice::getTransferMode() != nanos::NORMAL ) {
      // Wait for the input transfer stream to finish
      cudaStreamSynchronize( ( (GPUProcessor *) myThread->runningOn() )->getGPUProcessorInfo()->getInTransferStream() );
   }

   ( dd.getWorkFct() )( wd.getData() );

   if ( GPUDevice::getTransferMode() != nanos::NORMAL ) {
      // Get next task in order to prefetch data to device memory
      WD *next = Scheduler::prefetch( ( nanos::BaseThread * ) this, wd );
      setNextWD( next );
      if ( next != 0 ) {
         next->start(false);
      }

      /*
      while ( !_pendingCopies.empty() ) {
         PendingCopy & copy = _pendingCopies[0];
         // Execute the memory copy
         copy.executeAsyncCopy();
         // Finish DO and WG done
         _pendingCopies.erase(_pendingCopies.begin());
      }
      */

      if ( _pendingCopies.size() == 1 ) {
         PendingCopy & copy = _pendingCopies[0];
         copy.executeSyncCopy();
         _pendingCopies.erase(_pendingCopies.begin());
      }
      else if ( !_pendingCopies.empty() ) {
         // First copy
         PendingCopy & copy1 = _pendingCopies[0];
         GPUDevice::copyOutAsyncToBuffer( copy1.getDst(), copy1.getSrc(), copy1.getSize() );

         while ( _pendingCopies.size() > 1) {
            // First copy
            GPUDevice::copyOutAsyncWait();
            // Second copy
            PendingCopy & copy2 = _pendingCopies[1];
            GPUDevice::copyOutAsyncToBuffer( copy2.getDst(), copy2.getSrc(), copy2.getSize() );

            // First copy
            GPUDevice::copyOutAsyncToHost( copy1.getDst(), copy1.getSrc(), copy1.getSize() );

            // Finish DO and WG done of first copy
            _pendingCopies.erase(_pendingCopies.begin());

            // Update second copy to be first copy at next iteration
            copy1 = _pendingCopies[0];
         }

         GPUDevice::copyOutAsyncWait();
         GPUDevice::copyOutAsyncToHost( copy1.getDst(), copy1.getSrc(), copy1.getSize() );
         _pendingCopies.erase(_pendingCopies.begin());
      }
   }

   // Wait for the GPU kernel to finish
   cudaThreadSynchronize();
}

void GPUThread::yield()
{
   if ( !_pendingCopies.empty() ) {
      PendingCopy & copy = _pendingCopies[0];
      copy.executeSyncCopy();
      _pendingCopies.erase(_pendingCopies.begin());
   }

   SMPThread::yield();
}

void GPUThread::PendingCopy::executeAsyncCopy()
{
   // Asynchronous copy-out
   GPUDevice::copyOutAsyncToBuffer( _dst, _src, _size );
   // Asynchronous wait
   GPUDevice::copyOutAsyncWait();
   // Memcpy
   GPUDevice::copyOutAsyncToHost( _dst, _src, _size );
}

void GPUThread::PendingCopy::executeSyncCopy()
{
   // Synchronous copy-out
   GPUDevice::copyOutSyncToHost( _dst, _src, _size );
}
