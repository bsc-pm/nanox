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


void GPUThread::initializeDependent ()
{
   // Bind the thread to a GPU device
   cudaError_t err = cudaSetDevice( _gpuDevice );
   if ( err != cudaSuccess )
      warning( "Couldn't set the GPU device for the thread: " << cudaGetErrorString( err ) );

   // Configure some GPU device flags
   if ( GPUConfig::getTransferMode() == NANOS_GPU_TRANSFER_PINNED_CUDA
         || GPUConfig::getTransferMode() == NANOS_GPU_TRANSFER_WC ) {
      err = cudaSetDeviceFlags( cudaDeviceMapHost | cudaDeviceBlockingSync );
      if ( err != cudaSuccess )
         warning( "Couldn't set the GPU device flags: " << cudaGetErrorString( err ) );
   }
   else {
      err = cudaSetDeviceFlags( cudaDeviceScheduleSpin );
      if ( err != cudaSuccess )
         warning( "Couldn't set the GPU device flags:" << cudaGetErrorString( err ) );
   }

   // Warming up GPU's...
   if ( GPUConfig::isGPUWarmupDefined() ) {
      int n = 65536;
      int *warmup = NEW int[n];
      int *warmup_d;
      err = cudaMalloc( &warmup_d, n * sizeof( int ) );
      err = cudaMemcpy( warmup_d, warmup, n * sizeof( int ) , cudaMemcpyHostToDevice );
      err = cudaMemcpy( warmup, warmup_d, n * sizeof( int ) , cudaMemcpyDeviceToHost );
      if ( err != cudaSuccess ) {
         if ( err == CUDANODEVERR ) {
            fatal( "Error while warming up the GPUs: all CUDA-capable devices are busy or unavailable" );
         }
         warning( "Error while warming up the GPUs: " << cudaGetErrorString( err ) );
      }
      cudaFree( warmup_d );
      delete warmup;
   }

   // Initialize GPUProcessor
   ( ( GPUProcessor * ) myThread->runningOn() )->init();
}

void GPUThread::runDependent ()
{
   WD &work = getThreadWD();
   setCurrentWD( work );
   SMPDD &dd = ( SMPDD & ) work.activateDevice( SMP );
   dd.getWorkFct()( work.getData() );
   ( ( GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->destroyTransferStreams();
}

void GPUThread::inlineWorkDependent ( WD &wd )
{
   GPUDD &dd = ( GPUDD & )wd.getActiveDevice();
   GPUProcessor &myGPU = * ( GPUProcessor * ) myThread->runningOn();

   if ( GPUConfig::isOverlappingInputsDefined() ) {
      // Wait for the input transfer stream to finish
      cudaStreamSynchronize( myGPU.getGPUProcessorInfo()->getInTransferStream() );
      // Erase the wait input list and synchronize it with cache
      myGPU.getInTransferList()->clearMemoryTransfers();
   }

   // We wait for wd inputs, but as we have just waited for them, we could skip this step
   wd.start( WD::IsNotAUserLevelThread );

   NANOS_INSTRUMENT ( InstrumentStateAndBurst inst1( "user-code", wd.getId(), NANOS_RUNNING ) );
   ( dd.getWorkFct() )( wd.getData() );

   if ( !GPUConfig::isOverlappingOutputsDefined() && !GPUConfig::isOverlappingInputsDefined() ) {
      // Wait for the GPU kernel to finish
      cudaThreadSynchronize();

      // Normally this instrumentation code is inserted by the compiler in the task outline.
      // But because the kernel call is asynchronous for GPUs we need to raise them manually here
      // when we know the kernel has really finished
      NANOS_INSTRUMENT ( raiseWDClosingEvents() );
   }

   // Copy out results from tasks executed previously
   // Do it always, as another GPU may be waiting for results
   myGPU.getOutTransferList()->executeMemoryTransfers();

   if ( GPUConfig::isPrefetchingDefined() ) {
      NANOS_INSTRUMENT ( InstrumentSubState inst2( NANOS_RUNTIME ) );
      // Get next task in order to prefetch data to device memory
      WD *next = Scheduler::prefetch( ( nanos::BaseThread * ) this, wd );

      setNextWD( next );
      if ( next != 0 ) {
         next->init();
      }
   }

   if ( GPUConfig::isOverlappingOutputsDefined() || GPUConfig::isOverlappingInputsDefined() ) {
      // Wait for the GPU kernel to finish, if we have not waited before
      cudaThreadSynchronize();

      // Normally this instrumentation code is inserted by the compiler in the task outline.
      // But because the kernel call is asynchronous for GPUs we need to raise them manually here
      // when we know the kernel has really finished
      NANOS_INSTRUMENT ( raiseWDClosingEvents() );
   }
}

void GPUThread::yield()
{
   ( ( GPUProcessor * ) runningOn() )->getInTransferList()->executeMemoryTransfers();
   ( ( GPUProcessor * ) runningOn() )->getOutTransferList()->executeMemoryTransfers();
}

void GPUThread::idle()
{
   ( ( GPUProcessor * ) runningOn() )->getInTransferList()->executeMemoryTransfers();
   ( ( GPUProcessor * ) runningOn() )->getOutTransferList()->removeMemoryTransfer();
}


void GPUThread::raiseWDClosingEvents ()
{
   if ( _wdClosingEvents ) {
      NANOS_INSTRUMENT(
            Instrumentation::Event e[2];
            sys.getInstrumentation()->closeBurstEvent( &e[0],
                  sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "user-funct-name" ) );
            sys.getInstrumentation()->closeBurstEvent( &e[1],
                  sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "user-funct-location" ) );

            sys.getInstrumentation()->addEventList( 2, e );
      );
      _wdClosingEvents = false;
   }
}
