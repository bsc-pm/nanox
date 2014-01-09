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
#include "asyncthread.hpp"
#include "gpuprocessor.hpp"
#include "gpuutils.hpp"
#include "gpucallback.hpp"
#include "instrumentationmodule_decl.hpp"
#include "schedule.hpp"
#include "system.hpp"

#include <cuda_runtime.h>
#ifdef NANOS_GPU_USE_CUDA32
extern void cublasShutdown();
extern void cublasSetKernelStream( cudaStream_t );
#else
#include <cublas.h>
#include <cublas_v2.h>
#endif

using namespace nanos;
using namespace nanos::ext;



void * nanos::ext::gpu_bootthread ( void *arg )
{
   GPUThread *self = static_cast<GPUThread *>( arg );

   self->run();

   pthread_exit ( 0 );
}

void GPUThread::start()
{
   pthread_attr_t attr;
   pthread_attr_init( &attr );

   if ( pthread_create( &_pth, &attr, gpu_bootthread, this ) )
      fatal( "couldn't create thread" );
}
void GPUThread::join()
{
   pthread_join( _pth, NULL );
   joined();
}

void GPUThread::switchTo( WD *work, SchedulerHelper *helper )
{
   fatal("A GPUThread cannot call switchTo function.");
}
void GPUThread::exitTo( WD *work, SchedulerHelper *helper )
{
   fatal("A GPUThread cannot call exitTo function.");
}

void GPUThread::switchHelperDependent( WD* oldWD, WD* newWD, void *arg )
{
   fatal("A GPUThread cannot call switchHelperDependent function.");
}



void GPUThread::initializeDependent ()
{

#ifdef NANOS_INSTRUMENTATION_ENABLED
   GPUUtils::GPUInstrumentationEventKeys::_wd_id =
         sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "wd-id" );

   GPUUtils::GPUInstrumentationEventKeys::_in_cuda_runtime =
         sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-cuda-runtime" );

   GPUUtils::GPUInstrumentationEventKeys::_user_funct_location =
         sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "user-funct-location" );

   GPUUtils::GPUInstrumentationEventKeys::_copy_in_gpu =
         sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "copy-in-gpu" );

   GPUUtils::GPUInstrumentationEventKeys::_copy_out_gpu =
         sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "copy-out-gpu" );
#endif

   // Bind the thread to a GPU device
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_SET_DEVICE_EVENT );
   cudaError_t err = cudaSetDevice( _gpuDevice );
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
   if ( err != cudaSuccess )
      warning( "Couldn't set the GPU device for the thread: " << cudaGetErrorString( err ) );

   // Initialize GPUProcessor
   ( ( GPUProcessor * ) myThread->runningOn() )->init();

   // Warming up GPU's...
   if ( GPUConfig::isGPUWarmupDefined() ) {
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_FREE_EVENT );
      cudaFree(0);
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
   }

#ifndef NANOS_GPU_USE_CUDA32
   // Initialize CUBLAS handle in case of potentially using CUBLAS
   if ( GPUConfig::isCUBLASInitDefined() ) {
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_GENERIC_EVENT );
      cublasStatus_t cublasErr = cublasCreate( ( cublasHandle_t * ) &_cublasHandle );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
      if ( cublasErr != CUBLAS_STATUS_SUCCESS ) {
         if ( cublasErr == CUBLAS_STATUS_NOT_INITIALIZED ) {
            warning( "Couldn't set the context handle for CUBLAS library: the CUDA Runtime initialization failed" );
         } else if ( cublasErr == CUBLAS_STATUS_ALLOC_FAILED ) {
            warning( "Couldn't set the context handle for CUBLAS library: the resources could not be allocated" );
         } else {
            warning( "Couldn't set the context handle for CUBLAS library: unknown error" );
         }
      } else {
         // It seems like it is useless, but still do it in case it works some time...
         // This call is causing a segmentation fault inside CUBLAS library...
         //cublasErr = cublasSetStream( * ( ( cublasHandle_t * ) _cublasHandle ),
         //      ( ( ( GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getKernelExecStream() ) );
         //if ( cublasErr != CUBLAS_STATUS_SUCCESS ) {
         //   warning( "Error setting the CUDA stream for the CUBLAS library" );
         //}
      }
   }
#endif

   // Reset CUDA errors that may have occurred inside the runtime initialization
   NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_GET_LAST_ERROR_EVENT );
   err = cudaGetLastError();
   NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
   if ( err != cudaSuccess )
      warning( "CUDA errors occurred during initialization:" << cudaGetErrorString( err ) );

   // Set the number of look ahead (prefetching) tasks
   // Add one to also count current workdescriptor
   setMaxPrefetch( GPUConfig::getNumPrefetch() + 1 );
}

void GPUThread::runDependent ()
{
   WD &work = getThreadWD();
   setCurrentWD( work );
   SMPDD &dd = ( SMPDD & ) work.activateDevice( SMP );

   if ( getTeam() == NULL ) {
      warning( "This GPUThread needs a team to work, but no team was found. The thread will exit.");
      return;
   }


   dd.getWorkFct()( work.getData() );

   if ( GPUConfig::isCUBLASInitDefined() ) {
#ifdef NANOS_GPU_USE_CUDA32
      cublasShutdown();
#else
      cublasDestroy( ( cublasHandle_t ) _cublasHandle );
#endif
   }

   ( ( GPUProcessor * ) myThread->runningOn() )->cleanUp();
}


//bool GPUThread::inlineWorkDependent ( WD &wd )
bool GPUThread::runWDDependent( WD &wd )
{
   GPUDD &dd = ( GPUDD & ) wd.getActiveDevice();
   GPUProcessor &myGPU = * ( GPUProcessor * ) myThread->runningOn();

#ifdef NANOS_INSTRUMENTATION_ENABLED
   // CUDA events and callbacks to instrument kernel execution on GPU
   cudaEvent_t evtk1;
   cudaEventCreate( &evtk1, 0 );
   cudaEventRecord( evtk1, myGPU.getGPUProcessorInfo()->getKernelExecStream( _kernelStreamIdx ) );

   cudaStreamWaitEvent( myGPU.getGPUProcessorInfo()->getTracingKernelStream( _kernelStreamIdx ), evtk1, 0 );

   GPUCallbackData * cbd = NEW GPUCallbackData( this, &wd );

   //cudaStreamAddCallback( myGPU.getGPUProcessorInfo()->getKernelExecStream(), beforeTaskCallback, ( void * ) cbd, 0 );
   cudaStreamAddCallback( myGPU.getGPUProcessorInfo()->getTracingKernelStream( _kernelStreamIdx ), beforeWDRunCallback, ( void * ) cbd, 0 );
#endif

#ifdef NANOS_INSTRUMENTATION_ENABLED
   // Instrumenting task number (WorkDescriptor ID)
   Instrumentation::Event e1;
   sys.getInstrumentation()->createBurstEvent( &e1, GPUUtils::GPUInstrumentationEventKeys::_wd_id, wd.getId() );
   sys.getInstrumentation()->addEventList( 1, &e1 );
#endif

   NANOS_INSTRUMENT ( InstrumentStateAndBurst inst1( "user-code", wd.getId(), NANOS_RUNNING ) );
   ( dd.getWorkFct() )( wd.getData() );

#ifdef NANOS_INSTRUMENTATION_ENABLED
   // Instrumenting task number (WorkDescriptor ID)
   Instrumentation::Event e2;
   sys.getInstrumentation()->closeBurstEvent( &e2, GPUUtils::GPUInstrumentationEventKeys::_wd_id );
   sys.getInstrumentation()->addEventList( 1, &e2 );
#endif

   //NANOS_INSTRUMENT( cudaStreamAddCallback( myGPU.getGPUProcessorInfo()->getKernelExecStream(), afterTaskCallback, ( void * ) cbd2, 0 ); )

#ifdef NANOS_INSTRUMENTATION_ENABLED
   // CUDA events and callbacks to instrument kernel execution on GPU
  cudaEvent_t evtk2;
   cudaEventCreate( &evtk2, 0 );
   cudaEventRecord( evtk2, myGPU.getGPUProcessorInfo()->getKernelExecStream( _kernelStreamIdx ) );

   cudaStreamWaitEvent( myGPU.getGPUProcessorInfo()->getTracingKernelStream( _kernelStreamIdx ), evtk2, 0 );

   GPUCallbackData * cbd2 = NEW GPUCallbackData( this, &wd );

   //cudaStreamAddCallback( myGPU.getGPUProcessorInfo()->getKernelExecStream(), afterTaskCallback, ( void * ) cbd2, 0 );
   cudaStreamAddCallback( myGPU.getGPUProcessorInfo()->getTracingKernelStream( _kernelStreamIdx ), afterWDRunCallback, ( void * ) cbd2, 0 );
#endif

   _kernelStreamIdx++;

   if ( _kernelStreamIdx == myGPU.getGPUProcessorInfo()->getNumExecStreams() ) _kernelStreamIdx = 0;

   return false;
}

void GPUThread::yield()
{
   cudaFree(0);
   ( ( GPUProcessor * ) runningOn() )->getInTransferList()->executeMemoryTransfers();
   ( ( GPUProcessor * ) runningOn() )->getOutTransferList()->executeMemoryTransfers();

   AsyncThread::yield();
}

void GPUThread::idle()
{
   cudaFree(0);
   ( ( GPUProcessor * ) runningOn() )->getInTransferList()->executeMemoryTransfers();
   ( ( GPUProcessor * ) runningOn() )->getOutTransferList()->removeMemoryTransfer();

   AsyncThread::idle();
}

void GPUThread::processTransfers()
{
   ( ( GPUProcessor * ) runningOn() )->getInTransferList()->executeMemoryTransfers();
   ( ( GPUProcessor * ) runningOn() )->getOutTransferList()->removeMemoryTransfer();

   this->checkEvents();
}


unsigned int GPUThread::getCurrentKernelExecStreamIdx()
{
   return _kernelStreamIdx;
}


void * GPUThread::getCUBLASHandle()
{
   ensure( _cublasHandle, "Trying to use CUBLAS handle without initializing CUBLAS library (please, use NX_GPUCUBLASINIT=yes)" );

   // Set the appropriate stream for CUBLAS handle
   cublasSetStream( ( cublasHandle_t ) _cublasHandle,
         ( ( GPUProcessor * ) myThread->runningOn() )->getGPUProcessorInfo()->getKernelExecStream( _kernelStreamIdx ));

   return _cublasHandle;
}


void GPUThread::raiseKernelLaunchEvent()
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   Instrumentation::Event e;

   sys.getInstrumentation()->createBurstEvent( &e,
         GPUUtils::GPUInstrumentationEventKeys::_in_cuda_runtime,
         GPUUtils::NANOS_GPU_CUDA_KERNEL_LAUNCH_EVENT );

   sys.getInstrumentation()->addEventList( 1, &e );
#endif
}


void GPUThread::closeKernelLaunchEvent()
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   Instrumentation::Event e;
   sys.getInstrumentation()->closeBurstEvent( &e, GPUUtils::GPUInstrumentationEventKeys::_in_cuda_runtime, 0 );

   sys.getInstrumentation()->addEventList( 1, &e );
#endif
}


void GPUThread::raiseWDRunEvent ( WD * wd )
{
   //double tstart = nanos::OS::getMonotonicTimeUs();

#ifdef NANOS_INSTRUMENTATION_ENABLED
   WD * oldwd = getCurrentWD();
   setCurrentWD( *wd );

   Instrumentation::Event e;

   GPUDD &dd = ( GPUDD & ) wd->getActiveDevice();
   nanos_event_value_t value = ( nanos_event_value_t ) * ( dd.getWorkFct() );

   //sys.getInstrumentation()->createBurstEvent( &e, GPUUtils::GPUInstrumentationEventKeys::_kernel_launch, value );
   sys.getInstrumentation()->createBurstEvent( &e, GPUUtils::GPUInstrumentationEventKeys::_user_funct_location, value );

   sys.getInstrumentation()->addEventList( 1, &e );

   sys.getInstrumentation()->flushDeferredEvents( wd );

   setCurrentWD( *oldwd );
#endif

   //double tend = nanos::OS::getMonotonicTimeUs() - tstart;

   //std::cout << "Start  " << ( int ) tend << std::endl;

}

void GPUThread::closeWDRunEvent ( WD * wd )
{

   //double tstart = nanos::OS::getMonotonicTimeUs();

#ifdef NANOS_INSTRUMENTATION_ENABLED
   WD * oldwd = getCurrentWD();
   setCurrentWD( *wd );

   Instrumentation::Event e;

   //sys.getInstrumentation()->closeBurstEvent( &e, GPUUtils::GPUInstrumentationEventKeys::_kernel_launch );
   sys.getInstrumentation()->closeBurstEvent( &e, GPUUtils::GPUInstrumentationEventKeys::_user_funct_location, 0 );

   sys.getInstrumentation()->addEventList( 1, &e );

   setCurrentWD( *oldwd );
#endif

   //double tend = nanos::OS::getMonotonicTimeUs() - tstart;

   //std::cout << "Finish " << ( int ) tend << std::endl;
}


void GPUThread::raiseAsyncInputEvent ( size_t size )
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   //WD * oldwd = getCurrentWD();
   //setCurrentWD( *wd );

   Instrumentation::Event e;

   nanos_event_value_t value = size;

   //InstrumentationContextData *icd = wd->getInstrumentationContextData();

   sys.getInstrumentation()->createBurstEvent( &e, GPUUtils::GPUInstrumentationEventKeys::_copy_in_gpu, value );
   //sys.getInstrumentation()->createBurstEvent( &e, _key, value, icd );

   sys.getInstrumentation()->addEventList( 1, &e );

   //setCurrentWD( *oldwd );
#endif
}


void GPUThread::closeAsyncInputEvent ( size_t size )
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   //WD * oldwd = getCurrentWD();
   //setCurrentWD( *wd );

   Instrumentation::Event e;

   //InstrumentationContextData *icd = wd->getInstrumentationContextData();

   sys.getInstrumentation()->closeBurstEvent( &e, GPUUtils::GPUInstrumentationEventKeys::_copy_in_gpu, 0 );
   //sys.getInstrumentation()->closeBurstEvent( &e, _key, icd );

   sys.getInstrumentation()->addEventList( 1, &e );

   //setCurrentWD( *oldwd );
#endif
}


void GPUThread::raiseAsyncOutputEvent ( size_t size )
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   //WD * oldwd = getCurrentWD();
   //setCurrentWD( *wd );

   Instrumentation::Event e;

   nanos_event_value_t value = size;

   //InstrumentationContextData *icd = wd->getInstrumentationContextData();

   sys.getInstrumentation()->createBurstEvent( &e, GPUUtils::GPUInstrumentationEventKeys::_copy_out_gpu, value );
   //sys.getInstrumentation()->createBurstEvent( &e, _key, value, icd );

   sys.getInstrumentation()->addEventList( 1, &e );

   //setCurrentWD( *oldwd );
#endif
}


void GPUThread::closeAsyncOutputEvent ( size_t size )
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   //WD * oldwd = getCurrentWD();
   //setCurrentWD( *wd );

   Instrumentation::Event e;

   //InstrumentationContextData *icd = wd->getInstrumentationContextData();

   sys.getInstrumentation()->closeBurstEvent( &e, GPUUtils::GPUInstrumentationEventKeys::_copy_out_gpu, 0 );
   //sys.getInstrumentation()->closeBurstEvent( &e, _key, icd );

   sys.getInstrumentation()->addEventList( 1, &e );

   //setCurrentWD( *oldwd );
#endif
}
