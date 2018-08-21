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

#include "gpuprocessor.hpp"
#include "gpumemoryspace_decl.hpp"
#include "debug.hpp"
#include "gpudd.hpp"
#include "gpuutils.hpp"
#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "simpleallocator.hpp"
#include "basethread.hpp"

#include "cuda_runtime.h"

using namespace nanos;
using namespace nanos::ext;

Atomic<int> GPUProcessor::_deviceSeed = 0;


GPUProcessor::GPUProcessor( int gpuId, memory_space_id_t memId, SMPProcessor *core, GPUMemorySpace &gpuMem ) :
      ProcessingElement( &GPU, memId, 0 /* local node */, core->getNumaNode() /* numa */, true, 0 /* socket: n/a */, false ),
      _gpuDevice( _deviceSeed++ ), _gpuProcessorStats(),
      _initialized( false ), _gpuMemory( gpuMem ), _core( core ), _thread( NULL)
{
   _gpuProcessorInfo = NEW GPUProcessorInfo( gpuId );
}

GPUProcessor::~GPUProcessor()
{
   delete _gpuProcessorInfo;
}

void GPUProcessor::init ()
{
   // Each thread initializes its own GPUProcessor so that initialization
   // can be done in parallel

   struct cudaDeviceProp gpuProperties;
   GPUConfig::getGPUsProperties( _gpuDevice, ( void * ) &gpuProperties );

   // Check if the user has set the amount of memory to use (and the value is valid)
   // Otherwise, use 95% of the total GPU global memory [Discrete GPUs] or
   //                45% of the total GPU global memory [integrated GPUs]
   bool isIntegrated = gpuProperties.integrated;
   float defaultGpuAllocRatio = (isIntegrated) ? 0.30 : 0.95;
   //TODO: this gpuProperties.totalGlobalMem can induce to allocation errors when there is other GPU applications running or when we use integrated GPUs and the User already allocated memory
   size_t maxMemoryAvailable = ( size_t ) ( gpuProperties.totalGlobalMem * defaultGpuAllocRatio );
   size_t userDefinedMem = GPUConfig::getGPUMaxMemory();

   if ( userDefinedMem > 0 ) {
      if ( userDefinedMem <= 100 ) {
         userDefinedMem = ( size_t ) ( maxMemoryAvailable * ( userDefinedMem / 100.0 ) );
      }
      if ( userDefinedMem > maxMemoryAvailable ) {
         warning( "Could not set memory size to " << GPUUtils::bytesToHumanReadable( userDefinedMem ) << " for GPU #" << _gpuDevice
               << " because maximum memory available is " << GPUUtils::bytesToHumanReadable( maxMemoryAvailable ) << ". Using "
               << GPUUtils::bytesToHumanReadable( maxMemoryAvailable ) );
      }
      else {
         maxMemoryAvailable = userDefinedMem;
      }
   }

   bool inputStream = GPUConfig::isOverlappingInputsDefined();
   bool outputStream = GPUConfig::isOverlappingOutputsDefined();

   if ( !gpuProperties.deviceOverlap ) {
      // It does not support stream overlapping, disable this feature
      warning( "Device #" << _gpuDevice
            << " does not support computation and data transfer overlapping" );
      inputStream = false;
      outputStream = false;
   }
   _gpuProcessorInfo->initTransferStreams( inputStream, outputStream );

   GPUConfig::setOverlappingInputs( inputStream );
   GPUConfig::setOverlappingOutputs( outputStream );

   // Get GPU memory alignment to allow the use of textures
   //_memoryAlignment = gpuProperties.textureAlignment;
   _gpuProcessorInfo->setMemoryAlignment( gpuProperties.textureAlignment );

   // We allocate the whole GPU memory
   // WARNING: GPUDevice::allocateWholeMemory() must be called first, as it may
   // modify maxMemoryAvailable, in the case of not being able to allocate as
   // much bytes as we have asked
   void * baseAddress = GPUDevice::allocateWholeMemory( maxMemoryAvailable );

   //std::cerr << "GPU memory: baseAddr=" << baseAddress << " size=" << maxMemoryAvailable << std::endl;

   //_allocator.init( ( uint64_t ) baseAddress, maxMemoryAvailable );
   //configureCache( maxMemoryAvailable, GPUConfig::getCachePolicy() );

   _gpuProcessorInfo->setBaseAddress( baseAddress );
   _gpuProcessorInfo->setMaxMemoryAvailable( maxMemoryAvailable );

   _gpuMemory.initialize( inputStream, outputStream, this );
   // If some kind of overlapping is defined, allocate some pinned memory

   // WARNING: initTransferStreams() can modify inputStream's and outputStream's value,
   // so call it first

   if ( outputStream ) {
      // If we have a stream for outputs, create the list with asynchronous behaviour
      // There is no need to do it for inputs, as it is already asynchronous
      delete _gpuProcessorTransfers._pendingCopiesOut;
      _gpuProcessorTransfers._pendingCopiesOut = NEW GPUMemoryTransferOutAsyncList();
   }
}

void GPUProcessor::cleanUp()
{
   cudaError_t err = cudaGetLastError();
   if ( err != cudaSuccess ) {
      warning( "WARNING: CUDA reported errors during application's execution: " << cudaGetErrorString( err ) );
   }
   _gpuProcessorInfo->destroyTransferStreams();
   // When cache is disabled, calling this function hangs the execution
   // Otherwise, it can take ages to finish, so we avoid calling it by now
   //if ( sys.isCacheEnabled() ) freeWholeMemory();
   printStats();
}

void GPUProcessor::freeWholeMemory()
{
   void * baseAddress = ( void * ) _gpuMemory.getAllocator()->getBaseAddress();
   GPUDevice::freeWholeMemory( baseAddress );
   _gpuMemory.getAllocator()->free( baseAddress );
}

std::size_t GPUProcessor::getMaxMemoryAvailable () const
{
   return _gpuProcessorInfo->getMaxMemoryAvailable();
}

WorkDescriptor & GPUProcessor::getWorkerWD () const
{
   SMPDD * dd = NEW SMPDD( ( SMPDD::work_fct ) Scheduler::asyncWorkerLoop );
   WD *wd = NEW WD( dd );
   return *wd;
}

WorkDescriptor & GPUProcessor::getMasterWD () const
{
   fatal( "Attempting to create a GPU master thread" );
}

BaseThread &GPUProcessor::createThread ( WorkDescriptor &helper, SMPMultiThread *parent )
{
   // In fact, the GPUThread will run on the CPU, so make sure it canRunIn( SMP )
   ensure( helper.canRunIn( getSMPDevice() ), "Incompatible worker thread" );
   GPUThread &th = *NEW GPUThread( helper, this, _core, _gpuDevice );

   return ( BaseThread& )  th;
}

void GPUProcessor::printStats ()
{
   message("GPU " << _gpuDevice << " TRANSFER STATISTICS");
   message("    Total input transfers: " << GPUUtils::bytesToHumanReadable( _gpuProcessorStats._bytesIn.value() ) );
   message("    Total output transfers: " << GPUUtils::bytesToHumanReadable( _gpuProcessorStats._bytesOut.value() ) );
   message("    Total device transfers: " << GPUUtils::bytesToHumanReadable( _gpuProcessorStats._bytesDevice.value() ) );
}


void GPUProcessor::GPUProcessorInfo::initTransferStreams ( bool &inputStream, bool &outputStream )
{
   if ( inputStream ) {
      // Initialize the CUDA streams used for input data transfers
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_CREATE_EVENT );
      cudaError_t err = cudaStreamCreate( &_inTransferStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         // If an error occurred, disable stream overlapping
         _inTransferStream = 0;
         inputStream = false;
         if ( err == CUDANODEVERR ) {
            fatal( "Error while creating the CUDA input transfer stream: all CUDA-capable devices are busy or unavailable" );
         }
         warning( "Error while creating the CUDA input transfer stream: " << cudaGetErrorString( err ) );
      }

#ifdef NANOS_INSTRUMENTATION_ENABLED
      // Initialize the CUDA stream used for tracing input data transfers
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_CREATE_EVENT );
      err = cudaStreamCreate( &_tracingInStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         _tracingInStream = 0;
         warning( "CUDA stream creation for tracing input transfers failed: " << cudaGetErrorString( err ) );
         warning( "Tracing information may differ from reality." );
      }
#endif

   }

   if ( outputStream ) {
      // Initialize the CUDA streams used for output data transfers
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_CREATE_EVENT );
      cudaError_t err = cudaStreamCreate( &_outTransferStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         // If an error occurred, disable stream overlapping
         _outTransferStream = 0;
         outputStream = false;
         if ( err == CUDANODEVERR ) {
            fatal( "Error while creating the CUDA output transfer stream: all CUDA-capable devices are busy or unavailable" );
         }
         warning( "Error while creating the CUDA output transfer stream: " << cudaGetErrorString( err ) );
      }

#ifdef NANOS_INSTRUMENTATION_ENABLED
      // Initialize the CUDA stream used for tracing output data transfers
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_CREATE_EVENT );
      err = cudaStreamCreate( &_tracingOutStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         _tracingOutStream = 0;
         warning( "CUDA stream creation for tracing output transfers failed: " << cudaGetErrorString( err ) );
         warning( "Tracing information may differ from reality." );
      }
#endif

   }

   if ( inputStream || outputStream ) {
      // Initialize the CUDA streams used for local data transfers and kernel execution
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_CREATE_EVENT );
      cudaError_t err = cudaStreamCreate( &_localTransferStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         // If an error occurred, disable stream overlapping
         _localTransferStream = 0;
         if ( err == CUDANODEVERR ) {
            fatal( "Error while creating the CUDA local transfer stream: all CUDA-capable devices are busy or unavailable" );
         }
         warning( "Error while creating the CUDA local transfer stream: " << cudaGetErrorString( err ) );
      }

      // Create as many kernel streams as the number of prefetching tasks
      _numExecStreams = GPUConfig::isConcurrentExecutionEnabled() ? GPUConfig::getNumPrefetch() + 1 : 1;

      _kernelExecStream = ( cudaStream_t * ) malloc( _numExecStreams * sizeof( cudaStream_t ) );
      for ( int i = 0; i < _numExecStreams; i++ ) {
         NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_CREATE_EVENT );
         err = cudaStreamCreate( &_kernelExecStream[i] );
         NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

         if ( err != cudaSuccess ) {
            // If an error occurred, disable stream overlapping for that kernel stream
            _kernelExecStream[i] = 0;
            if ( err == CUDANODEVERR ) {
               fatal( "Error while creating the CUDA kernel transfer streams: all CUDA-capable devices are busy or unavailable" );
            }
            warning( "Error while creating the CUDA kernel execution stream #" << i << ": " << cudaGetErrorString( err ) );
         }
      }

#ifdef NANOS_INSTRUMENTATION_ENABLED
      // Initialize the CUDA streams used for tracing kernel launches
      _tracingKernelStream = ( cudaStream_t * ) malloc( _numExecStreams * sizeof( cudaStream_t ) );
      for ( int i = 0; i < _numExecStreams; i++ ) {
         NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_CREATE_EVENT );
         err = cudaStreamCreate( &_tracingKernelStream[i] );
         NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

         if ( err != cudaSuccess ) {
            _tracingKernelStream[i] = 0;
            warning( "CUDA stream creation for tracing kernel launches failed: " << cudaGetErrorString( err ) );
            warning( "Tracing information may differ from reality." );
         }
      }
#endif

   } else {
      _kernelExecStream = ( cudaStream_t * ) malloc( sizeof( cudaStream_t ) );
      _kernelExecStream[0] = 0;

#ifdef NANOS_INSTRUMENTATION_ENABLED
      _tracingKernelStream = ( cudaStream_t * ) malloc( sizeof( cudaStream_t ) );
      _tracingKernelStream[0] = 0;
#endif

   }
}

void GPUProcessor::GPUProcessorInfo::destroyTransferStreams ()
{
   if ( _inTransferStream ) {
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_DESTROY_EVENT );
      cudaError_t err = cudaStreamDestroy( _inTransferStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         warning( "Error while destroying the CUDA input transfer stream: " << cudaGetErrorString( err ) );
      }

#ifdef NANOS_INSTRUMENTATION_ENABLED
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_DESTROY_EVENT );
      err = cudaStreamDestroy( _tracingInStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         warning( "Error while destroying the CUDA stream for tracing input transfers: " << cudaGetErrorString( err ) );
      }
#endif
   }

   if ( _outTransferStream ) {
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_DESTROY_EVENT );
      cudaError_t err = cudaStreamDestroy( _outTransferStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         warning( "Error while destroying the CUDA output transfer stream: " << cudaGetErrorString( err ) );
      }

#ifdef NANOS_INSTRUMENTATION_ENABLED
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_DESTROY_EVENT );
      err = cudaStreamDestroy( _tracingOutStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         warning( "Error while destroying the CUDA stream for tracing output transfers: " << cudaGetErrorString( err ) );
      }
#endif
   }

   if ( _localTransferStream ) {
      NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_DESTROY_EVENT );
      cudaError_t err = cudaStreamDestroy( _localTransferStream );
      NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

      if ( err != cudaSuccess ) {
         warning( "Error while destroying the CUDA local transfer stream: " << cudaGetErrorString( err ) );
      }
   }

   for ( int i = 0; i < _numExecStreams; i++ ) {
      if ( _kernelExecStream[i] ) {
         NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_DESTROY_EVENT );
         cudaError_t err = cudaStreamDestroy( _kernelExecStream[i] );
         NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

         if ( err != cudaSuccess ) {
            warning( "Error while destroying the CUDA kernel execution stream #" << i << ": " << cudaGetErrorString( err ) );
         }
      }
   }

#ifdef NANOS_INSTRUMENTATION_ENABLED
   for ( int i = 0; i < _numExecStreams; i++ ) {
      if ( _tracingKernelStream[i] ) {
         NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_STREAM_DESTROY_EVENT );
         cudaError_t err = cudaStreamDestroy( _tracingKernelStream[i] );
         NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

         if ( err != cudaSuccess ) {
            warning( "Error while destroying the CUDA stream for tracing kernel launches: " << cudaGetErrorString( err ) );
         }
      }
   }
#endif

}

cudaStream_t GPUProcessor::GPUProcessorInfo::getKernelExecStream ()
{
   unsigned int index = ( ( GPUThread * ) myThread )->getCurrentKernelExecStreamIdx();
   return _kernelExecStream[index];
}

BaseThread & GPUProcessor::startGPUThread()
{
   WD & worker = getWorkerWD();

   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );

   _thread = &_core->startThread( *this, worker, NULL );

   return *_thread;
}

void GPUProcessor::stopAllThreads ()
{
   _core->stopAllThreads();
}

BaseThread * GPUProcessor::getFirstThread()
{
   return _thread;
}

//xteruel
#if 0
BaseThread * GPUProcessor::getFirstRunningThread_FIXME()
{
   return _core->getFirstRunningThread_FIXME();
}

BaseThread * GPUProcessor::getFirstStoppedThread_FIXME()
{
   return _core->getFirstStoppedThread_FIXME();
}
BaseThread * GPUProcessor::getUnassignedThread()
{
   return _core->getUnassignedThread();
}
#endif
