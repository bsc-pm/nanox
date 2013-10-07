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

#include "gpuprocessor.hpp"
#include "debug.hpp"
#include "gpudd.hpp"
#include "schedule.hpp"
#include "simpleallocator.hpp"

#include "cuda_runtime.h"

using namespace nanos;
using namespace nanos::ext;

Atomic<int> GPUProcessor::_deviceSeed = 0;


GPUProcessor::GPUProcessor( int id, int gpuId, int uid ) : CachedAccelerator<GPUDevice>( id, &GPU, uid ),
      _gpuDevice( _deviceSeed++ ), _gpuProcessorStats(), _gpuProcessorTransfers(), _allocator(),
      _inputPinnedMemoryBuffer()
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
   // Otherwise, use 95% of the total GPU global memory
   size_t userDefinedMem = GPUConfig::getGPUMaxMemory();
   size_t maxMemoryAvailable = ( size_t ) ( gpuProperties.totalGlobalMem * 0.95 );

   if ( userDefinedMem > 0 ) {
      if ( userDefinedMem <= 100 ) {
         userDefinedMem = ( size_t ) ( maxMemoryAvailable * ( userDefinedMem / 100.0 ) );
      }
      if ( userDefinedMem > maxMemoryAvailable ) {
         warning( "Could not set memory size to " << bytesToHumanReadable( userDefinedMem ) << " for GPU #" << _gpuDevice
               << " because maximum memory available is " << bytesToHumanReadable( maxMemoryAvailable ) << ". Using "
               << bytesToHumanReadable( maxMemoryAvailable ) );
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
   _memoryAlignment = gpuProperties.textureAlignment;

   // We allocate the whole GPU memory
   // WARNING: GPUDevice::allocateWholeMemory() must be called first, as it may
   // modify maxMemoryAvailable, in the case of not being able to allocate as
   // much bytes as we have asked
   void * baseAddress = GPUDevice::allocateWholeMemory( maxMemoryAvailable );
   _allocator.init( ( uint64_t ) baseAddress, maxMemoryAvailable );
   configureCache( maxMemoryAvailable, GPUConfig::getCachePolicy() );
   _gpuProcessorInfo->setMaxMemoryAvailable( maxMemoryAvailable );

   // If some kind of overlapping is defined, allocate some pinned memory

   if ( inputStream ) {
      size_t pinnedSize = std::min( maxMemoryAvailable, ( size_t ) 256*1024*1024 );
      void * pinnedAddress = GPUDevice::allocatePinnedMemory( pinnedSize );
      _inputPinnedMemoryBuffer.init( pinnedAddress, pinnedSize );
   }

   if ( outputStream ) {
      size_t pinnedSize = std::min( maxMemoryAvailable, ( size_t ) 256*1024*1024 );
      void * pinnedAddress = GPUDevice::allocatePinnedMemory( pinnedSize );
      _outputPinnedMemoryBuffer.init( pinnedAddress, pinnedSize );
   }
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
      warning("WARNING: CUDA reported errors during application's execution: " << cudaGetErrorString(err));
   }
   _gpuProcessorInfo->destroyTransferStreams();
   // When cache is disabled, calling this function hangs the execution
   // Otherwise, it can take ages to finish, so we avoid calling it by now
   //if ( sys.isCacheEnabled() ) freeWholeMemory();
   printStats();
}

void GPUProcessor::freeWholeMemory()
{
   void * baseAddress = ( void * ) _allocator.getBaseAddress();
   GPUDevice::freeWholeMemory( baseAddress );
   _allocator.free( baseAddress );
}

size_t GPUProcessor::getMaxMemoryAvailable ( int id )
{
   return _gpuProcessorInfo->getMaxMemoryAvailable();
}

WorkDescriptor & GPUProcessor::getWorkerWD () const
{
   SMPDD * dd = NEW SMPDD( ( SMPDD::work_fct )Scheduler::workerLoop );
   WD *wd = NEW WD( dd );
   return *wd;
}

WorkDescriptor & GPUProcessor::getMasterWD () const
{
   fatal("Attempting to create a GPU master thread");
}

BaseThread &GPUProcessor::createThread ( WorkDescriptor &helper )
{
   // In fact, the GPUThread will run on the CPU, so make sure it canRunIn( SMP )
   ensure( helper.canRunIn( SMP ), "Incompatible worker thread" );
   GPUThread &th = *NEW GPUThread( helper, this, _gpuDevice );

   return th;
}


void GPUProcessor::GPUProcessorInfo::initTransferStreams ( bool &inputStream, bool &outputStream )
{
   if ( inputStream ) {
      // Initialize the CUDA streams used for input data transfers
      cudaError_t err = cudaStreamCreate( &_inTransferStream );
      if ( err != cudaSuccess ) {
         // If an error occurred, disable stream overlapping
         _inTransferStream = 0;
         inputStream = false;
         if ( err == CUDANODEVERR ) {
            fatal( "Error while creating the CUDA input transfer stream: all CUDA-capable devices are busy or unavailable" );
         }
         warning( "Error while creating the CUDA input transfer stream: " << cudaGetErrorString( err ) );
      }
   }

   if ( outputStream ) {
      // Initialize the CUDA streams used for output data transfers
      cudaError_t err = cudaStreamCreate( &_outTransferStream );
      if ( err != cudaSuccess ) {
         // If an error occurred, disable stream overlapping
         _outTransferStream = 0;
         outputStream = false;
         if ( err == CUDANODEVERR ) {
            fatal( "Error while creating the CUDA output transfer stream: all CUDA-capable devices are busy or unavailable" );
         }
         warning( "Error while creating the CUDA output transfer stream: " << cudaGetErrorString( err ) );
      }
   }

   if ( inputStream || outputStream ) {
      // Initialize the CUDA streams used for local data transfers and kernel execution
      cudaError_t err = cudaStreamCreate( &_localTransferStream );
      if ( err != cudaSuccess ) {
         // If an error occurred, disable stream overlapping
         _localTransferStream = 0;
         if ( err == CUDANODEVERR ) {
            fatal( "Error while creating the CUDA output transfer stream: all CUDA-capable devices are busy or unavailable" );
         }
         warning( "Error while creating the CUDA local transfer stream: " << cudaGetErrorString( err ) );
      }
      err = cudaStreamCreate( &_kernelExecStream );
      if ( err != cudaSuccess ) {
         // If an error occurred, disable stream overlapping
         _kernelExecStream = 0;
         if ( err == CUDANODEVERR ) {
            fatal( "Error while creating the CUDA output transfer stream: all CUDA-capable devices are busy or unavailable" );
         }
         warning( "Error while creating the CUDA kernel execution stream: " << cudaGetErrorString( err ) );
      }
   }
}

void GPUProcessor::GPUProcessorInfo::destroyTransferStreams ()
{
   if ( _inTransferStream ) {
      cudaError_t err = cudaStreamDestroy( _inTransferStream );
      if ( err != cudaSuccess ) {
         warning( "Error while destroying the CUDA input transfer stream: " << cudaGetErrorString( err ) );
      }
   }

   if ( _outTransferStream ) {
      cudaError_t err = cudaStreamDestroy( _outTransferStream );
      if ( err != cudaSuccess ) {
         warning( "Error while destroying the CUDA output transfer stream: " << cudaGetErrorString( err ) );
      }
   }

   if ( _localTransferStream ) {
      cudaError_t err = cudaStreamDestroy( _localTransferStream );
      if ( err != cudaSuccess ) {
         warning( "Error while destroying the CUDA local transfer stream: " << cudaGetErrorString( err ) );
      }
   }

   if ( _kernelExecStream ) {
      cudaError_t err = cudaStreamDestroy( _kernelExecStream );
      if ( err != cudaSuccess ) {
         warning( "Error while destroying the CUDA kernel execution stream: " << cudaGetErrorString( err ) );
      }
   }
}
