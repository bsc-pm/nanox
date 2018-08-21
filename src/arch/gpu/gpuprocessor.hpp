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

#ifndef _NANOS_GPU_PROCESSOR
#define _NANOS_GPU_PROCESSOR

#include "gpuprocessor_decl.hpp"

#include <cuda_runtime.h>

#include <iostream>


namespace nanos {
namespace ext {

   class GPUProcessor::GPUProcessorInfo
   {
      private:
         // Device #
         unsigned int   _deviceId;

         // Memory
         size_t         _maxMemoryAvailable;
         void          *_baseAddress;
         std::size_t    _memoryAlignment;

         // Transfers
         cudaStream_t   _inTransferStream;
         cudaStream_t   _outTransferStream;
         cudaStream_t   _localTransferStream;

         // Execution
         int            _numExecStreams;
         cudaStream_t * _kernelExecStream;

         // Tracing
#ifdef NANOS_INSTRUMENTATION_ENABLED
         cudaStream_t   _tracingInStream;
         cudaStream_t   _tracingOutStream;
         cudaStream_t * _tracingKernelStream;
#endif

      public:
         GPUProcessorInfo ( int device ) : _deviceId( device ), _maxMemoryAvailable( 0 ),
            _inTransferStream( 0 ), _outTransferStream( 0 ), _localTransferStream( 0 ), _kernelExecStream( NULL )
#ifdef NANOS_INSTRUMENTATION_ENABLED
         , _tracingInStream( 0 ), _tracingOutStream( 0 ), _tracingKernelStream( NULL )
#endif
         {}

         ~GPUProcessorInfo () {}

         void initTransferStreams ( bool &inputStream, bool &outputStream );
         void destroyTransferStreams ();

         size_t getMaxMemoryAvailable ()
         {
            return _maxMemoryAvailable;
         }

         void setMaxMemoryAvailable ( size_t maxMemory )
         {
            _maxMemoryAvailable = maxMemory;
         }

         cudaStream_t getInTransferStream ()
         {
            return _inTransferStream;
         }

         cudaStream_t getOutTransferStream ()
         {
            return _outTransferStream;
         }

         cudaStream_t getLocalTransferStream ()
         {
            return _localTransferStream;
         }

         int getNumExecStreams ()
         {
            return _numExecStreams;
         }

         cudaStream_t getKernelExecStream ( unsigned int index )
         {
            return _kernelExecStream[index];
         }

         cudaStream_t getKernelExecStream ();


#ifdef NANOS_INSTRUMENTATION_ENABLED
         cudaStream_t getTracingInputStream ()
         {
            return _tracingInStream;
         }

         cudaStream_t getTracingOutputStream ()
         {
            return _tracingOutStream;
         }

         cudaStream_t getTracingKernelStream ( unsigned int index )
         {
            return _tracingKernelStream[index];
         }


#endif

         void setBaseAddress( void *addr ) {
            _baseAddress = addr;
         }
         
         void setMemoryAlignment( size_t align ) {
            _memoryAlignment = align;
         }

         void *getBaseAddress() const {
            return _baseAddress;
         }

         size_t getMemoryAlignment() const {
            return _memoryAlignment;
         }
   };


inline GPUMemoryTransferList * GPUProcessor::getInTransferList ()
{
   return _gpuProcessorTransfers._pendingCopiesIn;
}

inline GPUMemoryTransferList * GPUProcessor::getOutTransferList ()
{
   return _gpuProcessorTransfers._pendingCopiesOut;
}

} // namespace ext
} // namespace nanos

#endif
