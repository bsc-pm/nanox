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

#ifndef _NANOS_GPU_PROCESSOR
#define _NANOS_GPU_PROCESSOR

#include "gpuprocessor_decl.hpp"

#include <cuda_runtime.h>

#include <iostream>


namespace nanos {
namespace ext
{

   class GPUProcessor::GPUProcessorInfo
   {
      private:
         // Device #
         unsigned int   _deviceId;

         // Memory
         size_t         _maxMemoryAvailable;

         // Transfers
         cudaStream_t   _inTransferStream;
         cudaStream_t   _outTransferStream;
         cudaStream_t   _localTransferStream;
         cudaStream_t   _kernelExecStream;

      public:
         GPUProcessorInfo ( int device ) : _deviceId( device ), _maxMemoryAvailable( 0 ),
            _inTransferStream( 0 ), _outTransferStream( 0 ), _localTransferStream( 0 ), _kernelExecStream( 0 )
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

         cudaStream_t getKernelExecStream ()
         {
            return _kernelExecStream;
         }
   };
}
}

#endif
