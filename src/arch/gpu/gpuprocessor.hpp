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

#include "gpuprocessor_fwd.hpp"

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
         bool           _overlap;
         cudaStream_t   _transferStream;

      public:
         GPUProcessorInfo ( int device ) : _deviceId ( device ), _maxMemoryAvailable ( 0 ),
            _overlap ( GPUDD::isOverlappingDefined() ), _transferStream ( 0 )
         {}

         ~GPUProcessorInfo ()
         {
            if ( _transferStream ) {
               cudaError_t err = cudaStreamDestroy( _transferStream );
               if ( err != cudaSuccess ) {
                  warning( "Error while destroying the CUDA stream: " << cudaGetErrorString( err ) );
               }
            }
         }

         void init ()
         {
            // Each thread initializes its own GPUProcessor so that initialization
            // can be done in parallel

            struct cudaDeviceProp gpuProperties;
            cudaGetDeviceProperties( &gpuProperties, _deviceId );

            // Use 70% of the total GPU global memory
            _maxMemoryAvailable = gpuProperties.totalGlobalMem * 0.7;

            if ( !gpuProperties.deviceOverlap ) {
               // It does not support stream overlapping, disable this feature
               if ( _overlap ) {
                  warning( "Device #" << _deviceId <<
                        " does not support computation and data transfer overlapping" );
                  _overlap = false;
               }
            }

            if ( _overlap ) {
               // Initialize the CUDA stream used for data transfers
               cudaError_t err = cudaStreamCreate( &_transferStream );
               if ( err != cudaSuccess ) {
                  // If an error occurred, disable stream overlapping
                  _overlap = false;
                  _transferStream = 0;
                  warning( "Error while creating the CUDA stream: " << cudaGetErrorString( err ) );
               }
            }
            else {
               _transferStream = 0;
            }
         }

         size_t getMaxMemoryAvailable ()
         {
            return _maxMemoryAvailable;
         }

         cudaStream_t getTransferStream ()
         {
            return _transferStream;
         }
   };
}
}

#endif
