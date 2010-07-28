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

   class GPUProcessor::TransferInfo
   {
      private:
         bool           _overlap;
         cudaStream_t   _transferStream;

      public:

         TransferInfo () : _overlap( GPUDD::isOverlappingDefined() ) {}

         void init ()
         {
            if ( _overlap ) {
               cudaError_t err = cudaStreamCreate( &_transferStream );
               if ( err != cudaSuccess ) {
                  _transferStream = 0;
                  warning( "Error while creating the CUDA stream: " << cudaGetErrorString( err ) );
               }
            }
            else {
               _transferStream = 0;
            }
         }

         cudaStream_t getTransferStream ()
         {
            return _transferStream;
         }
   };

}
}

#endif
