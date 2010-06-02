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

#include "plugin.hpp"
#include "gpudd.hpp"
#include "gpuprocessor.hpp"
#include "system.hpp"

#include "cuda_runtime.h"

namespace nanos {
namespace ext {

PE * gpuProcessorFactory ( int id )
{
   return new GPUProcessor( id );
}


class GPUPlugin : public Plugin
{
   private:
      int _numGPUs;
      bool _prefetch;

   public:
      GPUPlugin() : Plugin( "GPU PE Plugin", 1 ), _numGPUs( -1 ), _prefetch( true ) {}

      virtual void config( Config& config )
      {
         config.setOptionsSection( "GPU Arch", "GPU specific options" );
         config.registerConfigOption ( "num-gpus", new Config::IntegerVar( _numGPUs ),
                                       "Defines the maximum number of GPUs to use" );
         config.registerArgOption ( "num-gpus", "gpus" );
         config.registerEnvOption ( "num-gpus", "NX_GPUS" );

         config.registerConfigOption( "gpu-prefetch", new Config::FlagOption( _prefetch ),
                                       "Set whether data prefetching must be activated or not" );
         config.registerEnvOption( "gpu-prefetch", "NX_GPUPREFETCH" );
         config.registerArgOption( "gpu-prefetch", "gpu-prefetch" );
      }

      virtual void init()
      {
         // Find out how many CUDA-capable GPUs the system has
         int totalCount, device, deviceCount = 0;
         struct cudaDeviceProp gpuProperties;

         cudaError_t cudaErr = cudaGetDeviceCount( &totalCount );

         if ( cudaErr != cudaSuccess ) {
            totalCount = 0;
         }

         // Machines with no GPUs can still report one emulation device
         for ( device = 0; device < totalCount; device++ ) {
            cudaGetDeviceProperties( &gpuProperties, device );
            if ( gpuProperties.major != 9999 ) {
               // 9999 means emulation only
               deviceCount++;
            }
         }

         //displayAllGPUsProperties();

         // Check if the user has set a different number of GPUs to use
         if ( _numGPUs >= 0 ) {
            deviceCount = std::min( _numGPUs, deviceCount );
         }

         GPUDD::_gpuCount = deviceCount;

         // Check if the user wants data to be prefetched or not
         GPUDD::_prefetch = _prefetch;
      }

};

}
}

nanos::ext::GPUPlugin NanosXPlugin;

