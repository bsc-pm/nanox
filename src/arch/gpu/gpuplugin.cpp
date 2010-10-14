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
#include "throttle.hpp"

#include "cuda_runtime.h"

namespace nanos {
namespace ext {

PE * gpuProcessorFactory ( int id )
{
   return new GPUProcessor( id, 0 );
}


class GPUTransferModeOption : public Config::MapAction<transfer_mode>
{
   public:
      GPUTransferModeOption( ) : Config::MapAction<transfer_mode>() {}

      // copy constructor
      GPUTransferModeOption( const GPUTransferModeOption &opt ) : Config::MapAction<transfer_mode>( opt ) {}

      // destructor
      ~GPUTransferModeOption() {}

      void setValue ( const transfer_mode &value ) { GPUDevice::setTransferMode( value ); }
      GPUTransferModeOption * clone () { return new GPUTransferModeOption( *this ); }
};

class GPUPlugin : public Plugin
{
   private:
      bool              _disableCUDA;
      int               _numGPUs;
      bool              _prefetch;
      bool              _overlap;
      bool              _overlapInputs;
      bool              _overlapOutputs;
      size_t            _maxGPUMemory;

   public:
      GPUPlugin() : Plugin( "GPU PE Plugin", 1 ), _disableCUDA( false ), _numGPUs( -1 ), _prefetch( true ),
                      _overlap( true ), _overlapInputs( true ), _overlapOutputs( true ), _maxGPUMemory( 0 )
      {}

      virtual void config( Config& config )
      {
         config.setOptionsSection( "GPU Arch", "GPU specific options" );

         // Enable / disable CUDA
         config.registerConfigOption( "disable-cuda", new Config::FlagOption( _disableCUDA ),
                                       "Enable or disable the use of GPUs with CUDA" );
         config.registerEnvOption( "disable-cuda", "NX_DISABLECUDA" );
         config.registerArgOption( "disable-cuda", "disable-cuda" );

         // Set #GPUs
         config.registerConfigOption ( "num-gpus", new Config::IntegerVar( _numGPUs ),
                                       "Defines the maximum number of GPUs to use" );
         config.registerEnvOption ( "num-gpus", "NX_GPUS" );
         config.registerArgOption ( "num-gpus", "gpus" );

         // Enable / disable prefetching
         config.registerConfigOption( "gpu-prefetch", new Config::FlagOption( _prefetch ),
                                       "Set whether data prefetching must be activated or not" );
         config.registerEnvOption( "gpu-prefetch", "NX_GPUPREFETCH" );
         config.registerArgOption( "gpu-prefetch", "gpu-prefetch" );

         // Enable / disable overlapping
         config.registerConfigOption( "gpu-overlap", new Config::FlagOption( _overlap ),
                                       "Set whether GPU computation should be overlapped with\n\
                                                       all data transfers, whenever possible, or not" );
         config.registerEnvOption( "gpu-overlap", "NX_GPUOVERLAP" );
         config.registerArgOption( "gpu-overlap", "gpu-overlap" );

         // Enable / disable overlapping of inputs
         config.registerConfigOption( "gpu-overlap-inputs", new Config::FlagOption( _overlapInputs ),
                                       "Set whether GPU computation should be overlapped with\n\
                                                       host --> device data transfers, whenever possible, or not" );
         config.registerEnvOption( "gpu-overlap-inputs", "NX_GPUOVERLAP_INPUTS" );
         config.registerArgOption( "gpu-overlap-inputs", "gpu-overlap-inputs" );

         // Enable / disable overlapping of outputs
         config.registerConfigOption( "gpu-overlap-outputs", new Config::FlagOption( _overlapOutputs ),
                                       "Set whether GPU computation should be overlapped with\n\
                                                       device --> host data transfers, whenever possible, or not" );
         config.registerEnvOption( "gpu-overlap-outputs", "NX_GPUOVERLAP_OUTPUTS" );
         config.registerArgOption( "gpu-overlap-outputs", "gpu-overlap-outputs" );

         // Set maximum GPU memory available for each GPU
         config.registerConfigOption ( "gpu-max-memory", new Config::SizeVar( _maxGPUMemory ),
                                       "Defines the maximum amount of GPU memory (in bytes) to use for each GPU" );
         config.registerEnvOption ( "gpu-max-memory", "NX_GPUMAXMEM" );
         config.registerArgOption ( "gpu-max-memory", "gpu-max-memory" );

         /*
         GPUTransferModeOption map;
         map.addOption("wc", WC)
            .addOption("normal",NORMAL);
         config.registerConfigOption ( "gpu-transfer-mode", &map, "Data transfer modes" );
         config.registerArgOption ( "gpu-transfer-mode", "gpu-transfer-mode" );
*/
         //sys.setThrottlePolicy( nanos::ext::createDummyThrottle() );
      }

      virtual void init()
      {
         if ( _disableCUDA ) {
            GPUDD::_gpuCount = 0;
            GPUDD::_prefetch = false;
            GPUDD::_overlap = false;
            GPUDD::_overlapInputs = false;
            GPUDD::_overlapOutputs = false;
            GPUDD::_maxGPUMemory = 0;
         }
         else {
            // Find out how many CUDA-capable GPUs the system has
            int totalCount, device, deviceCount = 0;

            cudaError_t cudaErr = cudaGetDeviceCount( &totalCount );

            // Keep the information of GPUs in GPUDD, in order to avoid a second call to
            // 'cudaGetDeviceProperties()' for each GPU device
            GPUDD::_gpusProperties = malloc( totalCount * sizeof( struct cudaDeviceProp ) );
            struct cudaDeviceProp * gpuProperties = (struct cudaDeviceProp *) GPUDD::_gpusProperties;

            if ( cudaErr != cudaSuccess ) {
               totalCount = 0;
            }

            // Machines with no GPUs can still report one emulation device
            for ( device = 0; device < totalCount; device++ ) {
               cudaGetDeviceProperties( &gpuProperties[deviceCount], device );
               if ( gpuProperties[deviceCount].major != 9999 ) {
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

            // Check if the user wants computation and data transfers to be overlapped
            GPUDD::_overlap = _overlap;

            // If _overlap is defined to false, disable any kind of overlapping
            GPUDD::_overlapInputs = _overlap ? _overlapInputs : false;
            GPUDD::_overlapOutputs = _overlap ? _overlapOutputs : false;

            if ( GPUDD::_overlapInputs || GPUDD::_overlapOutputs ) {
               GPUDevice::setTransferMode( ASYNC );
            }
            else {
               GPUDevice::setTransferMode( NORMAL );
            }

            // Check if the user has defined the maximum GPU memory to use
            GPUDD::_maxGPUMemory = _maxGPUMemory;
         }

         GPUDD::printConfiguration();
      }
};

}
}

nanos::ext::GPUPlugin NanosXPlugin;

