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

#include "gpuconfig.hpp"
#include "gpuutils.hpp"
#include "plugin.hpp"
// We need to include system.hpp (to use verbose0(msg)), as debug.hpp does not include it
#include "system.hpp"

#include <cuda_runtime.h>

namespace nanos {
namespace ext {

bool GPUConfig::_enableCUDA = false;
bool GPUConfig::_forceDisableCUDA = false;
int  GPUConfig::_numGPUs = -1;
System::CachePolicyType GPUConfig::_cachePolicy = System::DEFAULT;
bool GPUConfig::_prefetch = false;
bool GPUConfig::_overlap = false;
bool GPUConfig::_overlapInputs = false;
bool GPUConfig::_overlapOutputs = false;
transfer_mode GPUConfig::_transferMode = NANOS_GPU_TRANSFER_NORMAL;
size_t GPUConfig::_maxGPUMemory = 0;
bool GPUConfig::_gpuWarmup = true;
bool GPUConfig::_initCublas = false;
void * GPUConfig::_gpusProperties = NULL;

void GPUConfig::prepare( Config& config )
{
   config.setOptionsSection( "GPU Arch", "GPU specific options" );

   // Enable / disable CUDA
   config.registerConfigOption( "enable-cuda", NEW Config::FlagOption( _enableCUDA ),
                                "Enable the use of GPUs with CUDA" );
   config.registerEnvOption( "enable-cuda", "NX_ENABLECUDA" );
   config.registerArgOption( "enable-cuda", "enable-cuda" );
   
   config.registerConfigOption( "disable-cuda", NEW Config::FlagOption( _forceDisableCUDA ),
                                "Disable the use of GPUs with CUDA" );
   config.registerEnvOption( "disable-cuda", "NX_DISABLECUDA" );
   config.registerArgOption( "disable-cuda", "disable-cuda" );

   // Set #GPUs
   config.registerConfigOption ( "num-gpus", NEW Config::IntegerVar( _numGPUs ),
                                 "Defines the maximum number of GPUs to use (defaults to the available number of GPUs in the system)" );
   config.registerEnvOption ( "num-gpus", "NX_GPUS" );
   config.registerArgOption ( "num-gpus", "gpus" );

   // Set the cache policy for GPU devices
   System::CachePolicyConfig *cachePolicyCfg = NEW System::CachePolicyConfig ( _cachePolicy );
   cachePolicyCfg->addOption("wt", System::WRITE_THROUGH );
   cachePolicyCfg->addOption("wb", System::WRITE_BACK );
   cachePolicyCfg->addOption( "nocache", System::NONE );
   config.registerConfigOption ( "gpu-cache-policy", cachePolicyCfg, "Defines the cache policy for GPU architectures: write-through / write-back (wb by default)" );
   config.registerEnvOption ( "gpu-cache-policy", "NX_GPU_CACHE_POLICY" );
   config.registerArgOption( "gpu-cache-policy", "gpu-cache-policy" );

   // Enable / disable prefetching
   config.registerConfigOption( "gpu-prefetch", NEW Config::FlagOption( _prefetch ),
                                "Set whether data prefetching must be activated or not (disabled by default)" );
   config.registerEnvOption( "gpu-prefetch", "NX_GPUPREFETCH" );
   config.registerArgOption( "gpu-prefetch", "gpu-prefetch" );

   // Enable / disable overlapping
   config.registerConfigOption( "gpu-overlap", NEW Config::FlagOption( _overlap ),
                                "Set whether GPU computation should be overlapped with\n\
                                     all data transfers, whenever possible, or not (disabled by default)" );
   config.registerEnvOption( "gpu-overlap", "NX_GPUOVERLAP" );
   config.registerArgOption( "gpu-overlap", "gpu-overlap" );

   // Enable / disable overlapping of inputs
   config.registerConfigOption( "gpu-overlap-inputs", NEW Config::FlagOption( _overlapInputs ),
                                "Set whether GPU computation should be overlapped with\n\
                                     host --> device data transfers, whenever possible, or not (disabled by default)" );
   config.registerEnvOption( "gpu-overlap-inputs", "NX_GPUOVERLAP_INPUTS" );
   config.registerArgOption( "gpu-overlap-inputs", "gpu-overlap-inputs" );

   // Enable / disable overlapping of outputs
   config.registerConfigOption( "gpu-overlap-outputs", NEW Config::FlagOption( _overlapOutputs ),
                                "Set whether GPU computation should be overlapped with\n\
                                     device --> host data transfers, whenever possible, or not (disabled by default)" );
   config.registerEnvOption( "gpu-overlap-outputs", "NX_GPUOVERLAP_OUTPUTS" );
   config.registerArgOption( "gpu-overlap-outputs", "gpu-overlap-outputs" );

   // Set maximum GPU memory available for each GPU
   config.registerConfigOption ( "gpu-max-memory", NEW Config::SizeVar( _maxGPUMemory ),
                                 "Defines the maximum amount of GPU memory (in bytes) to use for each GPU (defaults to the total amount of shared memory that each GPU has). If this number is below 100, the amount of memory is taken as a percentage of the total device memory" );
   config.registerEnvOption ( "gpu-max-memory", "NX_GPUMAXMEM" );
   config.registerArgOption ( "gpu-max-memory", "gpu-max-memory" );

   // Enable / disable GPU warmup
   config.registerConfigOption( "gpu-warmup", NEW Config::FlagOption( _gpuWarmup ),
                                "Enable or disable warming up the GPU (enabled by default)" );
   config.registerEnvOption( "gpu-warmup", "NX_GPUWARMUP" );
   config.registerArgOption( "gpu-warmup", "gpu-warmup" );

   // Enable / disable CUBLAS initialization
   config.registerConfigOption( "gpu-cublas-init", NEW Config::FlagOption( _initCublas ),
                                "Enable or disable CUBLAS initialization (disabled by default)" );
   config.registerEnvOption( "gpu-cublas-init", "NX_GPUCUBLASINIT" );
   config.registerArgOption( "gpu-cublas-init", "gpu-cublas-init" );
}

void GPUConfig::apply()
{
   //Auto-enable CUDA if it was not done before
   if ( !_enableCUDA ) {
       //ompss_uses_cuda pointer will be null (it's extern) if the compiler did not fill it
      _enableCUDA = ( sys.getOmpssUsesCuda() != 0 );
   }

   if ( _forceDisableCUDA || !_enableCUDA || _numGPUs == 0 ) {
      bool mercuriumHasTasks = ( sys.getOmpssUsesCuda() != 0 );

      if ( mercuriumHasTasks ) {
         message0( " CUDA tasks were compiled and CUDA was disabled, execution"
               " could have unexpected behavior and can even hang, check configuration parameters" );
      }
      _numGPUs = 0;
      _cachePolicy = System::DEFAULT;
      _prefetch = false;
      _overlap = false;
      _overlapInputs = false;
      _overlapOutputs = false;
      _maxGPUMemory = 0;
      _gpuWarmup = false;
      _initCublas = false;
      _gpusProperties = NULL;

   } else {
      verbose0( "Initializing GPU support component" );
      // Find out how many CUDA-capable GPUs the system has
      int totalCount, device, deviceCount = 0;

      cudaError_t cudaErr = cudaGetDeviceCount( &totalCount );
      if ( cudaErr != cudaSuccess ) {
         totalCount = 0;
         _numGPUs = 0;
         _cachePolicy = System::DEFAULT;
         _prefetch = false;
         _overlap = false;
         _overlapInputs = false;
         _overlapOutputs = false;
         _maxGPUMemory = 0;
         _gpuWarmup = false;
         _initCublas = false;
         _gpusProperties = NULL;
         warning0( "Couldn't initialize the GPU support component at runtime startup: " << cudaGetErrorString( cudaErr ) );

         return;
      }

      // Keep the information of GPUs in GPUDD, in order to avoid a second call to
      // 'cudaGetDeviceProperties()' for each GPU device
      _gpusProperties = NEW cudaDeviceProp[totalCount];
      struct cudaDeviceProp * gpuProperties = ( cudaDeviceProp * ) _gpusProperties;

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
         _numGPUs = std::min( _numGPUs, deviceCount );
      } else {
         _numGPUs = deviceCount;
      }

      // Check if the use of caches has been disabled
      if ( sys.isCacheEnabled() ) {
         // Check if the cache policy for GPUs has been defined
         if ( _cachePolicy == System::DEFAULT ) {
            // The user has not defined a specific cache policy for GPUs,
            // check if he has defined a global cache policy
            _cachePolicy = sys.getCachePolicy();
            if ( _cachePolicy == System::DEFAULT ) {
               // There is no global cache policy specified, assign it the default value (write-back)
               _cachePolicy = System::WRITE_BACK;
            }
         }
      } else {
         _cachePolicy = System::NONE;
      }

      // Check overlappings
      _overlapInputs = _overlap ? true : _overlapInputs;
      _overlapOutputs = _overlap ? true : _overlapOutputs;

      if ( _overlapInputs || _overlapOutputs ) {
         _transferMode = NANOS_GPU_TRANSFER_ASYNC;
      }

      // Configure some GPU device flags before initializing any CUDA context
      if ( _transferMode == NANOS_GPU_TRANSFER_PINNED_CUDA
            || _transferMode == NANOS_GPU_TRANSFER_WC ) {
         // Cannot trace events, as instrumentation has not been initialized yet
         //NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( NANOS_GPU_CUDA_SET_DEVICE_FLAGS_EVENT );
         cudaErr = cudaSetDeviceFlags( cudaDeviceMapHost | cudaDeviceBlockingSync );
         //NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
         if ( cudaErr != cudaSuccess )
            warning( "Couldn't set the GPU device flags: " << cudaGetErrorString( cudaErr ) );
      }
      else {
         // Cannot trace events, as instrumentation has not been initialized yet
         //NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( NANOS_GPU_CUDA_SET_DEVICE_FLAGS_EVENT );
         cudaErr = cudaSetDeviceFlags( cudaDeviceScheduleSpin );
         //NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;
         if ( cudaErr != cudaSuccess )
            warning( "Couldn't set the GPU device flags:" << cudaGetErrorString( cudaErr ) );
      }

      if ( _initCublas || ( sys.getOmpssUsesCublas() != 0 ) ) {
         //gpu_cublas_init pointer will be null (it's extern) if the compiler did not fill it
         _initCublas = true;
         verbose0( "Initializing CUBLAS Library" );
         if ( !sys.loadPlugin( "gpu-cublas" ) ) {
            _initCublas = false;
            warning0( "Couldn't initialize CUBLAS library at runtime startup" );
         }
      }
      
      if ( _numGPUs == 0 ) {
         bool mercuriumHasTasks = ( sys.getOmpssUsesCuda() != 0 );
         if ( mercuriumHasTasks ) {
            message0( " CUDA tasks were compiled and no CUDA devices were found, execution"
                    " could have unexpected behavior and can even hang" );
         } else {
             message0( " CUDA plugin was enabled and no CUDA devices were found " );
         }
      }
   }

   printConfiguration();
}

void GPUConfig::printConfiguration()
{
   verbose0( "--- GPUDD configuration ---" );
   verbose0( "  Number of GPU's: " << _numGPUs );
   verbose0( "  GPU cache policy: " << ( _cachePolicy == System::WRITE_THROUGH ? "write-through" : "write-back" ) );
   verbose0( "  Prefetching: " << ( _prefetch ? "Enabled" : "Disabled" ) );
   verbose0( "  Overlapping: " << ( _overlap ? "Enabled" : "Disabled" ) );
   verbose0( "  Overlapping inputs: " << ( _overlapInputs ? "Enabled" : "Disabled" ) );
   verbose0( "  Overlapping outputs: " << ( _overlapOutputs ? "Enabled" : "Disabled" ) );
   verbose0( "  Transfer mode: " << ( _transferMode == NANOS_GPU_TRANSFER_NORMAL ? "Sync" : "Async" ) );
   if ( _maxGPUMemory != 0 ) {
      if ( _maxGPUMemory > 100 ) {
         verbose0( "  Limited memory: Enabled: " << bytesToHumanReadable( _maxGPUMemory ) );
      } else {
         verbose0( "  Limited memory: Enabled: " << _maxGPUMemory << "% of the total device memory" );
      }
   }
   else {
      verbose0( "  Limited memory: Disabled" );
   }
   verbose0( "  GPU warm up: " << ( _gpuWarmup ? "Enabled" : "Disabled" ) );
   verbose0( "  CUBLAS initialization: " << ( _initCublas ? "Enabled" : "Disabled" ) );

   verbose0( "--- end of GPUDD configuration ---" );
}

void GPUConfig::getGPUsProperties( int device, void * deviceProps )
{
   void * props = &( ( cudaDeviceProp * ) _gpusProperties)[device];
   memcpy( deviceProps, props, sizeof( cudaDeviceProp ) );
}

}
}

