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

#ifndef _NANOS_GPU_CFG
#define _NANOS_GPU_CFG

#include "config.hpp"
#include "system_decl.hpp"

#if __CUDA_API_VERSION < 3020
#define CUDANODEVERR cudaErrorNoDevice
#else
#define CUDANODEVERR cudaErrorDevicesUnavailable
#endif


namespace nanos {
namespace ext {

   typedef enum {
      NANOS_GPU_TRANSFER_NORMAL,         //!  Basic transfer mode with no overlap
      NANOS_GPU_TRANSFER_ASYNC,          //! -- A little bit better (gives bad results from time to time)
      NANOS_GPU_TRANSFER_PINNED_CUDA,    //! -- Slowdown of ~10x (gives always bad results)
      NANOS_GPU_TRANSFER_PINNED_OS,      //! -- Similar to NANOS_GPU_TRANSFER_NORMAL (correct results though mlock fails)
      NANOS_GPU_TRANSFER_WC              //! -- Same as NANOS_GPU_TRANSFER_PINNED_CUDA: Slowdown of ~10x (gives always bad results)
   } transfer_mode;

   class GPUPlugin;

   /*! Contains the general configuration for the GPU module */
   class GPUConfig
   {
      friend class GPUPlugin;
      private:
         static bool                      _enableCUDA; //! Enable all CUDA support
         static bool                      _forceDisableCUDA; //! Force disable all CUDA support
         static int                       _numGPUs; //! Number of CUDA-capable GPUs
         static System::CachePolicyType   _cachePolicy; //! Defines the cache policy used by GPU devices
         static int                       _numPrefetch; //! Maximum number of tasks to prefetch (set by the user)
         static bool                      _concurrentExec; //! Create more than one execution stream to enable concurrent kernels
         static bool                      _overlap; //! Enable / disable computation and data transfer overlapping (set by the user)
         static bool                      _overlapInputs;
         static bool                      _overlapOutputs;
         static transfer_mode             _transferMode; //! Data transfer's mode (synchronous / asynchronous, ...)
         static size_t                    _maxGPUMemory; //! Maximum amount of memory for each GPU to use
         static size_t                    _maxPinnedMemory; //! Maximum amount of pinned memory for each GPU
         static bool                      _allocatePinnedBuffers; //! Enable / disable allocation of pinned memory buffers used by transfers
         static bool                      _gpuWarmup; //! Enable / disable driver warmup (during runtime startup)
         static bool                      _initCublas; //! Init CUBLAS library during runtime startup
         static bool                      _initCuSparse; //! Init cuSPARSE library during runtime startup
         static void *                    _gpusProperties; //! Array of structs of cudaDeviceProp
         static bool                      _allocWide; //! Use wide allocation policy for the region cache

         /*! Parses the GPU user options */
         static void prepare ( Config &config );
         /*! Applies the configuration options and retrieves the information of the GPUs of the system */
         static void apply ( void );

      public:
         GPUConfig() {}
         ~GPUConfig() {}

         /*! return the number of available GPUs */
         static int getGPUCount ( void ) { return _numGPUs; }

         static System::CachePolicyType getCachePolicy ( void ) { return _cachePolicy; }

         static int getNumPrefetch ( void ) { return _numPrefetch; }

         static bool isConcurrentExecutionEnabled ( void ) { return _concurrentExec; }

         static void setConcurrentExecution ( bool concurrent ) { _concurrentExec = concurrent; }

         static bool isOverlappingInputsDefined ( void ) { return _overlapInputs; }

         static void setOverlappingInputs ( bool overlap ) { _overlapInputs = overlap; }

         static bool isOverlappingOutputsDefined ( void ) { return _overlapOutputs; }

         static void setOverlappingOutputs ( bool overlap ) { _overlapOutputs = overlap; }

         /* \brief get the transfer mode for GPU devices */
         static transfer_mode getTransferMode ( void ) { return _transferMode; }

         static size_t getGPUMaxMemory( void ) { return _maxGPUMemory; }

         static size_t getGPUMaxPinnedMemory( void ) { return _maxPinnedMemory; }

         static bool isAllocatePinnedBuffersEnabled ( void ) { return _allocatePinnedBuffers; }

         static bool isGPUWarmupDefined ( void ) { return _gpuWarmup; }

         static bool isCUBLASInitDefined ( void ) { return _initCublas; }

         static bool isCUSPARSEInitDefined ( void ) { return _initCuSparse; }

         static void getGPUsProperties( int device, void * deviceProps );

         static bool getAllocWide( void );

         static void printConfiguration( void );
   };

} // namespace ext
} // namespace nanos

#endif
