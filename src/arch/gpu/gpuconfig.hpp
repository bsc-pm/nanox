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

#ifndef _NANOS_GPU_CFG
#define _NANOS_GPU_CFG

#include "config.hpp"

namespace nanos {
namespace ext
{
   class GPUPlugin;

   /*! Contains the general configuration for the GPU module */
   class GPUConfig
   {
      friend class GPUPlugin;
      private:
	 static bool    _disableCUDA; //! Enable/disable all CUDA support
         static int     _numGPUs; //! Number of CUDA-capable GPUs
         static bool    _prefetch; //! Enable / disable data prefetching (set by the user)
         static bool    _overlap; //! Enable / disable computation and data transfer overlapping (set by the user)
         static bool    _overlapInputs;
         static bool    _overlapOutputs;
         static size_t  _maxGPUMemory; //! Maximum amount of memory for each GPU to use
         static void *  _gpusProperties; //! Array of structs of cudaDeviceProp

         /*! Parses the GPU user options */
         static void prepare ( Config &config );
         /*! Applies the configuration options and retrieves the information of the GPUs of the system */
         static void apply ( void );

      public:
	 GPUConfig() {}
	 ~GPUConfig() {}

	 /*! return the number of available GPUs */
         static int getGPUCount ( void ) { return _numGPUs; }

         static bool isPrefetchingDefined ( void ) { return _prefetch; }

         static bool isOverlappingInputsDefined ( void ) { return _overlapInputs; }

         static bool isOverlappingOutputsDefined ( void ) { return _overlapOutputs; }

         static size_t getGPUMaxMemory( void ) { return _maxGPUMemory; }

         static void getGPUsProperties( int device, void * deviceProps );

         static void printConfiguration( void );

   };

}
}

#endif
