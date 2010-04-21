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

#include "gpudd.hpp"
#include "gpuutils.hpp"

#include <cuda_runtime.h>

using namespace nanos;
using namespace nanos::ext;

GPUDevice nanos::ext::GPU( "GPU" );

int GPUDD::_gpuCount = 0;

/*!
  \brief Registers the Device's configuration options
  \param reference to a configuration object.
  \sa Config System
 */
void GPUDD::prepareConfig( Config &config )
{
   // Find out how many CUDA-capable GPUs the system has
   int deviceCount, device;
   struct cudaDeviceProp properties;
   cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
   if (cudaErr != cudaSuccess)
      deviceCount = 0;

   // Machines with no GPUs can still report one emulation device
   for (device = 0; device < deviceCount; ++device) {
      cudaGetDeviceProperties(&properties, device);
      if (properties.major != 9999) // 9999 means emulation only
         ++_gpuCount;
   }

   //displayAllGPUsProperties();
}

GPUDD * GPUDD::copyTo ( void *toAddr )
{
   GPUDD *dd = new (toAddr) GPUDD(*this);
   return dd;
}

