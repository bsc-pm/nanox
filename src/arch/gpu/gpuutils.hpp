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

#ifndef _NANOS_GPU_UTILS
#define _NANOS_GPU_UTILS


#include <iostream>
#include <iomanip>

#include "cuda_runtime.h"


namespace nanos {
namespace ext
{

void displayProperties( cudaDeviceProp* pDeviceProp )
{
   if( !pDeviceProp )
      return;

   std::cout << std::endl << "************************************************";
   std::cout << std::endl << std::setw(37)<< std::left << "Device Name" << "- " << pDeviceProp->name << " ";
   std::cout << std::endl << std::setw(37) << "Total Global Memory" << "- " << pDeviceProp->totalGlobalMem/1024 << " KB";
   std::cout << std::endl << std::setw(37) << "Shared memory available per block" << "- " << pDeviceProp->sharedMemPerBlock/1024 << " KB";
   std::cout << std::endl << std::setw(37) << "Number of registers per thread block" << "- " << pDeviceProp->regsPerBlock;
   std::cout << std::endl << std::setw(37) << "Warp size in threads" << "- " << pDeviceProp->warpSize;
   std::cout << std::endl << std::setw(37) << "Memory Pitch" << "- " << pDeviceProp->memPitch << " bytes";
   std::cout << std::endl << std::setw(37) << "Maximum threads per block" << "- " << pDeviceProp->maxThreadsPerBlock;
   std::cout << std::endl << std::setw(37) << "Maximum Thread Dimension (block)" << "- " << pDeviceProp->maxThreadsDim[0] << " " << pDeviceProp->maxThreadsDim[1] << " " << pDeviceProp->maxThreadsDim[2];
   std::cout << std::endl << std::setw(37) << "Maximum Thread Dimension (grid)" << "- " << pDeviceProp->maxGridSize[0] << " " << pDeviceProp->maxGridSize[1] << " " << pDeviceProp->maxGridSize[2];
   std::cout << std::endl << std::setw(37) << "Total constant memory" << "- " << pDeviceProp->totalConstMem << " bytes";
   std::cout << std::endl << std::setw(37) << "CUDA version" << "- " << pDeviceProp->major << "." << pDeviceProp->minor;
   std::cout << std::endl << std::setw(37) << "Clock rate" << "- " << pDeviceProp->clockRate << " KHz";
   std::cout << std::endl << std::setw(37) << "Texture Alignment" << "- " << pDeviceProp->textureAlignment << " bytes";
   std::cout << std::endl << std::setw(37) << "Device Overlap" << "- "<< ( pDeviceProp-> deviceOverlap ? "Allowed" : "Not Allowed" );
   std::cout << std::endl << std::setw(37) << "Number of Multiprocessors" << "- " << pDeviceProp->multiProcessorCount;
   std::cout << std::endl << "************************************************";
}

void displayAllGPUsProperties( void )
{
   cudaDeviceProp deviceProp;
   int idx, deviceCount = 0;

   cudaGetDeviceCount( &deviceCount );
   std::cout << "Total number of GPUs found: " << deviceCount;
   for ( idx = 0; idx < deviceCount; idx++ ) {
      memset( &deviceProp, 0, sizeof(deviceProp));
      if( cudaSuccess == cudaGetDeviceProperties( &deviceProp, idx ) ) {
         displayProperties( &deviceProp );
      }
      else {
         std::cout << cudaGetErrorString( cudaGetLastError() );
      }
   }

   std::cout << std::endl;
}

}
}

#endif


