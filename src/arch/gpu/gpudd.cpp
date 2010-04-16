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

#include "gpuprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "gpuutils.hpp"

// CUDA
#include <cuda_runtime.h>

using namespace nanos;
using namespace nanos::ext;

Device nanos::ext::GPU( "GPU" );

int GPUDD::_gpuCount = 0;
size_t GPUDD::_stackSize = 16*1024;

/*!
  \brief Registers the Device's configuration options
  \param reference to a configuration object.
  \sa Config System
 */
void GPUDD::prepareConfig( Config &config )
{
   /*!
      Get the stack size from system configuration
    */
   size_t size = sys.getDeviceStackSize(); 
   if ( size > 0 )
      _stackSize = size;

   /*!
      Get the stack size for this device
    */
   config.registerConfigOption ( "gpu-stack-size", new Config::SizeVar( _stackSize ), "Defines GPU workdescriptor stack size" );
   config.registerArgOption ( "gpu-stack-size", "gpu-stack-size" );
   config.registerEnvOption ( "gpu-stack-size", "NX_GPU_STACK_SIZE" );

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

void GPUDD::initStack ( void *data )
{
   //_state = ::initContext( _stack, _stackSize, ( void * )getWorkFct(),data,( void * )Scheduler::exit, 0 );
}

void GPUDD::lazyInit (WD &wd, bool isUserLevelThread, WD *previous)
{
/*   if (isUserLevelThread) {
      if ( previous == NULL )
         _stack = new intptr_t[_stackSize];
      else {
         GPUDD &oldDD = (GPUDD &) previous->getActiveDevice();

         std::swap(_stack,oldDD._stack);
      }

      initStack(wd.getData());
   }
*/
}

GPUDD * GPUDD::copyTo ( void *toAddr )
{
   GPUDD *dd = new (toAddr) GPUDD(*this);
   return dd;
}

