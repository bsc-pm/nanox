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

#ifndef GPUMEMORYSPACE_H
#define GPUMEMORYSPACE_H

#include "gpumemoryspace_decl.hpp"
#include "gpuprocessor.hpp"

namespace nanos {
namespace ext {

GPUMemorySpace::GPUMemorySpace() : _initialized( false ),  _allocator( NULL ), _inputPinnedMemoryBuffer(), _outputPinnedMemoryBuffer(), _gpu( NULL ) {
}

void GPUMemorySpace::initialize( bool allocateInputMem, bool allocateOutputMem, GPUProcessor *gpu ) {

   if ( !_initialized ) {
      _gpu = gpu;
      std::size_t maxGPUMem = gpu->getGPUProcessorInfo()->getMaxMemoryAvailable();
      _allocator = NEW SimpleAllocator( (uint64_t) gpu->getGPUProcessorInfo()->getBaseAddress(), maxGPUMem );

      if ( GPUConfig::isAllocatePinnedBuffersEnabled() ) {
         std::size_t maxPinnedMem = GPUConfig::getGPUMaxPinnedMemory() * 1024 * 1024; //(convert to bytes)
         std::size_t pinnedSize = std::min( maxGPUMem, maxPinnedMem );

         if ( allocateInputMem && allocateOutputMem ) pinnedSize /= 2;

         if ( allocateInputMem ) {
            void * pinnedAddress = GPUDevice::allocatePinnedMemory( pinnedSize );
            _inputPinnedMemoryBuffer.init( pinnedAddress, pinnedSize );
         }

         if ( allocateOutputMem ) {
            void * pinnedAddress = GPUDevice::allocatePinnedMemory( pinnedSize );
            _outputPinnedMemoryBuffer.init( pinnedAddress, pinnedSize );
         }
      }

      _initialized = true;
   }
}

SimpleAllocator *GPUMemorySpace::getAllocator() const {
   return _allocator;
}

std::size_t GPUMemorySpace::getAlignment() const {
   return _gpu->getGPUProcessorInfo()->getMemoryAlignment();;
}

void * GPUMemorySpace::allocateInputPinnedMemory ( std::size_t size )
{
   return _inputPinnedMemoryBuffer.allocate( size );
}

void GPUMemorySpace::freeInputPinnedMemory ()
{
   _inputPinnedMemoryBuffer.reset();
}

void * GPUMemorySpace::allocateOutputPinnedMemory ( std::size_t size )
{
   return _outputPinnedMemoryBuffer.allocate( size );
}

void GPUMemorySpace::freeOutputPinnedMemory ()
{
   _outputPinnedMemoryBuffer.reset();
}

GPUProcessor *GPUMemorySpace::getGPU() const {
   return _gpu;
}

}
}

#endif /* GPUMEMORYSPACE_H */
