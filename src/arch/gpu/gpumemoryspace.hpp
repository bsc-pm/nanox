/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#ifndef GPUMEMORYSPACE_H
#define GPUMEMORYSPACE_H

#include "gpumemoryspace_decl.hpp"
#include "gpuprocessor_decl.hpp"

namespace nanos {
namespace ext {

inline GPUMemorySpace::GPUMemorySpace() : _initialized( false ),  _allocator( NULL ), _memoryAlignment( 0 ), _inputPinnedMemoryBuffer(), _outputPinnedMemoryBuffer() {
}

inline void GPUMemorySpace::initialize( bool allocateInputMem, bool allocateOutputMem, GPUProcessor::GPUProcessorInfo &gpuInfo ) {
   
   if ( !_initialized ) {
      _allocator = NEW SimpleAllocator( (uint64_t) gpuInfo.getBaseAddress(), gpuInfo.getMaxMemoryAvailable() );
      _memoryAlignment = gpuInfo.getMemoryAlignment();

      if ( allocateInputMem ) {
         size_t pinnedSize = std::min( gpuInfo.getMaxMemoryAvailable(), ( size_t ) 2*1024*1024*1024 );
         void * pinnedAddress = GPUDevice::allocatePinnedMemory( pinnedSize );
         _inputPinnedMemoryBuffer.init( pinnedAddress, pinnedSize );
      }

      if ( allocateOutputMem ) {
         size_t pinnedSize = std::min( gpuInfo.getMaxMemoryAvailable(), ( size_t ) 2*1024*1024*1024 );
         void * pinnedAddress = GPUDevice::allocatePinnedMemory( pinnedSize );
         _outputPinnedMemoryBuffer.init( pinnedAddress, pinnedSize );
      }
      _initialized = true;
   }
}

inline SimpleAllocator *GPUMemorySpace::getAllocator() const {
   return _allocator;
}

inline std::size_t GPUMemorySpace::getAlignment() const {
   return _memoryAlignment;
}

inline void * GPUMemorySpace::allocateInputPinnedMemory ( std::size_t size )
{
   return _inputPinnedMemoryBuffer.allocate( size );
}

inline void GPUMemorySpace::freeInputPinnedMemory ()
{
   _inputPinnedMemoryBuffer.reset();
}

inline void * GPUMemorySpace::allocateOutputPinnedMemory ( std::size_t size )
{
   return _outputPinnedMemoryBuffer.allocate( size );
}

inline void GPUMemorySpace::freeOutputPinnedMemory ()
{
   _outputPinnedMemoryBuffer.reset();
}

} // namespace ext
} // namespace nanos

#endif /* GPUMEMORYSPACE_H */
