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

}
}

#endif /* GPUMEMORYSPACE_H */
