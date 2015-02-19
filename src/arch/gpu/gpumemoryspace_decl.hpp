#ifndef GPUMEMORYSPACE_DECL_H
#define GPUMEMORYSPACE_DECL_H

#include "simpleallocator_fwd.hpp"
#include "gpuprocessor_decl.hpp"

namespace nanos {
namespace ext {

   class GPUMemorySpace {
      bool             _initialized;
      SimpleAllocator *_allocator;
      std::size_t      _memoryAlignment;
      BufferManager    _inputPinnedMemoryBuffer;
      BufferManager    _outputPinnedMemoryBuffer;
      GPUProcessor    *_gpu;

      public:
      GPUMemorySpace();
      void initialize( bool allocateInputMem, bool allocateOutputMem, GPUProcessor *gpu );
      SimpleAllocator *getAllocator() const;
      std::size_t getAlignment() const;
      void freeOutputPinnedMemory ();
      void * allocateOutputPinnedMemory ( std::size_t size );
      void freeInputPinnedMemory ();
      void * allocateInputPinnedMemory ( std::size_t size );
      GPUProcessor * getGPU() const;
   };
}
}

#endif /* GPUMEMORYSPACE_DECL_H */
