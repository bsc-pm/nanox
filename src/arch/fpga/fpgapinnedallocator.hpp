#ifndef _NANOS_FPGA_PINNED_ALLOCATOR
#define _NANOS_FPGA_PINNED_ALLOCATOR

#include <map>

#include "atomic.hpp"

#include "libxdma.h"

namespace nanos {

   //   class FPGAPinnedAllocator;
   //
   //   class FPGAPinnedMemoryManager : public PinnedMemoryManager
   //   {
   //      public:
   //         FPGAPinnedMemoryManager( FPGAPinnedAllocator *allocator );
   //         ~FPGAPinnedMemoryManager();
   //
   //         virtual void *allocate( size_t size );
   //         virtual void free( void * address );
   //
   //      private:
   //         FPGAPinnedAllocator *_allocator;
   //   };

   //class FPGAPinnedAllocator: public PinnedAllocator
   class FPGAPinnedAllocator
   {
      private:
         typedef std::map < void *, size_t > PinnedMemoryMap;
         std::map< void *, xdma_buf_handle > _handleMap;
         std::map< void *, size_t > _pinnedChunks;
         Lock _lock;

      public:
         FPGAPinnedAllocator() {}
         ~FPGAPinnedAllocator() {}

         void *allocate( size_t size );
         void free( void * address );
         void * getBasePointer( void *address, size_t size );
         void addBufferHandle( void * address, xdma_buf_handle handle );
         xdma_buf_handle getBufferHandle( void *address );
         void delBufferHandle( void *address );
   };
} // namespace nanos

#endif //_NANOS_PINNED_ALLOCATOR
