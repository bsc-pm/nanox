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
