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

#ifndef _NANOS_PINNEDALLOCATOR_DECL
#define _NANOS_PINNEDALLOCATOR_DECL

#include <stdint.h>
#include <map>

#include "atomic_decl.hpp"
#include "lock_decl.hpp"

namespace nanos {


   /*! \brief Specialized class to allocate pinned memory depending on how this memory
    *         will be used afterwards
    */
   class PinnedMemoryManager
   {
      public:
         PinnedMemoryManager();
         virtual ~PinnedMemoryManager();

         virtual void * allocate( size_t size ) = 0;
         virtual void free( void * address ) = 0;

   };

   /*! \brief Class to allocate and free pinned memory using CUDA runtime
    *
    */
   class CUDAPinnedMemoryManager : public PinnedMemoryManager
   {
      public:
         CUDAPinnedMemoryManager();
         ~CUDAPinnedMemoryManager();

         void * allocate( size_t size );
         void free( void * address );
   };


   /*! \brief Memory allocator to manage pinned memory allocations
    */
   class PinnedAllocator
   {
      private:
         typedef std::map < void *, size_t > PinnedMemoryMap;

         PinnedMemoryMap            _pinnedChunks;
         PinnedMemoryManager     *  _manager;

         Lock                       _lock;

      public:
         PinnedAllocator ( PinnedMemoryManager * manager );
        ~PinnedAllocator () { delete _manager; }

         void * allocate( size_t size );

         void free( void * address );

         bool isPinned( void * address, size_t size );

         void printPinnedMemoryMap();
   };

} // namespace nanos

#endif /* _NANOS_PINNEDALLOCATOR */
