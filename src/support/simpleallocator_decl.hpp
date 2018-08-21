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

#ifndef _NANOS_SIMPLEALLOCATOR_DECL
#define _NANOS_SIMPLEALLOCATOR_DECL

#include <stdint.h>
#include <map>
#include <list>
#include <ostream>

#include "atomic_decl.hpp"
#include "lock_decl.hpp"

namespace nanos {

   /*! \brief Simple memory allocator to manage a given contiguous memory area
    */
   class SimpleAllocator
   {
      private:
         typedef std::map < uint64_t, std::size_t > SegmentMap;

         SegmentMap _allocatedChunks;
         SegmentMap _freeChunks;

         uint64_t _baseAddress;
         Lock     _lock;
         std::size_t _remaining;
         std::size_t _capacity;

      public:
         typedef std::list< std::pair< uint64_t, std::size_t > > ChunkList;

         SimpleAllocator( uint64_t baseAddress, std::size_t len );

         // WARNING: Calling this constructor requires calling init() at some time
         // before any allocate() or free() methods are called
         SimpleAllocator() : _baseAddress( 0 ), _remaining( 0 )  { }

         void init( uint64_t baseAddress, std::size_t len );
         uint64_t getBaseAddress ();
         void lock();
         void unlock();

         void * allocate( std::size_t len );
         void * allocateSizeAligned( std::size_t len );
         std::size_t free( void *address );

         void canAllocate( std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) const;
         void getFreeChunksList( ChunkList &list ) const;

         void printMap( std::ostream &o );
         std::size_t getCapacity() const;
         uint64_t getBasePointer( uint64_t address, size_t size );

   };

   class BufferManager
   {
      private:
         void *      _baseAddress;
         std::size_t _index;
         std::size_t _size;

      public:
         BufferManager( void * address, std::size_t size );
         BufferManager() : _baseAddress(0),_index(0),_size(0) {} 

         ~BufferManager() {}

         void init ( void * address, std::size_t size );

         void * getBaseAddress ();

         void * allocate ( std::size_t size );

         void reset ();
   };

} // namespace nanos
#endif /* _NANOS_SIMPLEALLOCATOR */
