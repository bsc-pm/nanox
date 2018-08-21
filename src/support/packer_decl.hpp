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

#ifndef PACKER_DECL_H
#define PACKER_DECL_H

#include <stdint.h>
#include <map>
#include "simpleallocator_decl.hpp"

namespace nanos {

class Packer {

   class PackInfo {
      uint64_t    _addr;
      std::size_t _len;
      std::size_t _count;

      public:
         PackInfo() : _addr( 0 ), _len ( 0 ), _count( 0 ) { }
         PackInfo( uint64_t addr, std::size_t len, std::size_t count ) : _addr( addr ), _len ( len ), _count( count ) { }
         PackInfo( PackInfo const &pack ) : _addr( pack._addr ), _len ( pack._len ), _count( pack._count ) { }
         bool operator<( PackInfo const &pack ) const;
         bool overlaps( uint64_t addr ) const;
         bool sizeMatch( std::size_t len, std::size_t count ) const;
   };

   class PackMemory {
      void * _memory;
      Atomic<unsigned int> _refCounter;

      public:
         PackMemory() : _memory( NULL ), _refCounter( 0 ) { }
         PackMemory( void *mem ) : _memory( mem ), _refCounter( 1 ) { }
         PackMemory( PackMemory const &pm ) : _memory( pm._memory ), _refCounter( pm._refCounter.value() ) { }
         void *getMemoryAndIncreaseReferences() { _refCounter++; return _memory; }
         unsigned int decreaseReferences() { unsigned int value = --_refCounter; return value; }
         void *getMemory() const { return _memory; }
   };

   std::map< PackInfo, PackMemory > _packs;
   SimpleAllocator *_allocator;
   Lock _lock;

   typedef std::map< PackInfo, PackMemory >::iterator mapIterator;

   private:
      Packer( Packer const &p );
      bool operator=( Packer const &p );

   public:
      Packer() : _packs(), _allocator( NULL ) {}
      void *give_pack( uint64_t addr, std::size_t len, std::size_t count );
      bool free_pack( uint64_t addr, std::size_t len, std::size_t count, void *allocAddr );
      void setAllocator( SimpleAllocator *alloc );
};

} // namespace nanos

#endif
