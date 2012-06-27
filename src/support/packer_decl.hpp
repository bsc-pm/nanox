
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

   std::map< PackInfo, void * > _packs;
   SimpleAllocator *_allocator;
   Lock _lock;

   typedef std::map< PackInfo, void * >::iterator mapIterator;

   private:
      Packer( Packer const &p );
      bool operator=( Packer const &p );

   public:
      Packer() : _packs(), _allocator( NULL ) {}
      void *give_pack( uint64_t addr, std::size_t len, std::size_t count );
      void free_pack( uint64_t addr, std::size_t len, std::size_t count );
      void setAllocator( SimpleAllocator *alloc );
};

}

#endif
