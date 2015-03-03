
#include "packer_decl.hpp"
#include "system.hpp"

#include <iostream>

using namespace nanos;


bool Packer::PackInfo::operator<( Packer::PackInfo const &pack ) const {
   return _addr < pack._addr;
}

bool Packer::PackInfo::overlaps( uint64_t addr ) const {
   return addr < ( _addr + _len * _count );
}

bool Packer::PackInfo::sizeMatch( std::size_t len, std::size_t count ) const {
   return len == _len && count == _count ;
}

void * Packer::give_pack( uint64_t addr, std::size_t len, std::size_t count ) {
   void *result = NULL;
   PackInfo key( addr, len, count );

#if 1 /* simple implementation */
   if ( _allocator == NULL ) _allocator = sys.getNetwork()->getPackerAllocator();
   _allocator->lock();
   result = _allocator->allocate( len * count );
   _allocator->unlock();
#else
   _lock.acquire();
   //std::map< PackInfo, void *>::iterator it = _packs.lower_bound( key );
   mapIterator it = _packs.lower_bound( key );
   if ( it == _packs.end() || _packs.key_comp()( key, it->first ) || !it->first.sizeMatch( len, count ) ) {
      if ( it == _packs.end() ) { /* not found */
         if ( it != _packs.begin() ) {
            mapIterator previous = it;
            //std::map< PackInfo, void *>::iterator previous = it;
            previous--;
            //if ( previous->first.overlaps( addr ) ) {
            //   std::cerr << "overlap with previous" << std::endl;
            //} else {
               if ( _allocator == NULL ) _allocator = sys.getNetwork()->getPackerAllocator();
               _allocator->lock();
               result = _allocator->allocate( len * count );
               _allocator->unlock();
               _packs.insert( it, std::make_pair( key, PackMemory( result ) ) );
            //}
         } else { 
            //std::cerr << "begin chunk" << std::endl;
            if ( _allocator == NULL ) _allocator = sys.getNetwork()->getPackerAllocator();
            _allocator->lock();
            result = _allocator->allocate( len * count );
            _allocator->unlock();
            _packs.insert( it, std::make_pair( key, PackMemory( result ) ) );
         }
      } else if ( _packs.key_comp()( key, it->first )  ) { /* not equal: key < it */
         //std::cerr << "not equal addr equal addr" << std::endl;
         if ( _allocator == NULL ) _allocator = sys.getNetwork()->getPackerAllocator();
         _allocator->lock();
         result = _allocator->allocate( len * count );
         _allocator->unlock();
         _packs.insert( it, std::make_pair( key, result ) );
      } else { /* eq addr, no size match*/
         if (sys.getNetwork()->getNodeNum() == 0) std::cerr << "equal addr, different size" << std::endl;
         fatal("Unhandled case\n");
      }
   } else { /* exact match */
      if (sys.getNetwork()->getNodeNum() == 0) std::cerr << "exact match chunk " << (void *) addr << " pack addr "<< it->second.getMemory()  << std::endl;
      result = it->second.getMemoryAndIncreaseReferences();
   }
   _lock.release();
#endif
   if ( result == NULL ) {
      std::cerr << "Error: could not get a memory area to pack data. Requested " << ( len*count) << " bytes, capacity " << _allocator->getCapacity() << " bytes."<< std::endl;
      printBt(std::cerr);
   }
   //std::cerr <<"pack returrns "<<  result << std::endl;
   return result;
}

void Packer::free_pack( uint64_t addr, std::size_t len, std::size_t count, void *allocAddr ) {
#if 1
   _allocator->lock();
   _allocator->free( allocAddr );
   _allocator->unlock();
#else
   PackInfo key( addr, len, count );
   _lock.acquire();
   _allocator->lock();
   mapIterator it = _packs.find( key );
   //if (sys.getNetwork()->getNodeNum() == 0) std::cerr <<"pack frees h: " << (void *) addr << " pack addr " <<  it->second << std::endl;
   if ( it->second.decreaseReferences() == 0 ) {
      _allocator->free( it->second.getMemory() );
      _packs.erase( it );
   }
   _allocator->unlock();
   _lock.release();
#endif
}

void Packer::setAllocator( SimpleAllocator *alloc ) {
   _allocator = alloc;
}
