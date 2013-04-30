#ifndef REGIONCACHE_HPP
#define REGIONCACHE_HPP

#include "regioncache_decl.hpp"
#include "processingelement_decl.hpp"


inline uint64_t AllocatedChunk::getAddress() const {
   return _address;
}

inline uint64_t AllocatedChunk::getHostAddress() const {
   return _hostAddress;
}

inline void AllocatedChunk::setHostAddress( uint64_t addr ) {
   _hostAddress = addr;
}

inline std::size_t AllocatedChunk::getSize() const {
   return _size;
}

inline void AllocatedChunk::addReference() {
   _refs += 1;
}

inline void AllocatedChunk::removeReference() {
   _refs -= 1;
   /*
   if ( _refs == (unsigned int) -1 ) {
      std::cerr << "overflow at references chunk "<< (void*)this << std::endl; sys.printBt();
   } else if ( _refs == 0 ) {
      std::cerr << "zeroed at references chunk "<< (void*)this << std::endl; sys.printBt();
   }
   */
}

inline unsigned int AllocatedChunk::getReferenceCount() const { return _refs.value(); }

inline bool AllocatedChunk::isDirty() const {
   return _dirty;
}

inline unsigned int AllocatedChunk::getLruStamp() const {
   return _lruStamp;
}

inline void AllocatedChunk::increaseLruStamp() {
   _lruStamp += 1;
}


inline Device const &RegionCache::getDevice() const {
   return _device;
}

inline bool RegionCache::canCopyFrom( RegionCache const &from ) const {
   return _device == from._device;
}

inline unsigned int RegionCache::getLruTime() const {
   return _lruTime;
}

inline void RegionCache::increaseLruTime() {
   _lruTime += 1;
}

inline bool RegionCache::pin( global_reg_t const &hostMem ) {
   bool result = false;
   AllocatedChunk *entry = this->getAllocatedChunk( hostMem );
   if ( entry ) {
      entry->addReference();
      entry->unlock();
      result = true;
   }
   return result;
}

inline void RegionCache::unpin( global_reg_t const &hostMem ) {
   AllocatedChunk *entry = this->getAllocatedChunk( hostMem );
   if ( entry ) {
      entry->removeReference();
      entry->unlock();
   } else {
      fprintf(stderr, "could not get a CacheEntry!\n");
   }
}

#endif /* REGIONCACHE_HPP */
