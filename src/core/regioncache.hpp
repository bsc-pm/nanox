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

#if 0
inline Region const &CacheCopy::getRegion() const {
   return _region;
}

inline unsigned int CacheCopy::getVersion() const {
   return _version;
}

inline unsigned int CacheCopy::getNewVersion() const {
   return _newVersion;
}

inline CopyData const &CacheCopy::getCopyData() const {
   return _copy;
}

inline reg_t CacheCopy::getRegId() const {
   return _reg.id;
}

inline NewNewRegionDirectory::RegionDirectoryKey CacheCopy::getRegionDirectoryKey() const {
   return _reg.key;
}

inline uint64_t CacheCopy::getDeviceAddress() const {
   uint64_t addr = 0;
   if ( _cacheEntry ) {
      addr = ( _cacheEntry->getAddress() - ( _cacheEntry->getHostAddress() - (uint64_t) _copy.getBaseAddress() ) ) + _copy.getOffset();
   } else {
      addr = ( (uint64_t) _copy.getBaseAddress() + _copy.getOffset() );
   }
   return addr;
}

inline DeviceOps *CacheCopy::getOperations() {
   return &_operations;
}

inline NewRegionDirectory::LocationInfoList const &CacheCopy::getLocations() const {
   return _locations;
}

inline NewLocationInfoList const &CacheCopy::getNewLocations() const {
   return _newLocations;
}

#endif

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
   //uint64_t targetHostAddr = hostMem.getFirstAddress();
   //std::size_t allocSize = hostMem.getDataSize();
   AllocatedChunk *entry = this->getAllocatedChunk( hostMem );
   if ( entry ) {
      entry->addReference();
      entry->unlock();
      result = true;
   }
   return result;
}

inline void RegionCache::unpin( global_reg_t const &hostMem ) {
   //uint64_t targetHostAddr = hostMem.getFirstAddress();
   //std::size_t allocSize = hostMem.getDataSize();
   AllocatedChunk *entry = this->getAllocatedChunk( hostMem );
   if ( entry ) {
      entry->removeReference();
      entry->unlock();
   } else {
      fprintf(stderr, "could not get a CacheEntry!\n");
   }
}

#if 0
inline CacheCopy *CacheController::getCacheCopies() const {
   return _cacheCopies;
}

inline RegionCache *CacheController::getTargetCache() const {
   return _targetCache;
}
#endif

#endif /* REGIONCACHE_HPP */
