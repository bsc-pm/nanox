#ifndef REGIONCACHE_HPP
#define REGIONCACHE_HPP

#include "regioncache_decl.hpp"

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

inline bool AllocatedChunk::isDirty() const {
   return _dirty;
}

inline ProcessingElement &RegionCache::getPE() const {
   return _pe;
}

inline unsigned int RegionCache::getNodeNumber() const {
   return _pe.getMyNodeNumber();
}

inline Region const &CacheCopy::getRegion() const {
   return _region;
}

inline Device const &RegionCache::getDevice() const {
   return _device;
}

inline bool RegionCache::canCopyFrom( RegionCache const &from ) const {
   return _pe.supportsDirectTransfersWith( from._pe );
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

inline CacheCopy *CacheController::getCacheCopies() const {
   return _cacheCopies;
}

inline RegionCache *CacheController::getTargetCache() const {
   return _targetCache;
}

#endif /* REGIONCACHE_HPP */
