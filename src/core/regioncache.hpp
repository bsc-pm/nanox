#ifndef REGIONCACHE_HPP
#define REGIONCACHE_HPP

#include "regioncache_decl.hpp"

inline uint64_t AllocatedChunk::getAddress() const {
   return _address;
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

inline CopyData const &CacheCopy::getCopyData() const {
   return _copy;
}

inline uint64_t CacheCopy::getDeviceAddress() const {
   uint64_t addr = 0;
   if ( _cacheEntry ) {
      addr = _cacheEntry->getAddress() + _offset + _copy.getOffset();
   } else {
      addr = ( (uint64_t) _copy.getBaseAddress() ) + _copy.getOffset();
   }
   return addr;
}

inline uint64_t CacheCopy::getDevBaseAddress() const {
   return _devBaseAddr;
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
