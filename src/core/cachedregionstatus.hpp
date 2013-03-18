#ifndef CACHEDREGIONSTATUS_HPP
#define CACHEDREGIONSTATUS_HPP
#include "cachedregionstatus_decl.hpp"
#include "version.hpp"
using namespace nanos; 

inline CachedRegionStatus::CachedRegionStatus() : Version(), _ops() {
}

inline CachedRegionStatus::CachedRegionStatus( CachedRegionStatus const &rs ) : Version( rs ), _ops ( ) {
}

inline DeviceOps *CachedRegionStatus::getDeviceOps() {
   return &_ops;
}

inline CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus const &rs ) {
   Version::operator=(rs);
   return *this;
}

inline CachedRegionStatus::CachedRegionStatus( CachedRegionStatus &rs ) : Version( rs ), _ops () {
}

inline CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus &rs ) {
   _waitObject = rs._waitObject;
   return *this;
}

inline void CachedRegionStatus::setCopying( DeviceOps *ops ) {
   _waitObject.set( ops );
}

inline bool CachedRegionStatus::isReady( ) {
   return _waitObject.isNotSet();
}
#endif /* CACHEDREGIONSTATUS_HPP */
