#ifndef CACHEDREGIONSTATUS_HPP
#define CACHEDREGIONSTATUS_HPP
#include "cachedregionstatus_decl.hpp"
#include "version.hpp"
using namespace nanos; 

inline CachedRegionStatus::CachedRegionStatus() : Version() {
}

inline CachedRegionStatus::CachedRegionStatus( CachedRegionStatus const &rs ) : Version( rs), _waitObject ( rs._waitObject ) {
}

inline CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus const &rs ) {
   Version::operator=(rs);
   _waitObject = rs._waitObject;
   return *this;
}

inline CachedRegionStatus::CachedRegionStatus( CachedRegionStatus &rs ) : Version( rs ), _waitObject ( rs._waitObject ) {
}

inline CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus &rs ) {
   _waitObject = rs._waitObject;
   return *this;
}

inline void CachedRegionStatus::setCopying( DeviceOps *ops ) {
   _waitObject.set( ops );
}

inline DeviceOps *CachedRegionStatus::getDeviceOps() {
   return _waitObject.get();
}

inline bool CachedRegionStatus::isReady( ) {
   return _waitObject.isNotSet();
}
#endif /* CACHEDREGIONSTATUS_HPP */
