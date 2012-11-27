#ifndef CACHEDREGIONSTATUS_HPP
#define CACHEDREGIONSTATUS_HPP

#include "deviceops_decl.hpp"

namespace nanos {

   class CachedRegionStatus {
      private:
         unsigned int _version;
         DeviceOpsPtr _waitObject;
      public:
         CachedRegionStatus();
         CachedRegionStatus( CachedRegionStatus const &rs );
         CachedRegionStatus &operator=( CachedRegionStatus const &rs );
         CachedRegionStatus( CachedRegionStatus &rs );
         CachedRegionStatus &operator=( CachedRegionStatus &rs );
         unsigned int getVersion();
         void setVersion( unsigned int version );
         void setCopying( DeviceOps *ops );
         DeviceOps * getDeviceOps();
         bool isReady();
   };
}

#endif /* CACHEDREGIONSTATUS_HPP */
