#ifndef CACHEDREGIONSTATUS_DECL_HPP
#define CACHEDREGIONSTATUS_DECL_HPP

#include "version_decl.hpp"
#include "deviceops_decl.hpp"

namespace nanos {

   class CachedRegionStatus : public Version {
      private:
         DeviceOpsPtr _waitObject;
      public:
         CachedRegionStatus();
         CachedRegionStatus( CachedRegionStatus const &rs );
         CachedRegionStatus &operator=( CachedRegionStatus const &rs );
         CachedRegionStatus( CachedRegionStatus &rs );
         CachedRegionStatus &operator=( CachedRegionStatus &rs );
         void setCopying( DeviceOps *ops );
         DeviceOps * getDeviceOps();
         bool isReady();
   };
}

#endif /* CACHEDREGIONSTATUS_DECL_HPP */
