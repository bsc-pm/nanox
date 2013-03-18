#ifndef CACHEDREGIONSTATUS_DECL_HPP
#define CACHEDREGIONSTATUS_DECL_HPP

#include "version_decl.hpp"
#include "deviceops_decl.hpp"

namespace nanos {

   class CachedRegionStatus : public Version {
      private:
         DeviceOps _ops;
         DeviceOpsPtr _waitObject;
      public:
         CachedRegionStatus();
         CachedRegionStatus( CachedRegionStatus const &rs );
         DeviceOps * getDeviceOps();

         CachedRegionStatus &operator=( CachedRegionStatus const &rs );
         CachedRegionStatus( CachedRegionStatus &rs );
         CachedRegionStatus &operator=( CachedRegionStatus &rs );
         bool isReady();
         void setCopying( DeviceOps * );
   };
}

#endif /* CACHEDREGIONSTATUS_DECL_HPP */
