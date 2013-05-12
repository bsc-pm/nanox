#ifndef DEVICEOPS_DECL_HPP
#define DEVICEOPS_DECL_HPP

#include <set>
#include "atomic_decl.hpp"
#include "deviceops_fwd.hpp"

namespace nanos {

   class DeviceOps {
      private:
         Atomic<unsigned int> _pendingDeviceOps;
         Lock _pendingCacheOp;
         Lock _lock;
      public:
         DeviceOps();
         ~DeviceOps();
         void completeOp();
         void addOp();
         bool allCompleted() ;

         bool addCacheOp();
         void completeCacheOp();
         bool allCacheOpsCompleted();

         bool setInvalidating();
         void clearInvalidating();

         void syncAndDisableInvalidations();
         void resumeInvalidations();
   };
}
#endif /* DEVICEOPS_DECL_HPP */
