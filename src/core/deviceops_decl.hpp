#ifndef DEVICEOPS_DECL_HPP
#define DEVICEOPS_DECL_HPP

#include <set>
#include "atomic_decl.hpp"
#include "deviceops_fwd.hpp"
#include "workdescriptor_fwd.hpp"

namespace nanos {

   class DeviceOps {
      private:
         Atomic<unsigned int> _pendingDeviceOps;
         Lock _pendingCacheOp;
         Lock _lock;
         /*debug:*/ int _owner;
         /*debug:*/ WorkDescriptor const *_wd;
         /*debug:*/ int _loc;
      public:
         DeviceOps();
         ~DeviceOps();
         void completeOp();
         void addOp();
         bool allCompleted() ;

         bool addCacheOp( /* debug: */ WorkDescriptor const *wd, int loc = -1 );
         void completeCacheOp( /* debug: */WorkDescriptor const *wd );
         bool allCacheOpsCompleted();

         bool setInvalidating();
         void clearInvalidating();

         void syncAndDisableInvalidations();
         void resumeInvalidations();
   };
}
#endif /* DEVICEOPS_DECL_HPP */
