#ifndef DEVICEOPS_DECL_HPP
#define DEVICEOPS_DECL_HPP

#include <set>
#include "atomic_decl.hpp"

namespace nanos {

   class DeviceOpsPtr;
   class DeviceOps {
      private:
         Atomic<unsigned int> _pendingDeviceOps;
         Lock _lock;
         std::set<DeviceOpsPtr *> _refs;
      public:
         DeviceOps();
         ~DeviceOps();
         void completeOp();
         void addOp();
         unsigned int getNumOps() const;
         bool allCompleted() const;
         bool addRef( DeviceOpsPtr *opsPtr, DeviceOpsPtr &p );
         void delRef( DeviceOpsPtr *opsPtr );
         void addFirstRef( DeviceOpsPtr *opsPtr );
   };
   
   class DeviceOpsPtr {
      private:
         DeviceOps *_value;
      public:
         DeviceOpsPtr() : _value( NULL ) {}
         DeviceOpsPtr( DeviceOpsPtr const &p );
         DeviceOpsPtr( DeviceOpsPtr &p );
         ~DeviceOpsPtr();
         DeviceOpsPtr & operator=( DeviceOpsPtr const &p );
         DeviceOpsPtr & operator=( DeviceOpsPtr &p );
         DeviceOps & operator*() const;
         DeviceOps * operator->() const;
         void set( DeviceOps *ops );
         DeviceOps *get() const;
         void clear();
         bool isNotSet() const;
   };
}
#endif /* DEVICEOPS_DECL_HPP */
