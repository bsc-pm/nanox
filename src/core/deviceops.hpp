#ifndef DEVICEOPS_HPP
#define DEVICEOPS_HPP

#include <iostream>

#include "deviceops_decl.hpp"
#include "atomic.hpp"

#define VERBOSE_CACHE_OPS 0

using namespace nanos;

inline DeviceOps::DeviceOps() : _pendingDeviceOps ( 0 ), _lock(), _owner( 0 )/*, _refs()*/ {
}

inline DeviceOps::~DeviceOps() {
}

inline void DeviceOps::addOp() {
   _pendingDeviceOps++;
}

inline bool DeviceOps::allCompleted() {
   bool b = ( _pendingDeviceOps.value() == 0);
   return b;
}

inline bool DeviceOps::addCacheOp( unsigned int owner ) {
   bool b = _pendingCacheOp.tryAcquire();
   ensure( owner > 0, "Invalid WD adding a Cache Op.")
   if ( b ) {
      if ( VERBOSE_CACHE_OPS ) {
         std::cerr << "[" << myThread->getId() << "] "<< (void *)this << " Added an op by " << owner << std::endl;
      }
      _owner = owner;
   }
   return b;
}

inline bool DeviceOps::allCacheOpsCompleted() {
   return _pendingCacheOp.getState() == NANOS_LOCK_FREE;
}

inline void DeviceOps::syncAndDisableInvalidations() {
   _lock.acquire();
}

inline void DeviceOps::resumeInvalidations() {
   _lock.release();
}

inline void DeviceOps::completeOp() {
   _pendingDeviceOps--;
}

inline void DeviceOps::completeCacheOp( unsigned int owner ) {
   ensure( owner == _owner, "Invalid owner clearing a cache op." );
   if ( VERBOSE_CACHE_OPS ) {
      std::cerr << "[" << myThread->getId() << "] "<< (void *)this << " cleared an op by " << owner << std::endl;
   }
   _owner = 0;
   _pendingCacheOp.release();
}

#endif /* DEVICEOPS_HPP */
