#ifndef DEVICEOPS_HPP
#define DEVICEOPS_HPP

#include <iostream>

#include "deviceops_decl.hpp"
#include "atomic.hpp"

#define VERBOSE_CACHE_OPS 0

using namespace nanos;

inline DeviceOps::DeviceOps() : _pendingDeviceOps ( 0 ), _lock() /* debug: , _owner( -1 ), _wd( NULL ), _loc( 0 ) */ {
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

inline bool DeviceOps::addCacheOp( /* debug: WorkDescriptor const *wd, int loc */ ) {
   bool b = _pendingCacheOp.tryAcquire();
   //debug: ensure( wd != NULL, "Invalid WD adding a Cache Op.")
   //debug: if ( b ) {
   //debug:    if ( VERBOSE_CACHE_OPS ) {
   //debug:       std::cerr << "[" << myThread->getId() << "] "<< (void *)this << " Added an op by " << wd->getId() << " at loc " << loc << std::endl;
   //debug:    }
   //debug:    _wd = wd;
   //debug:    _owner = wd->getId();
   //debug:    _loc = loc;
   //debug: }
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

inline void DeviceOps::completeCacheOp( /* debug: WorkDescriptor const *wd */ ) {
   ensure( _pendingCacheOp.getState() != NANOS_LOCK_FREE, "Already completed op!" );
   //debug: ensure( wd == _wd, "Invalid owner clearing a cache op." );
   //debug: if ( VERBOSE_CACHE_OPS ) {
   //debug:    std::cerr << "[" << myThread->getId() << "] "<< (void *)this << " cleared an op by " << wd->getId() << std::endl;
   //debug: }
   //debug: _wd = NULL;
   //debug: _owner = -1;
   //debug: _loc = 0;
   _pendingCacheOp.release();
}

#endif /* DEVICEOPS_HPP */
