#include "atomic.hpp"
#include "deviceops.hpp"
#include "system_decl.hpp"

using namespace nanos;

void DeviceOps::completeOp() {
   _lock.acquire();
   unsigned int value = _pendingDeviceOps--;
   if ( value == 1 ) {
      if ( !_refs.empty() ) {
         for ( std::set<DeviceOpsPtr *>::iterator it = _refs.begin(); it != _refs.end(); it++ ) {
            (*it)->clear();
         }
         _refs.clear();
      }
   } else if ( value == 0 ) { std::cerr << "overflow!!! "<< (void *)this << std::endl; sys.printBt(); }
   _lock.release();
}

void DeviceOps::completeCacheOp() {
   _pendingCacheOp.release();
}

bool DeviceOps::addRef( DeviceOpsPtr *opsPtr, DeviceOpsPtr &p ) {
  /* add the reference only if "p" is already inside */
   bool result = false;
   _lock.acquire();
   if ( _refs.count( &p ) == 1 ) {
      _refs.insert( opsPtr );
      result = true;
   }
   _lock.release();
   return result;
}

