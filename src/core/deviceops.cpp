#include "atomic.hpp"
#include "deviceops.hpp"

using namespace nanos;

void DeviceOps::completeOp() {
   unsigned int value = _pendingDeviceOps--;
   if ( value == 1 ) {
      _lock.acquire();
      if ( !_refs.empty() ) {
         //std::cerr <<"ive got " << _refs.size() << " refs to clear" << std::endl; 
         for ( std::set<DeviceOpsPtr *>::iterator it = _refs.begin(); it != _refs.end(); it++ ) {
            (*it)->clear();
         }
         _refs.clear();
      }
      _lock.release();
   } else if ( value == 0 ) std::cerr << "overflow!!! "<< (void *)this << std::endl;
   /*std::cerr << "op--! " << (void *) this <<std::endl; sys.printBt();*/
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

