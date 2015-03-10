/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#ifndef DEVICEOPS_HPP
#define DEVICEOPS_HPP

#include <iostream>

#include "deviceops_decl.hpp"
#include "atomic.hpp"

#define VERBOSE_CACHE_OPS 0

using namespace nanos;

inline DeviceOps::DeviceOps() : _pendingDeviceOps ( 0 ), _lock() /* debug: */ , _owner( -1 ), _wd( NULL ), _loc( 0 ) {
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

inline bool DeviceOps::addCacheOp( /* debug: */ WorkDescriptor const *wd, int loc ) {
   bool b = _pendingCacheOp.tryAcquire();
      ensure( wd != NULL, "Invalid WD adding a Cache Op.")
      if ( b ) {
         if ( VERBOSE_CACHE_OPS ) {
            *(myThread->_file) << "[" << myThread->getId() << "] "<< (void *)this << " Added an op by " << wd->getId() << " at loc " << loc << std::endl;
         }
         _wd = wd;
         _owner = wd->getId();
         _loc = loc;
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

inline void DeviceOps::completeCacheOp( /* debug: */ WorkDescriptor const *wd ) {
   ensure( _pendingCacheOp.getState() != NANOS_LOCK_FREE, "Already completed op!" );
        ensure( wd == _wd, "Invalid owner clearing a cache op." );
        if ( VERBOSE_CACHE_OPS ) {
           *(myThread->_file) << "[" << myThread->getId() << "] "<< (void *)this << " cleared an op by " << wd->getId() << std::endl;
        }
        _wd = NULL;
        _owner = -1;
        _loc = 0;
   _pendingCacheOp.release();
}

#endif /* DEVICEOPS_HPP */
