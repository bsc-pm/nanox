/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef DEVICEOPS_HPP
#define DEVICEOPS_HPP

#include <iostream>

#include "atomic.hpp"
#include "debug.hpp"
#include "deviceops_decl.hpp"
#include "os.hpp"

#define VERBOSE_CACHE_OPS 0

namespace nanos {

inline DeviceOps::DeviceOps() : _pendingDeviceOps ( 0 ) /* debug: */ , _owner( -1 ), _wd( NULL ), _loc( 0 ) {
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
      ensure( wd != NULL, "Invalid WD adding a Cache Op.");
      if ( b ) {
         if ( VERBOSE_CACHE_OPS ) {
            *(myThread->_file) << "[" << myThread->getId() << "] " << OS::getMonotonicTime() << " " << (void *)this << " Added an op by " << wd->getId() << " at loc " << loc << std::endl;
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

inline void DeviceOps::completeOp() {
   _pendingDeviceOps--;
}

inline void DeviceOps::completeCacheOp( /* debug: */ WorkDescriptor const *wd ) {
   ensure( _pendingCacheOp.getState() != NANOS_LOCK_FREE, "Already completed op!" );
        ensure( wd == _wd, "Invalid owner clearing a cache op." );
        if ( VERBOSE_CACHE_OPS ) {
           *(myThread->_file) << "[" << myThread->getId() << "] " << OS::getMonotonicTime() << " " << (void *)this << " cleared an op by " << wd->getId() << std::endl;
        }
        _wd = NULL;
        _owner = -1;
        _loc = 0;
   _pendingCacheOp.release();
}

inline std::ostream & operator<< ( std::ostream &o, nanos::DeviceOps const &ops ) {
   o << "{_pDeviceOps: " << ops._pendingDeviceOps.value() << " _pCacheOp: " << ops._pendingCacheOp.getState() << " _owner " << ops._owner <<"}";
   return o;
}

} // namespace nanos

#endif /* DEVICEOPS_HPP */
