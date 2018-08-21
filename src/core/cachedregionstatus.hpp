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

#ifndef CACHEDREGIONSTATUS_HPP
#define CACHEDREGIONSTATUS_HPP
#include "cachedregionstatus_decl.hpp"
#include "version.hpp"

namespace nanos {

inline CachedRegionStatus::CachedRegionStatus() : Version(), _ops(), _dirty( false ) {
}

inline CachedRegionStatus::CachedRegionStatus( CachedRegionStatus const &rs ) : Version( rs ), _ops ( ), _dirty( rs._dirty ) {
}

inline DeviceOps *CachedRegionStatus::getDeviceOps() {
   return &_ops;
}

inline CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus const &rs ) {
   Version::operator=(rs);
   _dirty = rs._dirty;
   return *this;
}

inline CachedRegionStatus::CachedRegionStatus( CachedRegionStatus &rs ) : Version( rs ), _ops (), _dirty( rs._dirty ) {
}

inline CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus &rs ) {
   Version::operator=(rs);
   _dirty = rs._dirty;
   return *this;
}

inline bool CachedRegionStatus::isDirty() const {
   return _dirty;
}
inline void CachedRegionStatus::setDirty() {
   _dirty = true;
}
inline void CachedRegionStatus::clearDirty() {
   _dirty = false;
}

} // namespace nanos

#endif /* CACHEDREGIONSTATUS_HPP */
