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

#ifndef REGIONCACHE_HPP
#define REGIONCACHE_HPP

#include <stdio.h>
#include "regioncache_decl.hpp"
#include "processingelement_decl.hpp"
#include "atomic.hpp"

namespace nanos {

inline uint64_t AllocatedChunk::getAddress() const {
   return _address;
}

inline uint64_t AllocatedChunk::getHostAddress() const {
   return _hostAddress;
}

inline void AllocatedChunk::setHostAddress( uint64_t addr ) {
   _hostAddress = addr;
}

inline std::size_t AllocatedChunk::getSize() const {
   return _size;
}

inline void AllocatedChunk::addReference( WD const &wd, unsigned int loc ) {
   _refs++;
   _refWdId[&wd]++;
   _refLoc[wd.getId()].insert(loc);
   //std::cerr << "add ref to chunk "<< (void*)this << " " << _refs.value() << std::endl;
}

inline void AllocatedChunk::removeReference( WD const &wd ) {
   //ensure(_refs > 0, "invalid removeReference, chunk has 0 references!");
   if ( _refs == 0 ) {
      *myThread->_file << " removeReference ON A CHUNK WITH 0 REFS!!!" << std::endl;
   }
   _refs--;
   _refWdId[&wd]--;
   if ( _refWdId[&wd] == 0 ) {
      _refLoc[wd.getId()].clear();
   }
   
   //std::cerr << "del ref to chunk "<< (void*)this << " " << _refs.value() << std::endl;
   //if ( _refs == (unsigned int) -1 ) {
   //   std::cerr << "overflow at references chunk "<< (void*)this << std::endl; sys.printBt();
   //} else if ( _refs == 0 ) {
   //   std::cerr << "zeroed at references chunk "<< (void*)this << std::endl;
   //}
   
}

inline unsigned int AllocatedChunk::getReferenceCount() const { return _refs.value(); }

inline bool AllocatedChunk::isDirty() const {
   return _dirty;
}

inline unsigned int AllocatedChunk::getLruStamp() const {
   return _lruStamp;
}

inline void AllocatedChunk::increaseLruStamp() {
   _lruStamp += 1;
}

inline global_reg_t AllocatedChunk::getAllocatedRegion() const {
   return _allocatedRegion;
}

inline bool AllocatedChunk::isRooted() const {
   return _rooted;
}

inline Device const &RegionCache::getDevice() const {
   return _device;
}

inline bool RegionCache::canCopyFrom( RegionCache const &from ) const {
   return _device == from._device;
}

inline unsigned int RegionCache::getLruTime() const {
   return _lruTime;
}

inline void RegionCache::increaseLruTime() {
   _lruTime += 1;
}

inline unsigned int RegionCache::getSoftInvalidationCount() const {
   return _softInvalidationCount.value();
}

inline void RegionCache::increaseSoftInvalidationCount(unsigned int v) {
   _softInvalidationCount += v;
}

inline unsigned int RegionCache::getHardInvalidationCount() const {
   return _hardInvalidationCount.value();
}

inline void RegionCache::increaseHardInvalidationCount(unsigned int v) {
   _hardInvalidationCount += v;
}

inline void RegionCache::increaseTransferredInData(size_t bytes) {
   _inBytes += bytes;

}
inline void RegionCache::increaseTransferredOutData(size_t bytes) {
   _outBytes += bytes;
}

inline void RegionCache::increaseTransferredReplacedOutData(size_t bytes) {
   _outRepalcementBytes += bytes;
}

inline size_t RegionCache::getTransferredInData() const {
   return _inBytes.value();
}

inline size_t RegionCache::getTransferredOutData() const {
   return _outBytes.value();
}

inline size_t RegionCache::getTransferredReplacedOutData() const {
   return _outRepalcementBytes.value();
}

inline unsigned int RegionCache::getCurrentAllocations() const {
   return _currentAllocations.value();
}

inline bool RegionCache::hasFreeMem() const {
   return _allocatedBytes < _device.getMemCapacity( sys.getSeparateMemory( _memorySpaceId ) );
}

inline std::size_t RegionCache::getUnallocatedBytes() const {
   return _device.getMemCapacity( sys.getSeparateMemory( _memorySpaceId ) ) - _allocatedBytes;
}

} // namespace nanos

#endif /* REGIONCACHE_HPP */
