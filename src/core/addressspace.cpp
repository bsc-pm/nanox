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

#include "addressspace_decl.hpp"
#include "regiondirectory.hpp"
#include "regioncache.hpp"
#include "system.hpp"
#include "regiondict.hpp"
namespace nanos {

template <>
MemSpace< HostAddressSpace >::MemSpace( Device &d ) : HostAddressSpace( d ) {
}

template <>
MemSpace< SeparateAddressSpace >::MemSpace( memory_space_id_t memSpaceId, Device &d, bool allocWide, std::size_t slabSize ) : SeparateAddressSpace( memSpaceId, d, allocWide, slabSize ) {
}

HostAddressSpace::HostAddressSpace( Device &d ) : _directory() {
}

void HostAddressSpace::doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const *wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, bool inval ) {
   ensure(destinationChunk == NULL, "Invalid argument");
   from.copyOut( reg, version, ops, wd, copyIdx, inval, sourceChunk );
}

void HostAddressSpace::failToLock( SeparateMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version ) {
   std::cerr << __FUNCTION__ << " @ " << __FILE__ << " : " << __LINE__ << " unimplemented" << std::endl;
}

void HostAddressSpace::synchronize( WD &wd, std::size_t numDataAccesses, DataAccess *data ) {
   _directory.synchronize( wd, numDataAccesses, data ); //needs wd
}

void HostAddressSpace::synchronize( WD &wd, void *addr ) {
   _directory.synchronize( wd, addr ); //needs wd
}

void HostAddressSpace::synchronize( WD &wd ) {
   _directory.synchronize( wd );
}

memory_space_id_t HostAddressSpace::getMemorySpaceId() const {
   return 0;
}

RegionDirectory::RegionDirectoryKey HostAddressSpace::getRegionDirectoryKey( uint64_t addr ) {
   return _directory.getRegionDirectoryKey( addr );
}

reg_t HostAddressSpace::getLocalRegionId( void *hostObject, reg_t hostRegionId ) {
   return _directory.getLocalRegionId( hostObject, hostRegionId );
}

void HostAddressSpace::registerObject( nanos_copy_data_internal_t *obj ) {
   _directory.registerObject( obj );
}

void HostAddressSpace::unregisterObject( void *baseAddr ) {
   _directory.unregisterObject( baseAddr );
}

RegionDirectory const &HostAddressSpace::getDirectory() const {
   return _directory;
}

SeparateAddressSpace::SeparateAddressSpace( memory_space_id_t memorySpaceId, Device &arch, bool allocWide, std::size_t slabSize ) : _cache( memorySpaceId, arch, allocWide ? RegionCache::ALLOC_WIDE : RegionCache::ALLOC_FIT, slabSize ), _nodeNumber( 0 ), _acceleratorNumber( 0 ), _isAccelerator( false ), _sdata( NULL ) {
}

void SeparateAddressSpace::copyOut( global_reg_t const &reg, unsigned int version, DeviceOps *ops, WD const *wd, unsigned int copyIdx, bool inval, AllocatedChunk *origChunk ) {
   _cache.NEWcopyOut( reg, version, wd, copyIdx, ops, inval, origChunk );
}

void SeparateAddressSpace::doOp( SeparateMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version, WD const *wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, bool inval ) {
   _cache.NEWcopyIn( from._cache.getMemorySpaceId(), reg, version, wd, copyIdx, ops, destinationChunk, sourceChunk );
}

void SeparateAddressSpace::doOp( HostMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version, WD const *wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, bool inval ) {
   ensure( sourceChunk == NULL, "invalid argument");
   _cache.NEWcopyIn( 0, reg, version, wd, copyIdx, ops, destinationChunk, sourceChunk );
}

void SeparateAddressSpace::failToLock( SeparateMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version ) {
   std::cerr << __FUNCTION__ << " @ " << __FILE__ << " : " << __LINE__ << " unimplemented" << std::endl;
}

void SeparateAddressSpace::failToLock( HostMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version ) {
   std::cerr << __FUNCTION__ << " @ " << __FILE__ << " : " << __LINE__ << " unimplemented" << std::endl;
}

bool SeparateAddressSpace::prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   return _cache.prepareRegions( memCopies, numCopies, wd );
}

//void SeparateAddressSpace::prepareRegion( global_reg_t const &reg, WD const &wd ) {
//   _cache.prepareRegion( reg, wd );
//}

unsigned int SeparateAddressSpace::getCurrentVersion( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   return _cache.getVersion( reg, wd, copyIdx );
}

void SeparateAddressSpace::releaseRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   _cache.releaseRegions( memCopies, numCopies, wd );
}

//void SeparateAddressSpace::releaseRegion( global_reg_t const &reg, WD const &wd, unsigned int copyIdx, enum RegionCache::CachePolicy policy ) {
//   _cache.releaseRegion( reg, wd, copyIdx, policy );
//}

void SeparateAddressSpace::copyFromHost( TransferList &list, WD const *wd ) {
   for ( TransferList::const_iterator it = list.begin(); it != list.end(); it++ ) {
      this->doOp( sys.getHostMemory(), it->getRegion(), it->getVersion(), wd, it->getCopyIndex(), it->getDeviceOps(), it->getDestinationChunk(), it->getSourceChunk(), false );
   }
}

uint64_t SeparateAddressSpace::getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress, AllocatedChunk *chunk ) const {
   return _cache.getDeviceAddress( reg, baseAddress, chunk );
}

unsigned int SeparateAddressSpace::getNodeNumber() const {
   return _nodeNumber;
}

void SeparateAddressSpace::setNodeNumber( unsigned int n ) {
   _nodeNumber = n;
}

unsigned int SeparateAddressSpace::getAcceleratorNumber() const {
   return _acceleratorNumber;
}

void SeparateAddressSpace::setAcceleratorNumber( unsigned int n ) {
   _acceleratorNumber = n;
   _isAccelerator = true;
}

bool SeparateAddressSpace::isAccelerator() const {
   return _isAccelerator;
}

void *SeparateAddressSpace::getSpecificData() const {
   return _sdata;
}

void SeparateAddressSpace::setSpecificData( void *data ) {
   _sdata = data;
}

RegionCache &SeparateAddressSpace::getCache() {
   return _cache;
}

ProcessingElement &SeparateAddressSpace::getPE() {
   memory_space_id_t id = _cache.getMemorySpaceId();
   return sys.getPEWithMemorySpaceId( id );
}

ProcessingElement const &SeparateAddressSpace::getConstPE() const {
   memory_space_id_t id = _cache.getMemorySpaceId();
   return sys.getPEWithMemorySpaceId( id );
}

memory_space_id_t SeparateAddressSpace::getMemorySpaceId() const {
   return _cache.getMemorySpaceId();
}

unsigned int SeparateAddressSpace::getSoftInvalidationCount() const {
   return _cache.getSoftInvalidationCount();
}

unsigned int SeparateAddressSpace::getHardInvalidationCount() const {
   return _cache.getHardInvalidationCount();
}

bool SeparateAddressSpace::canAllocateMemory( MemCacheCopy *memCopies, unsigned int numCopies, bool considerInvalidations, WD const &wd ) {
   return _cache.canAllocateMemory( memCopies, numCopies, considerInvalidations, wd );
}

void SeparateAddressSpace::invalidate( global_reg_t const &reg ) {
   _cache.invalidateObject( reg );
}

void SeparateAddressSpace::setRegionVersion( global_reg_t const &reg, AllocatedChunk *chunk, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   _cache.setRegionVersion( reg, chunk, version, wd, copyIdx );
}

Device const &SeparateAddressSpace::getDevice() const {
   return _cache.getDevice();
}

//AllocatedChunk *SeparateAddressSpace::getAndReferenceAllocatedChunk( global_reg_t reg, WD const *wd, unsigned int copyIdx ) {
//   return _cache.getAndReferenceAllocatedChunk( reg, wd, copyIdx );
//}

}
