#include "addressspace_decl.hpp"
#include "newregiondirectory.hpp"
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

void HostAddressSpace::doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *chunk, bool inval ) {
   from.copyOut( reg, version, ops, wd, copyIdx, inval );
}

void HostAddressSpace::getVersionInfo( global_reg_t const &reg, unsigned int &version, NewLocationInfoList &locations ) {
   do {
     NewNewRegionDirectory::tryGetLocation( reg.key, reg.id, locations, version, *((WD*)NULL) );
   } while ( version == 0 ); 
}

void HostAddressSpace::getRegionId( CopyData const &cd, global_reg_t &reg, WD const &wd, unsigned int idx ) {
   // *(myThread->_file) << "Registering CD with addr " << (void *) cd.getBaseAddress() << std::endl;
   // *(myThread->_file) << cd << std::endl;
   reg.key = _directory.getRegionDirectoryKeyRegisterIfNeeded( cd, &wd );
   reg.id = reg.key->obtainRegionId( cd, wd, idx );
   //*(myThread->_file) << "Got key " << (void *)reg.key << " got id " << (int)reg.id << std::endl;
}

void HostAddressSpace::failToLock( SeparateMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version ) {
   std::cerr << __FUNCTION__ << " @ " << __FILE__ << " : " << __LINE__ << " unimplemented" << std::endl;
}

void HostAddressSpace::synchronize( WD &wd ) {
   _directory.synchronize( wd );
}

memory_space_id_t HostAddressSpace::getMemorySpaceId() const {
   return 0;
}

NewNewRegionDirectory::RegionDirectoryKey HostAddressSpace::getRegionDirectoryKey( uint64_t addr ) {
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

SeparateAddressSpace::SeparateAddressSpace( memory_space_id_t memorySpaceId, Device &arch, bool allocWide, std::size_t slabSize ) : _cache( memorySpaceId, arch, allocWide ? RegionCache::ALLOC_WIDE : RegionCache::ALLOC_FIT, slabSize ), _nodeNumber( 0 ), _acceleratorNumber( 0 ), _isAccelerator( false ), _sdata( NULL ) {
}

void SeparateAddressSpace::copyOut( global_reg_t const &reg, unsigned int version, DeviceOps *ops, WD const &wd, unsigned int copyIdx, bool inval ) {
   _cache.NEWcopyOut( reg, version, wd, copyIdx, ops, inval );
}

void SeparateAddressSpace::doOp( SeparateMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *chunk, bool inval ) {
   _cache.NEWcopyIn( from._cache.getMemorySpaceId(), reg, version, wd, copyIdx, ops, chunk );
}

void SeparateAddressSpace::doOp( HostMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *chunk, bool inval ) {
   _cache.NEWcopyIn( 0, reg, version, wd, copyIdx, ops, chunk );
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

void SeparateAddressSpace::copyFromHost( TransferList &list, WD const &wd ) {
   for ( TransferList::const_iterator it = list.begin(); it != list.end(); it++ ) {
      this->doOp( sys.getHostMemory(), it->getRegion(), it->getVersion(), wd, it->getCopyIndex(), it->getDeviceOps(), it->getChunk(), false );
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

void SeparateAddressSpace::copyInputData( BaseAddressSpaceInOps &ops, global_reg_t const &reg, unsigned int version, NewLocationInfoList const &locations, enum RegionCache::CachePolicy policy, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx ) {
   _cache.copyInputData( ops, reg, version, locations, policy, chunk, wd, copyIdx );
}

void SeparateAddressSpace::copyOutputData( SeparateAddressSpaceOutOps &ops, global_reg_t const &reg, unsigned int version, bool output, enum RegionCache::CachePolicy policy, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx ) {
   _cache.copyOutputData( ops, reg, version, output, policy, chunk, wd, copyIdx );
}

void SeparateAddressSpace::allocateOutputMemory( global_reg_t const &reg, ProcessingElement *pe, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   _cache.allocateOutputMemory( reg, pe, version, wd, copyIdx );
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

void SeparateAddressSpace::setRegionVersion( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   _cache.setRegionVersion( reg, version, wd, copyIdx );
}

Device const &SeparateAddressSpace::getDevice() const {
   return _cache.getDevice();
}

}
