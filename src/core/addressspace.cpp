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
MemSpace< SeparateAddressSpace >::MemSpace( memory_space_id_t memSpaceId, Device &d, bool allocWide ) : SeparateAddressSpace( memSpaceId, d, allocWide ) {
}

HostAddressSpace::HostAddressSpace( Device &d ) : _directory() {
}

bool HostAddressSpace::lockForTransfer( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   return true;
}

void HostAddressSpace::releaseForTransfer( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
}

void HostAddressSpace::doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *chunk, bool inval ) {
   if ( reg.setCopying( from ) ) {
     from.copyOut( reg, version, ops, wd, copyIdx, inval );
   } else {
     reg.waitCopy();
   } 
}

void HostAddressSpace::getVersionInfo( global_reg_t const &reg, unsigned int &version, NewLocationInfoList &locations ) {
   do {
     NewNewRegionDirectory::tryGetLocation( reg.key, reg.id, locations, version, *((WD*)NULL) );
   } while ( version == 0 ); 
   reg.initializeGlobalEntryIfNeeded();
}

void HostAddressSpace::getRegionId( CopyData const &cd, global_reg_t &reg ) {
   //std::cerr << "Registering CD with addr " << (void *) cd.getBaseAddress() << std::endl;
   //std::cerr << cd << std::endl;
   reg.key = _directory.getRegionDirectoryKeyRegisterIfNeeded( cd );
   reg.id = reg.key->obtainRegionId( cd );
   //std::cerr << "Got key " << (void *)reg.key << " got id " << (int)reg.id << std::endl;
   reg_t master_id = cd.getHostRegionId();
   if ( master_id != 0 ) {
      NewNewRegionDirectory::addMasterRegionId( reg.key, master_id, reg.id );
   }
}

void HostAddressSpace::failToLock( SeparateMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version ) {
   std::cerr << __FUNCTION__ << " @ " << __FILE__ << " : " << __LINE__ << " unimplemented" << std::endl;
}

void HostAddressSpace::synchronize( WD const &wd ) {
   _directory.synchronize( wd );
}

memory_space_id_t HostAddressSpace::getMemorySpaceId() const {
   return 0;
}

NewNewRegionDirectory::RegionDirectoryKey HostAddressSpace::getRegionDirectoryKey( uint64_t addr ) const {
   return _directory.getRegionDirectoryKey( addr );
}

reg_t HostAddressSpace::getLocalRegionId( void *hostObject, reg_t hostRegionId ) const {
   return _directory.getLocalRegionId( hostObject, hostRegionId );
}

SeparateAddressSpace::SeparateAddressSpace( memory_space_id_t memorySpaceId, Device &arch, bool allocWide ) : _cache( memorySpaceId, arch, allocWide ? RegionCache::ALLOC_WIDE : RegionCache::ALLOC_FIT ), _nodeNumber( 0 )  {
}


bool SeparateAddressSpace::lockForTransfer( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   return _cache.pin( reg, wd, copyIdx );
}

void SeparateAddressSpace::releaseForTransfer( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   _cache.unpin( reg, wd, copyIdx );
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

void SeparateAddressSpace::releaseRegion( global_reg_t const &reg, WD const &wd, unsigned int copyIdx, enum RegionCache::CachePolicy policy ) {
   _cache.releaseRegion( reg, wd, copyIdx, policy );
}

void SeparateAddressSpace::copyFromHost( TransferList &list, WD const &wd ) {
   for ( TransferList::const_iterator it = list.begin(); it != list.end(); it++ ) {
      if ( sys.getHostMemory().lockForTransfer( it->getRegion(), it->getVersion(), wd, it->getCopyIndex() ) ) {
         this->doOp( sys.getHostMemory(), it->getRegion(), it->getVersion(), wd, it->getCopyIndex(), it->getDeviceOps(), it->getChunk(), false );
         sys.getHostMemory().releaseForTransfer( it->getRegion(), it->getVersion(), wd, it->getCopyIndex() );
      } else {
         this->failToLock( sys.getHostMemory(), it->getRegion(), it->getVersion() );
      }
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

void *SeparateAddressSpace::getSpecificData() const {
   return _sdata;
}

void SeparateAddressSpace::setSpecificData( void *data ) {
   _sdata = data;
}

void SeparateAddressSpace::copyInputData( BaseAddressSpaceInOps &ops, global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx ) {
   _cache.copyInputData( ops, reg, version, output, locations, chunk, wd, copyIdx );
}

void SeparateAddressSpace::copyOutputData( SeparateAddressSpaceOutOps &ops, global_reg_t const &reg, unsigned int version, bool output, enum RegionCache::CachePolicy policy, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx ) {
   _cache.copyOutputData( ops, reg, version, output, policy, chunk, wd, copyIdx );
}

void SeparateAddressSpace::allocateOutputMemory( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   _cache.allocateOutputMemory( reg, version, wd, copyIdx );
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

}
