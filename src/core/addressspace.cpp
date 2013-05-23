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
MemSpace< SeparateAddressSpace >::MemSpace( memory_space_id_t memSpaceId, Device &d ) : SeparateAddressSpace( memSpaceId, d ) {
}

HostAddressSpace::HostAddressSpace( Device &d ) : _directory() {
}

bool HostAddressSpace::lockForTransfer( global_reg_t const &reg, unsigned int version ) {
   return true;
}

void HostAddressSpace::releaseForTransfer( global_reg_t const &reg, unsigned int version ) {
}

void HostAddressSpace::doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const &wd, DeviceOps *ops ) {
   if ( reg.setCopying( from ) ) {
     from.copyOut( reg, version, ops, wd );
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
   reg.key = _directory.getRegionDirectoryKeyRegisterIfNeeded( cd );
   reg.id = reg.key->obtainRegionId( cd );
}

void HostAddressSpace::failToLock( SeparateMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version ) {
   std::cerr << "unimplemented" << std::endl;
}

void HostAddressSpace::synchronize( bool flushData ) {
   _directory.synchronize2( flushData );
}

memory_space_id_t HostAddressSpace::getMemorySpaceId() const {
   return 0;
}

SeparateAddressSpace::SeparateAddressSpace( memory_space_id_t memorySpaceId, Device &arch ) : _cache( memorySpaceId, arch, RegionCache::ALLOC_FIT ), _nodeNumber( 0 )  {
}


bool SeparateAddressSpace::lockForTransfer( global_reg_t const &reg, unsigned int version ) {
   return _cache.pin( reg );
}

void SeparateAddressSpace::releaseForTransfer( global_reg_t const &reg, unsigned int version ) {
   _cache.unpin( reg );
}

void SeparateAddressSpace::copyOut( global_reg_t const &reg, unsigned int version, DeviceOps *ops, WD const &wd ) {
   _cache.NEWcopyOut( reg, version, wd, ops );
}

void SeparateAddressSpace::doOp( SeparateMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version, WD const &wd, DeviceOps *ops ) {
   _cache.NEWcopyIn( from._cache.getMemorySpaceId(), reg, version, wd, ops );
}

void SeparateAddressSpace::doOp( HostMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version, WD const &wd, DeviceOps *ops ) {
   _cache.NEWcopyIn( 0, reg, version, wd, ops );
}

void SeparateAddressSpace::failToLock( SeparateMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version ) {
   std::cerr << "unimplemented" << std::endl;
}

void SeparateAddressSpace::failToLock( HostMemoryAddressSpace &from, global_reg_t const &reg, unsigned int version ) {
   std::cerr << "unimplemented" << std::endl;
}

bool SeparateAddressSpace::prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   return _cache.prepareRegions( memCopies, numCopies, wd );
}

//void SeparateAddressSpace::prepareRegion( global_reg_t const &reg, WD const &wd ) {
//   _cache.prepareRegion( reg, wd );
//}

unsigned int SeparateAddressSpace::getCurrentVersion( global_reg_t const &reg ) {
   return _cache.getVersion( reg );
}

void SeparateAddressSpace::releaseRegion( global_reg_t const &reg, WD const &wd ) {
   _cache.releaseRegion( reg, wd );
}

void SeparateAddressSpace::copyFromHost( TransferList list, WD const &wd ) {
   for ( TransferList::const_iterator it = list.begin(); it != list.end(); it++ ) {
      if ( sys.getHostMemory().lockForTransfer( it->getRegion(), it->getVersion() ) ) {
         this->doOp( sys.getHostMemory(), it->getRegion(), it->getVersion(), wd, it->getDeviceOps() );
         sys.getHostMemory().releaseForTransfer( it->getRegion(), it->getVersion() );
      } else {
         this->failToLock( sys.getHostMemory(), it->getRegion(), it->getVersion() );
      }
   }
}

uint64_t SeparateAddressSpace::getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress ) const {
   return _cache.getDeviceAddress( reg, baseAddress );
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

void SeparateAddressSpace::copyInputData( BaseAddressSpaceInOps &ops, global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations ) {
   _cache.copyInputData( ops, reg, version, output, locations );
}

void SeparateAddressSpace::allocateOutputMemory( global_reg_t const &reg, unsigned int version ) {
   _cache.allocateOutputMemory( reg, version );
}

RegionCache &SeparateAddressSpace::getCache() {
   return _cache;
}

ProcessingElement &SeparateAddressSpace::getPE() {
   memory_space_id_t id = _cache.getMemorySpaceId();
   return sys.getPEWithMemorySpaceId( id );
}

memory_space_id_t SeparateAddressSpace::getMemorySpaceId() const {
   return _cache.getMemorySpaceId();
}

unsigned int SeparateAddressSpace::getInvalidationCount() const {
   return _cache.getInvalidationCount();
}

bool SeparateAddressSpace::canAllocateMemory( MemCacheCopy *memCopies, unsigned int numCopies, bool considerInvalidations ) {
   return _cache.canAllocateMemory( memCopies, numCopies, considerInvalidations );
}


}
