#include <iostream>
#include "memoryops_decl.hpp"
#include "system_decl.hpp"
#include "addressspace.hpp"
#include "workdescriptor.hpp"
#include "deviceops.hpp"
#include "regiondict.hpp"
#include "regioncache.hpp"

namespace nanos {

BaseOps::OwnOp::OwnOp( DeviceOps *ops, global_reg_t reg, unsigned int version, memory_space_id_t location ) :
   _ops( ops ), _reg( reg ), _version( version ), _location( location ) {
}

BaseOps::OwnOp::OwnOp( BaseOps::OwnOp const &op ) :
   _ops( op._ops ), _reg( op._reg ), _version( op._version ), _location( op._location ) {
}

BaseOps::OwnOp &BaseOps::OwnOp::operator=( BaseOps::OwnOp const &op ) {
   _ops = op._ops;
   _reg = op._reg;
   _version = op._version;
   _location = op._location;
   return *this;
}

void BaseOps::OwnOp::commitMetadata() const {
   _reg.setLocationAndVersion( _location, _version );
}

BaseOps::BaseOps( bool delayedCommit ) : _delayedCommit( delayedCommit ), _ownDeviceOps(), _otherDeviceOps() {
}

BaseOps::~BaseOps() {
}

bool BaseOps::isDataReady( WD const &wd ) {
   bool allReady = true;

   std::set< OwnOp >::iterator it = _ownDeviceOps.begin();
   while ( it != _ownDeviceOps.end() && allReady ) {
      if ( it->_ops->allCompleted() ) {
         it++;
      } else {
         allReady = false;
      }
   }
   // do it this way because there may be dependencies between operations,
   // by clearing all when all are completed any dependence will be satisfied.
   if ( allReady ) {
      for ( it = _ownDeviceOps.begin(); it != _ownDeviceOps.end(); it++ ) {
         it->_ops->completeCacheOp( wd.getId() );
         if ( _delayedCommit ) { 
            it->commitMetadata();
         }
      }
      _ownDeviceOps.clear();
   }
   if ( allReady ) {
      std::set< DeviceOps * >::iterator otherIt = _otherDeviceOps.begin();
      while ( otherIt != _otherDeviceOps.end() && allReady ) {
         if ( (*otherIt)->allCacheOpsCompleted() ) {
            std::set< DeviceOps * >::iterator toBeRemovedIt = otherIt;
            otherIt++;
            _otherDeviceOps.erase( toBeRemovedIt );
         } else {
            allReady = false;
         }
      }
   }
   return allReady;
}

std::set< DeviceOps * > &BaseOps::getOtherOps() {
   return _otherDeviceOps;
}

std::set< BaseOps::OwnOp > &BaseOps::getOwnOps() {
   return _ownDeviceOps;
}

void BaseOps::insertOwnOp( DeviceOps *ops, global_reg_t reg, unsigned int version, memory_space_id_t location ) {
   OwnOp op( ops, reg, version, location );
   _ownDeviceOps.insert( op );
   if ( !_delayedCommit ) {
      op.commitMetadata();
   }
}

BaseAddressSpaceInOps::BaseAddressSpaceInOps( bool delayedCommit ) : BaseOps( delayedCommit ) , _separateTransfers() {
}

BaseAddressSpaceInOps::~BaseAddressSpaceInOps() {
}

void BaseAddressSpaceInOps::addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version, AllocatedChunk *chunk ) {
   TransferList &list = _separateTransfers[ from ];
   list.push_back( TransferListEntry( reg, version, NULL, chunk ) );
}

void BaseAddressSpaceInOps::addOpFromHost( global_reg_t const &reg, unsigned int version, AllocatedChunk *chunk ) {
   std::cerr << "Error, can not send data from myself." << std::endl; 
}

void BaseAddressSpaceInOps::issue( WD const &wd ) {
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
     sys.getHostMemory().copy( *(it->first) /* mem space */, it->second /* regions */, wd );
   }
}

bool BaseAddressSpaceInOps::prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   return true;
}

unsigned int BaseAddressSpaceInOps::getVersionNoLock( global_reg_t const &reg ) {
   return reg.getHostVersion(false);
}

void BaseAddressSpaceInOps::lockSourceChunks( global_reg_t const &reg, unsigned int version, NewLocationInfoList const &locations, memory_space_id_t thisLocation ) {
   std::map< memory_space_id_t, std::set< global_reg_t > > parts;
   for ( NewLocationInfoList::const_iterator it = locations.begin(); it != locations.end(); it++ ) {
      global_reg_t region_shape( it->first, reg.key );
      global_reg_t data_source( it->second, reg.key );
      //if ( !data_source.isLocatedIn( thisLocation ) ) {
         memory_space_id_t location = data_source.getFirstLocation();
         if ( location != thisLocation ) {
            parts[ location ].insert( data_source );
         }
      //}
   }
   if ( VERBOSE_CACHE ) { std::cerr << "avoiding... process region " << reg.id << " got locked chunks: " << std::endl; }
   for ( std::map< memory_space_id_t, std::set< global_reg_t > >::iterator mIt = parts.begin(); mIt != parts.end(); mIt++ ) {
      if ( VERBOSE_CACHE ) { std::cerr << " from location " << mIt->first << std::endl; }
      sys.getSeparateMemory( mIt->first ).getCache().prepareRegionsToCopyToHost( mIt->second, version, _lockedChunks );
   }
   if ( VERBOSE_CACHE ) {
      std::cerr << "safe from invalidations... process region " << reg.id << " got locked chunks: ";
      for ( std::set< AllocatedChunk * >::iterator it = _lockedChunks.begin(); it != _lockedChunks.end(); it++ ) {
         std::cerr << " " << *it;
      }
      std::cerr << std::endl;
   }
}

void BaseAddressSpaceInOps::releaseLockedSourceChunks() {
   for ( std::set< AllocatedChunk * >::iterator it = _lockedChunks.begin(); it != _lockedChunks.end(); it++ ) {
      //(*it)->unlock();
      (*it)->removeReference();
   }
   _lockedChunks.clear();
}

void BaseAddressSpaceInOps::copyInputData( MemCacheCopy const &memCopy, bool output, WD const &wd ) {

   //std::set< DeviceOps * > ops;
   //ops.insert( reg.getDeviceOps() );

   //for ( NewLocationInfoList::const_iterator it = locations.begin(); it != locations.end(); it++ ) {
   //   global_reg_t data_source( it->second, reg.key );
   //   ops.insert( data_source.getDeviceOps() );
   //}
 
   //for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
   //   (*opIt)->syncAndDisableInvalidations();
   //}

   lockSourceChunks( memCopy._reg, memCopy._version, memCopy._locations, 0 );

   DeviceOps *thisRegOps = memCopy._reg.getDeviceOps();
   if ( memCopy._reg.getHostVersion( false ) != memCopy._version ) {
      if ( VERBOSE_CACHE ) { std::cerr << "I have to copy region " << memCopy._reg.id << " dont have it "<<std::endl; }
      if ( thisRegOps->addCacheOp( wd.getId() ) ) {
         if ( VERBOSE_CACHE ) { std::cerr << "I will do the transfer for reg " << memCopy._reg.id << " dont have it "<<std::endl; }
         insertOwnOp( thisRegOps, memCopy._reg, memCopy._version, 0 ); //i've got the responsability of copying this region

         if ( memCopy._locations.size() == 1 ) {
            global_reg_t region_shape( memCopy._locations.begin()->first, memCopy._reg.key );
            global_reg_t data_source( memCopy._locations.begin()->second, memCopy._reg.key );
            ensure( region_shape.id == memCopy._reg.id, "Wrong region" );
            if ( !data_source.isLocatedIn( 0 ) ) {
               memory_space_id_t location = data_source.getFirstLocation();
               ensure( location > 0, "Wrong location.");
               this->addOp( &( sys.getSeparateMemory( location ) ), region_shape, memCopy._version, NULL );
            } else {
               //memory_space_id_t location = data_source.getFirstLocation();
               //std::cerr << "This should not happen, reg " << data_source.id << " reported to be in 0 (first loc " << location << " ) shape is " << region_shape.id << std::endl;
               //fatal("Impossible path!");
               getOtherOps().insert( data_source.getDeviceOps() );
            }
         } else {
            for ( NewLocationInfoList::const_iterator it = memCopy._locations.begin(); it != memCopy._locations.end(); it++ ) {
               global_reg_t region_shape( it->first, memCopy._reg.key );
               global_reg_t data_source( it->second, memCopy._reg.key );
               ensure( region_shape.id != memCopy._reg.id, "Wrong region" );
               //if ( region_shape.id == data_source.id ) {
                  if ( !data_source.isLocatedIn( 0 ) ) {
                     memory_space_id_t location = data_source.getFirstLocation();
                     DeviceOps *thisOps = region_shape.getDeviceOps(); //FIXME: we assume that region_shape has a directory entry, it may be a wrong assumption
                     int added = 0;
                     if ( thisOps->addCacheOp( wd.getId() ) ) {
                        insertOwnOp( thisOps, region_shape, memCopy._version, 0 );
                        added = 1;
                     } else {
                        std::cerr << "ERROR, could not add a cache op for a chunk!" << std::endl;
                     }
                     if ( VERBOSE_CACHE ) { std::cerr << " added a op! ds= " << it->second << " rs= " << it->first << " added= " << added << " so far we have ops: " << getOwnOps().size() << " this Obj "<< (void *) this << std::endl; }
                     this->addOp( &( sys.getSeparateMemory( location ) ), region_shape, memCopy._version, NULL );
                  } else {
                     if ( VERBOSE_CACHE ) { std::cerr << " sync with other op! ds= " << it->second << " rs= " << it->first <<std::endl; }
                     getOtherOps().insert( data_source.getDeviceOps() );
                  }
               //} else {
               //}
            }
         }
      } else {
         if ( VERBOSE_CACHE ) { std::cerr << "I will not do the transfer for reg " << memCopy._reg.id << " dont have it "<<std::endl; }
         getOtherOps().insert( thisRegOps );
      }
   } else {
      getOtherOps().insert( thisRegOps );
   }

   //for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
   //   (*opIt)->resumeInvalidations();
   //}
}

void BaseAddressSpaceInOps::allocateOutputMemory( global_reg_t const &reg, unsigned int version ) {
   //std::cerr << "FIXME "<< __FUNCTION__ << std::endl;
   reg.setLocationAndVersion( 0, version );
}

SeparateAddressSpaceInOps::SeparateAddressSpaceInOps( bool delayedCommit, MemSpace<SeparateAddressSpace> &destination ) : BaseAddressSpaceInOps( delayedCommit ), _destination( destination ), _hostTransfers() {
}

SeparateAddressSpaceInOps::~SeparateAddressSpaceInOps() {
}

void SeparateAddressSpaceInOps::addOpFromHost( global_reg_t const &reg, unsigned int version, AllocatedChunk *chunk ) {
   _hostTransfers.push_back( TransferListEntry( reg, version, NULL, chunk ) );
}

void SeparateAddressSpaceInOps::issue( WD const &wd ) {
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
      _destination.copy( *(it->first) /* mem space */, it->second /* list of regions */, wd );
   }
   _destination.copyFromHost( _hostTransfers, wd );
}

bool SeparateAddressSpaceInOps::prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   return _destination.prepareRegions( memCopies, numCopies, wd );
}

unsigned int SeparateAddressSpaceInOps::getVersionNoLock( global_reg_t const &reg ) {
   return _destination.getCurrentVersion( reg );
}

void SeparateAddressSpaceInOps::copyInputData( MemCacheCopy const& memCopy, bool output, WD const &wd ) {
   _destination.copyInputData( *this, memCopy._reg, memCopy._version, output, memCopy._locations, memCopy._chunk, wd );
}

void SeparateAddressSpaceInOps::allocateOutputMemory( global_reg_t const &reg, unsigned int version ) {
   _destination.allocateOutputMemory( reg, version );
}


SeparateAddressSpaceOutOps::SeparateAddressSpaceOutOps( bool delayedCommit ) : BaseOps( delayedCommit ), _transfers() {
}

SeparateAddressSpaceOutOps::~SeparateAddressSpaceOutOps() {
}

void SeparateAddressSpaceOutOps::addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version, DeviceOps *ops, AllocatedChunk *chunk ) {
   TransferList &list = _transfers[ from ];
   list.push_back( TransferListEntry( reg, version, ops, chunk ) );
}

void SeparateAddressSpaceOutOps::issue( WD const &wd ) {
   for ( MapType::iterator it = _transfers.begin(); it != _transfers.end(); it++ ) {
     sys.getHostMemory().copy( *(it->first) /* mem space */, it->second /* region */, wd );
   }
}

}
