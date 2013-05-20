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

bool BaseOps::isDataReady() {
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
         it->_ops->completeCacheOp();
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

void BaseAddressSpaceInOps::addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version ) {
   TransferList &list = _separateTransfers[ from ];
   list.push_back( TransferListEntry( reg, version, NULL ) );
}

void BaseAddressSpaceInOps::addOpFromHost( global_reg_t const &reg, unsigned int version ) {
   std::cerr << "Error, can not send data from myself." << std::endl; 
}

void BaseAddressSpaceInOps::issue( WD const &wd ) {
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
     sys.getHostMemory().copy( *(it->first) /* mem space */, it->second /* regions */, wd );
   }
}

void BaseAddressSpaceInOps::prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
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
      (*it)->unlock();
   }
   _lockedChunks.clear();
}

void BaseAddressSpaceInOps::copyInputData( global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations ) {

   //std::set< DeviceOps * > ops;
   //ops.insert( reg.getDeviceOps() );

   //for ( NewLocationInfoList::const_iterator it = locations.begin(); it != locations.end(); it++ ) {
   //   global_reg_t data_source( it->second, reg.key );
   //   ops.insert( data_source.getDeviceOps() );
   //}
 
   //for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
   //   (*opIt)->syncAndDisableInvalidations();
   //}

   lockSourceChunks( reg, version, locations, 0 );

   DeviceOps *thisRegOps = reg.getDeviceOps();
   if ( reg.getHostVersion( false ) != version ) {
      if ( VERBOSE_CACHE ) { std::cerr << "I have to copy region " << reg.id << " dont have it "<<std::endl; }
      if ( thisRegOps->addCacheOp() ) {
         insertOwnOp( thisRegOps, reg, version, 0 ); //i've got the responsability of copying this region

         if ( locations.size() == 1 ) {
            global_reg_t region_shape( locations.begin()->first, reg.key );
            global_reg_t data_source( locations.begin()->second, reg.key );
            ensure( region_shape.id == reg.id, "Wrong region" );
            if ( !data_source.isLocatedIn( 0 ) ) {
               memory_space_id_t location = data_source.getFirstLocation();
               ensure( location > 0, "Wrong location.");
               addOp( &( sys.getSeparateMemory( location ) ), region_shape, version );
            } else {
               //memory_space_id_t location = data_source.getFirstLocation();
               //std::cerr << "This should not happen, reg " << data_source.id << " reported to be in 0 (first loc " << location << " ) shape is " << region_shape.id << std::endl;
               //fatal("Impossible path!");
               getOtherOps().insert( data_source.getDeviceOps() );
            }
         } else {
            for ( NewLocationInfoList::const_iterator it = locations.begin(); it != locations.end(); it++ ) {
               global_reg_t region_shape( it->first, reg.key );
               global_reg_t data_source( it->second, reg.key );
               ensure( region_shape.id != reg.id, "Wrong region" );
               //if ( region_shape.id == data_source.id ) {
                  if ( !data_source.isLocatedIn( 0 ) ) {
                     memory_space_id_t location = data_source.getFirstLocation();
                     DeviceOps *thisOps = region_shape.getDeviceOps(); //FIXME: we assume that region_shape has a directory entry, it may be a wrong assumption
                     if ( thisOps->addCacheOp() ) {
                        insertOwnOp( thisOps, region_shape, version, 0 );
                     } else {
                        std::cerr << "ERROR, could not add a cache op for a chunk!" << std::endl;
                     }
                     if ( VERBOSE_CACHE ) { std::cerr << " added a op! ds= " << it->second << " rs= " << it->first <<std::endl; }
                     addOp( &( sys.getSeparateMemory( location ) ), region_shape, version );
                  } else {
                     if ( VERBOSE_CACHE ) { std::cerr << " sync with other op! ds= " << it->second << " rs= " << it->first <<std::endl; }
                     getOtherOps().insert( data_source.getDeviceOps() );
                  }
               //} else {
               //}
            }
         }
      } else {
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

void SeparateAddressSpaceInOps::addOpFromHost( global_reg_t const &reg, unsigned int version ) {
   _hostTransfers.push_back( TransferListEntry( reg, version, NULL ) );
}

void SeparateAddressSpaceInOps::issue( WD const &wd ) {
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
      _destination.copy( *(it->first) /* mem space */, it->second /* list of regions */, wd );
   }
   _destination.copyFromHost( _hostTransfers, wd );
}

void SeparateAddressSpaceInOps::prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   _destination.prepareRegions( memCopies, numCopies, wd );
}

unsigned int SeparateAddressSpaceInOps::getVersionNoLock( global_reg_t const &reg ) {
   return _destination.getCurrentVersion( reg );
}

void SeparateAddressSpaceInOps::copyInputData( global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations ) {
   _destination.copyInputData( *this, reg, version, output, locations );
}

void SeparateAddressSpaceInOps::allocateOutputMemory( global_reg_t const &reg, unsigned int version ) {
   _destination.allocateOutputMemory( reg, version );
}


SeparateAddressSpaceOutOps::SeparateAddressSpaceOutOps( bool delayedCommit ) : BaseOps( delayedCommit ), _transfers() {
}

SeparateAddressSpaceOutOps::~SeparateAddressSpaceOutOps() {
}

void SeparateAddressSpaceOutOps::addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version, DeviceOps *ops ) {
   TransferList &list = _transfers[ from ];
   list.push_back( TransferListEntry( reg, version, ops ) );
}

void SeparateAddressSpaceOutOps::issue( WD const &wd ) {
   for ( MapType::iterator it = _transfers.begin(); it != _transfers.end(); it++ ) {
     sys.getHostMemory().copy( *(it->first) /* mem space */, it->second /* region */, wd );
   }
}

}
