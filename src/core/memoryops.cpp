#include <iostream>
#include "memoryops_decl.hpp"
#include "system_decl.hpp"
#include "addressspace.hpp"
#include "workdescriptor.hpp"
#include "deviceops.hpp"
#include "regiondict.hpp"

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

BaseOps::BaseOps() : _ownDeviceOps(), _otherDeviceOps() {
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
         it->commitMetadata();
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
   _ownDeviceOps.insert( OwnOp( ops, reg, version, location ) );
}

BaseAddressSpaceInOps::BaseAddressSpaceInOps() : BaseOps() , _separateTransfers() {
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

void BaseAddressSpaceInOps::copyInputData( global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations ) {

   std::set< DeviceOps * > ops;
   ops.insert( reg.getDeviceOps() );

   for ( NewLocationInfoList::const_iterator it = locations.begin(); it != locations.end(); it++ ) {
      global_reg_t data_source( it->second, reg.key );
      ops.insert( data_source.getDeviceOps() );
   }
 
   reg.key->invalLock();
   for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
      (*opIt)->syncAndDisableInvalidations();
   }
   reg.key->invalUnlock();

   DeviceOps *thisRegOps = reg.getDeviceOps();
   if ( reg.getHostVersion( false ) != version ) {
      if ( thisRegOps->addCacheOp() ) {
         insertOwnOp( thisRegOps, reg, version, 0 );
         for ( NewLocationInfoList::const_iterator it = locations.begin(); it != locations.end(); it++ ) {
            global_reg_t region_shape( it->first, reg.key );
            global_reg_t data_source( it->second, reg.key );
            memory_space_id_t location = data_source.getFirstLocation();
            if ( location != 0 ) {
               if ( region_shape.id != reg.id ) {
                  DeviceOps *thisOps = region_shape.getDeviceOps();
                  if ( thisOps->addCacheOp() ) {
                     insertOwnOp( thisOps, region_shape, version, 0 );
                  } else {
                     std::cerr << "ERROR, could not add a cache op for a chunk!" << std::endl;
                  }
               }
               //std::cerr << "HOST mustcopy: reg " << reg.id << " version " << version << "  region shape: " << region_shape.id << " data source: " << data_source.id << " location "<< location << std::endl;
               ensure( location > 0, "Wrong location.");
               addOp( &( sys.getSeparateMemory( location ) ), region_shape, version );
            }
         }
      } else {
         getOtherOps().insert( thisRegOps );
      }
   } else {
      getOtherOps().insert( thisRegOps );
   }

   for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
      (*opIt)->resumeInvalidations();
   }
}

void BaseAddressSpaceInOps::allocateOutputMemory( global_reg_t const &reg, unsigned int version ) {
   //std::cerr << "FIXME "<< __FUNCTION__ << std::endl;
   reg.setLocationAndVersion( 0, version );
}

SeparateAddressSpaceInOps::SeparateAddressSpaceInOps( MemSpace<SeparateAddressSpace> &destination ) : BaseAddressSpaceInOps(), _destination( destination ), _hostTransfers() {
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


SeparateAddressSpaceOutOps::SeparateAddressSpaceOutOps() : BaseOps(), _transfers() {
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
