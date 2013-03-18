#include <iostream>
#include "memoryops_decl.hpp"
#include "system_decl.hpp"
#include "addressspace.hpp"
#include "workdescriptor.hpp"
#include "deviceops.hpp"

namespace nanos {

BaseAddressSpaceInOps::BaseAddressSpaceInOps() : _separateTransfers(), _ownDeviceOps(), _otherDeviceOps() {
}

BaseAddressSpaceInOps::~BaseAddressSpaceInOps() {
}

void BaseAddressSpaceInOps::addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version ) {
   TransferListType &list = _separateTransfers[ from ];
   list.push_back( std::make_pair( reg, version ) );
}

bool BaseAddressSpaceInOps::isDataReady() {
   bool allReady = true;
   //std::cerr << "Own Objects to wait: "; 
   //for ( std::set< DeviceOps * >::iterator pit = _ownDeviceOps.begin(); pit != _ownDeviceOps.end(); pit++ ) {
   //   std::cerr << " " << (void *) *pit;
   //}
   //std::cerr << std::endl;

   std::set< DeviceOps * >::iterator it = _ownDeviceOps.begin();
   while ( it != _ownDeviceOps.end() && allReady ) {
      //if ( *it == NULL || ( *it != NULL && (*it)->allCompleted() ) ) {
      if ( (*it)->allCompleted() ) {
         //(*it)->completeCacheOp();
         //std::set< DeviceOps * >::iterator toBeRemovedIt = it;
         it++;
         //_ownDeviceOps.erase( toBeRemovedIt );
      } else {
         allReady = false;
      }
   }
   // do it this way because there may be dependencies between operations,
   // by clearing all when all are completed any dependence will be satisfied.
   if ( allReady ) {
      for ( it = _ownDeviceOps.begin(); it != _ownDeviceOps.end(); it++ ) {
         (*it)->completeCacheOp();
      }
      _ownDeviceOps.clear();
   }
   if ( allReady ) {
      it = _otherDeviceOps.begin();
      while ( it != _otherDeviceOps.end() && allReady ) {
         if ( (*it)->allCacheOpsCompleted() ) {
            std::set< DeviceOps * >::iterator toBeRemovedIt = it;
         it++;
            _otherDeviceOps.erase( toBeRemovedIt );
         } else {
            allReady = false;
         }
      }
   }
   if ( allReady ) {
      for ( MapType::iterator mit = _separateTransfers.begin(); mit != _separateTransfers.end(); mit++ ) {
         for ( TransferListType::iterator lit = mit->second.begin(); lit != mit->second.end(); lit++ ) {
            mit->first->releaseForTransfer( lit->first, lit->second );
         }
      }
   }
   return allReady;
}

std::set< DeviceOps * > &BaseAddressSpaceInOps::getOwnOps() {
   return _ownDeviceOps;
}
std::set< DeviceOps * > &BaseAddressSpaceInOps::getOtherOps() {
   return _otherDeviceOps;
}

void BaseAddressSpaceInOps::addOpFromHost( global_reg_t const &reg, unsigned int version ) {
   std::cerr << "Error, can not send data from myself." << std::endl; 
}

void BaseAddressSpaceInOps::issue(WD const &wd) {
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
     sys.getHostMemory().copy( *(it->first) /* mem space */, it->second /* region */, wd );
   }
}


void BaseAddressSpaceInOps::prepareRegion( global_reg_t const &reg, WD const &wd ) {
}

void BaseAddressSpaceInOps::setRegionVersion( global_reg_t const &reg, unsigned int version ) {
   reg.setLocationAndVersion( 0, version );
}

unsigned int BaseAddressSpaceInOps::getVersionSetVersion( global_reg_t const &reg, unsigned int newVersion) {
   unsigned int current_version = reg.getHostVersion(false);
   reg.setLocationAndVersion( 0, newVersion );
   return current_version;
}

unsigned int BaseAddressSpaceInOps::getVersionNoLock( global_reg_t const &reg ) {
   return reg.getHostVersion(false);
}

void BaseAddressSpaceInOps::copyInputData( global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations ) {
   std::cerr << "FIXME "<< __FUNCTION__ << std::endl;
   DeviceOps *thisRegOps = NULL;
   if ( reg.getHostVersion( false ) != version ) {
      thisRegOps = reg.getDeviceOps();
      if ( thisRegOps->addCacheOp() ) {
         _ownDeviceOps.insert( thisRegOps );
         for ( NewLocationInfoList::const_iterator it = locations.begin(); it != locations.end(); it++ ) {
            global_reg_t region_shape( it->first, reg.key );
            global_reg_t data_source( it->second, reg.key );
            memory_space_id_t location = data_source.getFirstLocation();
            if ( location != 0 ) {
               if ( region_shape.id != reg.id ) {
                  DeviceOps *thisOps = region_shape.getDeviceOps();
                  thisOps->addCacheOp();
                  _ownDeviceOps.insert( thisOps );
               }
               std::cerr << "HOST mustcopy: reg " << reg.id << " version " << version << "  region shape: " << region_shape.id << " data source: " << data_source.id << " location "<< location << std::endl;
               ensure( location > 0, "Wrong location.");
               addOp( &( sys.getSeparateMemory( location ) ), region_shape, version );
            }
         }
      }
   }
   reg.setLocationAndVersion( 0, version + ( output ? 1 : 0 ) );
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
   _hostTransfers.push_back( std::make_pair( reg, version ) );
}

void SeparateAddressSpaceInOps::issue( WD const &wd ) {
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
      _destination.copy( *(it->first) /* mem space */, it->second /* list of regions */, wd );
   }
   _destination.copyFromHost( _hostTransfers, wd );
}

void SeparateAddressSpaceInOps::prepareRegion( global_reg_t const &reg, WD const &wd ) {
   _destination.prepareRegion( reg, wd );
}

void SeparateAddressSpaceInOps::setRegionVersion( global_reg_t const &reg, unsigned int version ) {
   _destination.setRegionVersion( reg, version );
}

unsigned int SeparateAddressSpaceInOps::getVersionSetVersion( global_reg_t const &reg, unsigned int newVersion  ) {
   return _destination.getCurrentVersionSetVersion( reg, newVersion );
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


SeparateAddressSpaceOutOps::SeparateAddressSpaceOutOps( SeparateMemoryAddressSpace &source ) : _source ( source ){
}

void SeparateAddressSpaceOutOps::issue( WD const &wd, MemCacheCopy *memCacheCopies ) {
   //do copies back to memory

   for ( unsigned int index = 0; index < wd.getNumCopies(); index++ ) {
      _source.releaseRegion( memCacheCopies[ index ]._reg ) ;
   }
}

}
