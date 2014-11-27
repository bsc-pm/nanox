#include <iostream>
#include "memoryops_decl.hpp"
#include "system_decl.hpp"
#include "addressspace.hpp"
#include "workdescriptor.hpp"
#include "deviceops.hpp"
#include "regiondict.hpp"
#include "regioncache.hpp"

#if VERBOSE_CACHE
 #define _VERBOSE_CACHE 1
#else
 #define _VERBOSE_CACHE 0
#endif

namespace nanos {

BaseOps::OwnOp::OwnOp( DeviceOps *ops, global_reg_t reg, unsigned int version, memory_space_id_t loc ) :
   _ops( ops ), _reg( reg ), _version( version ), _location( loc ) {
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

void BaseOps::OwnOp::commitMetadata( ProcessingElement *pe ) const {
   _reg.setLocationAndVersion( pe, _location, _version );
}

BaseOps::BaseOps( ProcessingElement *pe, bool delayedCommit ) : _delayedCommit( delayedCommit )
   , _dataReady( false )
   , _pe( pe )
   , _ownDeviceOps()
   , _otherDeviceOps()
   , _amountOfTransferredData( 0 )
{
}

BaseOps::~BaseOps() {
}

bool BaseOps::isDataReady( WD const &wd, bool inval ) {
   if ( !_dataReady ) {
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
         //if (!inval) {
         //   *(myThread->_file) << "######################## COMPLETED OPS FOR WD " << wd.getId() << " ownOps is "<< _ownDeviceOps.size() << std::endl;
         //}
         for ( it = _ownDeviceOps.begin(); it != _ownDeviceOps.end(); it++ ) {
            it->_ops->completeCacheOp( /* debug: */ &wd );
            if ( _delayedCommit ) { 
               it->commitMetadata( _pe );
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
      if ( allReady ) {
         releaseLockedSourceChunks( wd );
      }
      _dataReady = allReady;
   }
   return _dataReady;
}

std::set< DeviceOps * > &BaseOps::getOtherOps() {
   return _otherDeviceOps;
}

std::set< BaseOps::OwnOp > &BaseOps::getOwnOps() {
   return _ownDeviceOps;
}

void BaseOps::insertOwnOp( DeviceOps *ops, global_reg_t reg, unsigned int version, memory_space_id_t loc ) {
   OwnOp op( ops, reg, version, loc );
   _ownDeviceOps.insert( op );
   if ( !_delayedCommit ) {
      op.commitMetadata( _pe );
   }
}

std::size_t BaseOps::getAmountOfTransferredData() const {
   return _amountOfTransferredData;
}

ProcessingElement *BaseOps::getPE() const {
   return _pe;
}

void BaseOps::addAmountTransferredData( std::size_t amount ) {
   _amountOfTransferredData += amount;
}

BaseAddressSpaceInOps::BaseAddressSpaceInOps( ProcessingElement *pe, bool delayedCommit ) : BaseOps( pe, delayedCommit )
   , _separateTransfers() {
}

BaseAddressSpaceInOps::~BaseAddressSpaceInOps() {
}

void BaseAddressSpaceInOps::addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version, AllocatedChunk *chunk, unsigned int copyIdx ) {
   TransferList &list = _separateTransfers[ from ];
   addAmountTransferredData( reg.getDataSize() );
   list.push_back( TransferListEntry( reg, version, NULL, chunk, copyIdx ) );
}

void BaseAddressSpaceInOps::addOpFromHost( global_reg_t const &reg, unsigned int version, AllocatedChunk *chunk, unsigned int copyIdx ) {
   std::cerr << "Error, can not send data from myself." << std::endl; 
}

void BaseAddressSpaceInOps::issue( WD const &wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_MEM_TRANSFER_ISSUE); );
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
     sys.getHostMemory().copy( *(it->first) /* mem space */, it->second /* regions */, wd );
   }
}

unsigned int BaseAddressSpaceInOps::getVersionNoLock( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   return reg.getHostVersion(false);
}

void BaseAddressSpaceInOps::lockSourceChunks( global_reg_t const &reg, unsigned int version, NewLocationInfoList const &locations, memory_space_id_t thisLocation, WD const &wd, unsigned int copyIdx ) {
   // FIXME: This should be more resillient code, first decide where to copy from, then try to lock the chunk (using .prepareRegionsToBeCopied) and if it fails then try wth a new location.
   // this could happen if an invalidation occurs and the data is only available on RegionCaches. Maybe we have read data_source.getFirstLocation() but the invalidation occured in between.
   std::map< memory_space_id_t, std::set< global_reg_t > > parts;
   for ( NewLocationInfoList::const_iterator it = locations.begin(); it != locations.end(); it++ ) {
      global_reg_t region_shape( it->first, reg.key );
      global_reg_t data_source( it->second, reg.key );
      if ( data_source.getFirstLocation() != 0 ) {
         memory_space_id_t location = data_source.getFirstLocation();
         if ( location != thisLocation ) {
            parts[ location ].insert( data_source );
         }
      }
   }
   if ( _VERBOSE_CACHE ) { std::cerr << "avoiding... process region " << reg.id << " got locked chunks: " << std::endl; }
   for ( std::map< memory_space_id_t, std::set< global_reg_t > >::iterator mIt = parts.begin(); mIt != parts.end(); mIt++ ) {
      if ( _VERBOSE_CACHE ) { std::cerr << " from location " << mIt->first << std::endl; }
      sys.getSeparateMemory( mIt->first ).getCache().prepareRegionsToBeCopied( mIt->second, version, _lockedChunks, wd, copyIdx );
   }
   if ( _VERBOSE_CACHE ) {
      std::cerr << "safe from invalidations... process region " << reg.id << " got locked chunks: ";
      for ( std::set< AllocatedChunk * >::iterator it = _lockedChunks.begin(); it != _lockedChunks.end(); it++ ) {
         std::cerr << " " << *it;
      }
      std::cerr << std::endl;
   }
}

void BaseOps::releaseLockedSourceChunks( WD const &wd ) {
   for ( std::set< AllocatedChunk * >::iterator it = _lockedChunks.begin(); it != _lockedChunks.end(); it++ ) {
      (*it)->removeReference( wd.getId() );
   }
   _lockedChunks.clear();
}

void BaseAddressSpaceInOps::copyInputData( MemCacheCopy const &memCopy, WD const &wd, unsigned int copyIdx ) {

   //std::set< DeviceOps * > ops;
   //ops.insert( reg.getDeviceOps() );

   //for ( NewLocationInfoList::const_iterator it = locations.begin(); it != locations.end(); it++ ) {
   //   global_reg_t data_source( it->second, reg.key );
   //   ops.insert( data_source.getDeviceOps() );
   //}
 
   //for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
   //   (*opIt)->syncAndDisableInvalidations();
   //}

   lockSourceChunks( memCopy._reg, memCopy.getVersion(), memCopy._locations, 0, wd, copyIdx );

   DeviceOps *thisRegOps = memCopy._reg.getDeviceOps();
   if ( memCopy._reg.getHostVersion( false ) != memCopy.getVersion() ) {
      if ( _VERBOSE_CACHE ) { std::cerr << "I have to copy region " << memCopy._reg.id << " dont have it, I want version " << memCopy.getVersion() << " host version is " << memCopy._reg.getHostVersion( false ) <<std::endl; }
      if ( thisRegOps->addCacheOp( /* debug: */ &wd ) ) {
         if ( _VERBOSE_CACHE ) { std::cerr << "I will do the transfer for reg " << memCopy._reg.id << " dont have it "<<std::endl; }

         if ( memCopy._locations.size() == 1 ) {
            global_reg_t region_shape( memCopy._locations.begin()->first, memCopy._reg.key );
            global_reg_t data_source( memCopy._locations.begin()->second, memCopy._reg.key );

            /* FIXME can this be moved after generating the ops?*/
            memory_space_id_t location = data_source.getFirstLocation();
            bool is_located_in_host = data_source.isLocatedIn( 0 );
            insertOwnOp( thisRegOps, memCopy._reg, memCopy.getVersion(), 0 ); //i've got the responsability of copying this region

            ensure( region_shape.id == memCopy._reg.id, "Wrong region" );
            if ( !is_located_in_host ) {
               ensure( location > 0, "Wrong location.");
               this->addOp( &( sys.getSeparateMemory( location ) ), region_shape, memCopy.getVersion(), NULL, copyIdx ); // inOp
            } else {
               *(myThread->_file) << "This should not happen, reg " << data_source.id << " reported to be in 0 (first loc " << location << " ) shape is " << region_shape.id << " wd: " << wd.getId() << " copyIdx " << copyIdx << " " << ((wd.getDescription()!=NULL) ? wd.getDescription() : "n/a") << std::endl;
               //fatal("Impossible path!");
               getOtherOps().insert( data_source.getDeviceOps() );
            }
         } else {
            /* FIXME can this be moved after generating the ops?*/
            insertOwnOp( thisRegOps, memCopy._reg, memCopy.getVersion(), 0 ); //i've got the responsability of copying this region

            for ( NewLocationInfoList::const_iterator it = memCopy._locations.begin(); it != memCopy._locations.end(); it++ ) {
               global_reg_t region_shape( it->first, memCopy._reg.key );
               global_reg_t data_source( it->second, memCopy._reg.key );
               ensure( region_shape.id != memCopy._reg.id, "Wrong region" );
               //if ( region_shape.id == data_source.id ) {
                  if ( !data_source.isLocatedIn( 0 ) ) {
                     memory_space_id_t location = data_source.getFirstLocation();
                     DeviceOps *thisOps = region_shape.getDeviceOps(); //FIXME: we assume that region_shape has a directory entry, it may be a wrong assumption
                     int added = 0;
                     if ( thisOps->addCacheOp( /* debug: */ &wd ) ) {
                        insertOwnOp( thisOps, region_shape, memCopy.getVersion(), 0 );
                        added = 1;
                     } else {
                        std::cerr << "ERROR, could not add a cache op for a chunk!" << std::endl;
                     }
                     if ( _VERBOSE_CACHE ) { std::cerr << " added a op! ds= " << it->second << " rs= " << it->first << " added= " << added << " so far we have ops: " << getOwnOps().size() << " this Obj "<< (void *) this << std::endl; }
                     this->addOp( &( sys.getSeparateMemory( location ) ), region_shape, memCopy.getVersion(), NULL, copyIdx ); //inOp
                  } else {
                     if ( _VERBOSE_CACHE ) { std::cerr << " sync with other op! ds= " << it->second << " rs= " << it->first <<std::endl; }
                     getOtherOps().insert( data_source.getDeviceOps() );
                  }
               //} else {
               //}
            }
         }
      } else {
         if ( _VERBOSE_CACHE ) { std::cerr << "I will not do the transfer for reg " << memCopy._reg.id << " dont have it "<<std::endl; }
         getOtherOps().insert( thisRegOps );
      }
   } else {
         if ( _VERBOSE_CACHE ) { std::cerr << "I will not do the transfer for reg " << memCopy._reg.id << " I have it at proper version " << memCopy.getVersion() <<std::endl; }
      getOtherOps().insert( thisRegOps );
   }

   //for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
   //   (*opIt)->resumeInvalidations();
   //}
}

void BaseAddressSpaceInOps::allocateOutputMemory( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   //std::cerr << "FIXME "<< __FUNCTION__ << std::endl;
   //reg.setLocationAndVersion( 0, version );
}

SeparateAddressSpaceInOps::SeparateAddressSpaceInOps( ProcessingElement *pe, bool delayedCommit, MemSpace<SeparateAddressSpace> &destination ) : BaseAddressSpaceInOps( pe, delayedCommit ), _destination( destination ), _hostTransfers() {
}

SeparateAddressSpaceInOps::~SeparateAddressSpaceInOps() {
}

void SeparateAddressSpaceInOps::addOpFromHost( global_reg_t const &reg, unsigned int version, AllocatedChunk *chunk, unsigned int copyIdx ) {
   addAmountTransferredData( reg.getDataSize() );
   _hostTransfers.push_back( TransferListEntry( reg, version, NULL, chunk, copyIdx ) );
}

void SeparateAddressSpaceInOps::issue( WD const &wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_MEM_TRANSFER_ISSUE); );
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
      _destination.copy( *(it->first) /* mem space */, it->second /* list of regions */, wd );
   }
   _destination.copyFromHost( _hostTransfers, wd );
}

unsigned int SeparateAddressSpaceInOps::getVersionNoLock( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   return _destination.getCurrentVersion( reg, wd, copyIdx );
}

void SeparateAddressSpaceInOps::copyInputData( MemCacheCopy const& memCopy, WD const &wd, unsigned int copyIdx ) {
   lockSourceChunks( memCopy._reg, memCopy.getVersion(), memCopy._locations, _destination.getMemorySpaceId(), wd, copyIdx );
   _destination.copyInputData( *this, memCopy._reg, memCopy.getVersion(), memCopy._locations, memCopy._chunk, wd, copyIdx );
}

void SeparateAddressSpaceInOps::allocateOutputMemory( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   _destination.allocateOutputMemory( reg, this->getPE(), version, wd, copyIdx );
}

SeparateAddressSpaceOutOps::SeparateAddressSpaceOutOps( ProcessingElement *pe, bool delayedCommit, bool isInval ) : BaseOps( pe, delayedCommit )
   , _invalidation( isInval )
   , _transfers()
{
}

SeparateAddressSpaceOutOps::~SeparateAddressSpaceOutOps() {
}

void SeparateAddressSpaceOutOps::addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version, DeviceOps *ops, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx ) {
   TransferList &list = _transfers[ from ];
   if ( _lockedChunks.count( chunk ) == 0 ) {
      chunk->lock();
      chunk->addReference( wd.getId(), 1 ); //Out addOp( with chunk )
      _lockedChunks.insert( chunk );
      chunk->unlock();
   }
   list.push_back( TransferListEntry( reg, version, ops, chunk, copyIdx ) );
}

void SeparateAddressSpaceOutOps::addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version, DeviceOps *ops, WD const &wd, unsigned int copyIdx ) {
   TransferList &list = _transfers[ from ];
   from->getCache().lock();
   from->getCache()._prepareRegionToBeCopied( reg, version, _lockedChunks, wd, copyIdx );
   from->getCache().unlock();
   list.push_back( TransferListEntry( reg, version, ops, NULL, copyIdx ) );
}

void SeparateAddressSpaceOutOps::issue( WD const &wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_MEM_TRANSFER_ISSUE); );
   for ( MapType::iterator it = _transfers.begin(); it != _transfers.end(); it++ ) {
     sys.getHostMemory().copy( *(it->first) /* mem space */, it->second /* region */, wd, _invalidation );
   }
}

void SeparateAddressSpaceOutOps::copyOutputData( SeparateMemoryAddressSpace *from, MemCacheCopy const &memCopy, bool output, WD const &wd, unsigned int copyIdx ) {
   from->copyOutputData( *this, memCopy._reg, memCopy.getVersion(), output, memCopy._policy, memCopy._chunk, wd, copyIdx );
}

}
