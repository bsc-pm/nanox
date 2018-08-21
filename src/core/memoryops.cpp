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

#include <iostream>
#include <limits>
#include <iomanip>

#include "memoryops_decl.hpp"
#include "system_decl.hpp"

#include "instrumentationmodule_decl.hpp"
#include "instrumentation.hpp"
#include "workdescriptor.hpp"

#include "addressspace.hpp"
#include "deviceops.hpp"
#include "regiondict.hpp"
#include "regioncache.hpp"
#include "trackableobject.hpp"
#include "memcachecopy.hpp"
#include "globalregt.hpp"

#if VERBOSE_CACHE
 #define _VERBOSE_CACHE 1
#else
 #define _VERBOSE_CACHE 0
#endif

using namespace nanos;

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
   _reg.setLocationAndVersion( pe, _location, _version ); //commitMetadata
}

BaseOps::BaseOps( ProcessingElement *pe, bool delayedCommit ) : 
     _delayedCommit( delayedCommit )
   , _dataReady( false )
   , _pe( pe )
   , _ownDeviceOps()
   , _otherDeviceOps()
   , _amountOfTransferredData( 0 )
{
}

BaseOps::~BaseOps() {
}

void BaseOps::cancelOwnOps( WD const &wd ) {
   for ( std::set< OwnOp >::iterator it = _ownDeviceOps.begin(); it != _ownDeviceOps.end(); it++ ) {
      it->_ops->completeCacheOp( /* debug: */ &wd );
   }
}

void BaseOps::print( std::ostream &out ) const {
   out << "own ops: " << _ownDeviceOps.size() << std::endl;
   for ( std::set< OwnOp >::const_iterator it = _ownDeviceOps.begin(); it != _ownDeviceOps.end(); it++ ) {
      out << "\tdev ops: " << it->_ops << " " << *(it->_ops) << " v: " << it->_version << " _loc " << it->_location << std::endl;
   }
   out << "other ops: " << _otherDeviceOps.size() << std::endl;
   for (std::set< DeviceOps * >::const_iterator it = _otherDeviceOps.begin(); it != _otherDeviceOps.end(); it++ ) {
      out << "\tdev ops: " << *it << " " << *(*it) << std::endl;
   }
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

bool BaseOps::checkDataReady() const {
   return _dataReady;
}

BaseAddressSpaceInOps::BaseAddressSpaceInOps( ProcessingElement *pe, bool delayedCommit ) : BaseOps( pe, delayedCommit )
   , _separateTransfers() {
}

BaseAddressSpaceInOps::~BaseAddressSpaceInOps() {
}

void BaseAddressSpaceInOps::addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, uint64_t srcDevAddr, WD const &wd, unsigned int copyIdx ) {
   TransferList &list = _separateTransfers[ from ];
   addAmountTransferredData( reg.getDataSize() );

   //AllocatedChunk *src_chunk = from.getAndReferenceAllocatedChunk(region_shape, wd, copyIdx); //FIXME (lock cache, getChunk, AddRef, release chunk, release cache)

   if ( sourceChunk == (AllocatedChunk *) -1) {
      printBt( *myThread->_file );
   }
   if ( destinationChunk == (AllocatedChunk *) -1) {
      printBt( *myThread->_file );
   }
   if ( _lockedChunks.count( sourceChunk ) == 0 ) {
      sourceChunk->lock();
      sourceChunk->addReference( wd, 133 ); //Out addOp( with chunk )
      _lockedChunks.insert( sourceChunk );
      sourceChunk->unlock();
   }
   list.push_back( TransferListEntry( reg, version, NULL, destinationChunk, sourceChunk, srcDevAddr, copyIdx ) );
}

void BaseAddressSpaceInOps::addOpFromHost( global_reg_t const &reg, unsigned int version, AllocatedChunk *chunk, unsigned int copyIdx ) {
   *myThread->_file << "Error, can not send data from myself." << std::endl; 
}

void BaseAddressSpaceInOps::issue( WD const *wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_MEM_TRANSFER_ISSUE, true); );
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
     sys.getHostMemory().copy( *(it->first) /* mem space */, it->second /* regions */, wd );
   }
}

unsigned int BaseAddressSpaceInOps::getVersionNoLock( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   return reg.getHostVersion(false);
}

void BaseOps::releaseLockedSourceChunks( WD const &wd ) {
   for ( std::set< AllocatedChunk * >::iterator it = _lockedChunks.begin(); it != _lockedChunks.end(); it++ ) {
      (*it)->removeReference( wd ); // releaseLockedSourceChunks
   }
   _lockedChunks.clear();
}

void BaseAddressSpaceInOps::copyInputData( MemCacheCopy const &memCopy, WD const &wd, unsigned int copyIdx ) {

   for ( NewLocationInfoList::const_iterator it = memCopy._locations.begin(); it != memCopy._locations.end(); it++ ) {
      global_reg_t region_shape( it->first, memCopy._reg.key );
      global_reg_t data_source( it->second, memCopy._reg.key );

      DeviceOps *rs_ops = region_shape.getDeviceOps();
      DeviceOps *ds_ops = data_source.getDeviceOps();
      unsigned int rs_version = region_shape.getVersion();
      unsigned int ds_version = data_source.getVersion();

      if ( ds_version > rs_version ) {
         if ( !data_source.isLocatedIn( 0 ) && rs_ops->addCacheOp( &wd ) ) {
            memory_space_id_t location = data_source.getPreferedSourceLocation(0);
            if ( location == 0 ) {
               DirectoryEntryData *entry = RegionDirectory::getDirectoryEntry( *data_source.key, data_source.id );
               *myThread->_file << "region_shape : " << region_shape.id << ", data_source " << data_source.id << " entry: " << *entry << " I want version " << memCopy.getVersion()<< std::endl;
            }
            ensure( location > 0, "Wrong location.");
            AllocatedChunk *source_chunk = sys.getSeparateMemory( location ).getCache().getAllocatedChunk( data_source, wd, copyIdx );
            uint64_t orig_dev_addr = source_chunk->getAddress() + ( region_shape.getRealFirstAddress() - source_chunk->getHostAddress() );
            source_chunk->unlock();
            insertOwnOp( rs_ops, region_shape, memCopy.getVersion(), 0 ); //i've got the responsability of copying this region
//(*myThreadfile) << std::setprecision(std::numeric_limits<double>::digits10) << OS::getMonotonicTime() << " adding op (copy to host from " << location << " using chunk " << source_chunk << " w/addr " << source_chunk->getHostAddress() << std::endl;
            this->addOp( &( sys.getSeparateMemory( location ) ), region_shape, memCopy.getVersion(), NULL, source_chunk, orig_dev_addr, wd, copyIdx ); // inOp
         } else {
            getOtherOps().insert( ds_ops );
         }
      } else if ( ds_version == rs_version ) {
         if ( !data_source.isLocatedIn( 0 ) && !region_shape.isLocatedIn( 0 ) && rs_ops->addCacheOp( &wd ) ) {
            memory_space_id_t location = data_source.getPreferedSourceLocation(0);
            if ( location == 0 ) {
               DirectoryEntryData *ds_entry = RegionDirectory::getDirectoryEntry( *data_source.key, data_source.id );
               DirectoryEntryData *rs_entry = RegionDirectory::getDirectoryEntry( *region_shape.key, region_shape.id );
               *myThread->_file << "region_shape : " << region_shape.id << " { " << *rs_entry << "}, data_source " << data_source.id << " { " << *ds_entry << " }, I want version " << memCopy.getVersion() << std::endl;
            }
            ensure( location > 0, "Wrong location.");
            AllocatedChunk *source_chunk = sys.getSeparateMemory( location ).getCache().getAllocatedChunk( data_source, wd, copyIdx );
            uint64_t orig_dev_addr = source_chunk->getAddress() + ( region_shape.getRealFirstAddress() - source_chunk->getHostAddress() );
            source_chunk->unlock();
            insertOwnOp( rs_ops, region_shape, memCopy.getVersion(), 0 ); //i've got the responsability of copying this region
//(*myThreadfile) << std::setprecision(std::numeric_limits<double>::digits10) << OS::getMonotonicTime() << " adding op (copy to host from " << location << " using chunk " << source_chunk << " w/addr " << source_chunk->getHostAddress() << std::endl;
            this->addOp( &( sys.getSeparateMemory( location ) ), region_shape, memCopy.getVersion(), NULL, source_chunk, orig_dev_addr, wd, copyIdx ); // inOp
         } else {
            getOtherOps().insert( rs_ops );
            getOtherOps().insert( ds_ops );
         }
      } else {
         fatal("Impossible path");
      }
   }
}

SeparateAddressSpaceInOps::SeparateAddressSpaceInOps( ProcessingElement *pe, bool delayedCommit, MemSpace<SeparateAddressSpace> &destination ) : BaseAddressSpaceInOps( pe, delayedCommit ), _destination( destination ), _hostTransfers() {
}

SeparateAddressSpaceInOps::~SeparateAddressSpaceInOps() {
}

void SeparateAddressSpaceInOps::addOpFromHost( global_reg_t const &reg, unsigned int version, AllocatedChunk *chunk, unsigned int copyIdx ) {
   addAmountTransferredData( reg.getDataSize() );
   _hostTransfers.push_back( TransferListEntry( reg, version, NULL, chunk, /* source chunk */ (AllocatedChunk *) NULL, /* srcDevAddr */ 0, copyIdx ) );
}

void SeparateAddressSpaceInOps::issue( WD const *wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_MEM_TRANSFER_ISSUE, true); );
   for ( MapType::iterator it = _separateTransfers.begin(); it != _separateTransfers.end(); it++ ) {
      _destination.copy( *(it->first) /* mem space */, it->second /* list of regions */, wd );
   }
   _destination.copyFromHost( _hostTransfers, wd );
}

unsigned int SeparateAddressSpaceInOps::getVersionNoLock( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   return _destination.getCurrentVersion( reg, wd, copyIdx );
}

SeparateAddressSpaceOutOps::SeparateAddressSpaceOutOps( ProcessingElement *pe, bool delayedCommit, bool isInval ) : BaseOps( pe, delayedCommit )
   , _invalidation( isInval )
   , _transfers()
{
}

SeparateAddressSpaceOutOps::~SeparateAddressSpaceOutOps() {
}

void SeparateAddressSpaceOutOps::addOutOp( memory_space_id_t to, memory_space_id_t from, global_reg_t const &reg, unsigned int version, DeviceOps *ops, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx ) {
   TransferList &list = _transfers[ std::make_pair(to, from) ];
   if ( _lockedChunks.count( chunk ) == 0 ) {
      chunk->lock();
      chunk->addReference( wd, 2 ); //Out addOp( with chunk )
      _lockedChunks.insert( chunk );
      chunk->unlock();
   }
   list.push_back( TransferListEntry( reg, version, ops, /* destination */ (AllocatedChunk *) NULL, chunk, /* FIXME: srcAddr */ 0, copyIdx ) );
}

void SeparateAddressSpaceOutOps::addOutOp( memory_space_id_t to, memory_space_id_t from, global_reg_t const &reg, unsigned int version, DeviceOps *ops, WD const &wd, unsigned int copyIdx ) {

   TransferList &list = _transfers[ std::make_pair(to, from) ];
   SeparateAddressSpace &sas = sys.getSeparateMemory( from );
   sas.getCache().lock();
   sas.getCache()._prepareRegionToBeCopied( reg, version, _lockedChunks, wd, copyIdx );
   sas.getCache().unlock();
   AllocatedChunk *chunk = sas.getCache()._getAllocatedChunk( reg, false, false, wd, copyIdx );
   list.push_back( TransferListEntry( reg, version, ops, /* destination */ (AllocatedChunk *) NULL, chunk, /* FIXME: srcAddr */ 0, copyIdx ) );
}

void SeparateAddressSpaceOutOps::issue( WD const *wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_MEM_TRANSFER_ISSUE, true); );
   for ( MapType::iterator it = _transfers.begin(); it != _transfers.end(); it++ ) {
      if (it->first.first == 0 ) {
         sys.getHostMemory().copy( sys.getSeparateMemory(it->first.second) /* mem space */, it->second /* region */, wd, _invalidation );
      } else {
         sys.getSeparateMemory( it->first.first ).copy( sys.getSeparateMemory(it->first.second) /* mem space */, it->second /* region */, wd, _invalidation );
      }
   }
}

bool SeparateAddressSpaceOutOps::hasPendingOps() const {
   return ( _transfers.size() > 0 && !this->checkDataReady() );
}

void SeparateAddressSpaceOutOps::cancel( WD const &wd ) {
   // for ( MapType::iterator it = _transfers.begin(); it != _transfers.end(); it++ ) {
   //    TransferList &list = it->second;
   //    for ( TransferList::iterator lit = list.begin(); lit != list.end(); lit++ ) {
   //       if ( lit->getDeviceOps() != NULL ) {
   //          lit->getDeviceOps()->completeCacheOp( &wd );
   //       }
   //    }
   // }
   this->cancelOwnOps(wd);
   releaseLockedSourceChunks( wd );
}

