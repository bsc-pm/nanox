/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "workdescriptor_decl.hpp"
#include "debug.hpp"
#include "memorymap.hpp"
#include "copydata.hpp"
#include "atomic.hpp"
#include "processingelement.hpp"
#include "regiondirectory.hpp"
#include "regiontree.hpp"
#include "system.hpp"
#include "deviceops.hpp"
#ifdef GPU_DEV
#include "gpudd.hpp"
#endif
#include "regioncache.hpp"

using namespace nanos; 

CachedRegionStatus::CachedRegionStatus() : _version( 0 ) {
}

CachedRegionStatus::CachedRegionStatus( CachedRegionStatus const &rs ) : _version( rs._version ), _waitObject ( rs._waitObject ) {
}

CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus const &rs ) {
   _version = rs._version;
   _waitObject = rs._waitObject;
   return *this;
}

CachedRegionStatus::CachedRegionStatus( CachedRegionStatus &rs ) : _version( rs._version ), _waitObject ( rs._waitObject ) {
}

CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus &rs ) {
   _version = rs._version;
   _waitObject = rs._waitObject;
   return *this;
}

unsigned int CachedRegionStatus::getVersion() {
   return _version;
}

void CachedRegionStatus::setVersion( unsigned int version ) {
   _version = version;
}

void CachedRegionStatus::setCopying( DeviceOps *ops ) {
   _waitObject.set( ops );
}

DeviceOps *CachedRegionStatus::getDeviceOps() {
   return _waitObject.get();
}

bool CachedRegionStatus::isReady( ) {
   return _waitObject.isNotSet();
}

AllocatedChunk::AllocatedChunk() : _lock(), _address( 0 ) {
}

AllocatedChunk::AllocatedChunk( uint64_t addr ) : _lock(), _address( addr ) {
}

AllocatedChunk::AllocatedChunk( AllocatedChunk const &chunk ) : _lock(), _address( chunk._address ) {
}

AllocatedChunk &AllocatedChunk::operator=( AllocatedChunk const &chunk ) {
   _address = chunk._address;
   return *this;
}

void AllocatedChunk::lock() {
   _lock.acquire();
}

void AllocatedChunk::unlock() {
   _lock.release();
}

void AllocatedChunk::addReadRegion( Region const &reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::list< Region > &notPresentRegions, DeviceOps *ops, bool alsoWriteRegion ) {
   RegionTree<CachedRegionStatus>::iterator_list_t insertOuts;
   RegionTree<CachedRegionStatus>::iterator ret;
   ret = _regions.findAndPopulate( reg, insertOuts );
   if ( !ret.isEmpty() ) insertOuts.push_back( ret );

   for ( RegionTree<CachedRegionStatus>::iterator_list_t::iterator it = insertOuts.begin();
         it != insertOuts.end();
         it++
       ) {
      RegionTree<CachedRegionStatus>::iterator &accessor = *it;
      CachedRegionStatus &cachedReg = *accessor;
      if ( version <= cachedReg.getVersion() )  {
         if ( cachedReg.isReady() ) {
            /* already in cache */
         } else {
            /* in cache but comming! */
            currentOps.insert( cachedReg.getDeviceOps() );
         }
      } else {
         /* not present! */
         notPresentRegions.push_back( accessor.getRegion() );
         cachedReg.setCopying( ops );
         cachedReg.setVersion(version);
      }
      if ( alsoWriteRegion ) cachedReg.setVersion( version + 1 );
   } 
}

void AllocatedChunk::addWriteRegion( Region const &reg, unsigned int version ) {
   RegionTree<CachedRegionStatus>::iterator_list_t insertOuts;
   RegionTree<CachedRegionStatus>::iterator ret;
   ret = _regions.findAndPopulate( reg, insertOuts );
   if ( !ret.isEmpty() ) insertOuts.push_back( ret );

   for ( RegionTree<CachedRegionStatus>::iterator_list_t::iterator it = insertOuts.begin();
         it != insertOuts.end();
         it++
       ) {
      RegionTree<CachedRegionStatus>::iterator &accessor = *it;
      CachedRegionStatus &cachedReg = *accessor;
      cachedReg.setVersion( version );
   } 
}

bool AllocatedChunk::isReady( Region reg ) {
   bool entryReady = true;
   RegionTree<CachedRegionStatus>::iterator_list_t outs;
   //RegionTree<CachedRegionStatus>::iterator ret;
   _regions.find( reg, outs );
   if ( outs.empty () ) {
      message0("ERROR: Got no regions from AllocatedChunk!!");
   } else {
      //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
      RegionTree<CachedRegionStatus>::iterator_list_t::iterator it = outs.begin();
      //RegionTree<CachedRegionStatus>::iterator &firstAccessor = *it;
      //Region tmpReg = firstAccessor.getRegion();
      //bool combiningIsGoingOk = true;

      for ( ; ( it != outs.end() ) /*&& ( combiningIsGoingOk )*/ && ( entryReady ); it++) {
         RegionTree<CachedRegionStatus>::iterator &accessor = *it;
         //combiningIsGoingOk = tmpReg.combine( accessor.getRegion(), tmpReg );
         CachedRegionStatus &status = *accessor;
         entryReady = entryReady && status.isReady();
      }
      //if ( combiningIsGoingOk ) {
      //   if ( tmpReg != reg && !tmpReg.contains( reg ) ) {
      //      message0("ERROR: Region not found in the Allocated chunk!!!");
      //   } else { }
      //} else {
      //   message0("ERROR: Region not found in the Allocated chunk!!! unable to combine return regions!");
      //}
   }

   return entryReady;
}

AllocatedChunk *RegionCache::getAddress( CopyData const &cd, uint64_t &offset ) {
   ChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   _chunks.getOrAddChunk( (uint64_t) cd.getBaseAddress(), cd.getMaxSize(), results ); //we dont want to create new entries if a bigger one already exists!!! FIXME
   if ( results.size() != 1 ) {
      message0( "Got results.size()="<< results.size() << " I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ChunkList::iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
   } else {
      if ( *(results.front().second) == NULL ) {
         *(results.front().second) = NEW AllocatedChunk( (uint64_t) _device.memAllocate( cd.getMaxSize(), _pe ) );
         allocChunkPtr = *(results.front().second);
         offset = 0;
      } else {
         allocChunkPtr = *(results.front().second);
         offset = ((uint64_t) cd.getBaseAddress() - (uint64_t) (results.front().first)->getAddress());
      }
   }
   if ( allocChunkPtr == NULL ) std::cerr << "WARNING: null RegionCache::getAddress()" << std::endl; 
   allocChunkPtr->lock();
   return allocChunkPtr;
}

AllocatedChunk *RegionCache::getAddress( uint64_t hostAddr, std::size_t len, uint64_t &offset ) {
   ConstChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   _chunks.getChunk3( hostAddr, len, results );
   if ( results.size() != 1 ) {
         std::cerr <<"Requested addr " << (void *) hostAddr << " size " <<len << std::endl;
      message0( "I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ConstChunkList::iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
   } else {
      if ( *(results.front().second) == NULL ) {
         message0("Address not found in cache, Error!! ");
      } else {
         allocChunkPtr = *(results.front().second);
         offset = hostAddr - (uint64_t) (results.front().first)->getAddress();
      }
   }
   if ( allocChunkPtr == NULL ) std::cerr << "WARNING: null RegionCache::getAddress()" << std::endl; 
   allocChunkPtr->lock();
   return allocChunkPtr;
}
   
void RegionCache::syncRegion( std::list< std::pair< Region, CacheCopy * > > const &regions, WD const &wd ) {
   std::list< std::pair< Region, CacheCopy *> >::const_iterator it;
   uint64_t offset = 0;
   DeviceOps localOps;

   
   for ( it = regions.begin(); it != regions.end(); it++ ) {
      Region const &reg = it->first;
      AllocatedChunk *origChunk = getAddress( ( uint64_t ) reg.getFirstValue(), ( std::size_t ) reg.getBreadth(), offset );
      uint64_t origDevAddr = origChunk->getAddress() + offset;
      origChunk->unlock();
      copyOut( reg, origDevAddr, ( it->second != NULL ) ? it->second->getOperations() : &localOps, wd );
   }

   while( !localOps.allCompleted() ) { myThread->idle(); }
}

void RegionCache::syncRegion( Region const &reg ) {
   std::list< std::pair< Region, CacheCopy * > > singleItemList;
   singleItemList.push_back( std::make_pair( reg, ( CacheCopy * ) NULL ) );
   syncRegion( singleItemList, *(( WD * ) NULL) );
}

void RegionCache::_generateRegionOps( Region const &reg, std::map< uintptr_t, MemoryMap< uint64_t > * > &opMap )
{
   uint64_t offset = 0, devAddr;
   AllocatedChunk *chunk = getAddress( (uint64_t) ((uint64_t) reg.getFirstValue()), ((std::size_t) reg.getBreadth()), offset );
   devAddr = chunk->getAddress() + offset;
   chunk->unlock();

   MemoryMap< uint64_t > *ops = opMap[ ( (uintptr_t) chunk ) ];
   if ( ops == NULL ) {
      ops = NEW MemoryMap< uint64_t >();
      opMap[ ( (uintptr_t) chunk ) ] = ops;
   }

   std::size_t skipBits = 0;
   std::size_t numChunks = reg.getNumNonContiguousChunks( skipBits );
   std::size_t contiguousSize = reg.getContiguousChunkLength();

   for ( unsigned int chunkIndex = 0; chunkIndex < numChunks; chunkIndex += 1 )
   {
      uint64_t address = reg.getNonContiguousChunk( chunkIndex, skipBits );
      ops->addChunk( address, contiguousSize, devAddr + ( address - reg.getFirstValue() ) );
   }
}

RegionCache::RegionCache( ProcessingElement &pe, Device &cacheArch ) : _device( cacheArch ), _pe( pe ),
   _copyInObj( *this ), _copyOutObj( *this ) {
}

unsigned int RegionCache::getMemorySpaceId() {
   return _pe.getMemorySpaceId();
}

void RegionCache::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd ) {
   ops->addOp();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   _device._copyIn( devAddr, hostAddr, len, _pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd ) {
   ops->addOp();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   _device._copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, _pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, WD const &wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   _device._copyOut( hostAddr, devAddr, len, _pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld,  DeviceOps *ops, WD const &wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   _device._copyOutStrided1D( hostAddr, devAddr, len, numChunks, ld, _pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd ) {
   uint64_t offset;
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->getAddress() + offset;
   origChunk->unlock();
   sys.getCaches()[ syncFrom ]->_copyOut( hostAddr, origDevAddr, len, cout, wd );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   delete cout;
   this->_copyIn( devAddr, hostAddr, len, ops, wd );
}

void RegionCache::_syncAndCopyInStrided1D( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd ) {
   uint64_t offset;
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->getAddress() + offset;
   origChunk->unlock();
   sys.getCaches()[ syncFrom ]->_copyOutStrided1D( hostAddr, origDevAddr, len, numChunks, ld, cout, wd );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   delete cout;
   this->_copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, ops, wd );
}

void RegionCache::_copyDevToDev( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd ) {
   uint64_t offset;
   AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->getAddress() + offset;
   origChunk->unlock();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   ops->addOp();
   _device._copyDevToDev( devAddr, origDevAddr, len, _pe, sys.getCaches()[ copyFrom ]->_pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyDevToDevStrided1D( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd ) {
   uint64_t offset;
   AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->getAddress() + offset;
   origChunk->unlock();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   ops->addOp();
   _device._copyDevToDevStrided1D( devAddr, origDevAddr, len, numChunks, ld, _pe, sys.getCaches()[ copyFrom ]->_pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::CopyIn::doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, WD const &wd ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyIn( devAddr, hostAddr, size, ops, wd );
   } else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
      getParent()._copyDevToDev( dataLocation, devAddr, hostAddr, size, ops, wd );
   } else {
      getParent()._syncAndCopyIn( dataLocation, devAddr, hostAddr, size, ops, wd );
   }
}

void RegionCache::CopyIn::doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, WD const &wd ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyInStrided1D( devAddr, hostAddr, size, count, ld, ops, wd );
   } else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
      getParent()._copyDevToDevStrided1D( dataLocation, devAddr, hostAddr, size, count, ld, ops, wd );
   } else {
      getParent()._syncAndCopyInStrided1D( dataLocation, devAddr, hostAddr, size, count, ld, ops, wd );
   }
}

void RegionCache::CopyOut::doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, WD const &wd ) {
   getParent()._copyOut( hostAddr, devAddr, size, ops, wd );
}
void RegionCache::CopyOut::doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, WD const &wd ) {
   getParent()._copyOutStrided1D( hostAddr, devAddr, size, count, ld, ops, wd );
}

void RegionCache::doOp( Op *opObj, Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ) {
   std::size_t skipBits = 0;
   std::size_t contiguousSize = hostMem.getContiguousChunkLength();
   std::size_t numChunks = hostMem.getNumNonContiguousChunks( skipBits );

   if ( numChunks > 1 && sys.usePacking() ) {
      uint64_t ld = hostMem.getNonContiguousChunk( 1, skipBits ) - hostMem.getNonContiguousChunk( 0, skipBits );
      uint64_t hostAddr = hostMem.getNonContiguousChunk( 0, skipBits );

      opObj->doStrided( location, devBaseAddr, hostAddr, contiguousSize, numChunks, ld, ops, wd );
   } else {
      for (unsigned int chunkIndex = 0; chunkIndex < numChunks; chunkIndex +=1 ) {
         uint64_t hostAddr = hostMem.getNonContiguousChunk( chunkIndex, skipBits );
         uint64_t devAddr = devBaseAddr + ( hostAddr - hostMem.getFirstValue() ); /* contiguous chunk offset */

         opObj->doNoStrided( location, devAddr, hostAddr, contiguousSize, ops, wd );
      }
   }
}

void RegionCache::copyIn( Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ) {
   doOp( &_copyInObj, hostMem, devBaseAddr, location, ops, wd );
}

void RegionCache::copyOut( Region const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, WD const &wd ) {
   doOp( &_copyOutObj, hostMem, devBaseAddr, /* locations unused, copyOut is always to 0 */ 0, ops, wd );
}

void RegionCache::lock() {
   _lock.acquire();
}
void RegionCache::unlock() {
   _lock.release();
}
bool RegionCache::tryLock() {
   return _lock.tryAcquire();
}

CacheCopy::CacheCopy() : _copy( *( (CopyData *) NULL ) ), _cacheEntry( NULL ), _cacheDataStatus(), _devRegion(), _devBaseAddr( 0 ),
   _region( ), _offset( 0 ), _version( 0 ), _locations(), _operations(), _otherPendingOps() {
}

CacheCopy::CacheCopy( WD const &wd , unsigned int copyIndex ) : _copy( wd.getCopies()[ copyIndex ] ), _cacheEntry( NULL ),
   _cacheDataStatus(), _devRegion(), _devBaseAddr( 0 ), _region( NewRegionDirectory::build_region( _copy ) ), _offset( 0 ),
   _version( 0 ), _locations(), _operations(), _otherPendingOps() {
   wd.getNewDirectory()->getLocation( _region, _locations, _version, wd );
}

bool CacheCopy::isReady() {
   bool allReady = true;
   if( !_operations.allCompleted() ) { 
      allReady = false;
   }

   if ( allReady ) { 
      std::set< DeviceOps * >::iterator it = _otherPendingOps.begin();
      while ( allReady && it != _otherPendingOps.end() ) {
         if ( (*it)->allCompleted() ) {
            std::set< DeviceOps * >::iterator toBeRemovedIt = it;
            it++;
            _otherPendingOps.erase( toBeRemovedIt );
         } else {
            allReady = false;
         }
      }
   }
   return allReady;
}

inline void CacheCopy::setUpDeviceAddress( RegionCache *targetCache ) {
   _cacheEntry = targetCache->getAddress( _copy, _offset );
   _devRegion = NewRegionDirectory::build_region_with_given_base_address( _copy, _offset );
   _devBaseAddr = _cacheEntry->getAddress() + _offset;
   _cacheEntry->unlock();
}

inline void CacheCopy::generateCopyInOps( RegionCache *targetCache, std::map<unsigned int, std::list< std::pair< Region, CacheCopy * > > > &opsBySourceRegions ) {
	NewRegionDirectory::LocationInfoList::const_iterator it;
   if ( targetCache ) _cacheEntry->lock();
   if ( _copy.isInput() )
   {
      for ( it = _locations.begin(); it != _locations.end(); it++ ) {
   
         if ( it->second.isLocatedIn( ( !targetCache ) ? 0 : targetCache->getMemorySpaceId() ) ) continue;
         /* FIXME: version info, (I think its not needed because directory stores
          * only the last version, if an old version is stored, it wont be reported
          * but _targetCache.getAddress will return the already allocated storage)
          */
         
         if ( !targetCache ) {
            /* No Cache scenario
             * we can not check if there are already comming ops for this region/sub-regions!! FIXME
             */
            opsBySourceRegions[ it->second.getFirstLocation() ].push_back( std::make_pair( it->first, this ) );
         } else { 
            std::list< Region > notPresentRegions;
            std::list< Region >::iterator notPresentRegionsIt;
            _cacheEntry->addReadRegion( it->first, it->second.getVersion(), _otherPendingOps,
               notPresentRegions, &_operations, _copy.isOutput() );
   
            for( notPresentRegionsIt = notPresentRegions.begin();
                 notPresentRegionsIt != notPresentRegions.end();
                 notPresentRegionsIt++ ) {
               std::list< std::pair< Region, CacheCopy * > > &thisCopyOpsRegions = opsBySourceRegions[ it->second.getFirstLocation() ];
   
               Region &origReg = *notPresentRegionsIt;
               thisCopyOpsRegions.push_back( std::make_pair( origReg, this ) );
            }
         }
      }
   } else if ( !_copy.isInput() && _copy.isOutput() && targetCache ) {
      unsigned int currentVersion = 1;
      for ( it = _locations.begin(); it != _locations.end(); it++ ) {
         currentVersion = std::max( currentVersion, it->second.getVersion() );
      }
      /* write only region */
      _cacheEntry->addWriteRegion( _region, currentVersion + 1 );
   }
   if ( targetCache ) _cacheEntry->unlock();
}

CacheController::CacheController( WD const &wd ) : _wd ( wd ), _numCopies ( wd.getNumCopies() ), _targetCache ( NULL ) {
   if ( _numCopies > 0 ) 
      _cacheCopies = NEW CacheCopy[ _numCopies ];
}

bool CacheController::isCreated() const {
   return _targetCache != NULL;
}

void CacheController::preInit() {
   unsigned int index;
   for ( index = 0; index < _numCopies; index += 1 ) {
      new ( &_cacheCopies[ index ] ) CacheCopy( _wd, index );
   }
}

void CacheController::copyDataIn(RegionCache *targetCache) {
   unsigned int index;
   _targetCache = targetCache;
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_CC_CDIN); );
   if ( _numCopies > 0 ) {
      /* Get device address, allocate if needed */
      NANOS_INSTRUMENT( InstrumentState inst3(NANOS_CC_CDIN_GET_ADDR); );
      for ( index = 0; index < _numCopies; index += 1 ) {
         CacheCopy &ccopy = _cacheCopies[ index ];
         if ( _targetCache ) ccopy.setUpDeviceAddress( _targetCache );
         
         // register version into this task directory
	 _wd.getNewDirectory()->addAccess( ccopy.getRegion(), ( !_targetCache ) ? 0 : _targetCache->getMemorySpaceId(),
            ccopy.getCopyData().isOutput() ? ccopy.getVersion() + 1 : ccopy.getVersion() );
      }
      NANOS_INSTRUMENT( inst3.close(); );

      /* COPY IN GENERATION */
      NANOS_INSTRUMENT( InstrumentState inst4(NANOS_CC_CDIN_OP_GEN); );
      std::map<unsigned int, std::list< std::pair< Region, CacheCopy * > > > opsBySourceRegions;
      for ( index = 0; index < _numCopies; index += 1 ) {
         _cacheCopies[ index ].generateCopyInOps( _targetCache, opsBySourceRegions );
      }
      NANOS_INSTRUMENT( inst4.close(); );
      /* END OF COPY IN GENERATION */

      /* ISSUE ACTUAL OPERATIONS */
      NANOS_INSTRUMENT( InstrumentState inst5(NANOS_CC_CDIN_DO_OP); );
      std::map< unsigned int, std::list< std::pair< Region, CacheCopy * > > >::iterator mapOpsStrIt;
      if ( targetCache ) {
         for ( mapOpsStrIt = opsBySourceRegions.begin(); mapOpsStrIt != opsBySourceRegions.end(); mapOpsStrIt++ ) {
            std::list< std::pair< Region, CacheCopy * > > &ops = mapOpsStrIt->second;
            for ( std::list< std::pair< Region, CacheCopy * > >::iterator listIt = ops.begin(); listIt != ops.end(); listIt++ ) {
               CacheCopy &ccopy = *listIt->second;
               uint64_t fragmentOffset = listIt->first.getFirstValue() - ( ( uint64_t ) ccopy.getCopyData().getBaseAddress() + ccopy.getCopyData().getOffset() ); /* displacement due to fragmented region */
               targetCache->copyIn( listIt->first, ccopy.getDevBaseAddress() + ccopy.getCopyData().getOffset() + fragmentOffset, mapOpsStrIt->first, ccopy.getOperations(), _wd );
            }
         }
      } else {
         for ( mapOpsStrIt = opsBySourceRegions.begin(); mapOpsStrIt != opsBySourceRegions.end(); mapOpsStrIt++ ) {
            sys.getCaches()[ mapOpsStrIt->first ]->syncRegion( mapOpsStrIt->second, _wd );
         }
      }
      NANOS_INSTRUMENT( inst5.close(); );
      /* END OF ISSUE OPERATIONS */
   }
   NANOS_INSTRUMENT( inst2.close(); );
}


bool CacheController::dataIsReady() const {
   bool allReady = true;
   unsigned int index;
   for ( index = 0; ( index < _numCopies ) && allReady; index += 1 ) {
      allReady = _cacheCopies[ index ].isReady();
   }
   return allReady;
}

uint64_t CacheController::getAddress( unsigned int copyId ) const {
   return _cacheCopies[ copyId ].getDeviceAddress();
}


CacheController::~CacheController() {
   if ( _numCopies > 0 ) 
      delete[] _cacheCopies;
}

void CacheController::copyDataOut() {
   unsigned int index;
  
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_CC_CDOUT); );

   for ( index = 0; index < _numCopies; index += 1 ) {
      CacheCopy &ccopy = _cacheCopies[ index ];

      if ( !ccopy.getCopyData().isOutput() ) continue;
   
      // TODO: WriteThrough code

      //Region reg = NewRegionDirectory::build_region( *ccopy._copy );
      //std::cerr << "Adding version "<< ccopy._version << " for addr " << ccopy._copy->getBaseAddress() << std::endl;
      //_directory->addAccess( reg, ccopy._copy->isInput(), ccopy._copy->isOutput(), 0, ((uint64_t)ccopy._copy->getBaseAddress()) + ccopy._copy->getOffset(), ccopy._version + 1 );

         //Region origReg = NewRegionDirectory::build_region( *ccopy._copy );
      //   _targetCache->syncRegion( reg, (ccopy._cacheEntry->address + ccopy._offset) + ccopy._copy->getOffset() );
   }
       
   NANOS_INSTRUMENT( inst2.close(); );
}
