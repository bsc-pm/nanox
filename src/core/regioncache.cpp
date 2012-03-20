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

#include "regioncache_decl.hpp"
#include "workdescriptor_decl.hpp"
#include "debug.hpp"
#include "memorymap.hpp"
#include "copydata.hpp"
#include "atomic.hpp"
#include "processingelement.hpp"
#include "regiondirectory.hpp"
#include "regiontree.hpp"

using namespace nanos; 

CachedRegionStatus::CachedRegionStatus() : _status( READY ), _version( 0 ) {}
CachedRegionStatus::CachedRegionStatus( CachedRegionStatus const &rs ) : _status( rs._status ), _version( rs._version ) {}
CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus const &rs ) { _status = rs._status; _version = rs._version; return *this; }
unsigned int CachedRegionStatus::getVersion() { return _version; }
//unsigned int CachedRegionStatus::getStatus() { return _status; }
void CachedRegionStatus::setVersion( unsigned int version ) { _version = version; }
//void CachedRegionStatus::setStatus( unsigned int status ) { _status = status; }
void CachedRegionStatus::setCopying( ) { _status = COPYING; }
bool CachedRegionStatus::isReady( ) { return _status == READY; }

void AllocatedChunk::addRegion( Region reg, std::list< std::pair< Region, CachedRegionStatus const &> > &outList ) {
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
      outList.push_back( std::make_pair( accessor.getRegion(), cachedReg ) );
   } 
}

bool AllocatedChunk::isReady( Region reg )
{
   bool entryReady = true;
   RegionTree<CachedRegionStatus>::iterator_list_t outs;
   //RegionTree<CachedRegionStatus>::iterator ret;
   _regions.find( reg, outs );
   if ( outs.empty () ) {
      message0("ERROR: Got no regions from AllocatedChunk!!");
   } else {
      //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
      RegionTree<CachedRegionStatus>::iterator_list_t::iterator it = outs.begin();
      RegionTree<CachedRegionStatus>::iterator &firstAccessor = *it;
      Region tmpReg = firstAccessor.getRegion();
      bool combiningIsGoingOk = true;

      for ( ; ( it != outs.end() ) && ( combiningIsGoingOk ) && ( entryReady ); it++) {
         RegionTree<CachedRegionStatus>::iterator &accessor = *it;
         combiningIsGoingOk = tmpReg.combine( accessor.getRegion(), tmpReg );
         CachedRegionStatus &status = *accessor;
         entryReady = entryReady && status.isReady();
      }
      if ( combiningIsGoingOk ) {
         if ( tmpReg != reg && !tmpReg.contains( reg ) ) {
            message0("ERROR: Region not found in the Allocated chunk!!!");
         } else { }
      } else {
         message0("ERROR: Region not found in the Allocated chunk!!! unable to combine return regions!");
      }
   }

   return entryReady;
}

void AllocatedChunk::setCopying( Region reg )
{
   RegionTree<CachedRegionStatus>::iterator_list_t outs;
   //RegionTree<CachedRegionStatus>::iterator ret;
   _regions.find( reg, outs );
   if ( outs.empty () ) {
      message0("ERROR: Got no regions from AllocatedChunk!!");
   } else {
      //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
      RegionTree<CachedRegionStatus>::iterator_list_t::iterator it = outs.begin();
      RegionTree<CachedRegionStatus>::iterator &firstAccessor = *it;
      Region tmpReg = firstAccessor.getRegion();
      bool combiningIsGoingOk = true;

      for ( ; ( it != outs.end() ) && ( combiningIsGoingOk ); it++) {
         RegionTree<CachedRegionStatus>::iterator &accessor = *it;
         combiningIsGoingOk = tmpReg.combine( accessor.getRegion(), tmpReg );
         CachedRegionStatus &status = *accessor;
         status.setCopying();
      }
      if ( combiningIsGoingOk ) {
         if ( tmpReg != reg && !tmpReg.contains( reg ) ) {
            message0("ERROR: Region not found in the Allocated chunk!!!");
         } else { }
      } else {
         message0("ERROR: Region not found in the Allocated chunk!!! unable to combine return regions!");
      }
   }
}

AllocatedChunk *RegionCache::getAddress( CopyData const &cd, uint64_t &offset ) {
   ChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   //_lock.acquire();
   //std::cerr << " addChunk " << ( void * ) cd.getBaseAddress() << " size " << cd.getMaxSize() << std::endl;
   _chunks.getOrAddChunk( (uint64_t) cd.getBaseAddress(), cd.getMaxSize(), results ); //we dont want to create new entries if a bigger one already exists!!! FIXME
   if ( results.size() != 1 ) {
      message0( "Got results.size()="<< results.size() << " I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ChunkList::iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
   } else {
      if ( *(results.front().second) == NULL ) {
         //message0("Address not found in cache, DeviceAlloc!! max size is " << cd.getMaxSize());
         *(results.front().second) = NEW AllocatedChunk();
         (*results.front().second)->address = (uint64_t) _device->memAllocate( cd.getMaxSize(), _pe );

         allocChunkPtr = *(results.front().second);

         offset = 0;
   //std::cerr << "offset(cd) is " << offset << std::endl;

         //allocChunkPtr->addRegion( devReg, outs  );
         
      } else {
         //addr = (*results.front().second)->address;
         allocChunkPtr = *(results.front().second);
         offset = ((uint64_t) cd.getBaseAddress() - (uint64_t) (results.front().first)->getAddress());
   //std::cerr << "offset(cd+alloc) is " << offset << " cd.Base " << (void *) cd.getBaseAddress() << " res " <<(void*)(results.front().first)->getAddress() << std::endl;

         //devReg = NewRegionDirectory::build_region_with_given_base_address( cd, 0 );
     //    std::cerr << " AllocatedChunk found in cache, this means that there is contiguous space already allocated, but region may not be present"<< std::endl; 

      }
   }
   //_lock.release();
   return allocChunkPtr;
}

AllocatedChunk *RegionCache::getAddress( uint64_t hostAddr, std::size_t len, uint64_t &offset ) {
   ConstChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   _chunks.getChunk3( hostAddr, len, results );
   if ( results.size() != 1 ) {
      message0( "I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
   } else {
      if ( *(results.front().second) == NULL ) {
         message0("Address not found in cache, Error!! ");
      } else {
         //addr = (*results.front().second)->address;
         allocChunkPtr = *(results.front().second);
         offset = hostAddr - (uint64_t) (results.front().first)->getAddress();
         //std::cerr << "Host addr requested is "<< (void *) hostAddr << " offset computed is " << offset << " - base host addr is " << (void *) (results.front().first)->getAddress() << " chunk addr is " << (void *)allocChunkPtr->address << std::endl;
      }
   }
   return allocChunkPtr;
}
   
void RegionCache::putRegion( CopyData const &cd, Region const &r ) {
   //uintptr_t baseAddress = d.getAddress();
}

void RegionCache::syncRegion( Region const &origReg, uint64_t devAddr ) {
   MemoryMap< uint64_t > thisCopysOps;
   uint64_t offset;
   DeviceOps *ops = NEW DeviceOps();

   //uint64_t fragmentOffset = origReg.getFirstValue() - (uint64_t)ccopy._copy->getBaseAddress();
   //std::cerr << "Copy base addr " <<  (void *) ccopy._copy->getBaseAddress() << std::endl;
   //std::cerr << "Copy  addr " <<  (void *) ccopy._copy->getAddress() << std::endl;
   //std::cerr << "First value " <<  (void *) origReg.getFirstValue() << std::endl;
   //(void) devAddr;

   AllocatedChunk *chunk = getAddress( (uint64_t) ((uint64_t) origReg.getFirstValue()), ((std::size_t) origReg.getBreadth()), offset );

   uint64_t devAddr2 = chunk->address + offset;
   //std::cerr << "Arguument addr " << (void *) devAddr <<" getAddr dev addr is " << (void *) devAddr2 << " origReg first value is "  << (void *)origReg.getFirstValue() << std::endl;

   std::size_t skipBits = 0;
   std::size_t numChunks = origReg.getNumNonContiguousChunks( skipBits );
   std::size_t contiguousSize = origReg.getContiguousChunkLength();

   //std::cerr << " Region chunk is " << origReg << std::endl;
   //std::cerr << " getNumNonContiguousChunks of this region is " << numChunks << std::endl;

   for ( unsigned int chunkIndex = 0; chunkIndex < numChunks; chunkIndex += 1 )
   {
      MemoryMap< uint64_t >::MemChunkList results;
      MemoryMap< uint64_t >::MemChunkList::iterator resultsIt;
      uint64_t address = origReg.getNonContiguousChunk( chunkIndex, skipBits ) /*+ fragmentOffset*/;
      uint64_t chunkDevAddr = devAddr2 + ( address - origReg.getFirstValue() );

      thisCopysOps.getOrAddChunk( address, contiguousSize, results );
      //std::cerr << "Sync haddr " << (void *) address <<" from dev addr " << (void *) chunkDevAddr << std::endl;
      for ( resultsIt = results.begin(); resultsIt != results.end(); resultsIt++ ) {
         if ( *(resultsIt->second) == NULL ) { ops->addOp(); *(resultsIt->second) = NEW uint64_t; **(resultsIt->second) = chunkDevAddr ; }
         else std::cerr << "Mmm..." <<std::endl;
      }
   }

   //std::cerr << "copy outs from " << _targetCache->getMemorySpaceId() << std::endl;
   MemoryMap< uint64_t >::iterator thisMapOpsIt;
   for ( thisMapOpsIt = thisCopysOps.begin(); thisMapOpsIt != thisCopysOps.end(); thisMapOpsIt++ ) {
      //std::cerr << "a copy out " << (void *) thisMapOpsIt->first.getAddress() << " from " << (void *)*(thisMapOpsIt->second) << " size " << thisMapOpsIt->first.getLength() << std::endl;
      this->copyOut(thisMapOpsIt->first.getAddress(), *(thisMapOpsIt->second), thisMapOpsIt->first.getLength(), ops );
   }

   while( !ops->allCompleted() ) {}
   delete ops;
}

void RegionCache::discardRegion( CopyData const &cd, Region const &r ) {
}

void RegionCache::setDevice( Device *d ) {
   _device = d;
}

void RegionCache::setPE( ProcessingElement *pe ) {
   _pe = pe;
}

unsigned int RegionCache::getMemorySpaceId() {
   return _pe->getMemorySpaceId();
}

void RegionCache::copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops ) {
   _device->_copyIn( devAddr, hostAddr, len, _pe, ops );
}
void RegionCache::copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops ) {
   _device->_copyOut( devAddr, hostAddr, len, _pe, ops );
}
void RegionCache::syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops ) {
   uint64_t offset;
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->address + offset;
   cout->addOp();
   sys.getCaches()[ syncFrom ]->copyOut( hostAddr, origDevAddr, len, cout );
   while ( !cout->allCompleted() ){}
   delete cout;
   this->copyIn( devAddr, hostAddr, len, ops );
}
void RegionCache::copyDevToDev( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops ) {
   uint64_t offset;
   AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->address + offset;
   std::cerr << "Copy Dev To Dev dest "<< _pe->getMemorySpaceId() << ": " << (void *) devAddr << " origAddr " << copyFrom <<": " << (void *) origDevAddr <<  std::endl;
   _device->_copyDevToDev( devAddr, origDevAddr, len, _pe, sys.getCaches()[ copyFrom ]->_pe, ops );
}
void RegionCache::lock() { _lock.acquire(); }
void RegionCache::unlock() { _lock.release(); }
bool RegionCache::tryLock() { return _lock.tryAcquire(); }
bool RegionCache::canCopyFrom( RegionCache const &from ) const { return _device == from._device; }


DeviceOps::DeviceOps() : _pendingDeviceOps ( 0 ) { }
void DeviceOps::completeOp() { _pendingDeviceOps--; }
void DeviceOps::addOp() { _pendingDeviceOps++; }
bool DeviceOps::allCompleted() { return _pendingDeviceOps.value() == 0; }

CacheControler::CacheControler() : _numCopies ( 0 ), _cacheCopies ( NULL ), _targetCache ( NULL ) {}

bool CacheControler::isCreated() const {
   return _targetCache != NULL;
}

void CacheControler::create( RegionCache *targetCache, NewDirectory *dir, std::size_t numCopies, CopyData *copies ) {
   unsigned int index;

   _directory = dir;
#if 0
   class AccessGenerator {
      void generateAccesses( uint64_t baseAddress, unsigned int currentDimension, nanos_region_dimension_internal_t *dimensions, MemoryMap< uint64_t > &thisCopysOps ) {
         if ( currentDimension == 1 ) {
            MemoryMap< uint64_t >::MemChunkList results;
            MemoryMap< uint64_t >::MemChunkList::iterator resultsIt;
               //for each dimension, etc
               thisCopysOps.getOrAddChunk( baseAddress + dimensions[ currentDimension ].lower_bound , dimensions[ currentDimension ].accessed_length, results );
               for ( resultsIt = results.begin(); resultsIt != results.end(); resultsIt++ ) {
                  if ( *(resultsIt->second) == NULL ) *(resultsIt->second) = NEW uint64_t( ccopy._cacheEntry->address + ccopy._offset );
                  else std::cerr << "Mmm..." <<std::endl;
               }
            
         } else {
            generateAccesses()
         }
      }
   }
#endif
   //std::cerr << " Creating cache controler!" << std::endl;
   if ( numCopies > 0 ) {
      _numCopies = numCopies;
      _targetCache = targetCache;
      _cacheCopies = NEW CacheCopy[ _numCopies ];
      for ( index = 0; index < _numCopies; index += 1 ) {
         //std::cerr << "Copy base addr " <<  (void *) copies[index].getBaseAddress() << std::endl;
         //std::cerr << "Copy base addr " <<  (void *) copies[index].getAddress() << std::endl;
         uint64_t devAddr;
         _cacheCopies[ index ]._copy = &copies[ index ];
         _cacheCopies[ index ]._devRegion = NewRegionDirectory::build_region_with_given_base_address( *_cacheCopies[ index ]._copy, 0 );

         _cacheCopies[ index ]._cacheEntry = _targetCache->getAddress( *_cacheCopies[ index ]._copy, _cacheCopies[ index ]._offset );

         _cacheCopies[ index ]._cacheEntry->addRegion( _cacheCopies[ index ]._devRegion, _cacheCopies[ index ]._cacheDataStatus );
         devAddr = _cacheCopies[ index ]._cacheEntry->address + _cacheCopies[ index ]._offset + _cacheCopies[ index ]._copy->getOffset();

         Region reg = NewRegionDirectory::build_region( copies[ index ] );
         _directory->registerAccess( reg, copies[ index ].isInput(), copies[ index ].isOutput(), _targetCache->getMemorySpaceId(), devAddr , _cacheCopies[ index ]._locations );
         //std::cerr << "Got locs: " << _cacheCopies[ index ]._locations.size() << " This copy is in " << *(_cacheCopies[ index ]._locations.front().second) << std::endl;
      }




      /* COPY IN GENERATION */


      // generate ops
      //class MergeableInt {
      //   private:
      //      int _datum;
      //   public:
      //      MergeableInt() : _datum( 0 ) {}
      //      MergeableInt( int datum ) : _datum( datum ) {}
      //      MergeableInt( MergeableInt const &mi ) : _datum( mi._datum ) {}
      //      MergeableInt &operator=( MergeableInt const &mi ) { datum = mi._datum; return *this; }
      //      bool equal( MergeableInt const &i ) { return i._datum == _datum; }
      //      void merge( MergeableInt const &i ) { if ( i._datum != _datum ) std::cerr << "Bad usage" << std:.endl; }
      //}
      std::map<unsigned int, MemoryMap< std::pair< uint64_t, DeviceOps *> > > opsBySource;

      for ( index = 0; index < _numCopies; index += 1 ) {
         CacheCopy &ccopy = _cacheCopies[ index ];
         ccopy._version = 1;

         if ( !ccopy._copy->isInput() ) continue;
         
         ccopy._cacheEntry->setCopying( ccopy._devRegion );

         NewRegionDirectory::LocationInfoList::iterator it;
         for ( it = ccopy._locations.begin(); it != ccopy._locations.end(); it++ ) {
            ccopy._version = std::max( ccopy._version, it->second->getVersion() );
            //if ( it->second->isLocatedIn( _targetCache->getMemorySpaceId() )) std::cerr << "To run in loc " << _targetCache->getMemorySpaceId() << " Region: " << (it->first) << " nded:  "<< *(it->second) <<  std::endl;
            if ( it->second->isLocatedIn( _targetCache->getMemorySpaceId() )) continue; // FIXME: version info, (I think its not needed because directory stores only the last version, if an old version is stored, it wont be reported but _targetCache.getAddress will return the already allocated storage)
            //std::cerr << " I have to copy region " << it->first <<  " into " << _targetCache->getMemorySpaceId() << " version is " << it->second->getVersion() << std::endl;
            //std::cerr << " Dest dev region is    " << ccopy._devRegion << " dev addr is " << (void *) (ccopy._cacheEntry->address + ccopy._offset) << std::endl;
            {
               MemoryMap< std::pair< uint64_t, DeviceOps *> > &thisCopysOps = opsBySource[ it->second->getFirstLocation() ];
               
               Region &origReg = it->first;
               //uint64_t fragmentOffset = origReg.getFirstValue() - (uint64_t)ccopy._copy->getBaseAddress();

               std::size_t skipBits = 0;
               std::size_t numChunks = origReg.getNumNonContiguousChunks( skipBits );
               std::size_t contiguousSize = origReg.getContiguousChunkLength();

               //std::cerr << " Region chunk is " << origReg << std::endl;
               //std::cerr << " getNumNonContiguousChunks of this region is " << numChunks << std::endl;

               for ( unsigned int chunkIndex = 0; chunkIndex < numChunks; chunkIndex += 1 )
               {
                  MemoryMap< std::pair< uint64_t, DeviceOps *> >::MemChunkList results;
                  MemoryMap< std::pair< uint64_t, DeviceOps *> >::MemChunkList::iterator resultsIt;
                  uint64_t address = origReg.getNonContiguousChunk( chunkIndex, skipBits ) /*+ fragmentOffset*/;
                  uint64_t devAddr = (ccopy._cacheEntry->address + ccopy._offset) + ccopy._copy->getOffset() + ( address - origReg.getFirstValue() );

                  thisCopysOps.getOrAddChunk( address, contiguousSize, results );
                  for ( resultsIt = results.begin(); resultsIt != results.end(); resultsIt++ ) {
                     if ( *(resultsIt->second) == NULL ) { ccopy._operations.addOp(); *(resultsIt->second) = NEW std::pair< uint64_t, DeviceOps *>( devAddr, &ccopy._operations ); }
                     else std::cerr << "Mmm..." <<std::endl;
                  }
               }
            }
         }
      }

      std::map<unsigned int, MemoryMap< std::pair< uint64_t, DeviceOps *> > >::iterator mapOpsIt;
      for ( mapOpsIt = opsBySource.begin(); mapOpsIt != opsBySource.end(); mapOpsIt++ ) {
         //std::cerr << "ops from " << mapOpsIt->first << " to " << _targetCache->getMemorySpaceId() << std::endl;
         unsigned int location = mapOpsIt->first;
         MemoryMap< std::pair< uint64_t, DeviceOps *> >::iterator thisMapOpsIt;
         for ( thisMapOpsIt = mapOpsIt->second.begin(); thisMapOpsIt != mapOpsIt->second.end(); thisMapOpsIt++ ) {
            //double *sample = (double *) thisMapOpsIt->first.getAddress();
            //std::cerr << "a copy "<< (void *) thisMapOpsIt->first.getAddress() << " ( " << *sample << " ) " << " to " << (void *)thisMapOpsIt->second->first << " size " << thisMapOpsIt->first.getLength() << std::endl;
            if ( location == 0 ) {
               _targetCache->copyIn(thisMapOpsIt->second->first, thisMapOpsIt->first.getAddress(), thisMapOpsIt->first.getLength(), thisMapOpsIt->second->second );
            } else if ( _targetCache->canCopyFrom( *sys.getCaches()[ location ] ) ) {
               _targetCache->copyDevToDev( location, thisMapOpsIt->second->first, thisMapOpsIt->first.getAddress(), thisMapOpsIt->first.getLength(), thisMapOpsIt->second->second );
            } else {
               //FIXME: actualitzar directory 0!
               std::cerr <<" copy Sync to loc 0 ............................. " << std::endl;
               _targetCache->syncAndCopyIn( location, thisMapOpsIt->second->first, thisMapOpsIt->first.getAddress(), thisMapOpsIt->first.getLength(), thisMapOpsIt->second->second );
            }
         }
      }
   }

}

bool CacheControler::dataIsReady() const {
   bool allReady = true;
   unsigned int index;
   if ( _targetCache == NULL ) return true;
   for ( index = 0; ( index < _numCopies ) && allReady; index += 1 ) {
      
      allReady = allReady && _cacheCopies[ index ]._operations.allCompleted();
   }
   //std::cerr << "Ckec readiness... " << allReady << std::endl;
   return allReady;
}

uint64_t CacheControler::getAddress( unsigned int copyId ) const {
   //std::cerr << "GetAddress " << copyId << " addr is " << (void *)_cacheCopies[ copyId ]._cacheEntry->address << " alloc_off " << _cacheCopies[ copyId ]._offset << " copy off " <<_cacheCopies[ copyId ]._copy->getOffset() << " all " << (void *) (_cacheCopies[ copyId ]._cacheEntry->address + _cacheCopies[ copyId ]._offset + _cacheCopies[ copyId ]._copy->getOffset() )<<std::endl;
   return _cacheCopies[ copyId ]._cacheEntry->address + _cacheCopies[ copyId ]._offset + _cacheCopies[ copyId ]._copy->getOffset();
}


void CacheControler::copyDataOut() {
   unsigned int index;
   for ( index = 0; index < _numCopies; index += 1 ) {
      CacheCopy &ccopy = _cacheCopies[ index ];

      if ( !ccopy._copy->isOutput() ) continue;

      //Region reg = NewRegionDirectory::build_region( *ccopy._copy );
      //std::cerr << "Adding version "<< ccopy._version << " for addr " << ccopy._copy->getBaseAddress() << std::endl;
      //_directory->addAccess( reg, ccopy._copy->isInput(), ccopy._copy->isOutput(), 0, ((uint64_t)ccopy._copy->getBaseAddress()) + ccopy._copy->getOffset(), ccopy._version + 1 );

         //Region origReg = NewRegionDirectory::build_region( *ccopy._copy );
      //   _targetCache->syncRegion( reg, (ccopy._cacheEntry->address + ccopy._offset) + ccopy._copy->getOffset() );
      //{
      //   MemoryMap< uint64_t > thisCopysOps;
      //   Region origReg = NewRegionDirectory::build_region( *ccopy._copy );
      //   //uint64_t fragmentOffset = origReg.getFirstValue() - (uint64_t)ccopy._copy->getBaseAddress();
      //   //std::cerr << "Copy base addr " <<  (void *) ccopy._copy->getBaseAddress() << std::endl;
      //   //std::cerr << "Copy  addr " <<  (void *) ccopy._copy->getAddress() << std::endl;
      //   //std::cerr << "First value " <<  (void *) origReg.getFirstValue() << std::endl;

      //   std::size_t skipBits = 0;
      //   std::size_t numChunks = origReg.getNumNonContiguousChunks( skipBits );
      //   std::size_t contiguousSize = origReg.getContiguousChunkLength();

      //   //std::cerr << " Region chunk is " << origReg << std::endl;
      //   //std::cerr << " getNumNonContiguousChunks of this region is " << numChunks << std::endl;

      //   for ( unsigned int chunkIndex = 0; chunkIndex < numChunks; chunkIndex += 1 )
      //   {
      //      MemoryMap< uint64_t >::MemChunkList results;
      //      MemoryMap< uint64_t >::MemChunkList::iterator resultsIt;
      //      uint64_t address = origReg.getNonContiguousChunk( chunkIndex, skipBits ) /*+ fragmentOffset*/;
      //      uint64_t devAddr = (ccopy._cacheEntry->address + ccopy._offset) + ccopy._copy->getOffset() + ( address - origReg.getFirstValue() );

      //      thisCopysOps.getOrAddChunk( address, contiguousSize, results );
      //      for ( resultsIt = results.begin(); resultsIt != results.end(); resultsIt++ ) {
      //         if ( *(resultsIt->second) == NULL ) { *(resultsIt->second) = NEW uint64_t; **(resultsIt->second) = devAddr ; }
      //         else std::cerr << "Mmm..." <<std::endl;
      //      }
      //   }

      //   //std::cerr << "copy outs from " << _targetCache->getMemorySpaceId() << std::endl;
      //   MemoryMap< uint64_t >::iterator thisMapOpsIt;
      //   for ( thisMapOpsIt = thisCopysOps.begin(); thisMapOpsIt != thisCopysOps.end(); thisMapOpsIt++ ) {
      //      //std::cerr << "a copy out " << (void *) thisMapOpsIt->first.getAddress() << " from " << (void *)*(thisMapOpsIt->second) << " size " << thisMapOpsIt->first.getLength() << std::endl;
      //      _targetCache->copyOut(thisMapOpsIt->first.getAddress(), *(thisMapOpsIt->second), thisMapOpsIt->first.getLength() );
      //   }
      //}
   }
}
//void CacheControler::executeOps()
//{
//}
