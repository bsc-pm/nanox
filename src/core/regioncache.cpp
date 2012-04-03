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
#include "system.hpp"
#ifdef GPU_DEV
#include "gpudd.hpp"
#endif
#ifdef CLUSTER_DEV
//#include "clusterdevice_decl.hpp"
#endif

using namespace nanos; 
DeviceOpsPtr::~DeviceOpsPtr() {
   if ( _value != NULL)  {
      _value->delRef( this );
      //sys.printBt();
      //std::cerr << __FUNCTION__ << " maybe I should delete my ref."<< std::endl;
   }
}

CachedRegionStatus::CachedRegionStatus() : _status( READY ), _version( 1 ) {}
CachedRegionStatus::CachedRegionStatus( CachedRegionStatus const &rs ) : _status( rs._status ), _version( rs._version ), _waitObject ( rs._waitObject ) {}
CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus const &rs ) { _status = rs._status; _version = rs._version; _waitObject = rs._waitObject;  return *this; }
CachedRegionStatus::CachedRegionStatus( CachedRegionStatus &rs ) : _status( rs._status ), _version( rs._version ), _waitObject ( rs._waitObject ) {}
CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus &rs ) { _status = rs._status; _version = rs._version; _waitObject = rs._waitObject;  return *this; }
unsigned int CachedRegionStatus::getVersion() { return _version; }
//unsigned int CachedRegionStatus::getStatus() { return _status; }
void CachedRegionStatus::setVersion( unsigned int version ) { _version = version; }
//void CachedRegionStatus::setStatus( unsigned int status ) { _status = status; }
void CachedRegionStatus::setCopying( DeviceOps *ops ) { _waitObject.set( ops );  }
DeviceOps *CachedRegionStatus::getDeviceOps() { return _waitObject.get();  }
bool CachedRegionStatus::isReady( ) { return _waitObject.isNotSet(); }


      AllocatedChunk::AllocatedChunk() : _lock(), address( NULL ) { }
      AllocatedChunk::AllocatedChunk( AllocatedChunk const &chunk ) : _lock(), address( chunk.address ) {}
      AllocatedChunk &AllocatedChunk::operator=( AllocatedChunk const &chunk ) { address = chunk.address; return *this; } 
void AllocatedChunk::lock() { _lock.acquire(); }
void AllocatedChunk::unlock() { _lock.release(); }
//void AllocatedChunk::addReadRegion( Region reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::list< std::pair< Region, tr1::shared_ptr< DeviceOps > > > &notPresentRegions, DeviceOps *ops ) {
void AllocatedChunk::addReadRegion( Region reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::list< Region > &notPresentRegions, DeviceOps *ops, bool alsoWriteRegion ) {
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
      //outList.push_back( std::make_pair( accessor.getRegion(), cachedReg ) );
      if ( version == cachedReg.getVersion() )  {
         if ( cachedReg.isReady() ) {
            /* already in cache */
            //std::cerr <<"cached region! v: "<< version <<" "<< accessor.getRegion()<<std::endl;
         } else {
            /* in cache but comming! */
            //incomingRegions.push_back( accessor.getRegion() );
            currentOps.insert( cachedReg.getDeviceOps() );
            //std::cerr <<"Already comming region! v: "<< version <<" "<< accessor.getRegion()<<std::endl;
         }
      } else {
         /* not present! */
         notPresentRegions.push_back( accessor.getRegion() );
         //cachedReg.setCopying();
         cachedReg.setCopying( ops );
         cachedReg.setVersion(version);
         //std::cerr <<"Not present region! v: "<< version <<" "<< accessor.getRegion()<<std::endl;
      }
      if ( alsoWriteRegion ) cachedReg.setVersion( version + 1 );
      //outList.push_back( std::make_pair( accessor.getRegion(), cachedReg ) );
   } 
}
void AllocatedChunk::addWriteRegion( Region reg, unsigned int version ) {
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
      cachedReg.setVersion( version + 1 );
      //outList.push_back( std::make_pair( accessor.getRegion(), cachedReg ) );
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

#if 0
void AllocatedChunk::setCopying( Region reg )
{
#if 0
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
#endif
}
#endif

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
         std::cerr <<"Allocated object " << (void *) results.front().first->getAddress() << " size " <<cd.getMaxSize() << std::endl;

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
   allocChunkPtr->lock();
   //_lock.release();
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
         //addr = (*results.front().second)->address;
         allocChunkPtr = *(results.front().second);
         offset = hostAddr - (uint64_t) (results.front().first)->getAddress();
         //std::cerr << "Host addr requested is "<< (void *) hostAddr << " offset computed is " << offset << " - base host addr is " << (void *) (results.front().first)->getAddress() << " chunk addr is " << (void *)allocChunkPtr->address << std::endl;
      }
   }
   allocChunkPtr->lock();
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
   chunk->unlock();
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

      thisCopysOps.getOrAddChunk2( address, contiguousSize, results );
      //std::cerr << "Sync haddr " << (void *) address <<" from dev addr " << (void *) chunkDevAddr << " loc " <<_pe->getMemorySpaceId() << std::endl;
      for ( resultsIt = results.begin(); resultsIt != results.end(); resultsIt++ ) {
         if ( *(resultsIt->second) == NULL ) { ops->addOp(); *(resultsIt->second) = NEW uint64_t; **(resultsIt->second) = chunkDevAddr ; }
         else if ( resultsIt->first->getAddress() != address && resultsIt->first->getLength() != contiguousSize  ) {
            std::cerr << "Mmm... SR size is " << results.size() << " addr is " << (void *) resultsIt->first->getAddress() << " requested " << (void *) address << " len " << resultsIt->first->getLength() << " requested " << contiguousSize<< std::endl;
         } else {
            std::cerr << "Mmm2... SR size is " << results.size() << " addr is " << (void *) resultsIt->first->getAddress() << " requested " << (void *) address << " len " << resultsIt->first->getLength() << " requested " << contiguousSize<< std::endl;
         }
      }
   }

   //std::cerr << "copy outs from " << _targetCache->getMemorySpaceId() << std::endl;
   MemoryMap< uint64_t >::iterator thisMapOpsIt;
   for ( thisMapOpsIt = thisCopysOps.begin(); thisMapOpsIt != thisCopysOps.end(); thisMapOpsIt++ ) {
      //std::cerr << "a copy out " << (void *) thisMapOpsIt->first.getAddress() << " from " << (void *)*(thisMapOpsIt->second) << " size " << thisMapOpsIt->first.getLength() << std::endl;
      this->copyOut(thisMapOpsIt->first.getAddress(), *(thisMapOpsIt->second), thisMapOpsIt->first.getLength(), ops );
   }
   
   while( !ops->allCompleted() ) { myThread->idle(); }
   delete ops;
}

void RegionCache::discardRegion( CopyData const &cd, Region const &r ) {
}

void RegionCache::setDevice( Device *d ) {
   _device = d;
}

void RegionCache::setPE( ProcessingElement *pe ) {
   _pe = pe;
if(sys.getNetwork()->getNodeNum() == 0) std::cerr << "IM CACHE "<< _pe->getMemorySpaceId() << ( _device == &ext::GPU  ? " GPU " : " Cluster" ) << std::endl;
}

unsigned int RegionCache::getMemorySpaceId() {
   return _pe->getMemorySpaceId();
}

void RegionCache::copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId ) {
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<"::copyIn "<< (void *) devAddr << " <=h " << (void *) hostAddr << " len " << len << " ops complete? "  << ( ops->allCompleted() ? "yes" : "no" )<<std::endl;
   _device->_copyIn( devAddr, hostAddr, len, _pe, ops, wdId );
}
void RegionCache::copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops ) {
   _device->_copyOut( hostAddr, devAddr, len, _pe, ops );
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<">::copyOut "<< (void *) devAddr << " =>h " << (void *) hostAddr << " len " << len << " ops complete? "  << ( ops->allCompleted() ? "yes" : "no" )<<std::endl;
}
void RegionCache::syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId ) {
   uint64_t offset;
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->address + offset;
   origChunk->unlock();
   cout->addOp();
   //if(sys.getNetwork()->getNodeNum() == 0) std::cerr <<sys.getNetwork()->getNodeNum() <<" Started a Copy out from " << syncFrom << std::endl;
   sys.getCaches()[ syncFrom ]->copyOut( hostAddr, origDevAddr, len, cout );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() <<" CopyOut from " << syncFrom << " completed" << std::endl;
   delete cout;
   this->copyIn( devAddr, hostAddr, len, ops, wdId );
}
void RegionCache::copyDevToDev( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId ) {
   uint64_t offset;
   AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->address + offset;
   origChunk->unlock();
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() << " Copy Dev To Dev dest "<< _pe->getMemorySpaceId() << ": " << (void *) devAddr << " origAddr " << copyFrom <<": " << (void *) origDevAddr <<  std::endl;
   _device->_copyDevToDev( devAddr, origDevAddr, len, _pe, sys.getCaches()[ copyFrom ]->_pe, ops, wdId );
}
void RegionCache::lock() { _lock.acquire(); }
void RegionCache::unlock() { _lock.release(); }
bool RegionCache::tryLock() { return _lock.tryAcquire(); }
bool RegionCache::canCopyFrom( RegionCache const &from ) const { return _device == from._device; }


unsigned int DeviceOps::getNumOps() { return _pendingDeviceOps.value(); }
DeviceOps::DeviceOps() : _pendingDeviceOps ( 0 ) { }
DeviceOps::DeviceOps(std::string &s) :  _pendingDeviceOps ( 0 ),_desc (s) { }
void DeviceOps::completeOp() {
   unsigned int value = _pendingDeviceOps--;
   if ( value == 1 ) {
      _lock.acquire();
      if ( !_refs.empty() ) {
         //std::cerr <<"ive got " << _refs.size() << " refs to clear" << std::endl; 
         for ( std::set<DeviceOpsPtr *>::iterator it = _refs.begin(); it != _refs.end(); it++ ) {
            (*it)->clear();
         }
         _refs.clear();
      }
      _lock.release();
   } else if ( value == 0 ) std::cerr << "overflow!!! "<< (void *)this << std::endl;
   /*std::cerr << "op--! " << (void *) this <<std::endl; sys.printBt();*/
}
void DeviceOps::addOp() {
   _pendingDeviceOps++;
   /*std::cerr << "op++! " << (void *) this <<std::endl;*/
}
bool DeviceOps::allCompleted() {
   return ( _pendingDeviceOps.value() == 0);
}
bool DeviceOps::addRef( DeviceOpsPtr *opsPtr, DeviceOpsPtr &p ) {
  /* add the reference only if "p" is already inside */
   bool result = false;
   _lock.acquire();
   if ( _refs.count( &p ) == 1 ) {
      _refs.insert( opsPtr );
      result = true;
   }
   _lock.release();
   return result;
}

void DeviceOps::delRef( DeviceOpsPtr *opsPtr ) {
  _lock.acquire();
  _refs.erase( opsPtr );
  _lock.release();
}
void DeviceOps::addFirstRef( DeviceOpsPtr *opsPtr ) {
   _lock.acquire();
  // std::cerr <<"Add first ref: object "<<(void *) opsPtr << std::endl;
   _refs.insert( opsPtr );
   _lock.release();
}
DeviceOps::~DeviceOps() {}

CacheControler::CacheControler() : _numCopies ( 0 ), _cacheCopies ( NULL ), _targetCache ( NULL ) {}

bool CacheControler::isCreated() const {
   return _targetCache != NULL;
}

void CacheControler::create( RegionCache *targetCache, NewDirectory *dir, std::size_t numCopies, CopyData *copies, unsigned int wdId ) {
   unsigned int index;

   _directory = dir;
   _wdId = wdId;
   //std::cerr << " Creating cache controler!" << std::endl;
   if ( numCopies > 0 ) {
      _numCopies = numCopies;
      _targetCache = targetCache;
      _cacheCopies = NEW CacheCopy[ _numCopies ];
      for ( index = 0; index < _numCopies; index += 1 ) {
         CacheCopy &ccopy = _cacheCopies[ index ];
         //std::cerr << "Copy base addr " <<  (void *) copies[index].getBaseAddress() << std::endl;
         //std::cerr << "Copy base addr " <<  (void *) copies[index].getAddress() << std::endl;
         uint64_t devAddr;
         ccopy._copy = &copies[ index ];

         ccopy._cacheEntry = _targetCache->getAddress( *ccopy._copy, ccopy._offset );
         ccopy._devRegion = NewRegionDirectory::build_region_with_given_base_address( *ccopy._copy, ccopy._offset );

         devAddr = ccopy._cacheEntry->address + ccopy._offset + ccopy._copy->getOffset();

         ccopy._region = NewRegionDirectory::build_region( copies[ index ] );
         //std::cerr << "copy " << index << " Region " << reg << std::endl;
         _directory->registerAccess( ccopy._region, copies[ index ].isInput(), copies[ index ].isOutput(), _targetCache->getMemorySpaceId(), devAddr , ccopy._locations );
         ccopy._cacheEntry->unlock();
         //std::cerr << "Got locs: " << ccopy._locations.size() << " This copy is in " << *(ccopy._locations.front().second) << std::endl;
      }

      /* COPY IN GENERATION */
      std::map<unsigned int, MemoryMap< std::pair< uint64_t, DeviceOps *> > > opsBySource;
      for ( index = 0; index < _numCopies; index += 1 ) {
	 NewRegionDirectory::LocationInfoList::iterator it;
         CacheCopy &ccopy = _cacheCopies[ index ];
         //ccopy._version = 1;
         ccopy._cacheEntry->lock();
         if ( ccopy._copy->isInput() )
         {
         //ccopy._cacheEntry->addRegion( ccopy._devRegion, ccopy._copy->isInput(),  ccopy._cacheDataStatus );
          //  ccopy._cacheEntry->setCopying( ccopy._devRegion );

//ccopy._operations._desc = "Copy in for wd ";
               //std::stringstream ss;
            for ( it = ccopy._locations.begin(); it != ccopy._locations.end(); it++ ) {

               
               //ccopy._version = std::max( ccopy._version, it->second.getVersion() );

               //if ( it->second->isLocatedIn( _targetCache->getMemorySpaceId() )) std::cerr << "To run in loc " << _targetCache->getMemorySpaceId() << " Region: " << (it->first) << " nded:  "<< *(it->second) <<  std::endl;
               //    if ( it->second.isLocatedIn( _targetCache->getMemorySpaceId() ))
               //    std::cerr << " I have NOT to copy region " << it->first <<  " into " << _targetCache->getMemorySpaceId() << " version is " << it->second.getVersion() << std::endl;
               if ( it->second.isLocatedIn( _targetCache->getMemorySpaceId() )) continue; // FIXME: version info, (I think its not needed because directory stores only the last version, if an old version is stored, it wont be reported but _targetCache.getAddress will return the already allocated storage)
               //std::cerr << " I have to copy region " << it->first <<  " into " << _targetCache->getMemorySpaceId() << " version is " << it->second.getVersion() << std::endl;
               std::list< Region > notPresentRegions;
               //std::list< std::pair < Region, tr1::shared_ptr< DeviceOps > > >::iterator notPresentRegionsIt;
               std::list< Region >::iterator notPresentRegionsIt;
               //std::set< DeviceOps * > currentOps;
               ccopy._cacheEntry->addReadRegion( it->first, it->second.getVersion(), ccopy._otherPendingOps, notPresentRegions, &ccopy._operations, ccopy._copy->isOutput() );

               //std::cerr << " Dest dev region is    " << ccopy._devRegion << " dev addr is " << (void *) (ccopy._cacheEntry->address + ccopy._offset) << std::endl;
               //ss << wdId << " res NotPresentRegsSize is " << notPresentRegions.size() ; 
               for( notPresentRegionsIt = notPresentRegions.begin(); notPresentRegionsIt != notPresentRegions.end(); notPresentRegionsIt++ ) {
               //if ( !ccopy._cacheEntry->alreadyContains( ccopy._devRegion ) )
               {
                  MemoryMap< std::pair< uint64_t, DeviceOps *> > &thisCopysOps = opsBySource[ it->second.getFirstLocation() ];

                  //Region &origReg = it->first;
                  Region &origReg = *notPresentRegionsIt;
                  //uint64_t fragmentOffset = origReg.getFirstValue() - (uint64_t)ccopy._copy->getBaseAddress();

                  std::size_t skipBits = 0;
                  std::size_t numChunks = origReg.getNumNonContiguousChunks( skipBits );
                  std::size_t contiguousSize = origReg.getContiguousChunkLength();

                  //std::cerr << " Region chunk is " << origReg << std::endl;
                  //std::cerr << " getNumNonContiguousChunks of this region is " << numChunks << std::endl;
                  

                  //ss << " contiguous chunks is " << numChunks ;

                  for ( unsigned int chunkIndex = 0; chunkIndex < numChunks; chunkIndex += 1 )
                  {
                     MemoryMap< std::pair< uint64_t, DeviceOps *> >::MemChunkList results;
                     MemoryMap< std::pair< uint64_t, DeviceOps *> >::MemChunkList::iterator resultsIt;
                     uint64_t address = origReg.getNonContiguousChunk( chunkIndex, skipBits ) /*+ fragmentOffset*/;
                     uint64_t devAddr = (ccopy._cacheEntry->address + ccopy._offset) /* entry addr and offset inside the chunk */
                        + (origReg.getFirstValue() - ( ((uint64_t)ccopy._copy->getBaseAddress() + ccopy._copy->getOffset()  )) ) /* displacement due to fragmented region */
                        + ccopy._copy->getOffset() + ( address - origReg.getFirstValue() ); /* contiguous chunk offset */
                     //std::cerr << "["<< index<<"] Entry addr " << (void *) ccopy._cacheEntry->address << " got offset " << ccopy._offset << " copy offset " << ccopy._copy->getOffset() << " addr - first value is "  << ( address - origReg.getFirstValue() ) << " total: " << (void *) devAddr <<" orig reg offset may be " << (origReg.getFirstValue() - ((uint64_t)ccopy._copy->getBaseAddress() + ccopy._copy->getOffset()  ) ) << (void *) devAddr << std::endl;

                     thisCopysOps.getOrAddChunk2( address, contiguousSize, results );
                     //std::cerr << "Copy IN addr=" << (void *) address << " size=" << contiguousSize <<  std::endl;
                     for ( resultsIt = results.begin(); resultsIt != results.end(); resultsIt++ ) {
                        if ( *(resultsIt->second) == NULL ) {
                           ccopy._operations.addOp();
                          // ss<< " "<<(void*)address << " size " << contiguousSize <<" from "<<it->second.getFirstLocation();
                           *(resultsIt->second) = NEW std::pair< uint64_t, DeviceOps *>( devAddr, &ccopy._operations );
                        }
                        else if ( resultsIt->first->getAddress() != address && resultsIt->first->getLength() != contiguousSize  ) {
                           // Ive got a different addr, and len
                          // std::cerr << "Mmm...  size is " << results.size() << " addr is " << (void *) resultsIt->first->getAddress() << " requested " << (void *) address << " len " << resultsIt->first->getLength() << " requested " << contiguousSize<< std::endl;
                        } else if (resultsIt->first->getAddress() == address && resultsIt->first->getLength() == contiguousSize ){
                        } else {
                           //std::cerr << "It doesnt seem to fit...  results size is " << results.size() << " addr is " << (void *) resultsIt->first->getAddress() << " requested " << (void *) address << " len " << resultsIt->first->getLength() << " requested " << contiguousSize<< std::endl;
                           //ss << " NASTY (deplasti) ";
                        }
                     }
                  }
               }
               }
            }
	       //ccopy._operations._desc.append( ss.str() );
               //std::cerr << _wdId << " so far copy " << index <<" waits for " <<ccopy._operations.getNumOps() << " addr is "<< &ccopy._operations<< std::endl;
         } else if ( !ccopy._copy->isInput() && ccopy._copy->isOutput() ) {
            unsigned int currentVersion = 1;
            for ( it = ccopy._locations.begin(); it != ccopy._locations.end(); it++ ) {
               currentVersion = std::max( currentVersion, it->second.getVersion() );
            }
            /* write only region */
             ccopy._cacheEntry->addWriteRegion( ccopy._region, currentVersion + 1 );
         }
         ccopy._cacheEntry->unlock();
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
               _targetCache->copyIn(thisMapOpsIt->second->first, thisMapOpsIt->first.getAddress(), thisMapOpsIt->first.getLength(), thisMapOpsIt->second->second, _wdId );
            } else if ( _targetCache->canCopyFrom( *sys.getCaches()[ location ] ) ) {
               _targetCache->copyDevToDev( location, thisMapOpsIt->second->first, thisMapOpsIt->first.getAddress(), thisMapOpsIt->first.getLength(), thisMapOpsIt->second->second, _wdId );
            } else {
               //FIXME: actualitzar directory 0! (not really needed, but it could be useful)
               //if(sys.getNetwork()->getNodeNum() == 0)std::cerr << sys.getNetwork()->getNodeNum() <<" sync and copyIn target loc "<< location << std::endl;
               _targetCache->syncAndCopyIn( location, thisMapOpsIt->second->first, thisMapOpsIt->first.getAddress(), thisMapOpsIt->first.getLength(), thisMapOpsIt->second->second, _wdId );
            }
         }
      }
   }

}

bool CacheControler::dataIsReady() const {
   bool allReady = true;
   unsigned int index;
   if ( _targetCache == NULL ) return true;
   //for ( index = 0; ( index < _numCopies ) && allReady; index += 1 ) { 
   //   std::cerr << _wdId << " copy " << index << " remain ops " << _cacheCopies[ index ]._operations.getNumOps() << " : " << _cacheCopies[ index ]._operations.allCompleted()<< " addr is "<< &_cacheCopies[index]._operations << std::endl;
   //}
   for ( index = 0; ( index < _numCopies ) && allReady; index += 1 ) {

      if( !_cacheCopies[ index ]._operations.allCompleted() ) { 
       //  std::cerr << "Waiting for " << _cacheCopies[ index ]._operations._desc << std::endl;
         allReady = false;
      }
      //allReady = allReady && _cacheCopies[ index ]._operations.allCompleted();

      if ( allReady ) { 
         std::set< DeviceOps * >::iterator it = _cacheCopies[ index ]._otherPendingOps.begin();
         while ( allReady && it != _cacheCopies[ index ]._otherPendingOps.end() ) {
            if ( (*it)->allCompleted() ) {
               std::set< DeviceOps * >::iterator toBeRemovedIt = it;
               it++;
               _cacheCopies[ index ]._otherPendingOps.erase( toBeRemovedIt );
            } else {
         //std::cerr << "Waiting for other " << (*it)->_desc << std::endl;
               allReady = false;
            }
         }
      }
   }
         //if (!allReady) std::cerr << "Waiting for copis undex " << index << " of " << _numCopies <<" still remain "<<_cacheCopies[ index ]._operations.getNumOps()<< std::endl;
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
