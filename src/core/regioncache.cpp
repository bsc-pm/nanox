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

CachedRegionStatus::CachedRegionStatus() : _status( READY ), _version( 0 ) {}
CachedRegionStatus::CachedRegionStatus( CachedRegionStatus const &rs ) : _status( rs._status ), _version( rs._version ), _waitObject ( rs._waitObject ) {}
CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus const &rs ) { _status = rs._status; _version = rs._version; _waitObject = rs._waitObject;  return *this; }
CachedRegionStatus::CachedRegionStatus( CachedRegionStatus &rs ) : _status( rs._status ), _version( rs._version ), _waitObject ( rs._waitObject ) {}
CachedRegionStatus &CachedRegionStatus::operator=( CachedRegionStatus &rs ) { _status = rs._status; _version = rs._version; _waitObject = rs._waitObject;  return *this; }
unsigned int CachedRegionStatus::getVersion() { return _version; }
void CachedRegionStatus::setVersion( unsigned int version ) { _version = version; }
void CachedRegionStatus::setCopying( DeviceOps *ops ) { _waitObject.set( ops );  }
DeviceOps *CachedRegionStatus::getDeviceOps() { return _waitObject.get();  }
bool CachedRegionStatus::isReady( ) { return _waitObject.isNotSet(); }

AllocatedChunk::AllocatedChunk() : _lock(), address( 0 ) { }
AllocatedChunk::AllocatedChunk( AllocatedChunk const &chunk ) : _lock(), address( chunk.address ) {}
AllocatedChunk &AllocatedChunk::operator=( AllocatedChunk const &chunk ) { address = chunk.address; return *this; } 
void AllocatedChunk::lock() { _lock.acquire(); }
void AllocatedChunk::unlock() { _lock.release(); }

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
         //std::cerr <<"Allocated object " << (void *) results.front().first->getAddress() << " size " <<cd.getMaxSize() << std::endl;

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

void RegionCache::syncRegion( std::list<Region> const &regions, DeviceOps *ops, WD* wd ) {
   std::list<Region>::const_iterator it;
   uint64_t offset = 0;

   for ( it = regions.begin(); it != regions.end(); it++ ) {
      Region const &reg = (*it);
      AllocatedChunk *origChunk = getAddress( ( uint64_t ) reg.getFirstValue(), ( std::size_t ) reg.getBreadth(), offset );
      uint64_t origDevAddr = origChunk->address + offset;
      origChunk->unlock();
      copyOut( reg, origDevAddr, ops, wd->getId(), wd );
   }
}

void RegionCache::syncRegion( std::list<Region> const &regions ) {
   DeviceOps *ops = NEW DeviceOps();
   std::list<Region>::const_iterator it;
   uint64_t offset = 0;
   for ( it = regions.begin(); it != regions.end(); it++ ) {
      Region const &reg = (*it);
      AllocatedChunk *origChunk = getAddress( ( uint64_t ) reg.getFirstValue(), ( std::size_t ) reg.getBreadth(), offset );
      uint64_t origDevAddr = origChunk->address + offset;
      origChunk->unlock();
      
      copyOut( reg, origDevAddr, ops, 0, NULL );
   }

   while( !ops->allCompleted() ) { myThread->idle(); }
   delete ops;
}

void RegionCache::syncRegion( Region const &reg ) {
   std::list<Region> singleItemList;
   singleItemList.push_back( reg );
   syncRegion( singleItemList );
}

void RegionCache::_generateRegionOps( Region const &reg, std::map< uintptr_t, MemoryMap< uint64_t > * > &opMap )
{
   uint64_t offset = 0, devAddr;
   AllocatedChunk *chunk = getAddress( (uint64_t) ((uint64_t) reg.getFirstValue()), ((std::size_t) reg.getBreadth()), offset );
   devAddr = chunk->address + offset;
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

void RegionCache::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId, WD * wd ) {
   ops->addOp();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   //NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   //NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-data-in"); )
   //NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) len ); )
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<"::copyIn "<< (void *) devAddr << " <=h " << (void *) hostAddr << " len " << len << " ops complete? "  << ( ops->allCompleted() ? "yes" : "no" )<<std::endl;
  // std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<"::copyIn "<< (void *) devAddr << " <=h " << (void *) hostAddr << " len " << len << " ops complete? "  << ( ops->allCompleted() ? "yes" : "no" )<<std::endl;
   _device->_copyIn( devAddr, hostAddr, len, _pe, ops, wdId, wd );
   //NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key ); )
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD * wd ) {
   ops->addOp();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   //NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   //NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-data-in"); )
   //NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) len ); )
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<"::copyIn "<< (void *) devAddr << " <=h " << (void *) hostAddr << " len " << len << " ops complete? "  << ( ops->allCompleted() ? "yes" : "no" )<<std::endl;
  // std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<"::copyIn "<< (void *) devAddr << " <=h " << (void *) hostAddr << " len " << len << " ops complete? "  << ( ops->allCompleted() ? "yes" : "no" )<<std::endl;
   _device->_copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, _pe, ops, wdId, wd );
   //NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key ); )
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   //NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   //NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-data-out"); )
   //NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) len ); )
   _device->_copyOut( hostAddr, devAddr, len, _pe, ops, wdId, wd );
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<">::copyOut "<< (void *) devAddr << " =>h " << (void *) hostAddr << " len " << len << " ops complete? "  << ( ops->allCompleted() ? "yes" : "no" )<<std::endl;
   //NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key ); )
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld,  DeviceOps *ops, unsigned int wdId, WD *wd ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   //ops->addOp();
   //NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   //NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-data-out"); )
   //NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) len ); )
   _device->_copyOutStrided1D( hostAddr, devAddr, len, numChunks, ld, _pe, ops, wdId, wd );
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<">::copyOut "<< (void *) devAddr << " =>h " << (void *) hostAddr << " len " << len << " ops complete? "  << ( ops->allCompleted() ? "yes" : "no" )<<std::endl;
   //NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key ); )
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId, WD * wd ) {
   uint64_t offset;
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->address + offset;
   origChunk->unlock();
   //if(sys.getNetwork()->getNodeNum() == 0) std::cerr <<sys.getNetwork()->getNodeNum() <<" Started a Copy out from " << syncFrom << std::endl;
   sys.getCaches()[ syncFrom ]->_copyOut( hostAddr, origDevAddr, len, cout, wdId, wd );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   //std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<"::syncAndcopyIn "<< (void *) origDevAddr << " dev=>host " << (void *) hostAddr << " host=>dev " << (void*)devAddr << " len " << len <<" data recvd "<< *((double *)hostAddr)<<std::endl;
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() <<" CopyOut from " << syncFrom << " completed" << std::endl;
   delete cout;
   this->_copyIn( devAddr, hostAddr, len, ops, wdId, wd );
}

void RegionCache::_syncAndCopyInStrided1D( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   uint64_t offset;
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->address + offset;
   origChunk->unlock();
   //if(sys.getNetwork()->getNodeNum() == 0) std::cerr <<sys.getNetwork()->getNodeNum() <<" Started a Copy out from " << syncFrom << std::endl;
   //sys.getCaches()[ syncFrom ]->copyOut( hostAddr, origDevAddr, len, cout );
   sys.getCaches()[ syncFrom ]->_copyOutStrided1D( hostAddr, origDevAddr, len, numChunks, ld, cout, wdId, wd );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   //std::cerr <<sys.getNetwork()->getNodeNum() << " RegionCache<"<< _pe->getMemorySpaceId() <<"::syncAndcopyIn "<< (void *) origDevAddr << " dev=>host " << (void *) hostAddr << " host=>dev " << (void*)devAddr << " len " << len <<" data recvd "<< *((double *)hostAddr)<<std::endl;
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() <<" CopyOut from " << syncFrom << " completed" << std::endl;
   delete cout;
   this->_copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, ops, wdId, wd );
}

void RegionCache::_copyDevToDev( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId, WD * wd ) {
   uint64_t offset;
   AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->address + offset;
   origChunk->unlock();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() << " wd " << wdId << " Copy Dev To Dev dest "<< _pe->getMemorySpaceId() << ": " << (void *) devAddr << " origAddr " << copyFrom <<": " << (void *) origDevAddr <<  std::endl;
   ops->addOp();
   _device->_copyDevToDev( devAddr, origDevAddr, len, _pe, sys.getCaches()[ copyFrom ]->_pe, ops, wdId, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyDevToDevStrided1D( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   uint64_t offset;
   AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len, offset );
   uint64_t origDevAddr = origChunk->address + offset;
   origChunk->unlock();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr <<sys.getNetwork()->getNodeNum() << " wd " << wdId << " Copy Dev To Dev dest "<< _pe->getMemorySpaceId() << ": " << (void *) devAddr << " origAddr " << copyFrom <<": " << (void *) origDevAddr <<  std::endl;
   ops->addOp();
   _device->_copyDevToDevStrided1D( devAddr, origDevAddr, len, numChunks, ld, _pe, sys.getCaches()[ copyFrom ]->_pe, ops, wdId, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::CopyIn::doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyIn( devAddr, hostAddr, size, ops, wdId, wd );
   } else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
      getParent()._copyDevToDev( dataLocation, devAddr, hostAddr, size, ops, wdId, wd );
   } else {
      getParent()._syncAndCopyIn( dataLocation, devAddr, hostAddr, size, ops, wdId, wd );
   }
}

void RegionCache::CopyIn::doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyInStrided1D( devAddr, hostAddr, size, count, ld, ops, wdId, wd );
   } else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
      getParent()._copyDevToDevStrided1D( dataLocation, devAddr, hostAddr, size, count, ld, ops, wdId, wd );
   } else {
      getParent()._syncAndCopyInStrided1D( dataLocation, devAddr, hostAddr, size, count, ld, ops, wdId, wd );
   }
}

void RegionCache::CopyOut::doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   getParent()._copyOut( hostAddr, devAddr, size, ops, wdId, wd );
}
void RegionCache::CopyOut::doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   getParent()._copyOutStrided1D( hostAddr, devAddr, size, count, ld, ops, wdId, wd );
}

void RegionCache::doOp( Op *opObj, Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   std::size_t skipBits = 0;
   std::size_t contiguousSize = hostMem.getContiguousChunkLength();
   std::size_t numChunks = hostMem.getNumNonContiguousChunks( skipBits );

   if ( numChunks > 1 && sys.usePacking() ) {
      uint64_t ld = hostMem.getNonContiguousChunk( 1, skipBits ) - hostMem.getNonContiguousChunk( 0, skipBits );
      uint64_t hostAddr = hostMem.getNonContiguousChunk( 0, skipBits );

      opObj->doStrided( location, devBaseAddr, hostAddr, contiguousSize, numChunks, ld, ops, wdId, wd );
   } else {
      for (unsigned int chunkIndex = 0; chunkIndex < numChunks; chunkIndex +=1 ) {
         uint64_t hostAddr = hostMem.getNonContiguousChunk( chunkIndex, skipBits );
         uint64_t devAddr = devBaseAddr + ( hostAddr - hostMem.getFirstValue() ); /* contiguous chunk offset */

         opObj->doNoStrided( location, devAddr, hostAddr, contiguousSize, ops, wdId, wd );
      }
   }
}

void RegionCache::copyIn( Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   doOp( &_copyInObj, hostMem, devBaseAddr, location, ops, wdId, wd );
}

void RegionCache::copyOut( Region const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   doOp( &_copyOutObj, hostMem, devBaseAddr, /* locations unused, copyOut is always to 0 */ 0, ops, wdId, wd );
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
bool RegionCache::canCopyFrom( RegionCache const &from ) const {
   return _pe->supportsDirectTransfersWith( *from._pe );
}
Device const *RegionCache::getDevice() const {
   return _device;
}

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
   _refs.insert( opsPtr );
   _lock.release();
}
DeviceOps::~DeviceOps() {}

CacheControler::CacheControler() : _numCopies ( 0 ), _cacheCopies ( NULL ), _targetCache ( NULL ) {}

bool CacheControler::isCreated() const {
   return _targetCache != NULL;
}

void CacheControler::preInit( NewDirectory *dir, std::size_t numCopies, CopyData *copies, unsigned int wdId, WD *wd ) {
   unsigned int index;
   _directory = dir;
   _wdId = wdId;
   _wd = wd;
   _numCopies = numCopies;
   if ( _numCopies > 0 ) {
      _cacheCopies = NEW CacheCopy[ _numCopies ];
      for ( index = 0; index < _numCopies; index += 1 ) {
         CacheCopy &ccopy = _cacheCopies[ index ];
         ccopy._copy = &copies[ index ];
         ccopy._region = NewRegionDirectory::build_region( copies[ index ] );
         ccopy._version = 0;
         _directory->getLocation( ccopy._region, ccopy._locations, ccopy._version, wd );
      }
   }
}

void CacheControler::copyDataInNoCache() {
   unsigned int index;
   for ( index = 0; index < _numCopies; index += 1 ) {
      CacheCopy &ccopy = _cacheCopies[ index ];
      _directory->addAccess( ccopy._region, 0, ccopy._copy->isOutput() ? ccopy._version + 1 : ccopy._version );

      if ( ccopy._copy->isInput() )
      {
         std::map<unsigned int, std::list<Region> > locationMap;

         for ( NewDirectory::LocationInfoList::iterator it = ccopy._locations.begin(); it != ccopy._locations.end(); it++ ) {
            if (!it->second.isLocatedIn( 0 ) ) { 
               int loc = it->second.getFirstLocation();
               locationMap[ loc ].push_back( it->first );
            }
         }

         if ( !locationMap.empty() ) {
            std::map<unsigned int, std::list<Region> >::iterator locIt;
            for( locIt = locationMap.begin(); locIt != locationMap.end(); locIt++ ) {
               sys.getCaches()[ locIt->first ]->syncRegion( locIt->second, &ccopy._operations, _wd );
            }
         }
      }
   }
}

void CacheControler::copyDataIn(RegionCache *targetCache) {
   unsigned int index;
   _targetCache = targetCache;
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_CC_CDIN); );
   if ( _numCopies > 0 ) {
      /* Get device address, allocate if needed */
      NANOS_INSTRUMENT( InstrumentState inst3(NANOS_CC_CDIN_GET_ADDR); );
      for ( index = 0; index < _numCopies; index += 1 ) {
         CacheCopy &ccopy = _cacheCopies[ index ];
         ccopy._cacheEntry = _targetCache->getAddress( *ccopy._copy, ccopy._offset );
         ccopy._devRegion = NewRegionDirectory::build_region_with_given_base_address( *ccopy._copy, ccopy._offset );
         ccopy._devBaseAddr = ccopy._cacheEntry->address + ccopy._offset;
         ccopy._cacheEntry->unlock();
         
         // register version into this task directory
	      _directory->addAccess( ccopy._region, _targetCache->getMemorySpaceId(),
            ccopy._copy->isOutput() ? ccopy._version + 1 : ccopy._version );
      }
      NANOS_INSTRUMENT( inst3.close(); );

      /* COPY IN GENERATION */
      NANOS_INSTRUMENT( InstrumentState inst4(NANOS_CC_CDIN_OP_GEN); );
      std::map<unsigned int, std::list< std::pair< Region, int > > > opsBySourceRegions;
      for ( index = 0; index < _numCopies; index += 1 ) {
	      NewRegionDirectory::LocationInfoList::iterator it;
         CacheCopy &ccopy = _cacheCopies[ index ];
         ccopy._cacheEntry->lock();
         unsigned int ncopies=0;
         if ( ccopy._copy->isInput() )
         {
            for ( it = ccopy._locations.begin(); it != ccopy._locations.end(); it++ ) {

               if ( it->second.isLocatedIn( _targetCache->getMemorySpaceId() )) continue;
               /* FIXME: version info, (I think its not needed because directory stores
                * only the last version, if an old version is stored, it wont be reported
                * but _targetCache.getAddress will return the already allocated storage)
                */
               std::list< Region > notPresentRegions;
               std::list< Region >::iterator notPresentRegionsIt;
               ccopy._cacheEntry->addReadRegion( it->first, it->second.getVersion(), ccopy._otherPendingOps,
                  notPresentRegions, &ccopy._operations, ccopy._copy->isOutput() );

               for( notPresentRegionsIt = notPresentRegions.begin();
                    notPresentRegionsIt != notPresentRegions.end();
                    notPresentRegionsIt++ ) {
                  std::list< std::pair< Region, int > > &thisCopyOpsRegions = opsBySourceRegions[ it->second.getFirstLocation() ];
                  ncopies++;

                  Region &origReg = *notPresentRegionsIt;
                  thisCopyOpsRegions.push_back( std::make_pair( origReg, index ) );
               }
            }
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
      NANOS_INSTRUMENT( inst4.close(); );
      /* END OF COPY IN GENERATION */

      /* ISSUE ACTUAL OPERATIONS */
      NANOS_INSTRUMENT( InstrumentState inst5(NANOS_CC_CDIN_DO_OP); );
      std::map< unsigned int, std::list< std::pair< Region, int > > >::iterator mapOpsStrIt;
      for ( mapOpsStrIt = opsBySourceRegions.begin(); mapOpsStrIt != opsBySourceRegions.end(); mapOpsStrIt++ ) {
         std::list< std::pair< Region, int > > &ops = mapOpsStrIt->second;
	      unsigned int location = mapOpsStrIt->first;
         for ( std::list< std::pair< Region, int > >::iterator listIt = ops.begin(); listIt != ops.end(); listIt++ ) {
            CacheCopy &ccopy = _cacheCopies[ listIt->second ];
            uint64_t fragmentOffset = listIt->first.getFirstValue() -
               ( ( uint64_t ) ccopy._copy->getBaseAddress() + ccopy._copy->getOffset() ); /* displacement due to fragmented region */
            _targetCache->copyIn( listIt->first, ccopy._devBaseAddr + ccopy._copy->getOffset() + fragmentOffset,
               location, &ccopy._operations, _wdId, _wd );
         }
      }
      NANOS_INSTRUMENT( inst5.close(); );
      /* END OF ISSUE OPERATIONS */
   }
   NANOS_INSTRUMENT( inst2.close(); );
}

unsigned int RegionCache::getNodeNumber() const {
   return _pe->getMyNodeNumber();
}

bool CacheControler::dataIsReady() const {
   bool allReady = true;
   unsigned int index;
   for ( index = 0; ( index < _numCopies ) && allReady; index += 1 ) {

      if( !_cacheCopies[ index ]._operations.allCompleted() ) { 
         allReady = false;
      }

      if ( allReady ) { 
         std::set< DeviceOps * >::iterator it = _cacheCopies[ index ]._otherPendingOps.begin();
         while ( allReady && it != _cacheCopies[ index ]._otherPendingOps.end() ) {
            if ( (*it)->allCompleted() ) {
               std::set< DeviceOps * >::iterator toBeRemovedIt = it;
               it++;
               _cacheCopies[ index ]._otherPendingOps.erase( toBeRemovedIt );
            } else {
               allReady = false;
            }
         }
      }
   }
   return allReady;
}

uint64_t CacheControler::getAddress( unsigned int copyId ) const {
   return _cacheCopies[ copyId ]._cacheEntry->address +
      _cacheCopies[ copyId ]._offset + _cacheCopies[ copyId ]._copy->getOffset();
}


void CacheControler::copyDataOut() {
   unsigned int index;
  
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_CC_CDOUT); );

   for ( index = 0; index < _numCopies; index += 1 ) {
      CacheCopy &ccopy = _cacheCopies[ index ];

      if ( !ccopy._copy->isOutput() ) continue;
   
      // TODO: WriteThrough code

      //Region reg = NewRegionDirectory::build_region( *ccopy._copy );
      //std::cerr << "Adding version "<< ccopy._version << " for addr " << ccopy._copy->getBaseAddress() << std::endl;
      //_directory->addAccess( reg, ccopy._copy->isInput(), ccopy._copy->isOutput(), 0, ((uint64_t)ccopy._copy->getBaseAddress()) + ccopy._copy->getOffset(), ccopy._version + 1 );

         //Region origReg = NewRegionDirectory::build_region( *ccopy._copy );
      //   _targetCache->syncRegion( reg, (ccopy._cacheEntry->address + ccopy._offset) + ccopy._copy->getOffset() );
   }
       
   NANOS_INSTRUMENT( inst2.close(); );
}
