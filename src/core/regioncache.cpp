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
#include "newregiondirectory.hpp"
#include "regioncache.hpp"
#include "cachedregionstatus.hpp"
#include "os.hpp"

AllocatedChunk::AllocatedChunk( uint64_t addr, uint64_t hostAddress, std::size_t size, CopyData const &cd ) :
   _lock(),
   _address( addr ),
   _hostAddress( hostAddress ),
   _size( size ),
   _dirty( false ),
   _roBytes( 0 ),
   _rwBytes( 0 ) {
   if ( sys.usingNewCache() ) {
      _newRegions = NEW RegionDictionary( sys.getMasterRegionDirectory().getDictionary( cd ),
                                        *( NEW RegionMap( sys.getMasterRegionDirectory().getDictionary( cd ).getContainer() ) ), true );
      //std::cerr << "Allocated chunk: with dict " << (void *)&_newRegions << std::endl;
   } else {
      _regions = NEW RegionTree< CachedRegionStatus >();
   }
}

AllocatedChunk::~AllocatedChunk() {
   if ( sys.usingNewCache() ) {
      delete _newRegions;
   } else {
      delete _regions;
   }
}

void AllocatedChunk::clearRegions() {
   _regions = NEW RegionTree< CachedRegionStatus >;
}

RegionTree< CachedRegionStatus > *AllocatedChunk::getRegions() {
   return _regions;
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
   ret = _regions->findAndPopulate( reg, insertOuts );
   if ( !ret.isEmpty() ) insertOuts.push_back( ret );

   for ( RegionTree<CachedRegionStatus>::iterator_list_t::iterator it = insertOuts.begin();
         it != insertOuts.end();
         it++
       ) {
      std::size_t bytes = 0;
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
         //std::cerr << " I dont have region version=" << version << " " << reg << std::endl;
         bytes += ( cachedReg.getVersion() == 0 ) ? accessor.getRegion().getBreadth() : 0;
         notPresentRegions.push_back( accessor.getRegion() );
         cachedReg.setCopying( ops );
         cachedReg.setVersion(version);
      }
      if ( alsoWriteRegion ) {
         cachedReg.setVersion( version + 1 );
         _rwBytes += bytes;
      } else {
         _roBytes += bytes;
      }
   }
   _dirty = _dirty || alsoWriteRegion;
}

void AllocatedChunk::NEWaddReadRegion( reg_t reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::list< reg_t > &notPresentRegions, DeviceOps *ops, bool alsoWriteRegion ) {

   unsigned int currentVersion = 0;
   std::list< std::pair< reg_t, reg_t > > components;
   _newRegions->addRegion( reg, components, currentVersion );

   //std::cerr << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ " << __FUNCTION__ << " reg " << reg << " ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"<< std::endl;
   
   if ( components.size() == 1 ) {
      ensure( components.begin()->first == reg, "Error, wrong region");
      CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( components.begin()->first );
      if ( !entry || version > entry->getVersion() ) {
         //std::cerr << "No entry for region " << components.begin()->first << " must copy from region " << components.begin()->second << " want version "<<version<< std::endl;
         CachedRegionStatus *copyFromEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( components.begin()->second );
         if ( !copyFromEntry || version > copyFromEntry->getVersion() ) {
            if ( !entry ) {
               entry = NEW CachedRegionStatus();
               _newRegions->setRegionData( reg, entry );
            }
            entry->setCopying( ops );
            notPresentRegions.push_back( reg );
         } else { // I have this region, as part of other region
            if ( !entry ) {
               entry = NEW CachedRegionStatus( *copyFromEntry );
               _newRegions->setRegionData( reg, entry );
            }
         }
      } else if ( version == entry->getVersion() ) {
         if ( entry->isReady() ) {
            /* already in cache */
         //std::cerr << "!!!!!!!!!!!! REGION READY " << components.begin()->first << std::endl;
         } else {
            /* in cache but comming */
        // std::cerr << "???????????? ON ITS WAY " << components.begin()->first << std::endl;
            currentOps.insert( entry->getDeviceOps() );
         }
      } else {
         std::cerr << "ERROR: version in cache > than version requested." << std::endl;
      }
      entry->setVersion( version + ( alsoWriteRegion ? 1 : 0) );
   } else if ( components.size() > 1 ) {
      std::cerr << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~ chunks! REGION "  << std::endl;
      for (std::list< std::pair< reg_t, reg_t > >::iterator it = components.begin(); it != components.end(); it++) {
         CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->first );
         if ( !entry ) {
            std::cerr << "m No entry for region " << it->first << " must copy from region " << it->second << std::endl;
            entry = NEW CachedRegionStatus();
            _newRegions->setRegionData( reg, entry );
            //entry->
         }
         else std::cerr << "GOT entry for region " << it->first << std::endl;
      } 
   } else {
      std::cerr << " ERROR " << std::endl;
   }
      
  // std::cerr << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[X]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"<< std::endl;
   _dirty = _dirty || alsoWriteRegion;
}

void AllocatedChunk::addWriteRegion( Region const &reg, unsigned int version ) {
   RegionTree<CachedRegionStatus>::iterator_list_t insertOuts;
   RegionTree<CachedRegionStatus>::iterator ret;
   ret = _regions->findAndPopulate( reg, insertOuts );
   if ( !ret.isEmpty() ) insertOuts.push_back( ret );

   for ( RegionTree<CachedRegionStatus>::iterator_list_t::iterator it = insertOuts.begin();
         it != insertOuts.end();
         it++
       ) {
      RegionTree<CachedRegionStatus>::iterator &accessor = *it;
      CachedRegionStatus &cachedReg = *accessor;
      cachedReg.setVersion( version );
      _rwBytes += ( cachedReg.getVersion() == 0 ) ? accessor.getRegion().getBreadth() : 0;
   } 
   _dirty = true;
}

void AllocatedChunk::NEWaddWriteRegion( reg_t reg, unsigned int version ) {
   unsigned int currentVersion = 0;
   std::list< std::pair< reg_t, reg_t > > components;
   _newRegions->addRegion( reg, components, currentVersion );

   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !entry ) {
      entry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, entry );
   }
   entry->setVersion( version );

   _dirty = true;
}

bool AllocatedChunk::isReady( Region reg ) {
   bool entryReady = true;
   RegionTree<CachedRegionStatus>::iterator_list_t outs;
   //RegionTree<CachedRegionStatus>::iterator ret;
   _regions->find( reg, outs );
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

AllocatedChunk *RegionCache::getAddress( CopyData const &cd, RegionTree< CachedRegionStatus > *&regsToInval) {
   ChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;

   std::size_t allocSize = 0;
   uint64_t targetHostAddr = 0;

   if ( _flags == ALLOC_WIDE ) {
      targetHostAddr = (uint64_t) cd.getBaseAddress();
      allocSize =  cd.getMaxSize();
   } else if ( _flags == ALLOC_FIT ) {
      targetHostAddr = cd.getFitAddress();
      allocSize =  cd.getFitSize();
   } else {
      std::cerr <<"RegionCache ERROR: Undefined _flags value."<<std::endl;
   }


  //std::cerr << "-----------------------------------------" << std::endl;
  //std::cerr << " Max " << cd.getMaxSize() << std::endl;
  //std::cerr << "WIDE targetHostAddr: "<< ((void *)targetHostAddr) << std::endl;
  //std::cerr << "WIDE allocSize     : "<< allocSize << std::endl;
  //std::cerr << "FIT  targetHostAddr: "<< ((void *)cd.getFitAddress()) << std::endl;
  //std::cerr << "FIT  allocSize     : "<< cd.getFitSize() << std::endl;
  //std::cerr << "-----------------------------------------" << std::endl;

   _chunks.getOrAddChunk2( targetHostAddr, allocSize, results );
   if ( results.size() != 1 ) {
      message0( "Got results.size()="<< results.size() << " I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ChunkList::iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
   } else {
      if ( *(results.front().second) == NULL ) {

         void *deviceMem = _device.memAllocate( allocSize, _pe );
         if ( deviceMem == NULL ) {
            // Device is full, free some chunk
            std::cerr << "Cache is full." << std::endl;
            MemoryMap<AllocatedChunk>::iterator it;
            bool done = false;
            int count = 0;
            for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
               //if ( it->second == NULL ) std::cerr << "["<< count << "] mmm this chunk: " << ((void *) it->second) << std::endl;  
               if ( it->second != NULL && !it->second->isDirty() && it->second->getSize() >= allocSize ) {
                  std::cerr << "["<< count << "] this chunk: " << ((void *) it->second) << std::endl;
                  done = true;
                  break;
               }
               count++;
            }
            if ( done ) {
               std::cerr << "IVE FOUND A CHUNK TO FREE (" << (void *) it->second << ")"<< std::endl;
               AllocatedChunk *chunkToReuse = it->second;
               it->second = NULL;
               chunkToReuse->setHostAddress( results.front().first->getAddress() );
               regsToInval = chunkToReuse->getRegions();
               chunkToReuse->clearRegions();
               allocChunkPtr = *(results.front().second) = chunkToReuse;
            } else {
               std::cerr << "IVE _not_ FOUND A CHUNK TO FREE" << std::endl;
            }
         } else {
            *(results.front().second) = NEW AllocatedChunk( (uint64_t) deviceMem, results.front().first->getAddress(), results.front().first->getLength(), cd );
            allocChunkPtr = *(results.front().second);
         }
      } else {
         if ( results.front().first->getAddress() == (uint64_t) cd.getBaseAddress() ) {
            if ( results.front().first->getLength() == allocSize ) {
               allocChunkPtr = *(results.front().second);
            } else {
               std::cerr << "I need a realloc of an allocated chunk!" << std::endl;
            }
         } else if ( results.front().first->getAddress() == targetHostAddr ) {
            allocChunkPtr = *(results.front().second);
         }
      }
   }
   if ( allocChunkPtr == NULL ) std::cerr << "WARNING: null RegionCache::getAddress()" << std::endl; 
   //std::cerr << __FUNCTION__ << " returns dev address " << (void *) allocChunkPtr->getAddress() << std::endl;
   allocChunkPtr->lock();
   return allocChunkPtr;
}

AllocatedChunk *RegionCache::getAddress( uint64_t hostAddr, std::size_t len ) {
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
      }
   }
   if ( allocChunkPtr == NULL ) std::cerr << "WARNING: null RegionCache::getAddress()" << std::endl; 
   allocChunkPtr->lock();
   return allocChunkPtr;
}
   
void RegionCache::syncRegion( std::list< std::pair< Region, CacheCopy * > > const &regions, WD const &wd ) {
   std::list< std::pair< Region, CacheCopy *> >::const_iterator it;
   DeviceOps localOps;
   
   for ( it = regions.begin(); it != regions.end(); it++ ) {
      Region const &reg = it->first;
      AllocatedChunk *origChunk = getAddress( ( uint64_t ) reg.getFirstValue(), ( std::size_t ) reg.getBreadth() );
      uint64_t origDevAddr = origChunk->getAddress() + ( ( uint64_t ) reg.getFirstValue() - origChunk->getHostAddress() );
      origChunk->unlock();
      //std::cerr << "OLD SYNC copy out DEV ADDR: " << ((void*)origDevAddr) << " origChunk->getHostAddr()= "<< ( (void*) origChunk->getHostAddress() ) << " firstHostAddr " << ((void*) reg.getFirstValue()) << std::endl;
      copyOut( reg, origDevAddr, ( it->second != NULL ) ? it->second->getOperations() : &localOps, wd );
   }

   while( !localOps.allCompleted() ) { myThread->idle(); }
}

void RegionCache::syncRegion( Region const &reg ) {
   std::list< std::pair< Region, CacheCopy * > > singleItemList;
   singleItemList.push_back( std::make_pair( reg, ( CacheCopy * ) NULL ) );
   syncRegion( singleItemList, *(( WD * ) NULL) );
}

void RegionCache::syncRegion( std::list< std::pair< global_reg_t, CacheCopy * > > const &regions, WD const &wd ) {
   std::list< std::pair< global_reg_t, CacheCopy *> >::const_iterator it;
   DeviceOps localOps;
   
   for ( it = regions.begin(); it != regions.end(); it++ ) {
      global_reg_t id = it->first;
      CacheCopy *ccopy = it->second;
      AllocatedChunk *origChunk = getAddress( id.getFirstAddress(), id.getBreadth() );
      uint64_t origDevAddr = origChunk->getAddress() + ( id.getFirstAddress() - origChunk->getHostAddress() );
      origChunk->unlock();
      //std::cerr << "NEW SYNC copy out reg " << (void*)id.key<< ","<< id.id << "DEV ADDR: " << ((void*)origDevAddr) << " origChunk->getHostAddr()= "<< ( (void*) origChunk->getHostAddress() ) << " firstHostAddr " << ((void*) id.getFirstAddress()) << std::endl;
      copyOut( id, origDevAddr, ( ccopy != NULL ) ? ccopy->getOperations() : &localOps, wd );
   }

   while( !localOps.allCompleted() ) { myThread->idle(); }
}
void RegionCache::syncRegion( global_reg_t const &id ) {
   std::list< std::pair< global_reg_t, CacheCopy * > > singleItemList;
   singleItemList.push_back( std::make_pair( id, ( CacheCopy * ) NULL ) );
   syncRegion( singleItemList, *(( WD * ) NULL) );
}

RegionCache::RegionCache( ProcessingElement &pe, Device &cacheArch, enum CacheOptions flags ) : _device( cacheArch ), _pe( pe ),
    _flags( flags ), _copyInObj( *this ), _copyOutObj( *this ) {
}

unsigned int RegionCache::getMemorySpaceId() {
   return _pe.getMemorySpaceId();
}

void RegionCache::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   //std::cerr << "_device._copyIn( copyTo=" << _pe.getMemorySpaceId() <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, _pe, ops, wd="<< wd.getId() <<" );";
   if (!fake) _device._copyIn( devAddr, hostAddr, len, _pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   if (!fake) _device._copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, _pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   //std::cerr << "_device._copyOut( copyFrom=" << _pe.getMemorySpaceId() <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, _pe, ops, wd="<< (&wd != NULL ? wd.getId() : -1 ) <<" );";
   if (!fake) _device._copyOut( hostAddr, devAddr, len, _pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld,  DeviceOps *ops, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   if (!fake) _device._copyOutStrided1D( hostAddr, devAddr, len, numChunks, ld, _pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake ) {
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   sys.getCaches()[ syncFrom ]->_copyOut( hostAddr, origDevAddr, len, cout, wd, fake );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   delete cout;
   this->_copyIn( devAddr, hostAddr, len, ops, wd, fake );
}

void RegionCache::_syncAndCopyInStrided1D( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) {
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   sys.getCaches()[ syncFrom ]->_copyOutStrided1D( hostAddr, origDevAddr, len, numChunks, ld, cout, wd, fake );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   delete cout;
   this->_copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, ops, wd, fake );
}

void RegionCache::_copyDevToDev( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake ) {
   AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   //std::cerr << "_device._copyDevToDev( copyFrom=" << copyFrom << ", copyTo=" << _pe.getMemorySpaceId() <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len, _pe, sys.getCaches()[ copyFrom="<< copyFrom<<" ]->_pe, ops, wd="<< wd.getId() <<" );";
   if (!fake) _device._copyDevToDev( devAddr, origDevAddr, len, _pe, sys.getCaches()[ copyFrom ]->_pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyDevToDevStrided1D( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) {
   AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   if (!fake) _device._copyDevToDevStrided1D( devAddr, origDevAddr, len, numChunks, ld, _pe, sys.getCaches()[ copyFrom ]->_pe, ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::CopyIn::doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, WD const &wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyIn( devAddr, hostAddr, size, ops, wd, fake );
   } else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
      getParent()._copyDevToDev( dataLocation, devAddr, hostAddr, size, ops, wd, fake );
   } else {
      getParent()._syncAndCopyIn( dataLocation, devAddr, hostAddr, size, ops, wd, fake );
   }
}

void RegionCache::CopyIn::doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyInStrided1D( devAddr, hostAddr, size, count, ld, ops, wd, fake );
   } else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
      getParent()._copyDevToDevStrided1D( dataLocation, devAddr, hostAddr, size, count, ld, ops, wd, fake );
   } else {
      getParent()._syncAndCopyInStrided1D( dataLocation, devAddr, hostAddr, size, count, ld, ops, wd, fake );
   }
}

void RegionCache::CopyOut::doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, WD const &wd, bool fake ) {
   getParent()._copyOut( hostAddr, devAddr, size, ops, wd, fake );
}
void RegionCache::CopyOut::doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) {
   getParent()._copyOutStrided1D( hostAddr, devAddr, size, count, ld, ops, wd, fake );
}

void RegionCache::doOp( Op *opObj, Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ) {
   std::size_t skipBits = 0;
   std::size_t contiguousSize = hostMem.getContiguousChunkLength();
   std::size_t numChunks = hostMem.getNumNonContiguousChunks( skipBits );

   if ( numChunks > 1 && sys.usePacking() ) {
      uint64_t ld = hostMem.getNonContiguousChunk( 1, skipBits ) - hostMem.getNonContiguousChunk( 0, skipBits );
      uint64_t hostAddr = hostMem.getNonContiguousChunk( 0, skipBits );

         //std::cerr <<"[OLD]opObj("<<opObj->getStr()<<")->doStrided( src="<<location<<", dst="<< getMemorySpaceId()<<", "<<(void*)(devBaseAddr)<<", "<<(void*)(hostAddr)<<", "<<contiguousSize<<", "<<numChunks << ", " << ld << ", _ops, _wd="<<(&wd != NULL ? wd.getId():-1)<<" )";
         opObj->doStrided( location, devBaseAddr, hostAddr, contiguousSize, numChunks, ld, ops, wd, false );
         //std::cerr << " done" << std::endl;
   } else {
      for (unsigned int chunkIndex = 0; chunkIndex < numChunks; chunkIndex +=1 ) {
         uint64_t hostAddr = hostMem.getNonContiguousChunk( chunkIndex, skipBits );
         uint64_t devAddr = devBaseAddr + ( hostAddr - hostMem.getFirstValue() ); /* contiguous chunk offset */

         //std::cerr <<"[OLD]opObj("<<opObj->getStr()<<")->doNoStrided( src="<<location<<", dst="<< getMemorySpaceId()<<", "<<(void*)(devAddr)<<", "<<(void*)(hostAddr)<<", "<<contiguousSize<<", _ops, _wd="<<(&wd != NULL ? wd.getId():-1)<<" )";
         opObj->doNoStrided( location, devAddr, hostAddr, contiguousSize, ops, wd, false );
         //std::cerr << " done" << std::endl;
      }
   }
}

void RegionCache::copyIn( Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ) {
   doOp( &_copyInObj, hostMem, devBaseAddr, location, ops, wd );
}

void RegionCache::copyOut( Region const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, WD const &wd ) {
   doOp( &_copyOutObj, hostMem, devBaseAddr, /* locations unused, copyOut is always to 0 */ 0, ops, wd );
}

void RegionCache::doOp( Op *opObj, global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ) {

   class LocalFunction {
      Op *_opObj;
      nanos_region_dimension_internal_t *_region;
      unsigned int _numDimensions;
      unsigned int _targetDimension;
      unsigned int _numChunks;
      std::size_t _contiguousChunkSize;
      unsigned int _location;
      DeviceOps *_ops;
      WD const &_wd;
      uint64_t _devBaseAddr;
      uint64_t _hostBaseAddr;
      public:
         LocalFunction( Op *opO, nanos_region_dimension_internal_t *r, unsigned int n, unsigned int t, unsigned int nc, std::size_t ccs, unsigned int loc, DeviceOps *operations, WD const &workdesc, uint64_t devAddr, uint64_t hostAddr )
               : _opObj( opO ), _region( r ), _numDimensions( n ), _targetDimension( t ), _numChunks( nc ), _contiguousChunkSize( ccs ), _location( loc ), _ops( operations ), _wd( workdesc ), _devBaseAddr( devAddr ), _hostBaseAddr( hostAddr) {
         }
         void issueOpsRecursive( unsigned int idx, std::size_t offset, std::size_t leadingDim ) {
            if ( idx == ( _numDimensions - 1 ) ) {
               //issue copy
               unsigned int L_numChunks = _numChunks; //_region[ idx ].accessed_length;
               if ( L_numChunks > 1 && sys.usePacking() ) {
                  //std::cerr << "[NEW]opObj("<<_opObj->getStr()<<")->doStrided( src="<<_location<<", dst="<< _opObj->getParent().getMemorySpaceId()<<", "<<(void*)(_devBaseAddr+offset)<<", "<<(void*)(_hostBaseAddr+offset)<<", "<<_contiguousChunkSize<<", "<<_numChunks<<", "<<leadingDim<<", _ops, _wd="<<(&_wd != NULL ? _wd.getId():-1)<<" )";
                  _opObj->doStrided( _location, _devBaseAddr+offset, _hostBaseAddr+offset, _contiguousChunkSize, _numChunks, leadingDim, _ops, _wd, false );
                  //std::cerr <<" done"<< std::endl;
               } else {
                  for (unsigned int chunkIndex = 0; chunkIndex < L_numChunks; chunkIndex +=1 ) {
                     //std::cerr <<"[NEW]opObj("<<_opObj->getStr()<<")->doNoStrided( src="<<_location<<", dst="<< _opObj->getParent().getMemorySpaceId()<<", "<<(void*)(_devBaseAddr+offset + chunkIndex*(leadingDim))<<", "<<(void*)(_hostBaseAddr+offset + chunkIndex*(leadingDim))<<", "<<_contiguousChunkSize<<", _ops, _wd="<<(&_wd != NULL ? _wd.getId():-1)<<" )";
                    _opObj->doNoStrided( _location, _devBaseAddr+offset + chunkIndex*(leadingDim), _hostBaseAddr+offset + chunkIndex*(leadingDim), _contiguousChunkSize, _ops, _wd, false );
                     //std::cerr <<" done"<< std::endl;
                  }
               }
            } else {
               for ( unsigned int i = 0; i < _region[ idx ].accessed_length; i += 1 ) {
                  //std::cerr <<"_recursive call " << idx << " "<< offset << " : " <<  offset + leadingDim * ( i + _region[ idx ].lower_bound ) <<std::endl;
                  issueOpsRecursive( idx + 1, offset + leadingDim * ( i /*+ _region[ idx ].lower_bound*/ ), leadingDim * _region[ idx ].size ); 
               }
            }
         }
   };
   nanos_region_dimension_internal_t region[ hostMem.getNumDimensions() ];
   hostMem.fillDimensionData( region );

   unsigned int dimIdx = 0;
   unsigned int numChunks = 1;
   std::size_t contiguousChunkSize = 1;
   std::size_t leadingDimension = 1;
   std::size_t offset = 0;

   do {
     offset += leadingDimension * region[ dimIdx ].lower_bound;
     contiguousChunkSize *= region[ dimIdx ].accessed_length;
     leadingDimension *= region[ dimIdx ].size;
     //std::cerr << dimIdx << " chunkSize=" << contiguousChunkSize << " leadingDim=" << leadingDimension << " thisDimACCCLEN=" << region[ dimIdx ].accessed_length  << " thisDimSIZE=" << region[ dimIdx ].size << std::endl;
     dimIdx += 1;
   } while ( ( region[ dimIdx - 1 ].accessed_length == region[ dimIdx - 1 ].size /*|| region[ dimIdx - 1 ].accessed_length == 1*/ ) && dimIdx < hostMem.getNumDimensions() );

   if ( dimIdx == hostMem.getNumDimensions() ) {
      // out because of dimIdx = NumDims
      numChunks = 1;
   } else {
      numChunks = region[ dimIdx ].accessed_length;
      dimIdx++;
   }

   //std::cerr << " NUM CHUNKS: " << numChunks << " of SIZE " << contiguousChunkSize << " dimIdx " << dimIdx << " leadingDim "<< leadingDimension << " numDimensions "<< hostMem.getNumDimensions() << " offset " << offset << std::endl;
   LocalFunction local( opObj, region, hostMem.getNumDimensions(), dimIdx, numChunks, contiguousChunkSize, location, ops, wd, devBaseAddr, hostMem.getFirstAddress() /* hostMem.key->getBaseAddress()*/ );
   local.issueOpsRecursive( dimIdx-1, 0, leadingDimension );
}

void RegionCache::copyIn( global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ) {
   doOp( &_copyInObj, hostMem, devBaseAddr, location, ops, wd );
}

void RegionCache::copyOut( global_reg_t const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, WD const &wd ) {
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

CacheCopy::CacheCopy() : _copy( *( (CopyData *) NULL ) ), _cacheEntry( NULL ), _cacheDataStatus(),
   _region( ), _offset( 0 ), _version( 0 ), _locations(), _operations(), _otherPendingOps(), _regId(0)/* gcc4.3 does not like it , _reg( (global_reg_t) { 0, NULL } )*/ {
   _reg.key = NULL;
   _reg.id = 0;
}

CacheCopy::CacheCopy( WD const &wd , unsigned int copyIndex, CacheController &ccontrol ) : _copy( wd.getCopies()[ copyIndex ] ), _cacheEntry( NULL ),
   _cacheDataStatus(), _region( NewRegionDirectory::build_region( _copy ) ), _offset( 0 ),
   _version( 0 ), _newVersion(0), _locations(), _newLocations(), _operations(), _otherPendingOps() {
   if ( !sys.usingNewCache() ) {
      wd.getNewDirectory()->getLocation( _region, _locations, _version, wd );
   } else {
      _reg.key = sys.getMasterRegionDirectory().getRegionDirectoryKeyRegisterIfNeeded( wd.getCopies()[ copyIndex ] );
      _reg.id = 0;
      _regId = _reg.key->tryObtainRegionId( wd.getCopies()[ copyIndex ] );

      if ( _regId != 0 && ccontrol.hasVersionInfoForRegion( global_reg_t( _regId, _reg.key ) , _newVersion, _newLocations ) ) {
         _reg.id = _regId;
         //_newLocations.push_back( std::make_pair( _regId, _regId ) );
      } else {
         tryGetLocation( wd, copyIndex );
      }
   }
   //std::cerr << "Region is " << _regId << " # Components: " << _newLocations.size() << " " << std::endl;
   //for ( NewNewRegionDirectory::NewLocationInfoList::iterator it = _newLocations.begin(); it != _newLocations.end(); it++ ) {
   //   std::cerr << "\tReg " << *it << std::endl;
   //}
}

bool CacheCopy::tryGetLocation( WD const &wd, unsigned int copyIndex ) {
   //do {
      _reg.id = NewNewRegionDirectory::tryGetLocation( _reg.key, wd.getCopies()[ copyIndex ], _newLocations, _newVersion, wd );
   //} while ( _reg.id == 0 );
   return (_reg.id != 0);
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

inline void CacheCopy::setUpDeviceAddress( RegionCache *targetCache, NewRegionDirectory *dir ) {
   RegionTree< CachedRegionStatus > *regsToInvalidate = NULL;
   _cacheEntry = targetCache->getAddress( _copy, regsToInvalidate );
   if ( regsToInvalidate ) {
      std::cerr << "Got to do something..." << std::endl;
      dir->invalidate( regsToInvalidate );
   }
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
         //std::cerr << "TO HOST I dont have region version=" << it->second.getVersion() << " " << it->first << std::endl;
         } else { 
            std::list< Region > notPresentRegions;
            std::list< Region >::iterator notPresentRegionsIt;
            _cacheEntry->addReadRegion( it->first, it->second.getVersion(), _otherPendingOps,
               notPresentRegions, &_operations, _copy.isOutput() );
   
            for( notPresentRegionsIt = notPresentRegions.begin();
                 notPresentRegionsIt != notPresentRegions.end();
                 notPresentRegionsIt++ ) {
         //std::cerr << " I dont have region version=" << it->second.getVersion() << " " << it->first << std::endl;
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

inline void CacheCopy::NEWgenerateCopyInOps( RegionCache *targetCache, std::map<unsigned int, std::list< std::pair< global_reg_t, CacheCopy * > > > &opsBySourceRegions ) {
	NewNewRegionDirectory::NewLocationInfoList::const_iterator it;
   if ( targetCache ) _cacheEntry->lock();
   if ( _copy.isInput() ) {
      for ( it = _newLocations.begin(); it != _newLocations.end(); it++ ) {

         global_reg_t gr( it->first, _reg.key );
         //if(sys.getNetwork()->getNodeNum() > 0)std::cerr <<"[" << sys.getNetwork()->getNodeNum()<<"] check location copy first: "<< it->first << " ("<< NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->first )<< ")[ "<<(*(NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->first )))<<" ]" << (void*) gr.getFirstAddress() << " second: " << it->second<< " (" << NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->second )<< ")[ " << (*(NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->second ))) << " do we have cache " << (void *)targetCache << " wanted version " << _newVersion << std::endl; 
         if ( NewNewRegionDirectory::isLocatedIn( _reg.key, it->first, ( !targetCache ) ? 0 : targetCache->getMemorySpaceId() /*, _newVersion*/ ) ) continue; //Version matching is already done in getLocation, if a region is listed, its because its a max version fragment
         //std::cerr <<"check location copy first: "<< it->first << " second: " << it->second << " do we have cache " << (void *)targetCache << std::endl; 

         //  if ( it->second.isLocatedIn( ( !targetCache ) ? 0 : targetCache->getMemorySpaceId() ) ) continue;
         /* FIXME: version info, (I think its not needed because directory stores
          * only the last version, if an old version is stored, it wont be reported
          * but _targetCache.getAddress will return the already allocated storage)
          */
         if ( !targetCache ) {
            /* No Cache scenario
             * we can not check if there are already comming ops for this region/sub-regions!! FIXME
             */
            opsBySourceRegions[ NewNewRegionDirectory::getFirstLocation( _reg.key, it->first ) ].push_back( std::make_pair( global_reg_t( it->first, _reg.key ) , this ) );
         } else { 
            std::list< reg_t > components;
            _cacheEntry->NEWaddReadRegion( it->first, NewNewRegionDirectory::getVersion( _reg.key, it->first ), _otherPendingOps, components, &_operations, _copy.isOutput() );
            std::list< reg_t >::iterator cit;
            for ( cit = components.begin(); cit != components.end(); cit++ ) {
               //std::cerr << "Reg " << *cit << " must be copied from loc " << NewNewRegionDirectory::getFirstLocation( _reg.key, it->first ) << " comes from reg " << it->second << std::endl;
               std::list< std::pair< global_reg_t, CacheCopy * > > &thisCopyOpsRegions = opsBySourceRegions[ NewNewRegionDirectory::getFirstLocation( _reg.key, it->first ) ];
               thisCopyOpsRegions.push_back( std::make_pair( global_reg_t( *cit, _reg.key ), this ) );
               //global_reg_t r( *cit, _reg.key );
            }
         }
      }
   } else { //else if ( !_copy.isInput() && _copy.isOutput() && targetCache ) {
      ensure( ( !_copy.isInput() && _copy.isOutput() ) , "Wrong copy directions.");
      unsigned int currentVersion = 1;
      for ( it = _newLocations.begin(); it != _newLocations.end(); it++ ) {
         currentVersion = std::max( currentVersion, NewNewRegionDirectory::getVersion( _reg.key, it->first ) );
      }
      /* write only region */
      if ( targetCache ) _cacheEntry->NEWaddWriteRegion( _reg.id, currentVersion + 1 );
   }
   if ( targetCache ) _cacheEntry->unlock();
}

CacheController::CacheController( WD const &wd ) : _wd( wd ), _numCopies( wd.getNumCopies() ), _targetCache( NULL ), _registered( false ), _provideLock(), _providedRegions() {
   if ( _numCopies > 0 ) {
      _cacheCopies = NEW CacheCopy[ _numCopies ];
   }
}

bool CacheController::isCreated() const {
   return _targetCache != NULL;
}

void CacheController::preInit() {
   unsigned int index;
   for ( index = 0; index < _numCopies; index += 1 ) {
      //std::cerr << "WD "<< _wd.getId() << " Depth: "<< _wd.getDepth() <<" Creating copy "<< index << std::endl;
      //std::cerr << _wd.getCopies()[ index ];
      new ( &_cacheCopies[ index ] ) CacheCopy( _wd, index, *this );
   }
}

void CacheController::copyDataIn(RegionCache *targetCache) {
   unsigned int index;
   _targetCache = targetCache;
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_CC_CDIN); );
   //fprintf(stderr, "%s for WD %d depth %u\n", __FUNCTION__, _wd.getId(), _wd.getDepth() );
   if( sys.usingNewCache() ) {
      unsigned int locationInfoReady = 0;
      while ( locationInfoReady < _numCopies ) {
         locationInfoReady = 0;
         for ( unsigned int idx = 0; idx < _numCopies; idx += 1 ) {
            if ( _cacheCopies[ idx ]._reg.id != 0 ) { locationInfoReady += 1;
            } else {
               if ( _cacheCopies[ idx ].tryGetLocation( _wd, idx ) ) {
                  locationInfoReady += 1;
               }
            }
         }
      }
   }

   if ( _numCopies > 0 ) {
      /* Get device address, allocate if needed */
      NANOS_INSTRUMENT( InstrumentState inst3(NANOS_CC_CDIN_GET_ADDR); );
      for ( index = 0; index < _numCopies; index += 1 ) {
         CacheCopy &ccopy = _cacheCopies[ index ];
         if ( _targetCache ) {
            //fprintf(stderr, "Set Up address index %d\n", index);
            ccopy.setUpDeviceAddress( _targetCache, _wd.getNewDirectory() );
         }

         // register version into this task directory
         if ( !sys.usingNewCache() )
            _wd.getNewDirectory()->addAccess( ccopy.getRegion(), ( !_targetCache ) ? 0 : _targetCache->getMemorySpaceId(),
                  ccopy.getCopyData().isOutput() ? ccopy.getVersion() + 1 : ccopy.getVersion() );


      }
      NANOS_INSTRUMENT( inst3.close(); );

      /* COPY IN GENERATION */
      NANOS_INSTRUMENT( InstrumentState inst4(NANOS_CC_CDIN_OP_GEN); );
      std::map<unsigned int, std::list< std::pair< Region, CacheCopy * > > > opsBySourceRegions;
      std::map<unsigned int, std::list< std::pair< global_reg_t, CacheCopy * > > > NEWopsBySourceRegions;
      //if(sys.getNetwork()->getNodeNum() > 0)std::cerr << _wd.getId()  << " "<< myThread->getId() << " ---------------------------- GEN COPIES WD "<< _wd.getId() <<" --------------------------" << std::endl;
      for ( index = 0; index < _numCopies; index += 1 ) {
         //if( sys.getNetwork()->getNodeNum() > 0 ) { 
         //   std::cerr << _wd.getId()  << " thd: "<< myThread->getId() <<" -- " << index << ": " << _cacheCopies[index]._reg.key << " "<< _cacheCopies[index]._reg.id;
         //   _cacheCopies[index]._reg.key->printRegion( _cacheCopies[index]._reg.id );
         //   std::cerr << std::endl;
         //}
         if (!sys.usingNewCache()) {
            _cacheCopies[ index ].generateCopyInOps( _targetCache, opsBySourceRegions );
         } else {
            _cacheCopies[ index ].NEWgenerateCopyInOps( _targetCache, NEWopsBySourceRegions );
         }
      }
      //if(sys.getNetwork()->getNodeNum() > 0)std::cerr << _wd.getId()  << " "<< myThread->getId() << " ------------------------------------------------------" << std::endl;
      NANOS_INSTRUMENT( inst4.close(); );
      /* END OF COPY IN GENERATION */

      NANOS_INSTRUMENT( InstrumentState inst5(NANOS_CC_CDIN_DO_OP); );
      if ( !sys.usingNewCache() ) {
         /* ISSUE ACTUAL OPERATIONS */
         //std::cerr <<"########## OLD copy in gen ######### running on thd " << myThread->getId() << " node " << sys.getNetwork()->getNodeNum() << std::endl;
         std::map< unsigned int, std::list< std::pair< Region, CacheCopy * > > >::iterator mapOpsStrIt;
         if ( targetCache ) {
            for ( mapOpsStrIt = opsBySourceRegions.begin(); mapOpsStrIt != opsBySourceRegions.end(); mapOpsStrIt++ ) {
               std::list< std::pair< Region, CacheCopy * > > &ops = mapOpsStrIt->second;
               for ( std::list< std::pair< Region, CacheCopy * > >::iterator listIt = ops.begin(); listIt != ops.end(); listIt++ ) {
                  CacheCopy &ccopy = *listIt->second;
                  uint64_t fragmentOffset = listIt->first.getFirstValue() - ( ( uint64_t ) ccopy.getCopyData().getBaseAddress() + ccopy.getCopyData().getOffset() ); /* displacement due to fragmented region */
                  uint64_t targetDevAddr = ccopy.getDeviceAddress() + fragmentOffset /* + ccopy.getCopyData().getOffset() */;
                  //if ( _wd.getDepth() == 1 ) std::cerr << "############################### CopyIn gen op: " << listIt->first.getLength() << " " << listIt->first.getBreadth() <<  " " << listIt->first << std::endl;
                  //         std::cerr << " OLD copy In, host: " << (void *) listIt->first.getFirstValue() << " dev " << (void *) targetDevAddr << " fragmentOff " << fragmentOffset << std::endl;
                  targetCache->copyIn( listIt->first, targetDevAddr, mapOpsStrIt->first, ccopy.getOperations(), _wd );
               }
            }
         } else {
            for ( mapOpsStrIt = opsBySourceRegions.begin(); mapOpsStrIt != opsBySourceRegions.end(); mapOpsStrIt++ ) {
               //         std::cerr << " OLD sync In, host: " << std::endl;
               sys.getCaches()[ mapOpsStrIt->first ]->syncRegion( mapOpsStrIt->second, _wd );
            }
         }
      } else {
         /* NEW REGIONS */
         //std::cerr <<"########## NEW copy in gen ######### "<<std::endl;

         //if ( sys.getNetwork()->getNodeNum() == 0 )  sys.getMasterRegionDirectory().print();
         std::map< unsigned int, std::list< std::pair< global_reg_t, CacheCopy * > > >::iterator NEWmapOpsStrIt;
         if ( targetCache ) {
            for ( NEWmapOpsStrIt = NEWopsBySourceRegions.begin(); NEWmapOpsStrIt != NEWopsBySourceRegions.end(); NEWmapOpsStrIt++ ) {
               std::list< std::pair< global_reg_t, CacheCopy * > > &ops = NEWmapOpsStrIt->second;
               //std::cerr << "Copies from loc " << NEWmapOpsStrIt->first << " " << ops.size() << std::endl;
               for ( std::list< std::pair< global_reg_t, CacheCopy * > >::iterator listIt = ops.begin(); listIt != ops.end(); listIt++ ) {
                  CacheCopy &ccopy = *listIt->second;
                  uint64_t fragmentOffset = listIt->first.getFirstAddress() - ( ( uint64_t ) ccopy.getCopyData().getBaseAddress() + ccopy.getCopyData().getOffset() ); /* displacement due to fragmented region */
                  uint64_t targetDevAddr = ccopy.getDeviceAddress() + fragmentOffset /* + ccopy.getCopyData().getOffset() */;
                  //std::cerr << " NEW copy In, host: " << (void *) listIt->first.getFirstAddress() << " dev " << (void *) targetDevAddr << " fragmentOff " << fragmentOffset << " reg " <<  listIt->first.key << ","<< listIt->first.id <<std::endl;
                  targetCache->copyIn( listIt->first, targetDevAddr, NEWmapOpsStrIt->first, ccopy.getOperations(), _wd );
               }
            }
         } else {
            for ( NEWmapOpsStrIt = NEWopsBySourceRegions.begin(); NEWmapOpsStrIt != NEWopsBySourceRegions.end(); NEWmapOpsStrIt++ ) {
               //std::cerr << "Copy from cache " <<  NEWmapOpsStrIt->first << std::endl;
               if ( NEWmapOpsStrIt->first == 0 ) {
                  std::list< std::pair< global_reg_t, CacheCopy * > > &ops = NEWmapOpsStrIt->second;
                  std::cerr << "DIRECTORY ERROR " << sys.getNetwork()->getNodeNum() << " regs: ";
                  for ( std::list< std::pair< global_reg_t, CacheCopy * > >::iterator listIt = ops.begin(); listIt != ops.end(); listIt++ ) {
                     std::cerr << "[" << (void *) listIt->first.key << "," << listIt->first.id << "]";
                  }
                  std::cerr << std::endl;
               } else {
                  sys.getCaches()[ NEWmapOpsStrIt->first ]->syncRegion( NEWmapOpsStrIt->second, _wd );
               }
            }
         }
         //std::cerr <<"########## ########### ######### "<<std::endl;
      }


      NANOS_INSTRUMENT( inst5.close(); );
      /* END OF ISSUE OPERATIONS */
   }
   NANOS_INSTRUMENT( inst2.close(); );
}


bool CacheController::dataIsReady() {
   bool allReady = true;
   unsigned int index;

   if ( _registered ) {
      return true;
   }

   for ( index = 0; ( index < _numCopies ) && allReady; index += 1 ) {
      allReady = _cacheCopies[ index ].isReady();
   }

   if ( allReady && _registered ) {
      if ( sys.usingNewCache() ) {
         for ( index = 0; index < _numCopies; index += 1 ) {
            CacheCopy &ccopy = _cacheCopies[ index ];
            NewNewRegionDirectory::addAccess( ccopy.getRegionDirectoryKey(), ccopy.getRegId(), ( !_targetCache ) ? 0 : _targetCache->getMemorySpaceId(), ccopy.getNewVersion() );
         }
      }
      _registered = true;
   }
   return allReady;
}

uint64_t CacheController::getAddress( unsigned int copyId ) const {
   return _cacheCopies[ copyId ].getDeviceAddress()  /* + ccopy.getCopyData().getOffset()  */;
}

CacheController::~CacheController() {
   if ( _numCopies > 0 ) 
      delete[] _cacheCopies;
}

void CacheController::copyDataOut() {
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_CC_CDOUT); );
   for ( unsigned int index = 0; index < _numCopies; index += 1 ) {
      CacheCopy &ccopy = _cacheCopies[ index ];

      if ( !ccopy.getCopyData().isOutput() ) continue;

      if ( sys.usingNewCache() ) {
         NANOS_INSTRUMENT( InstrumentState inst3(NANOS_POST_OUTLINE_WORK3); );
         NewNewRegionDirectory::addAccess( ccopy.getRegionDirectoryKey(), ccopy.getRegId(), ( !_targetCache ) ? 0 : _targetCache->getMemorySpaceId(), ccopy.getNewVersion() + 1 );
         NANOS_INSTRUMENT( inst3.close(); );
      }
   
      // TODO: WriteThrough code

      //Region reg = NewRegionDirectory::build_region( *ccopy._copy );
      //std::cerr << "Adding version "<< ccopy._version << " for addr " << ccopy._copy->getBaseAddress() << std::endl;
      //_directory->addAccess( reg, ccopy._copy->isInput(), ccopy._copy->isOutput(), 0, ((uint64_t)ccopy._copy->getBaseAddress()) + ccopy._copy->getOffset(), ccopy._version + 1 );
      //Region origReg = NewRegionDirectory::build_region( *ccopy._copy );
      //_targetCache->syncRegion( reg, (ccopy._cacheEntry->address + ccopy._offset) + ccopy._copy->getOffset() );
   }
       
   NANOS_INSTRUMENT( inst2.close(); );
}

void CacheController::getInfoFromPredecessor( CacheController const &predecessorController ) {
   //std::cerr << _wd.getId() <<" checking predecessor info from " << predecessorController._wd.getId() << std::endl;
   _provideLock.acquire();
   for( unsigned int index = 0; index < predecessorController._numCopies; index += 1) {
      std::map< reg_t, unsigned int > &regs = _providedRegions[ predecessorController._cacheCopies[ index ]._reg.key ];
      regs[ predecessorController._cacheCopies[ index ]._reg.id ] = ( ( predecessorController._cacheCopies[index].getCopyData().isOutput() ) ? predecessorController._cacheCopies[ index ].getNewVersion() + 1 : predecessorController._cacheCopies[ index ].getNewVersion() );
      //std::cerr << "provided data for copy " << index << " reg ("<<predecessorController._cacheCopies[ index ]._reg.key<<"," << predecessorController._cacheCopies[ index ]._reg.id << ") with version " << ( ( predecessorController._cacheCopies[index].getCopyData().isOutput() ) ? predecessorController._cacheCopies[ index ].getNewVersion() + 1 : predecessorController._cacheCopies[ index ].getNewVersion() ) << " isOut "<< predecessorController._cacheCopies[index].getCopyData().isOutput()<< " isIn "<< predecessorController._cacheCopies[index].getCopyData().isInput() << std::endl;
   }
   _provideLock.release();
}

bool CacheController::hasVersionInfoForRegion( global_reg_t reg, unsigned int &version, NewNewRegionDirectory::NewLocationInfoList &locations ) {
   bool resultHIT = false;
   bool resultSUBR = false;
   bool resultSUPER = false;
   std::map<NewNewRegionDirectory::RegionDirectoryKey, std::map< reg_t, unsigned int > >::iterator wantedDir = _providedRegions.find( reg.key );
   if ( wantedDir != _providedRegions.end() ) {
      unsigned int versionHIT = 0;
      std::map< reg_t, unsigned int >::iterator wantedReg = wantedDir->second.find( reg.id );
      if ( wantedReg != wantedDir->second.end() ) {
         versionHIT = wantedReg->second;
         resultHIT = true;
         wantedDir->second.erase( wantedReg );
      }

      unsigned int versionSUPER = 0;
      reg_t superPart = wantedDir->first->isThisPartOf( reg.id, wantedDir->second.begin(), wantedDir->second.end(), versionSUPER ); 
      if ( superPart != 0 ) {
         resultSUPER = true;
      }

      unsigned int versionSUBR = 0;
      if ( wantedDir->first->doTheseRegionsForm( reg.id, wantedDir->second.begin(), wantedDir->second.end(), versionSUBR ) ) {
         if ( versionHIT < versionSUBR && versionSUPER < versionSUBR ) {
            for ( std::map< reg_t, unsigned int >::const_iterator it = wantedDir->second.begin(); it != wantedDir->second.end(); it++ ) {
               global_reg_t r( it->first, wantedDir->first );
               reg_t intersect = r.key->computeIntersect( reg.id, r.id );
               if ( it->first == intersect ) {
                  locations.push_back( std::make_pair( it->first, it->first ) );
               }
            }
            NewNewDirectoryEntryData *firstEntry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( reg.id );
            if ( firstEntry == NULL ) {
               firstEntry = NEW NewNewDirectoryEntryData(  );
               firstEntry->addAccess( 0, 1 );
               wantedDir->first->setRegionData( reg.id, firstEntry );
            }
            resultSUBR = true;
            version = versionSUBR;
         }
      }
      if ( !resultSUBR && ( resultSUPER || resultHIT ) ) {
         if ( versionHIT >= versionSUPER ) {
            version = versionHIT;
            locations.push_back( std::make_pair( reg.id, reg.id ) );
         } else {
            version = versionSUPER;
            locations.push_back( std::make_pair( reg.id, superPart ) );
            NewNewDirectoryEntryData *firstEntry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( reg.id );
            NewNewDirectoryEntryData *secondEntry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( superPart );
            if ( firstEntry == NULL ) {
               firstEntry = NEW NewNewDirectoryEntryData( *secondEntry );
               wantedDir->first->setRegionData( reg.id, firstEntry );
            } else {
               if (secondEntry == NULL) std::cerr << "LOLWTF!"<< std::endl;
               *firstEntry = *secondEntry;
            }
         }
      }
   }
   return (resultSUBR || resultSUPER || resultHIT) ;
}
