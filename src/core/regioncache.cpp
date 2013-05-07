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
#include "system.hpp"
#include "deviceops.hpp"
#ifdef GPU_DEV
#include "gpudd.hpp"
#endif
#include "newregiondirectory.hpp"
#include "regioncache.hpp"
#include "cachedregionstatus.hpp"
#include "os.hpp"
#include "regiondict.hpp"
#include "memoryops_decl.hpp"

#define NEWINIT

AllocatedChunk::AllocatedChunk( RegionCache &owner, uint64_t addr, uint64_t hostAddress, std::size_t size, global_reg_t const &allocatedRegion ) :
   _owner( owner ),
   _lock(),
   _address( addr ),
   _hostAddress( hostAddress ),
   _size( size ),
   _dirty( false ),
   _invalidated( false ),
   _lruStamp( 0 ),
   _roBytes( 0 ),
   _rwBytes( 0 ),
   _refs( 0 ),
   _allocatedRegion( allocatedRegion ) {
      _newRegions = NEW CacheRegionDictionary( *(allocatedRegion.key) );
}

AllocatedChunk::~AllocatedChunk() {
      delete _newRegions;
}

void AllocatedChunk::clearNewRegions( global_reg_t const &reg ) {
   //delete _newRegions;
   _newRegions = NEW CacheRegionDictionary( *(reg.key) );
   _allocatedRegion = reg;
}


CacheRegionDictionary *AllocatedChunk::getNewRegions() {
   return _newRegions;
}

void AllocatedChunk::lock() {
   _lock.acquire();
}

void AllocatedChunk::unlock() {
   _lock.release();
}

bool AllocatedChunk::NEWaddReadRegion2( BaseAddressSpaceInOps &ops, reg_t reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::set< reg_t > &notPresentRegions, std::set< DeviceOps * > &thisRegOps, bool output, NewLocationInfoList const &locations ) {
   unsigned int currentVersion = 0;
   bool opEmitted = false;
   std::list< std::pair< reg_t, reg_t > > components;
   //std::cerr << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ " << __FUNCTION__ << " reg " << reg << " set rversion "<< version << " ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"<< std::endl;
   // lock / free needed for multithreading on the same cache.
   _newRegions->registerRegion( reg, components, currentVersion );
   NewNewRegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();

   //for ( CacheRegionDictionaryIterator it = _newRegions->begin(); it != _newRegions->end(); it++) {
   //   std::cerr << "Region: " << it->first << " "; _newRegions->getGlobalDirectoryKey()->printRegion( it->first ); std::cerr << " has entry with version " << (( (it->second).getData() ) ? (it->second).getData()->getVersion() : -1)<< std::endl;
   //}
   
   if ( components.size() == 1 ) {
      ensure( components.begin()->first == reg, "Error, wrong region");
   }

   for ( std::list< std::pair< reg_t, reg_t > >::iterator it = components.begin(); it != components.end(); it++ )
   {
      CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->first );
      if ( ( !entry || version > entry->getVersion() ) ) {
         //std::cerr << "No entry for region " << it->first << " must copy from region " << it->second << " want version "<< version << " entry version is " << ( (!entry) ? -1 : entry->getVersion() )<< std::endl;
         CachedRegionStatus *copyFromEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->second );
         if ( !copyFromEntry || version > copyFromEntry->getVersion() ) {
         //std::cerr << "I HAVE TO COPY: I dont have this region" << std::endl;

         global_reg_t chunkReg( it->first, key );
            
         NewLocationInfoList::const_iterator locIt;
         for ( locIt = locations.begin(); locIt != locations.end(); locIt++ ) {
            global_reg_t locReg( locIt->first, key );
            if ( locIt->first == it->first || chunkReg.contains( locReg ) ) {
               if ( reg != locIt->first || ( reg == locIt->first && _newRegions->getRegionData( locIt->first ) == NULL ) ) {
                  prepareRegion( locIt->first, version );
               }
               global_reg_t region_shape( locIt->first, key );
               global_reg_t data_source( locIt->second, key );
               //std::cerr << "shape: "<< it->first << " data source: " << it->second << std::endl;
               //std::cerr <<" CHECKING THIS SHIT ID " << data_source.id << std::endl;
               memory_space_id_t location = data_source.getFirstLocation();
               if ( location == 0 || location != _owner.getMemorySpaceId() ) {
                  //std::cerr << "add copy from host, reg " << region_shape.id << " version " << ops.getVersionNoLock( data_source ) << std::endl;
                  CachedRegionStatus *entryToCopy = ( CachedRegionStatus * ) _newRegions->getRegionData( locIt->first );
                  DeviceOps *entryToCopyOps = entryToCopy->getDeviceOps();
                  if ( !entryToCopyOps->addCacheOp() ) {
                     std::cerr << "ERROR " << __FUNCTION__ << std::endl;
                  }
                  if ( location == 0 ) {
                  ops.getOwnOps().insert( entryToCopyOps );
                  opEmitted = !( locIt->first == reg );
                     ops.addOpFromHost( region_shape, version );
                  } else if ( location != _owner.getMemorySpaceId() ) {
                  //std::cerr << "add copy from device, reg " << region_shape.id << " version " << ops.getVersionNoLock( data_source ) << std::endl;
                  ops.getOwnOps().insert( entryToCopyOps );
                  opEmitted = !( locIt->first == reg );
                     ops.addOp( &sys.getSeparateMemory( location ) , region_shape, version );
                  }
               }
            }
         }

            //if ( !entry ) {
            //   entry = NEW CachedRegionStatus();
            //   _newRegions->setRegionData( it->first, entry );
            //}
            //DeviceOps *entryOps = entry->getDeviceOps();
            //if ( entryOps->addCacheOp() ) {
            //   thisRegOps.insert( entryOps );
            //}
            //  notPresentRegions.insert( it->first );
         } else { // I have this region, as part of other region
         //std::cerr << "NO NEED TO COPY: I have this region as a part of region " << it->second << std::endl;
            //if ( !entry ) {
            //   entry = NEW CachedRegionStatus( *copyFromEntry );
            //   _newRegions->setRegionData( it->first, entry );
            //}
            currentOps.insert( copyFromEntry->getDeviceOps() );
         }
      } else if ( version == entry->getVersion() ) {
        // entry already at desired version.
         //std::cerr << "NO NEED TO COPY: I have this region already "  << std::endl;
         currentOps.insert( entry->getDeviceOps() );
      } else {
         std::cerr << "ERROR: version in cache > than version requested." << std::endl;
      }
   }

   CachedRegionStatus *thisRegEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !thisRegEntry ) {
      thisRegEntry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, thisRegEntry );
   }
   thisRegEntry->setVersion( version + ( output ? 1 : 0) );
   if ( opEmitted ) {
      DeviceOps *thisEntryOps = thisRegEntry->getDeviceOps();
      thisEntryOps->addCacheOp();
      ops.getOwnOps().insert( thisEntryOps );
   }
  // std::cerr << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[X]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"<< std::endl;
   _dirty = _dirty || output;
   return (!notPresentRegions.empty());
}

void AllocatedChunk::prepareRegion( reg_t reg, unsigned int version ) {
   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !entry ) {
      entry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, entry );
   }
   entry->setVersion( version );
  // std::cerr << "PREPARE REGION " << reg << " version " << version << " deviceOps " << (void *)entry->getDeviceOps() <<std::endl;
}

void AllocatedChunk::NEWaddWriteRegion( reg_t reg, unsigned int version ) {
   unsigned int currentVersion = 0;
   std::list< std::pair< reg_t, reg_t > > components;
   _newRegions->registerRegion( reg, components, currentVersion );


   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !entry ) {
      entry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, entry );
   }
   entry->setVersion( version );
   //std::cerr << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ " << __FUNCTION__ << " reg " << reg << " set version " << version << " entry " << (void *)entry <<" ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"<< std::endl;

   _dirty = true;
}

bool AllocatedChunk::isInvalidated() const {
   return _invalidated;
}

Atomic<int> AllocatedChunk::numCall(0);
void AllocatedChunk::invalidate( RegionCache *targetCache, WD const &wd ) {
   int call = numCall++;
   BaseAddressSpaceInOps invalOps;
   DeviceOps localOps;

   NewNewRegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();
   key->lock();


   //bool wholeRegionInval = false;
   //for ( std::map<reg_t, RegionVectorEntry>::const_iterator it = _newRegions->begin(); it != _newRegions->end() /*&& !wholeRegionInval*/; it++ ) {
   //   if ( it->first == _allocatedRegion.id ) wholeRegionInval = true;
   //      CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->first );
   //  NewNewDirectoryEntryData *dentry = NewNewRegionDirectory::getDirectoryEntry( *key, it->first );
   //   std::cerr << "Wholecheck: Reg " << it->first;
   //   key->printRegion( it->first );
   //   if (dentry) std::cerr << *dentry;
   //   else std::cerr << " n/a ";
   //   std::cerr << " cache v " << (entry ? entry->getVersion() : -1)  << std::endl;
   //}
   //if ( wholeRegionInval ) {
      //global_reg_t reg( _allocatedRegion, _newRegions->getGlobalDirectoryKey() );
      bool opEmitted = false;

      std::list< std::pair< reg_t, reg_t > > missing;
      unsigned int ver = 0;
      _allocatedRegion.key->registerRegion( _allocatedRegion.id, missing, ver );

      std::map< reg_t, std::set< reg_t > > fragmentedRegions;

      for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missing.begin(); lit != missing.end(); lit++ ) {
         NewNewDirectoryEntryData *dentry = NewNewRegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), lit->first );
         NewNewDirectoryEntryData *dsentry = NewNewRegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), lit->second );
         std::cerr << "missing registerReg: " << lit->first << " "; _allocatedRegion.key->printRegion( lit->first ); if (!dentry ) { std::cerr << " nul "; } else { std::cerr << *dentry; } 
                             std::cerr << "," << lit->second << " "; _allocatedRegion.key->printRegion( lit->second ); if (!dsentry ) { std::cerr << " nul "; } else { std::cerr << *dsentry; }
                             std::cerr <<  std::endl;
         global_reg_t region_shape( lit->first, key );
         global_reg_t data_source( lit->second, key );
         if ( region_shape.id == data_source.id ) {
            if ( data_source.isLocatedIn( _owner.getMemorySpaceId() ) ) {
               if ( NewNewRegionDirectory::delAccess( data_source.key, data_source.id, _owner.getMemorySpaceId() ) ) {
                  std::cerr << "has to be copied!, shape = dsrc and Im the only owner!" << std::endl;
                  DeviceOps *thisChunkOps = data_source.getDeviceOps();
                  opEmitted = !( data_source.id == _allocatedRegion.id );
                  if ( thisChunkOps->addCacheOp() ) {
                     invalOps.getOwnOps().insert( thisChunkOps );
                     CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( data_source.id );
                     invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), data_source, entry->getVersion() );
                     // update main directory
                     data_source.setLocationAndVersion( 0, entry->getVersion() );
                  } else {
                     invalOps.getOtherOps().insert( thisChunkOps );
                     // make sure the op we are waiting for its the same that we want to do, maybe it is impossible to reach this code
                     std::cerr << "FIXME " << __FUNCTION__ << std::endl;
                  }
               }
            }
         } else {
            fragmentedRegions[ data_source.id ].insert( region_shape.id );
         }
      }

      for ( std::map< reg_t, std::set< reg_t > >::iterator mit = fragmentedRegions.begin(); mit != fragmentedRegions.end(); mit++ ) {
         std::cerr << " fragmented region " << mit->first << " has #chunks " << mit->second.size() << std::endl;
         global_reg_t data_source( mit->first, key );
         if ( data_source.isLocatedIn( _owner.getMemorySpaceId() ) ) {
            if ( NewNewRegionDirectory::delAccess( key, mit->first, _owner.getMemorySpaceId() ) ) {
               for ( std::set< reg_t >::iterator sit = mit->second.begin(); sit != mit->second.end(); sit++ ) {
                  std::cerr << "    this is region " << *sit << std::endl;
                  global_reg_t subReg( *sit, key );
                  subReg.initializeGlobalEntryIfNeeded();
                  DeviceOps *subRegOps = subReg.getDeviceOps();
                  opEmitted = !( *sit == _allocatedRegion.id );
                  if ( subRegOps->addCacheOp() ) {
                     invalOps.getOwnOps().insert( subRegOps );
                     CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( mit->first );
                     invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), subReg, entry->getVersion() );
                     subReg.setLocationAndVersion( 0, entry->getVersion() );
                  } else {
                     invalOps.getOtherOps().insert( subRegOps );
                     // make sure the op we are waiting for its the same that we want to do, maybe it is impossible to reach this code
                     std::cerr << "FIXME " << __FUNCTION__ << std::endl;
                  }
               }
            }
         }
      }

      _allocatedRegion.initializeGlobalEntryIfNeeded();
      NewNewRegionDirectory::delAccess( _allocatedRegion.key, _allocatedRegion.id, _owner.getMemorySpaceId() );

      if ( opEmitted ) {
         DeviceOps *thisEntryOps = _allocatedRegion.getDeviceOps();
         std::cerr << " add cache op to obj " << (void*)thisEntryOps << std::endl;
         thisEntryOps->addCacheOp();
         invalOps.getOwnOps().insert( thisEntryOps );
      }

      std::cerr << call << "========== whole reg "<< _newRegions->getRegionNodeCount() <<"===========> Invalidate region "<< (void*) key << ":" << _allocatedRegion.id << " reg: "; _allocatedRegion.key->printRegion( _allocatedRegion.id ); std::cerr << std::endl;

   key->unlock();
   invalOps.issue( wd );
   while ( !invalOps.isDataReady() ) { myThread->idle(); }
      //while( !localOps.allCompleted() ) { myThread->idle(); }

}

AllocatedChunk **RegionCache::selectChunkToInvalidate( /*CopyData const &cd, uint64_t addr,*/ std::size_t allocSize/*, RegionTree< CachedRegionStatus > *&regsToInval, CacheRegionDictionary *&newRegsToInval*/ ) {
   AllocatedChunk **allocChunkPtrPtr = NULL;
   //std::cerr << "Cache is full." << std::endl;
   MemoryMap<AllocatedChunk>::iterator it;
   bool done = false;
   int count = 0;
   //for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
   //   std::cerr << "["<< count << "] this chunk: " << ((void *) it->second) << " refs: " << (int)( (it->second != NULL) ? it->second->getReferenceCount() : -1 ) << " dirty? " << (int)( (it->second != NULL) ? it->second->isDirty() : -1 )<< std::endl;
   //   count++;
   //}
   //count = 0;
   //AllocatedChunk *chunkToReuseNoLru = NULL;
   AllocatedChunk **chunkToReuseNoLruPtr = NULL;
   MemoryMap<AllocatedChunk>::iterator itNoLru;
   //AllocatedChunk *chunkToReuse = NULL;
   AllocatedChunk **chunkToReusePtr = NULL;
   //AllocatedChunk *chunkToReuseDirtyNoLru = NULL;
   AllocatedChunk **chunkToReuseDirtyNoLruPtr = NULL;
   MemoryMap<AllocatedChunk>::iterator itDirtyNoLru;
   //AllocatedChunk *chunkToReuseDirty = NULL;
   AllocatedChunk **chunkToReuseDirtyPtr = NULL;
   MemoryMap<AllocatedChunk>::iterator itDirty;
   for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
      if ( it->second != NULL ) {
         global_reg_t reg = it->second->getAllocatedRegion();
         std::cerr << "["<< count << "] mmm this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << " ";
         reg.key->printRegion( reg.id );
         std::cerr << std::endl;
      }
      if ( it->second != NULL && it->second->getReferenceCount() == 0 && it->second->getSize() >= allocSize ) {
         if ( !it->second->isDirty() ) {
            if ( _lruTime == it->second->getLruStamp() ) {
               //std::cerr << "["<< count << "] this chunk: " << ((void *) it->second) << std::endl;
               //chunkToReuse = it->second;
               chunkToReusePtr = &(it->second);
               done = true;
               break;
            } else if ( chunkToReuseNoLruPtr == NULL ) {
               //chunkToReuseNoLru = it->second;
               chunkToReuseNoLruPtr = &(it->second);
               itNoLru = it;
            }
         } else {
            if ( _lruTime == it->second->getLruStamp() ) {
               //std::cerr << "["<< count << "] this chunk: " << ((void *) it->second) << std::endl;
               //chunkToReuseDirty = it->second;
               chunkToReuseDirtyPtr = &(it->second);
               itDirty = it;
            } else if ( chunkToReuseDirtyNoLruPtr == NULL ) {
               //chunkToReuseDirtyNoLru = it->second;
               chunkToReuseDirtyNoLruPtr = &(it->second);
               itDirtyNoLru = it;
            }
         }
      }
      count++;
   }
   if ( chunkToReusePtr == NULL ) {
      if ( chunkToReuseNoLruPtr != NULL ) {
         std::cerr << "LRU clean chunk"<< std::endl;
         //chunkToReuse = chunkToReuseNoLru;
         chunkToReusePtr = chunkToReuseNoLruPtr;
         done = true;
         it = itNoLru;
         increaseLruTime();
      } else if ( chunkToReuseDirtyPtr != NULL ) {
         std::cerr << "Dirty chunk"<< std::endl;
         //chunkToReuse = chunkToReuseDirty;
         chunkToReusePtr = chunkToReuseDirtyPtr;
         it = itDirty;
         done = true;
      } else if ( chunkToReuseDirtyNoLruPtr != NULL ) {
         std::cerr << "LRU Dirty chunk"<< std::endl;
         //chunkToReuse = chunkToReuseDirtyNoLru;
         chunkToReusePtr = chunkToReuseDirtyNoLruPtr;
         it = itDirtyNoLru;
         done = true;
         increaseLruTime();
      }
   } else {
      std::cerr << "clean chunk"<< std::endl;
   }
   if ( done ) {
      //std::cerr << _memorySpaceId << "IVE FOUND A CHUNK TO FREE (" << (void *) chunkToReusePtr << ") addr=" << it->first.getAddress() << " size="<< it->first.getLength() <<std::endl;
      fprintf(stderr, "[%s] Im cache with id %d, I've found a chunk to free, addr=%p size=%zu\n",  __FUNCTION__, _memorySpaceId, (void*)it->first.getAddress(), it->first.getLength());
      allocChunkPtrPtr = chunkToReusePtr;
      //chunkToReuse->setHostAddress( addr );
      //regsToInval = chunkToReuse->getRegions();
      //newRegsToInval = chunkToReuse->getNewRegions();
      //*chunkToReusePtr = NULL;
      //chunkToReuse->increaseLruStamp();
      //chunkToReuse->clearRegions();
      //chunkToReuse->clearNewRegions( cd );
      //allocChunkPtr = chunkToReuse;
   } else {
      fatal("IVE _not_ FOUND A CHUNK TO FREE");
   }
   return allocChunkPtrPtr;
}

void AllocatedChunk::confirmCopyIn( reg_t id, unsigned int version ) {
   unsigned int currentVersion = 0;
   std::list< std::pair< reg_t, reg_t > > components;
   _newRegions->registerRegion( id, components, currentVersion );

   std::cerr << __FUNCTION__ << " reg " << id << std::endl;

   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( id );
   if ( !entry ) {
      entry = NEW CachedRegionStatus();
      _newRegions->setRegionData( id, entry );
   }
   entry->setVersion( version );
}

unsigned int AllocatedChunk::getVersion( global_reg_t const &reg ) {
   unsigned int version = 0;
   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg.id );
   if ( entry ) {
      version = entry->getVersion();
   }
   return version;
}

DeviceOps *AllocatedChunk::getDeviceOps( global_reg_t const &reg ) {
   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg.id );
   return entry->getDeviceOps();
}

unsigned int AllocatedChunk::getVersionSetVersion( global_reg_t const &reg, unsigned int newVersion ) {
   //CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg.id );
   //if ( !entry ) {
   //   entry = NEW CachedRegionStatus();
   //   entry->setVersion( 0 );
   //   _newRegions->setRegionData( reg.id, entry );
   //}
   //unsigned int version = entry->getVersion();
   //entry->setVersion( 0 );
   //return version;
   std::list< std::pair< reg_t, reg_t > > components;
   unsigned int version;
   _newRegions->registerRegion( reg.id, components, version );
   CachedRegionStatus *entry = NULL;
   if ( components.size() == 1 ) {
      ensure( components.begin()->first == reg.id, "Error, wrong region");
      entry = ( CachedRegionStatus * ) _newRegions->getRegionData( components.begin()->first );
      if ( !entry ) {
         CachedRegionStatus *copyFromEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( components.begin()->second );
         if ( !copyFromEntry ) {
            entry = NEW CachedRegionStatus();
         } else {
            entry = NEW CachedRegionStatus( *copyFromEntry );
         }
         _newRegions->setRegionData( reg.id , entry );
      }
   } else {
      std::cerr << __FUNCTION__ << " unhandled case."<< std::endl;
   }
   version = entry->getVersion();
   entry->setVersion( newVersion );
   return version;
}

AllocatedChunk *RegionCache::getAddress( global_reg_t const &reg, CacheRegionDictionary *&newRegsToInval, WD const &wd ) {
   ChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   AllocatedChunk **allocChunkPtrPtr = NULL;
   global_reg_t allocatedRegion;
   allocatedRegion.key = reg.key;

   std::size_t allocSize = 0;
   uint64_t targetHostAddr = 0;
   //std::cerr << __FUNCTION__ << " num dimensions " << cd.getNumDimensions() << std::endl;

   if ( _flags == ALLOC_WIDE ) {
      allocatedRegion.id = 1;
   } else if ( _flags == ALLOC_FIT ) {
      allocatedRegion.id = reg.getFitRegionId();
   } else {
      std::cerr <<"RegionCache ERROR: Undefined _flags value."<<std::endl;
   }

   targetHostAddr = allocatedRegion.getFirstAddress();
   allocSize      = allocatedRegion.getDataSize();

  //std::cerr << "-----------------------------------------" << std::endl;
  //std::cerr << " Max " << cd.getMaxSize() << std::endl;
  //std::cerr << "WIDE targetHostAddr: "<< ((void *)cd.getBaseAddress()) << std::endl;
  //std::cerr << "WIDE allocSize     : "<< cd.getMaxSize() << std::endl;
  //std::cerr << "FIT  targetHostAddr: "<< ((void *)cd.getFitAddress()) << std::endl;
  //std::cerr << "FIT  allocSize     : "<< cd.getFitSize() << std::endl;
  //std::cerr << "-----------------------------------------" << std::endl;
  //
  //std::cerr << "Requesting a chunk with targetHostAddr=" << (void *) targetHostAddr << " and size=" << allocSize << " allocRegionId " << allocatedRegion.id << " "; allocatedRegion.key->printRegion( allocatedRegion.id ); std::cerr << std::endl;

   _chunks.getOrAddChunk2( targetHostAddr, allocSize, results );
   if ( results.size() != 1 ) {
      message0( "Got results.size()="<< results.size() << " I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ChunkList::iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
   } else {
      //if ( results.front().first->getAddress() != targetHostAddr || results.front().first->getLength() < allocSize ) {
      //   std::cerr << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ERROR, realloc needed>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      //}
      if ( *(results.front().second) == NULL ) {

         void *deviceMem = _device.memAllocate( allocSize, sys.getSeparateMemory( _memorySpaceId ) );
         //std::cerr << "malloc returns " << (void *)deviceMem << std::endl;
         if ( deviceMem == NULL ) {
            // Device is full, free some chunk
            //std::cerr << " INVAL ! "<<std::endl;
            allocChunkPtrPtr = selectChunkToInvalidate( /*cd, results.front().first->getAddress(),*/ allocSize/*, regsToInval, newRegsToInval*/ );
            if ( allocChunkPtrPtr != NULL ) {
               allocChunkPtr = *allocChunkPtrPtr;
               allocChunkPtr->invalidate( this, wd );
               allocChunkPtr->increaseLruStamp();
               allocChunkPtr->clearNewRegions( allocatedRegion );
               allocChunkPtr->setHostAddress( results.front().first->getAddress() );
               *(results.front().second) = allocChunkPtr;
               *allocChunkPtrPtr = NULL;
            }
         } else {
            *(results.front().second) = NEW AllocatedChunk( *this, (uint64_t) deviceMem, results.front().first->getAddress(), results.front().first->getLength(), allocatedRegion );
            allocChunkPtr = *(results.front().second);
            //*(results.front().second) = allocChunkPtr;
         }
      } else {
         if ( results.front().first->getAddress() <= targetHostAddr ) {
            if ( results.front().first->getLength() + results.front().first->getAddress() >= (targetHostAddr + allocSize) ) {
               allocChunkPtr = *(results.front().second);
            } else {
               std::cerr << "I need a realloc of an allocated chunk!" << std::endl;
            }
         }
      }
   }
   if ( allocChunkPtr == NULL ) { 
      std::cerr << "WARNING: null RegionCache::getAddress()" << std::endl;
   } else {
     //std::cerr << __FUNCTION__ << " returns dev address " << (void *) allocChunkPtr->getAddress() << std::endl;
     allocChunkPtr->lock();
     allocChunkPtr->addReference();
   }
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

AllocatedChunk *RegionCache::getAllocatedChunk( global_reg_t const &reg ) const {
   ConstChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   _chunks.getChunk3( reg.getFirstAddress(), reg.getBreadth(), results );
   if ( results.size() != 1 ) {
         std::cerr <<"Requested addr " << (void *) reg.getFirstAddress() << " size " << reg.getBreadth() << std::endl;
      message0( "I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ConstChunkList::const_iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
   } else {
      if ( results.front().second )
         allocChunkPtr = *(results.front().second);
      else
         allocChunkPtr = NULL;
   }
   if ( !allocChunkPtr ) { sys.printBt(); std::cerr << "Error, null region "; reg.key->printRegion( reg.id ); std::cerr << std::endl; }
   ensure(allocChunkPtr != NULL, "Chunk not found!");
   if ( allocChunkPtr ) {
      //std::cerr << "AllocChunkPtr is " << allocChunkPtr << std::endl;
      allocChunkPtr->lock(); 
   }
   return allocChunkPtr;
}

void RegionCache::NEWcopyIn( unsigned int srcLocation, global_reg_t const &reg, unsigned int version, WD const &wd ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   uint64_t origDevAddr = chunk->getAddress() + ( reg.getFirstAddress() - chunk->getHostAddress() );
   DeviceOps *ops = chunk->getDeviceOps( reg );
   chunk->unlock();
   //std::cerr << " COPY REGION ID " << reg.id << " OPS " << (void*)ops << std::endl;
   copyIn( reg, origDevAddr, srcLocation, ops, NULL, wd );
}

void RegionCache::NEWcopyOut( global_reg_t const &reg, unsigned int version, WD const &wd ) {
   AllocatedChunk *origChunk = getAllocatedChunk( reg );
   uint64_t origDevAddr = origChunk->getAddress() + ( reg.getFirstAddress() - origChunk->getHostAddress() );
   DeviceOps *ops = reg.getDeviceOps();
   origChunk->unlock();
   CompleteOpFunctor *f = NEW CompleteOpFunctor( ops, origChunk );
   copyOut( reg, origDevAddr, ops, f, wd );
}

RegionCache::RegionCache( memory_space_id_t memSpaceId, Device &cacheArch, enum CacheOptions flags ) : _device( cacheArch ), _memorySpaceId( memSpaceId ),
    _flags( flags ), _lruTime( 0 ), _copyInObj( *this ), _copyOutObj( *this ) {
}

unsigned int RegionCache::getMemorySpaceId() {
   return _memorySpaceId;
}

void RegionCache::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   //std::cerr << "_device._copyIn( copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, _pe, ops, wd="<< wd.getId() <<" );";
   if (!fake) _device._copyIn( devAddr, hostAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, (CompleteOpFunctor *) NULL, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   if (!fake) _device._copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, (CompleteOpFunctor *) NULL, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   ensure( f != NULL, " Error, functor received is null.");
   //std::cerr << "_device._copyOut( copyFrom=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, _pe, ops, wd="<< (&wd != NULL ? wd.getId() : -1 ) <<" );";
   if (!fake) _device._copyOut( hostAddr, devAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, f, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld,  DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f != NULL, " Error, functor received is null.");
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   if (!fake) _device._copyOutStrided1D( hostAddr, devAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, f, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   sys.getCaches()[ syncFrom ]->_copyOut( hostAddr, origDevAddr, len, cout, fsource, wd, fake );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   delete cout;
   this->_copyIn( devAddr, hostAddr, len, ops, (CompleteOpFunctor *) NULL, wd, fake );
}

void RegionCache::_syncAndCopyInStrided1D( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getCaches()[ syncFrom ]->getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   sys.getCaches()[ syncFrom ]->_copyOutStrided1D( hostAddr, origDevAddr, len, numChunks, ld, cout, fsource, wd, fake );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   delete cout;
   this->_copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, ops, (CompleteOpFunctor *) NULL, wd, fake );
}

void RegionCache::_copyDevToDev( memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   //AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len );
   AllocatedChunk *origChunk = sys.getSeparateMemory( copyFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   //std::cerr << "_device._copyDevToDev( copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len, _pe, sys.getSeparateMemory( copyFrom="<< copyFrom<<" ), ops, wd="<< wd.getId() << ", f="<< f <<" );" <<std::endl;
   if (!fake) _device._copyDevToDev( devAddr, origDevAddr, len, sys.getSeparateMemory( _memorySpaceId ), sys.getSeparateMemory( copyFrom ), ops, fsource, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyDevToDevStrided1D( memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   //AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len );
   AllocatedChunk *origChunk = sys.getSeparateMemory( copyFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   ensure( f == NULL, " Error, functor received is not null.");
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   //std::cerr << "_device._copyDevToDevStrided1D( copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len, _pe, sys.getCaches()[ copyFrom="<< copyFrom<<" ]->_pe, ops, wd="<< wd.getId() <<", f="<< f <<" );"<<std::endl;
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   if (!fake) _device._copyDevToDevStrided1D( devAddr, origDevAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), sys.getSeparateMemory( copyFrom ), ops, fsource, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::CopyIn::doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyIn( devAddr, hostAddr, size, ops, f, wd, fake );
   //} else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
   } else if ( sys.canCopy( dataLocation, getParent().getMemorySpaceId() ) ) { 
      getParent()._copyDevToDev( dataLocation, devAddr, hostAddr, size, ops, f, wd, fake );
   } else {
      getParent()._syncAndCopyIn( dataLocation, devAddr, hostAddr, size, ops, f, wd, fake );
   }
}

void RegionCache::CopyIn::doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyInStrided1D( devAddr, hostAddr, size, count, ld, ops, f, wd, fake );
   //} else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
   } else if ( sys.canCopy( dataLocation, getParent().getMemorySpaceId() ) ) { 
      getParent()._copyDevToDevStrided1D( dataLocation, devAddr, hostAddr, size, count, ld, ops, f, wd, fake );
   } else {
      getParent()._syncAndCopyInStrided1D( dataLocation, devAddr, hostAddr, size, count, ld, ops, f, wd, fake );
   }
}

void RegionCache::CopyOut::doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   getParent()._copyOut( hostAddr, devAddr, size, ops, f, wd, fake );
}
void RegionCache::CopyOut::doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   getParent()._copyOutStrided1D( hostAddr, devAddr, size, count, ld, ops, f, wd, fake );
}

void RegionCache::doOp( Op *opObj, global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd ) {

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
      CompleteOpFunctor *_f;
      public:
         LocalFunction( Op *opO, nanos_region_dimension_internal_t *r, unsigned int n, unsigned int t, unsigned int nc, std::size_t ccs, unsigned int loc, DeviceOps *operations, WD const &workdesc, uint64_t devAddr, uint64_t hostAddr, CompleteOpFunctor *f )
               : _opObj( opO ), _region( r ), _numDimensions( n ), _targetDimension( t ), _numChunks( nc ), _contiguousChunkSize( ccs ), _location( loc ), _ops( operations ), _wd( workdesc ), _devBaseAddr( devAddr ), _hostBaseAddr( hostAddr ), _f( f ) {
         }
         void issueOpsRecursive( unsigned int idx, std::size_t offset, std::size_t leadingDim ) {
            if ( idx == ( _numDimensions - 1 ) ) {
               //issue copy
               unsigned int L_numChunks = _numChunks; //_region[ idx ].accessed_length;
               if ( L_numChunks > 1 && sys.usePacking() ) {
                  //std::cerr << "[NEW]opObj("<<_opObj->getStr()<<")->doStrided( src="<<_location<<", dst="<< _opObj->getParent().getMemorySpaceId()<<", "<<(void*)(_devBaseAddr+offset)<<", "<<(void*)(_hostBaseAddr+offset)<<", "<<_contiguousChunkSize<<", "<<_numChunks<<", "<<leadingDim<<", _ops="<< (void*)_ops<<", _wd="<<(&_wd != NULL ? _wd.getId():-1)<<" )";
                  _opObj->doStrided( _location, _devBaseAddr+offset, _hostBaseAddr+offset, _contiguousChunkSize, _numChunks, leadingDim, _ops, _f, _wd, false );
                  //std::cerr <<" done"<< std::endl;
               } else {
                  for (unsigned int chunkIndex = 0; chunkIndex < L_numChunks; chunkIndex +=1 ) {
                     //std::cerr <<"[NEW]opObj("<<_opObj->getStr()<<")->doNoStrided( src="<<_location<<", dst="<< _opObj->getParent().getMemorySpaceId()<<", "<<(void*)(_devBaseAddr+offset + chunkIndex*(leadingDim))<<", "<<(void*)(_hostBaseAddr+offset + chunkIndex*(leadingDim))<<", "<<_contiguousChunkSize<<", _ops="<< (void*)_ops<< ", _wd="<<(&_wd != NULL ? _wd.getId():-1)<<" )";
                    _opObj->doNoStrided( _location, _devBaseAddr+offset + chunkIndex*(leadingDim), _hostBaseAddr+offset + chunkIndex*(leadingDim), _contiguousChunkSize, _ops, _f, _wd, false );
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
   LocalFunction local( opObj, region, hostMem.getNumDimensions(), dimIdx, numChunks, contiguousChunkSize, location, ops, wd, devBaseAddr, hostMem.getFirstAddress(), f /* hostMem.key->getBaseAddress()*/ );
   local.issueOpsRecursive( dimIdx-1, 0, leadingDimension );
}

void RegionCache::copyIn( global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd ) {
   doOp( &_copyInObj, hostMem, devBaseAddr, location, ops, NULL, wd );
}

void RegionCache::copyOut( global_reg_t const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd ) {
   doOp( &_copyOutObj, hostMem, devBaseAddr, /* locations unused, copyOut is always to 0 */ 0, ops, f, wd );
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

CompleteOpFunctor::CompleteOpFunctor( DeviceOps *ops, AllocatedChunk *chunk ) : _ops( ops ), _chunk( chunk ) {
}

CompleteOpFunctor::~CompleteOpFunctor() {
}

void CompleteOpFunctor::operator()() {
   _chunk->removeReference();
}

unsigned int RegionCache::getVersionSetVersion( global_reg_t const &reg, unsigned int newVersion ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   unsigned int version = chunk->getVersionSetVersion( reg, newVersion );
   chunk->unlock();
   return version;
}

unsigned int RegionCache::getVersion( global_reg_t const &reg ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   unsigned int version = chunk->getVersion( reg );
   chunk->unlock();
   return version;
}

void RegionCache::releaseRegion( global_reg_t const &reg, WD const &wd ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   std::cerr << "- [" << wd.getId() << "," << _memorySpaceId << "," << chunk->getReferenceCount() <<"] " << (void *) chunk << " "; reg.key->printRegion( reg.id ); std::cerr << std::endl;
   chunk->removeReference();
   chunk->unlock();
}

uint64_t RegionCache::getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress ) const {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   //uint64_t addr = chunk->getAddress() + ( reg.getFirstAddress() - chunk->getHostAddress() );
   uint64_t addr = ( chunk->getAddress() - ( chunk->getHostAddress() - baseAddress ) ) + 
      ( reg.getFirstAddress() - reg.getBaseAddress() ); /* this is the copy original offset, getBaseAddress does not return the correct value in slave nodes, 
                                                         * in this case getFirstAddress is also based om it so it corrects the error */
   //std::cerr << "getDevice Address= "<< (void*)addr <<" for reg "; reg.key->printRegion( reg.id ); std::cerr << std::endl;
   chunk->unlock();
   return addr;
}

void RegionCache::prepareRegion( global_reg_t const &reg, WD const &wd ) {
   _lock.acquire();
   CacheRegionDictionary *newRegsToInvalidate = NULL;
   AllocatedChunk *chunk = getAddress( reg, newRegsToInvalidate, wd );
   std::cerr << "+ [" << wd.getId() << "," << _memorySpaceId << "," << (chunk->getReferenceCount()-1) <<"] "<< (void *) chunk << " " ; reg.key->printRegion( reg.id ); std::cerr << std::endl;
   chunk->unlock();
   _lock.release();
}

void RegionCache::setRegionVersion( global_reg_t const &hostMem, unsigned int version ) {
   AllocatedChunk *chunk = getAllocatedChunk( hostMem );
   chunk->NEWaddWriteRegion( hostMem.id, version );
   chunk->unlock();
}

void RegionCache::copyInputData( BaseAddressSpaceInOps &ops, global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations ) {
   _lock.acquire();
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   std::set< reg_t > notPresentParts;
   //      std::cerr << "locations:  ";
   //      for ( NewLocationInfoList::const_iterator it2 = locations.begin(); it2 != locations.end(); it2++ ) {
   //         std::cerr << "[ " << it2->first << "," << it2->second << " ] ";
   //      }
   //      std::cerr << std::endl;
   if ( chunk->NEWaddReadRegion2( ops, reg.id, version, ops.getOtherOps(), notPresentParts, ops.getOwnOps(), output, locations ) ) {
   }

   //reg.setLocationAndVersion( _memorySpaceId, version + (output ? 1 : 0) );
   chunk->unlock();
   _lock.release();
}

void RegionCache::allocateOutputMemory( global_reg_t const &reg, unsigned int version ) {
   _lock.acquire();
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   chunk->NEWaddWriteRegion( reg.id, version );
   reg.setLocationAndVersion( _memorySpaceId, version );
   chunk->unlock();
   _lock.release();
}


