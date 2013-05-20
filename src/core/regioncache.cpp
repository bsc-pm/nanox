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

//#define VERBOSE_DEV_OPS

AllocatedChunk::AllocatedChunk( RegionCache &owner, uint64_t addr, uint64_t hostAddress, std::size_t size, global_reg_t const &allocatedRegion ) :
   _owner( owner ),
   _lock(),
   _address( addr ),
   _hostAddress( hostAddress ),
   _size( size ),
   _dirty( false ),
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
   //std::cerr << ": " << myThread->getId() <<" : Locked chunk " << (void *) this << std::endl;
   _lock.acquire();
   //sys.printBt();
   //std::cerr << "x " << myThread->getId() <<" x Locked chunk " << (void *) this << std::endl;
}

void AllocatedChunk::unlock() {
   //sys.printBt();
   //std::cerr << "x " << myThread->getId() << " x Unlocked chunk " << (void *) this << std::endl;
   _lock.release();
   //std::cerr << ": " << myThread->getId() << " : Unlocked chunk " << (void *) this << std::endl;
}

bool AllocatedChunk::locked() const {
   return _lock.getState() != NANOS_LOCK_FREE;
}

bool AllocatedChunk::NEWaddReadRegion2( BaseAddressSpaceInOps &ops, reg_t reg, unsigned int version, std::set< reg_t > &notPresentRegions, bool output, NewLocationInfoList const &locations ) {
   unsigned int currentVersion = 0;
   bool opEmitted = false;
   std::list< std::pair< reg_t, reg_t > > components;

   CachedRegionStatus *thisRegEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !thisRegEntry ) {
      thisRegEntry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, thisRegEntry );
   }

   DeviceOps *thisEntryOps = thisRegEntry->getDeviceOps();
   if ( thisEntryOps->addCacheOp() ) {
      opEmitted = true;
      ops.insertOwnOp( thisEntryOps, global_reg_t( reg, _newRegions->getGlobalDirectoryKey() ), version, _owner.getMemorySpaceId() );

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
                        if ( entryToCopy != thisRegEntry ) {
                           if ( !entryToCopyOps->addCacheOp() ) {
                              std::cerr << "ERROR " << __FUNCTION__ << std::endl;
                           }
                           ops.insertOwnOp( entryToCopyOps, global_reg_t( locIt->first, _newRegions->getGlobalDirectoryKey() ), version, _owner.getMemorySpaceId() );
                        }

                        if ( location == 0 ) {
                           ops.addOpFromHost( region_shape, version );
                        } else if ( location != _owner.getMemorySpaceId() ) {
                           ops.addOp( &sys.getSeparateMemory( location ) , region_shape, version );
                        }
                     }
                  }
               }
            } else { // I have this region, as part of other region
               //std::cerr << "NO NEED TO COPY: I have this region as a part of region " << it->second << std::endl;
               //if ( !entry ) {
               //   entry = NEW CachedRegionStatus( *copyFromEntry );
               //   _newRegions->setRegionData( it->first, entry );
               //}
               ops.getOtherOps().insert( copyFromEntry->getDeviceOps() );
            }
         } else if ( version == entry->getVersion() ) {
            // entry already at desired version.
            //std::cerr << "NO NEED TO COPY: I have this region already "  << std::endl;
            ops.getOtherOps().insert( entry->getDeviceOps() );
         } else {
            std::cerr << "ERROR: version in cache (" << entry->getVersion() << ") > than version requested ("<< version <<")." << std::endl;
            key->printRegion( reg );
            std::cerr << std::endl;
         }
      }
      thisRegEntry->setVersion( version + ( output ? 1 : 0) );
   } else {
      ops.getOtherOps().insert( thisEntryOps );
   }
   //std::cerr << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[X]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"<< std::endl;
   _dirty = _dirty || output;
   return opEmitted;
}

void AllocatedChunk::prepareRegion( reg_t reg, unsigned int version ) {
   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !entry ) {
      entry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, entry );
   }
   entry->setVersion( version );
}

//void AllocatedChunk::clearDirty( global_reg_t const &reg ) {
//   unsigned int currentVersion = 0;
//   std::list< std::pair< reg_t, reg_t > > components;
//   _newRegions->registerRegion( reg.id, components, currentVersion );
//
//   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg.id );
//   if ( !entry ) {
//      entry = NEW CachedRegionStatus();
//      _newRegions->setRegionData( reg.id, entry );
//   }
//   entry->clearDirty();
//}

void AllocatedChunk::NEWaddWriteRegion( reg_t reg, unsigned int version ) {
   unsigned int currentVersion = 0;
   std::list< std::pair< reg_t, reg_t > > components;
   _newRegions->registerRegion( reg, components, currentVersion );

   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !entry ) {
      entry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, entry );
   }
   //entry->setDirty();
   entry->setVersion( version );
   if ( VERBOSE_CACHE ) { std::cerr << "[[[[[[[[[[[[[[[[[[[[ " << __FUNCTION__ << " reg " << reg << " set version " << version << " entry " << (void *)entry << " components size " << components.size() <<" ]]]]]]]]]]]]]]]]]]]]"<< std::endl; }

   _dirty = true;
}

Atomic<int> AllocatedChunk::numCall(0);
void AllocatedChunk::invalidate( RegionCache *targetCache, WD const &wd ) {
   SeparateAddressSpaceOutOps invalOps( true );

   NewNewRegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();
   key->lock();

   std::list< std::pair< reg_t, reg_t > > missing;
   unsigned int ver = 0;
   //_allocatedRegion.key->registerRegionReturnSameVersionSubparts( _allocatedRegion.id, missing, ver );
   _newRegions->registerRegionReturnSameVersionSubparts( _allocatedRegion.id, missing, ver );

   //std::set<DeviceOps *> ops;
   //ops.insert( _allocatedRegion.getDeviceOps() );
   //for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missing.begin(); lit != missing.end(); lit++ ) {
   //   global_reg_t data_source( lit->second, key );
   //   ops.insert( data_source.getDeviceOps() );
   //}

   //for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
   //   (*opIt)->syncAndDisableInvalidations();
   //}

   std::set< reg_t > regions_to_remove_access;

   DeviceOps *thisChunkOps = _allocatedRegion.getDeviceOps();
   if ( missing.size() == 1 ) {
      ensure( _allocatedRegion.id == missing.begin()->first, "Wrong region." );
      if ( _allocatedRegion.isLocatedIn( _owner.getMemorySpaceId() ) ) {
         if ( NewNewRegionDirectory::isOnlyLocated( _allocatedRegion.key, _allocatedRegion.id, _owner.getMemorySpaceId() ) ) {
            //std::cerr << "AC: has to be copied!, shape = dsrc and Im the only owner!" << std::endl;
            //if ( thisChunkOps->addCacheOp() ) {
               CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( _allocatedRegion.id );
               invalOps.insertOwnOp( thisChunkOps, _allocatedRegion, entry->getVersion(), 0 );
               regions_to_remove_access.insert( _allocatedRegion.id );
               invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), _allocatedRegion, entry->getVersion(), NULL );
               entry->resetVersion();
            //} else {
            //   std::cerr << " ERROR: could not add a cache op to my ops!"<<std::endl;
            //}
         }
      }
   } else {
      std::map< reg_t, std::set< reg_t > > fragmented_regions;
      for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missing.begin(); lit != missing.end(); lit++ ) {
         NewNewDirectoryEntryData *dentry = NewNewRegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), lit->first );
         if ( VERBOSE_CACHE ) {
            NewNewDirectoryEntryData *dsentry = NewNewRegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), lit->second );
            std::cerr << (void *)_newRegions << " missing registerReg: " << lit->first << " "; _allocatedRegion.key->printRegion( lit->first ); if (!dentry ) { std::cerr << " nul "; } else { std::cerr << *dentry; } 
            std::cerr << "," << lit->second << " "; _allocatedRegion.key->printRegion( lit->second ); if (!dsentry ) { std::cerr << " nul "; } else { std::cerr << *dsentry; }
            std::cerr <<  std::endl;
         }
         global_reg_t region_shape( lit->first, key );
         global_reg_t data_source( lit->second, key );
         if ( region_shape.id == data_source.id ) {
            ensure( _allocatedRegion.id != data_source.id, "Wrong region" );
            if ( data_source.isLocatedIn( _owner.getMemorySpaceId() ) ) {
               if ( NewNewRegionDirectory::isOnlyLocated( data_source.key, data_source.id, _owner.getMemorySpaceId() ) ) {
                  if ( VERBOSE_CACHE ) {
                     for ( CacheRegionDictionary::citerator pit = _newRegions->begin(); pit != _newRegions->end(); pit++ ) {
                        std::cerr << " reg " << pit->first << " "; key->printRegion( pit->first); std::cerr << " has entry " << (void *) &pit->second << std::endl;
                     }
                  }
                  regions_to_remove_access.insert( data_source.id );
                  DeviceOps *fragment_ops = data_source.getDeviceOps();
                  CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( data_source.id );
                  if ( VERBOSE_CACHE ) { std::cerr << data_source.id << " has to be copied!, shape = dsrc and Im the only owner! "<< (void *)entry << std::endl; }
                  //if ( fragment_ops->addCacheOp() ) {
                     invalOps.insertOwnOp( fragment_ops, data_source, entry->getVersion(), 0 );
                     invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), data_source, entry->getVersion(), NULL );
                  //} else {
                  //   invalOps.getOtherOps().insert( fragment_ops );
                  //   // make sure the op we are waiting for its the same that we want to do, maybe it is impossible to reach this code
                  //   std::cerr << "FIXME " << __FUNCTION__ << " this is memspace "<< _owner.getMemorySpaceId() << std::endl;
                  //}
                  entry->resetVersion();
               }
            }
         } else {
            if ( dentry == NULL ||
                  ( data_source.getVersion() <= region_shape.getVersion() && NewNewRegionDirectory::isOnlyLocated( region_shape.key, region_shape.id, _owner.getMemorySpaceId() ) ) ||
                  ( data_source.getVersion() >  region_shape.getVersion() && NewNewRegionDirectory::isOnlyLocated( data_source.key,  data_source.id,  _owner.getMemorySpaceId() ) )
               ) {
               fragmented_regions[ data_source.id ].insert( region_shape.id );
            }
         }
      }

      for ( std::map< reg_t, std::set< reg_t > >::iterator mit = fragmented_regions.begin(); mit != fragmented_regions.end(); mit++ ) {
         if ( VERBOSE_CACHE ) { std::cerr << " fragmented region " << mit->first << " has #chunks " << mit->second.size() << std::endl; }
         global_reg_t data_source( mit->first, key );
         if ( NewNewRegionDirectory::isOnlyLocated( key, data_source.id, _owner.getMemorySpaceId() ) ) {
            bool subChunkInval = false;
            CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( data_source.id );

            for ( std::set< reg_t >::iterator sit = mit->second.begin(); sit != mit->second.end(); sit++ ) {
               if ( VERBOSE_CACHE ) { std::cerr << "    this is region " << *sit << std::endl; }
               global_reg_t subReg( *sit, key );
               NewNewDirectoryEntryData *dentry = NewNewRegionDirectory::getDirectoryEntry( *key, *sit );
               if ( dentry == NULL ) { //FIXME: maybe we need a version check to handle when the dentry exists but is old?
                  std::list< std::pair< reg_t, reg_t > > missingSubReg;
                  _allocatedRegion.key->registerRegion( subReg.id, missingSubReg, ver );
                  for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missingSubReg.begin(); lit != missingSubReg.end(); lit++ ) {
                     global_reg_t region_shape( lit->first, key );
                     global_reg_t new_data_source( lit->second, key );
                     if ( VERBOSE_CACHE ) { std::cerr << " DIR CHECK WITH FRAGMENT: "<< lit->first << " - " << lit->second << " " << std::endl; }

                     NewNewDirectoryEntryData *subEntry = NewNewRegionDirectory::getDirectoryEntry( *key, lit->first );
                     if ( subEntry ) {
                        std::cerr << "FIXME: Invalidation, and found a region shape with no entry, a new Entry may be needed." << std::endl;
                     }
                     if ( new_data_source.id == data_source.id || NewNewRegionDirectory::isOnlyLocated( key, new_data_source.id, _owner.getMemorySpaceId() ) ) {
                        subChunkInval = true;
                        if ( VERBOSE_CACHE ) { std::cerr << " COPY subReg " << lit->first << " comes from subreg "<< subReg.id << " new DS " << new_data_source.id << std::endl; }
                        invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), region_shape, entry->getVersion(), thisChunkOps );
                     }
                  }
               } else {
                  DeviceOps *subRegOps = subReg.getDeviceOps();
                  //if ( subRegOps->addCacheOp() ) {
                  regions_to_remove_access.insert( subReg.id );
                  invalOps.insertOwnOp( subRegOps, data_source, entry->getVersion(), 0 );
                  invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), subReg, entry->getVersion(), subRegOps );
                  //} else {
                  //   invalOps.getOtherOps().insert( subRegOps );
                  //   std::cerr << "FIXME " << __FUNCTION__ << std::endl;
                  //}
               }
            }

            if ( subChunkInval ) {
               regions_to_remove_access.insert( data_source.id );
               invalOps.insertOwnOp( thisChunkOps, data_source, entry->getVersion(), 0 );
               //FIXME I think this is wrong, can potentially affect regions that are not there, 
            }
            entry->resetVersion();
         }
      }
   }

   //std::cerr << numCall++ << "=============> " << "Cache " << _owner.getMemorySpaceId() << " Invalidate region "<< (void*) key << ":" << _allocatedRegion.id << " reg: "; _allocatedRegion.key->printRegion( _allocatedRegion.id ); std::cerr << std::endl;

   key->unlock();
   invalOps.issue( wd );
   while ( !invalOps.isDataReady() ) { myThread->idle(); }
   if ( VERBOSE_CACHE ) { std::cerr << "===> Invalidation complete " << std::endl; }
   for ( std::set< reg_t >::iterator it = regions_to_remove_access.begin(); it != regions_to_remove_access.end(); it++ ) {
      NewNewRegionDirectory::delAccess( key, *it,  _owner.getMemorySpaceId() );
   }

   //for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
   //   (*opIt)->resumeInvalidations();
   //}
}

AllocatedChunk **RegionCache::selectChunkToInvalidate( std::size_t allocSize ) {
   AllocatedChunk **allocChunkPtrPtr = NULL;
   MemoryMap<AllocatedChunk>::iterator it;
   bool done = false;
   int count = 0;
   //for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
   //   std::cerr << "["<< count << "] this chunk: " << ((void *) it->second) << " refs: " << (int)( (it->second != NULL) ? it->second->getReferenceCount() : -1 ) << " dirty? " << (int)( (it->second != NULL) ? it->second->isDirty() : -1 )<< std::endl;
   //   count++;
   //}
   //count = 0;
   AllocatedChunk **chunkToReuseNoLruPtr = NULL;
   MemoryMap<AllocatedChunk>::iterator itNoLru;
   AllocatedChunk **chunkToReusePtr = NULL;
   AllocatedChunk **chunkToReuseDirtyNoLruPtr = NULL;
   MemoryMap<AllocatedChunk>::iterator itDirtyNoLru;
   AllocatedChunk **chunkToReuseDirtyPtr = NULL;
   MemoryMap<AllocatedChunk>::iterator itDirty;
   for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
      // if ( it->second != NULL ) {
      //    global_reg_t reg = it->second->getAllocatedRegion();
      //    std::cerr << "["<< count << "] mmm this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << " size is " << it->second->getSize() << " vs " << allocSize << " ";
      //    reg.key->printRegion( reg.id );
      //    std::cerr << std::endl;
      // }
      if ( it->second != NULL && it->second->getReferenceCount() == 0 && it->second->getSize() == allocSize ) {
         if ( !it->second->isDirty() ) {
            if ( _lruTime == it->second->getLruStamp() ) {
               //std::cerr << "["<< count << "] this chunk: " << ((void *) it->second) << std::endl;
               chunkToReusePtr = &(it->second);
               done = true;
               break;
            } else if ( chunkToReuseNoLruPtr == NULL ) {
               chunkToReuseNoLruPtr = &(it->second);
               itNoLru = it;
            }
         } else {
            if ( _lruTime == it->second->getLruStamp() ) {
               //std::cerr << "["<< count << "] this chunk: " << ((void *) it->second) << std::endl;
               chunkToReuseDirtyPtr = &(it->second);
               itDirty = it;
            } else if ( chunkToReuseDirtyNoLruPtr == NULL ) {
               chunkToReuseDirtyNoLruPtr = &(it->second);
               itDirtyNoLru = it;
            }
         }
      }
      count++;
   }
   if ( chunkToReusePtr == NULL ) {
      if ( chunkToReuseNoLruPtr != NULL ) {
         //std::cerr << "LRU clean chunk"<< std::endl;
         chunkToReusePtr = chunkToReuseNoLruPtr;
         done = true;
         it = itNoLru;
         increaseLruTime();
      } else if ( chunkToReuseDirtyPtr != NULL ) {
         //std::cerr << "Dirty chunk"<< std::endl;
         chunkToReusePtr = chunkToReuseDirtyPtr;
         it = itDirty;
         done = true;
      } else if ( chunkToReuseDirtyNoLruPtr != NULL ) {
         //std::cerr << "LRU Dirty chunk"<< std::endl;
         chunkToReusePtr = chunkToReuseDirtyNoLruPtr;
         it = itDirtyNoLru;
         done = true;
         increaseLruTime();
      }
   } else {
      //std::cerr << "clean chunk"<< std::endl;
   }
   if ( done ) {
      allocChunkPtrPtr = chunkToReusePtr;
      if ( VERBOSE_CACHE ) { fprintf(stderr, "[%s] Thd %d Im cache with id %d, I've found a chunk to free, %p (locked? %d) region %d addr=%p size=%zu\n",  __FUNCTION__, myThread->getId(), _memorySpaceId, *allocChunkPtrPtr, ((*allocChunkPtrPtr)->locked()?1:0), (*allocChunkPtrPtr)->getAllocatedRegion().id, (void*)it->first.getAddress(), it->first.getLength()); }
      (*allocChunkPtrPtr)->lock();
   } else {
      fatal("IVE _not_ FOUND A CHUNK TO FREE");
      allocChunkPtrPtr = NULL;
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

AllocatedChunk *RegionCache::tryGetAddress( global_reg_t const &reg, WD const &wd ) {
   ChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   global_reg_t allocatedRegion;
   allocatedRegion.key = reg.key;

   std::size_t allocSize = 0;
   uint64_t targetHostAddr = 0;

   if ( _flags == ALLOC_WIDE ) {
      allocatedRegion.id = 1;
   } else if ( _flags == ALLOC_FIT ) {
      allocatedRegion.id = reg.getFitRegionId();
   } else {
      std::cerr <<"RegionCache ERROR: Undefined _flags value."<<std::endl;
   }

   targetHostAddr = allocatedRegion.getFirstAddress();
   allocSize      = allocatedRegion.getDataSize();

   _chunks.getOrAddChunk2( targetHostAddr, allocSize, results );
   if ( results.size() != 1 ) {
      message0( "Got results.size()="<< results.size() << " I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ChunkList::iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
   } else {
      if ( *(results.front().second) == NULL ) {

         void *deviceMem = _device.memAllocate( allocSize, sys.getSeparateMemory( _memorySpaceId ) );
         if ( deviceMem != NULL ) {
            *(results.front().second) = NEW AllocatedChunk( *this, (uint64_t) deviceMem, results.front().first->getAddress(), results.front().first->getLength(), allocatedRegion );
            allocChunkPtr = *(results.front().second);
            //*(results.front().second) = allocChunkPtr;
         } else {
            // I have not been able to allocate a chunk, just return NULL;
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
   if ( allocChunkPtr != NULL ) {
     //std::cerr << __FUNCTION__ << " returns dev address " << (void *) allocChunkPtr->getAddress() << std::endl;
     allocChunkPtr->lock();
     allocChunkPtr->addReference();
   }
   return allocChunkPtr;
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
            //reg.key->invalLock();
            allocChunkPtrPtr = selectChunkToInvalidate( allocSize );
            if ( allocChunkPtrPtr != NULL ) {
               allocChunkPtr = *allocChunkPtrPtr;
               allocChunkPtr->invalidate( this, wd );
               _invalidationCount++;
               allocChunkPtr->increaseLruStamp();
               allocChunkPtr->clearNewRegions( allocatedRegion );
               allocChunkPtr->setHostAddress( results.front().first->getAddress() );
               *(results.front().second) = allocChunkPtr;
               *allocChunkPtrPtr = NULL;
            }
            //reg.key->invalUnlock();
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
     if ( !allocChunkPtr->locked() ) 
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
   return _getAllocatedChunk( reg, true, true );
}

AllocatedChunk *RegionCache::getAllocatedChunk( global_reg_t const &reg, bool complain ) {
   _lock.acquire();
   AllocatedChunk *chunk = _getAllocatedChunk( reg, complain, true );
   _lock.release();
   return chunk;
}

AllocatedChunk *RegionCache::_getAllocatedChunk( global_reg_t const &reg, bool complain, bool lockChunk ) const {
   ConstChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   _chunks.getChunk3( reg.getFirstAddress(), reg.getBreadth(), results );
   if ( results.size() == 1 ) {
      if ( results.front().second )
         allocChunkPtr = *(results.front().second);
      else
         allocChunkPtr = NULL;
   } else if ( results.size() > 1 ) {
         std::cerr <<"Requested addr " << (void *) reg.getFirstAddress() << " size " << reg.getBreadth() << std::endl;
      message0( "I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ConstChunkList::const_iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
   }
   if ( !allocChunkPtr && complain ) {
      sys.printBt(); std::cerr << "Error, null region "; reg.key->printRegion( reg.id ); std::cerr << std::endl;
      ensure(allocChunkPtr != NULL, "Chunk not found!");
   }
   if ( allocChunkPtr && lockChunk ) {
      //std::cerr << "AllocChunkPtr is " << allocChunkPtr << std::endl;
      allocChunkPtr->lock(); 
   }
   return allocChunkPtr;
}

void RegionCache::NEWcopyIn( unsigned int srcLocation, global_reg_t const &reg, unsigned int version, WD const &wd, DeviceOps *givenOps ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   uint64_t origDevAddr = chunk->getAddress() + ( reg.getFirstAddress() - chunk->getHostAddress() );
   DeviceOps *ops = ( givenOps != NULL ) ? givenOps : chunk->getDeviceOps( reg );
   chunk->unlock();
   //std::cerr << " COPY REGION ID " << reg.id << " OPS " << (void*)ops << std::endl;
   copyIn( reg, origDevAddr, srcLocation, ops, NULL, wd );
}

void RegionCache::NEWcopyOut( global_reg_t const &reg, unsigned int version, WD const &wd, DeviceOps *givenOps ) {
   AllocatedChunk *origChunk = getAllocatedChunk( reg );
   uint64_t origDevAddr = origChunk->getAddress() + ( reg.getFirstAddress() - origChunk->getHostAddress() );
   DeviceOps *ops = ( givenOps != NULL ) ? givenOps : reg.getDeviceOps();
   //origChunk->clearDirty( reg );
   origChunk->NEWaddWriteRegion( reg.id, version );// this is needed in case we are copying out a fragment of a region
   origChunk->unlock();
   CompleteOpFunctor *f = NEW CompleteOpFunctor( ops, origChunk );
   copyOut( reg, origDevAddr, ops, f, wd );
}

RegionCache::RegionCache( memory_space_id_t memSpaceId, Device &cacheArch, enum CacheOptions flags ) : _chunks(), _lock(), _device( cacheArch ), _memorySpaceId( memSpaceId ),
    _flags( flags ), _lruTime( 0 ), _invalidationCount( 0 ), _copyInObj( *this ), _copyOutObj( *this ) {
}

unsigned int RegionCache::getMemorySpaceId() const {
   return _memorySpaceId;
}

void RegionCache::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
#ifdef VERBOSE_DEV_OPS
   std::cerr << "_device._copyIn( copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, _pe, ops, wd="<< wd.getId() <<" );" <<std::endl;
#endif
   if (!fake) _device._copyIn( devAddr, hostAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, (CompleteOpFunctor *) NULL, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
#ifdef VERBOSE_DEV_OPS
   std::cerr << "_device._copyInStrided1D( copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<" ["<< *((double*) hostAddr) <<"]"<<", devAddr="<< (void*)devAddr <<", len, numChunks, ld, _pe, ops, wd="<< wd.getId() <<" );" <<std::endl;
#endif
   if (!fake) _device._copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, (CompleteOpFunctor *) NULL, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   ensure( f != NULL, " Error, functor received is null.");
#ifdef VERBOSE_DEV_OPS
   std::cerr << "_device._copyOut( copyFrom=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, _pe, ops, wd="<< (&wd != NULL ? wd.getId() : -1 ) <<" );" <<std::endl;
#endif
   if (!fake) _device._copyOut( hostAddr, devAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, f, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld,  DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f != NULL, " Error, functor received is null.");
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
#ifdef VERBOSE_DEV_OPS
   std::cerr << "_device._copyOutStrided1D( copyFrom=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, numChunks, ld, _pe, ops, wd="<< (&wd != NULL ? wd.getId() : -1 ) <<" );" <<std::endl;
#endif
   if (!fake) _device._copyOutStrided1D( hostAddr, devAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, f, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getSeparateMemory( syncFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   sys.getSeparateMemory( syncFrom ).getCache()._copyOut( hostAddr, origDevAddr, len, cout, fsource, wd, fake );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   delete cout;
   this->_copyIn( devAddr, hostAddr, len, ops, (CompleteOpFunctor *) NULL, wd, fake );
}

void RegionCache::_syncAndCopyInStrided1D( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getSeparateMemory( syncFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   sys.getSeparateMemory( syncFrom ).getCache()._copyOutStrided1D( hostAddr, origDevAddr, len, numChunks, ld, cout, fsource, wd, fake );
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
#ifdef VERBOSE_DEV_OPS
   std::cerr << "_device._copyDevToDev( copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len, _pe, sys.getSeparateMemory( copyFrom="<< copyFrom<<" ), ops, wd="<< wd.getId() << ", f="<< f <<" );" <<std::endl;
#endif
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
#ifdef VERBOSE_DEV_OPS
   std::cerr << "_device._copyDevToDevStrided1D( copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len, _pe, sys.getCaches()[ copyFrom="<< copyFrom<<" ]->_pe, ops, wd="<< wd.getId() <<", f="<< f <<" );"<<std::endl;
#endif
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

void RegionCache::doOp( Op *opObj, global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, CompleteOpFunctor *functor, WD const &wd ) {

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
   LocalFunction local( opObj, region, hostMem.getNumDimensions(), dimIdx, numChunks, contiguousChunkSize, location, ops, wd, devBaseAddr, hostMem.getFirstAddress(), functor /* hostMem.key->getBaseAddress()*/ );
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

//unsigned int RegionCache::getVersionSetVersion( global_reg_t const &reg, unsigned int newVersion ) {
//   AllocatedChunk *chunk = getAllocatedChunk( reg );
//   unsigned int version = chunk->getVersionSetVersion( reg, newVersion );
//   chunk->unlock();
//   return version;
//}

unsigned int RegionCache::getVersion( global_reg_t const &reg ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   unsigned int version = chunk->getVersion( reg );
   chunk->unlock();
   return version;
}

void RegionCache::releaseRegion( global_reg_t const &reg, WD const &wd ) {
   AllocatedChunk *chunk = _getAllocatedChunk( reg, true, false );
   chunk->removeReference();
   //chunk->unlock();
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

void RegionCache::prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   _lock.acquire();
   //attempt to allocate regions without triggering invalidations, this will reserve any chunk used by this WD
   for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
      memCopies[ idx ]._chunk = tryGetAddress( memCopies[ idx ]._reg, wd );
      if ( memCopies[ idx ]._chunk != NULL ) {
         memCopies[ idx ]._chunk->unlock();
      }
   }
   for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
      if ( memCopies[ idx ]._chunk == NULL ) {
         CacheRegionDictionary *newRegsToInvalidate = NULL;
         memCopies[ idx ]._chunk = getAddress( memCopies[ idx ]._reg, newRegsToInvalidate, wd );
         memCopies[ idx ]._chunk->unlock();
      }
   }
   _lock.release();
}

void RegionCache::prepareRegionsToCopyToHost( std::set< global_reg_t > const &regs, unsigned int version, std::set< AllocatedChunk * > &chunks  ) {
   _lock.acquire();
   for ( std::set< global_reg_t >::iterator it = regs.begin(); it != regs.end(); it++ ) {
      AllocatedChunk *chunk = _getAllocatedChunk( *it, false, false );
      if ( VERBOSE_CACHE ) { std::cerr << " reg " << it->id << " got chunk " << chunk << std::endl; }
      if ( chunk != NULL ) {
         if ( chunks.count( chunk ) == 0 ) {
            chunk->lock();
            chunks.insert( chunk );
         }
      }
   }
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
   if ( chunk->NEWaddReadRegion2( ops, reg.id, version, notPresentParts, output, locations ) ) {
   }
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

std::size_t RegionCache::getAllocatableSize( global_reg_t const &reg ) const {
   global_reg_t allocated_region;
   allocated_region.key = reg.key;
   if ( _flags == ALLOC_WIDE ) {
      allocated_region.id = 1;
   } else if ( _flags == ALLOC_FIT ) {
      allocated_region.id = reg.getFitRegionId();
   } else {
      std::cerr <<"RegionCache ERROR: Undefined _flags value."<<std::endl;
   }
   return allocated_region.getDataSize();
}

bool RegionCache::canAllocateMemory( MemCacheCopy *memCopies, unsigned int numCopies, bool considerInvalidations ) {
   bool result = true;
   bool *present_regions = (bool *) alloca( numCopies * sizeof(bool) );
   std::size_t *sizes = (std::size_t *) alloca( numCopies * sizeof(std::size_t) );
   unsigned int needed_chunks = 0;
   _lock.acquire();
   
   /* check if the desired region is already allocated */
   for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
      AllocatedChunk *chunk = _getAllocatedChunk( memCopies[ idx ]._reg , false, false );
      if ( chunk != NULL ) {
         present_regions[ idx ] = true;
         sizes[ idx ] = 0;
         //chunk->unlock();
      } else {
         present_regions[ idx ] = false;
         sizes[ needed_chunks ] = getAllocatableSize( memCopies[ idx ]._reg );
         needed_chunks += 1;
      }
   }

   _lock.release();
   //std::cerr << __FUNCTION__ << " needed chunks is " << needed_chunks << std::endl;

   if ( needed_chunks != 0 ) {
      std::size_t *remaining_sizes = (std::size_t *) alloca( needed_chunks * sizeof(std::size_t) );
      /* compute if missing chunks can be allocated in the device memory */
      _device._canAllocate( sys.getSeparateMemory( _memorySpaceId ), sizes, needed_chunks, remaining_sizes );

      unsigned int remaining_count;
      for ( remaining_count = 0; remaining_count < needed_chunks && remaining_sizes[ remaining_count ] != 0; remaining_count +=1 );

      if ( remaining_count > 0 ) {
         /* check if data can be invalidated in order to allocate the memory */
         if ( considerInvalidations ) {
            result = canInvalidateToFit( remaining_sizes, remaining_count );
         } else {
            result = false;
         }
      }
   }

   return result;
}

bool RegionCache::canInvalidateToFit( std::size_t *sizes, unsigned int numChunks ) const {
   unsigned int allocated_count = 0;
   bool *allocated = (bool *) alloca( numChunks * sizeof(bool) );
   for (unsigned int idx = 0; idx < numChunks; idx += 1) {
      allocated[ idx ] = false;
   }

   MemoryMap<AllocatedChunk>::const_iterator it;
   //int count =0;
   for ( it = _chunks.begin(); it != _chunks.end() && ( allocated_count < numChunks ); it++ ) {
      // if ( it->second != NULL ) {
      //    global_reg_t thisreg = it->second->getAllocatedRegion();
      //    std::cerr << "["<< count++ << "] mmm this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << " size is " << it->second->getSize() << " ";
      //    thisreg.key->printRegion( thisreg.id );
      //    std::cerr << std::endl;
      // }
      if ( it->second != NULL && it->second->getReferenceCount() == 0 ) {
         for ( unsigned int idx = 0; idx < numChunks && ( allocated_count < numChunks ); idx += 1 ) {
            if ( !allocated[ idx ] && it->second->getSize() == sizes[ idx ] ) {
               allocated[ idx ] = true;
               allocated_count += 1;
            }
         }
      }
   }
   
   return ( allocated_count == numChunks );
}

