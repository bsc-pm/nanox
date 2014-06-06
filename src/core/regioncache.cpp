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

#define VERBOSE_DEV_OPS ( sys.getVerboseDevOps() )
#define VERBOSE_INVAL 0

#if VERBOSE_CACHE
 #define _VERBOSE_CACHE 1
#else
 #define _VERBOSE_CACHE 0
#endif

AllocatedChunk::AllocatedChunk( RegionCache &owner, uint64_t addr, uint64_t hostAddress, std::size_t size, global_reg_t const &allocatedRegion, bool rooted ) :
   _owner( owner ),
   _lock(),
   _address( addr ),
   _hostAddress( hostAddress ),
   _size( size ),
   _dirty( false ),
   _rooted( rooted ),
   _lruStamp( 0 ),
   _roBytes( 0 ),
   _rwBytes( 0 ),
   _refs( 0 ),
   _refWdId(),
   _refLoc(),
   _allocatedRegion( allocatedRegion ) {
      //std::cerr << "region " << allocatedRegion.id << " addr " << (void *) addr<<" hostAddr is " << (void*)hostAddress << " key " << allocatedRegion.key << std::endl;
      _newRegions = NEW CacheRegionDictionary( *(allocatedRegion.key) );
}

AllocatedChunk::~AllocatedChunk() {
   //std::cerr << "Im being released! "<< (void *) this << std::endl;
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

bool AllocatedChunk::trylock() {
   bool res = _lock.tryAcquire();
   //std::cerr << "x " << myThread->getId() <<" x Locked chunk " << (void *) this << std::endl;
   return res;
}

void AllocatedChunk::lock( bool setVerbose ) {
   //std::cerr << ": " << myThread->getId() <<" : Locked chunk " << (void *) this << std::endl;
   //_lock.acquire();
   while ( !_lock.tryAcquire() ) {
      myThread->idle();
   }
   //sys.printBt();
   //std::cerr << "x " << myThread->getId() <<" x Locked chunk " << (void *) this << std::endl;
}

void AllocatedChunk::unlock( bool unsetVerbose ) {
   //sys.printBt();
   //std::cerr << "x " << myThread->getId() << " x Unlocked chunk " << (void *) this << std::endl;
   _lock.release();
   //std::cerr << ": " << myThread->getId() << " : Unlocked chunk " << (void *) this << std::endl;
}

bool AllocatedChunk::locked() const {
   return _lock.getState() != NANOS_LOCK_FREE;
}

void AllocatedChunk::copyRegionToHost( SeparateAddressSpaceOutOps &ops, reg_t reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   NewNewRegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();
   global_reg_t greg( reg, key );
   DeviceOps * dops = greg.getDeviceOps();
   if ( dops->addCacheOp( &wd, 8 ) ) {
      ops.insertOwnOp( dops, greg, version, 0 );
   } else {
      ops.getOtherOps().insert( dops );
   }

}

bool AllocatedChunk::NEWaddReadRegion2( BaseAddressSpaceInOps &ops, reg_t reg, unsigned int version, std::set< reg_t > &notPresentRegions, bool output, NewLocationInfoList const &locations, WD const &wd, unsigned int copyIdx ) {
   unsigned int currentVersion = 0;
   bool opEmitted = false;
   std::list< std::pair< reg_t, reg_t > > components;

   std::ostream &o = *(myThread->_file);

//if ( sys.getNetwork()->getNodeNum() > 0 ) {
//   std::cerr << __FUNCTION__ << " reg " << reg << std::endl;
//}
   CachedRegionStatus *thisRegEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !thisRegEntry ) {
//if ( sys.getNetwork()->getNodeNum() > 0 ) {
//   std::cerr << __FUNCTION__ << " thisEntry is null " << reg << std::endl;
//}
      thisRegEntry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, thisRegEntry );
   } else {
//if ( sys.getNetwork()->getNodeNum() > 0 ) {
//   std::cerr << __FUNCTION__ << " thisEntry is not null " << reg << std::endl;
//}
   }

   DeviceOps *thisEntryOps = thisRegEntry->getDeviceOps();
   if ( thisEntryOps->addCacheOp( /* debug: */ &wd, 1 ) ) {
      opEmitted = true;

      //std::cerr << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ " << __FUNCTION__ << " " << (void*) this << " reg " << reg << " set rversion "<< version << " ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]] This chunk key: " << (void *) _newRegions->getGlobalDirectoryKey()<< std::endl;
      // lock / free needed for multithreading on the same cache.
      _newRegions->registerRegion( reg, components, currentVersion );
      NewNewRegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();

      //for ( CacheRegionDictionaryIterator it = _newRegions->begin(); it != _newRegions->end(); it++) {
      //   o << "Region: " << it->first << " "; _newRegions->printRegion( o, it->first ); o << " has entry with version " << (( (it->second).getData() ) ? (it->second).getData()->getVersion() : -1)<< std::endl;
      //}

      //o << "Asked for region " << reg << " got: " << std::endl;
      //for ( std::list< std::pair< reg_t, reg_t > >::const_iterator it = components.begin(); it != components.end(); it++ ) {
      //   o << "component: " << it->first << ", " << it->second << std::endl;
      //}

      if ( components.size() == 1 ) {
         ensure( components.begin()->first == reg, "Error, wrong region");
      }

      for ( std::list< std::pair< reg_t, reg_t > >::iterator it = components.begin(); it != components.end(); it++ )
      {
         CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->first );
         if ( ( !entry || version > entry->getVersion() ) ) {
            //o << "No entry for region " << it->first << " "; _newRegions->printRegion( o, it->first); o << " must copy from region " << it->second << " "; _newRegions->printRegion(o, it->second); o << " want version "<< version << " entry version is " << ( (!entry) ? -1 : entry->getVersion() )<< std::endl;
            CachedRegionStatus *copyFromEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->second );
            if ( !copyFromEntry || version > copyFromEntry->getVersion() ) {
               //o << "I HAVE TO COPY: I dont have this region" << std::endl;

               global_reg_t chunkReg( it->first, key );

               NewLocationInfoList::const_iterator locIt;
               for ( locIt = locations.begin(); locIt != locations.end(); locIt++ ) {
                  global_reg_t locReg( locIt->first, key );

                  //o << "+ Location ( " << locIt->first << ", " << locIt->second << " ) This reg " << chunkReg.id << " :: ";
                  //if ( locReg.id != chunkReg.id ) {
                  //   if ( chunkReg.contains( locReg ) ) {
                  //      o << " " << it->first << " contains(1) " << locIt->first << std::endl;
                  //   } else if ( locReg.contains( chunkReg ) ) {
                  //      o << " " << locIt->first << " contains(2) " << it->first << std::endl;
                  //   } else if ( locReg.key->checkIntersect( locReg.id, chunkReg.id ) ) {
                  //   o << " intersect! " << std::endl;
                  //   } else {
                  //   o << " unrelated " << std::endl;
                  //   }
                  //} else {
                  //   o << " same " << std::endl;
                  //}

                  //if ( locIt->first == it->first || chunkReg.contains( locReg ) ) {
                  if ( locReg.id == chunkReg.id || locReg.key->checkIntersect( locReg.id, chunkReg.id ) ) {

                     reg_t target = 0;
                     if ( locReg.id == chunkReg.id ) {
                        target = locReg.id;
                     } else {
                        target = locReg.key->computeIntersect( locReg.id, chunkReg.id );
                     }

                     global_reg_t region_shape( target , key );
                     global_reg_t data_source( locIt->second, key );

                     if ( reg != region_shape.id || ( reg == region_shape.id && _newRegions->getRegionData( region_shape.id ) == NULL ) ) {
                        prepareRegion( region_shape.id, version );
                     }
                     //o << "shape: "<< it->first << " data source: " << it->second << std::endl;
                     //o <<" CHECKING THIS SHIT ID " << data_source.id << std::endl;
                     memory_space_id_t location = data_source.getFirstLocation();
                     if ( location == 0 || location != _owner.getMemorySpaceId() ) {
                        //o << "add copy from host, reg " << region_shape.id << " version " << ops.getVersionNoLock( data_source, wd, copyIdx ) << std::endl;
                        if ( _VERBOSE_CACHE ) {
                           NewNewDirectoryEntryData *dentry = NewNewRegionDirectory::getDirectoryEntry( *(data_source.key), data_source.id );
                           NewNewDirectoryEntryData *dentry2 = NewNewRegionDirectory::getDirectoryEntry( *(data_source.key), region_shape.id );
                           if ( dentry2 && dentry ) o << "I have to copy region " << region_shape.id << " from location " << location << " (data_source is " << data_source.id << ")" << *dentry << " region_shape: "<< *dentry2<< std::endl;
                        }
                        CachedRegionStatus *entryToCopy = ( CachedRegionStatus * ) _newRegions->getRegionData( region_shape.id );
                        DeviceOps *entryToCopyOps = entryToCopy->getDeviceOps();
                        if ( entryToCopy != thisRegEntry ) {
                           if ( !entryToCopyOps->addCacheOp( /* debug: */ &wd, 2 ) ) {
                              std::cerr << "ERROR " << __FUNCTION__ << std::endl;
                           }
                      //FIXME: this now updates the metadata of reg: which is redundant but updating the metadata of region_shape it's not possible since it could not have a directory entry, and its not right to update the source region since we may be copying just a piece.
                           ops.insertOwnOp( entryToCopyOps, global_reg_t( reg, _newRegions->getGlobalDirectoryKey() ), version + (output ? 1 : 0), _owner.getMemorySpaceId() ); 
                           //ops.insertOwnOp( entryToCopyOps, global_reg_t( locIt->first, _newRegions->getGlobalDirectoryKey() ), version, _owner.getMemorySpaceId() );
                        }

                        if ( location == 0 ) {
                           ops.addOpFromHost( region_shape, version, this, copyIdx );
                        } else if ( location != _owner.getMemorySpaceId() ) {
                           ops.addOp( &sys.getSeparateMemory( location ) , region_shape, version, this, copyIdx );
                        }
                     }// else {
                      //  o << "Ooops! no copy!" << std::endl;
                      // }
                  }
               }
            } else { // I have this region, as part of other region
               //o << "NO NEED TO COPY: I have this region as a part of region " << it->second << std::endl;
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
            std::cerr << "ERROR: version in cache (" << entry->getVersion() << ") > than version requested ("<< version <<"). WD id: "<< wd.getId() << " desc: " << (wd.getDescription() ? wd.getDescription() : "n/a") << std::endl;
            key->printRegion( std::cerr, reg );
            std::cerr << std::endl;
         }
      }
      thisRegEntry->setVersion( version + ( output ? 1 : 0) );
      ops.insertOwnOp( thisEntryOps, global_reg_t( reg, _newRegions->getGlobalDirectoryKey() ), version + (output ? 1 : 0), _owner.getMemorySpaceId() );
      //ops.insertOwnOp( thisEntryOps, global_reg_t( reg, _newRegions->getGlobalDirectoryKey() ), version, _owner.getMemorySpaceId() );
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
   if ( _VERBOSE_CACHE ) { std::cerr << "[[[[[[[[[[[[[[[[[[[[ " << __FUNCTION__ << " reg " << reg << " set version " << version << " entry " << (void *)entry << " components size " << components.size() <<" ]]]]]]]]]]]]]]]]]]]]"<< std::endl; }

   _dirty = true;
}

Atomic<int> AllocatedChunk::numCall(0);
bool AllocatedChunk::invalidate( RegionCache *targetCache, WD const &wd, unsigned int copyIdx, SeparateAddressSpaceOutOps &invalOps, std::set< global_reg_t > &regionsToRemoveAccess, std::set< NewNewRegionDirectory::RegionDirectoryKey > &alreadyLockedObjects ) {
   bool hard=false;
   NewNewRegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();

   std::set< NewNewRegionDirectory::RegionDirectoryKey >::iterator dict_it = alreadyLockedObjects.find( key );
   if ( dict_it == alreadyLockedObjects.end() ) {
      //not present, lock and insert
      key->lock();
      alreadyLockedObjects.insert( key );
   } //else no need to lock

   std::list< std::pair< reg_t, reg_t > > missing;
   unsigned int ver = 0;
   _allocatedRegion.key->registerRegionReturnSameVersionSubparts( _allocatedRegion.id, missing, ver );
   //_newRegions->registerRegionReturnSameVersionSubparts( _allocatedRegion.id, missing, ver );

   //std::set<DeviceOps *> ops;
   //ops.insert( _allocatedRegion.getDeviceOps() );
   //for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missing.begin(); lit != missing.end(); lit++ ) {
   //   global_reg_t data_source( lit->second, key );
   //   ops.insert( data_source.getDeviceOps() );
   //}

   //for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
   //   (*opIt)->syncAndDisableInvalidations();
   //}

   DeviceOps *thisChunkOps = _allocatedRegion.getDeviceOps();
   CachedRegionStatus *alloc_entry = ( CachedRegionStatus * ) _newRegions->getRegionData( _allocatedRegion.id );
   //bool alloc_entry_not_present = false;

   if (alloc_entry != NULL) {
      regionsToRemoveAccess.insert( _allocatedRegion );
   } /*else {
      alloc_entry_not_present = true;
   }*/

   //std::cerr << "Missing pieces are: " << missing.size() << std::endl;

   if ( missing.size() == 1 ) {
      ensure( _allocatedRegion.id == missing.begin()->first, "Wrong region." );
      if ( _allocatedRegion.isLocatedIn( _owner.getMemorySpaceId() ) ) {
     //    regionsToRemoveAccess.insert( _allocatedRegion );
         if ( NewNewRegionDirectory::isOnlyLocated( _allocatedRegion.key, _allocatedRegion.id, _owner.getMemorySpaceId() ) ) {
            //std::cerr << "AC: has to be copied!, shape = dsrc and Im the only owner!" << std::endl;
            hard = true;
            if ( thisChunkOps->addCacheOp( /* debug: */ &wd, 3 ) ) {
               invalOps.insertOwnOp( thisChunkOps, _allocatedRegion, alloc_entry->getVersion(), 0 );
               invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), _allocatedRegion, alloc_entry->getVersion(), NULL, this, wd, copyIdx );
               alloc_entry->resetVersion();
            } else {
               std::cerr << " ERROR: could not add a cache op to my ops!"<<std::endl;
            }
         }
      }
   } else {
      std::map< reg_t, std::set< reg_t > > fragmented_regions;
      for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missing.begin(); lit != missing.end(); lit++ ) {
         NewNewDirectoryEntryData *dentry = NewNewRegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), lit->first );
         if ( VERBOSE_INVAL ) {
            NewNewDirectoryEntryData *dsentry = NewNewRegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), lit->second );
            std::cerr << (void *)_newRegions << " missing registerReg: " << lit->first << " "; _allocatedRegion.key->printRegion( std::cerr, lit->first ); if (!dentry ) { std::cerr << " nul "; } else { std::cerr << *dentry; } 
            std::cerr << "," << lit->second << " "; _allocatedRegion.key->printRegion( std::cerr, lit->second ); if (!dsentry ) { std::cerr << " nul "; } else { std::cerr << *dsentry; }
            std::cerr <<  std::endl;
         }
         global_reg_t region_shape( lit->first, key );
         global_reg_t data_source( lit->second, key );
         if ( region_shape.id == data_source.id ) {
            ensure( _allocatedRegion.id != data_source.id, "Wrong region" );
            if ( data_source.isLocatedIn( _owner.getMemorySpaceId() ) ) {
               regionsToRemoveAccess.insert( data_source );
               //if ( NewNewRegionDirectory::isOnlyLocated( data_source.key, data_source.id, _owner.getMemorySpaceId() ) )
               if ( ! data_source.isLocatedIn( 0 ) ) { // FIXME: not optimal, but we write metadata to "allocatedRegion" entry so we must copy all to node 0 if its not there to keep it consistent!
                  if ( VERBOSE_INVAL ) {
                     for ( CacheRegionDictionary::citerator pit = _newRegions->begin(); pit != _newRegions->end(); pit++ ) {
                        NewNewDirectoryEntryData *d = NewNewRegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), pit->first );
                        CachedRegionStatus *c = ( CachedRegionStatus * ) _newRegions->getRegionData( pit->first );
                        std::cerr << " reg " << pit->first << " "; key->printRegion( std::cerr, pit->first); std::cerr << " has entry " << (void *) &pit->second << " CaheVersion: "<< (int)( c!=NULL ? c->getVersion() : -1) ;
                        if ( d ) std::cerr << *d << std::endl;
                        else std::cerr << " n/a " << std::endl;
                     }
                  }
                  DeviceOps *fragment_ops = data_source.getDeviceOps();
                  CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( data_source.id );
                  if ( VERBOSE_INVAL ) { std::cerr << data_source.id << " has to be copied!, shape = dsrc and Im the only owner! "<< (void *)entry << std::endl; }
                  unsigned int version;
                  if ( entry ) {
                     version = entry->getVersion();
                     entry->resetVersion();
                  } else {
                     version = NewNewRegionDirectory::getVersion( data_source.key, data_source.id, false );
                  }
                  hard = true;
                  if ( fragment_ops->addCacheOp( /* debug: */ &wd, 4 ) ) {
                     invalOps.insertOwnOp( fragment_ops, data_source, version, 0 );
                     invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), data_source, version, NULL, this, wd, copyIdx );
                  } else {
                     invalOps.getOtherOps().insert( fragment_ops );
                     // make sure the op we are waiting for its the same that we want to do, maybe it is impossible to reach this code
                     std::cerr << "FIXME " << __FUNCTION__ << " this is memspace "<< _owner.getMemorySpaceId() << std::endl;
                  }
               }
            }
         } else {
            CachedRegionStatus *c_ds_entry = ( CachedRegionStatus * ) _newRegions->getRegionData( data_source.id );
            if ( c_ds_entry != NULL && 
                  ( dentry == NULL ||
                    ( data_source.getVersion() <= region_shape.getVersion() && NewNewRegionDirectory::isOnlyLocated( region_shape.key, region_shape.id, _owner.getMemorySpaceId() ) ) ||
                    ( data_source.getVersion() >  region_shape.getVersion() && NewNewRegionDirectory::isOnlyLocated( data_source.key,  data_source.id,  _owner.getMemorySpaceId() ) )
                  )
               ) {
               fragmented_regions[ data_source.id ].insert( region_shape.id );
            }
         }
      }

      for ( std::map< reg_t, std::set< reg_t > >::iterator mit = fragmented_regions.begin(); mit != fragmented_regions.end(); mit++ ) {
         if ( VERBOSE_INVAL ) { std::cerr << " fragmented region " << mit->first << " has #chunks " << mit->second.size() << std::endl; }
         global_reg_t data_source( mit->first, key );
         regionsToRemoveAccess.insert( data_source );
         //if ( NewNewRegionDirectory::isOnlyLocated( key, data_source.id, _owner.getMemorySpaceId() ) )
         if ( ! data_source.isLocatedIn( 0 ) ) { // FIXME: not optimal, but we write metadata to "allocatedRegion" entry so we must copy all to node 0 if its not there to keep it consistent!
            bool subChunkInval = false;
            CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( data_source.id );
            if ( VERBOSE_INVAL ) { std::cerr << "data source is " << data_source.id << " with entry "<< entry << std::endl; }

            for ( std::set< reg_t >::iterator sit = mit->second.begin(); sit != mit->second.end(); sit++ ) {
               if ( VERBOSE_INVAL ) { std::cerr << "    this is region " << *sit << std::endl; }
               global_reg_t subReg( *sit, key );
               NewNewDirectoryEntryData *dentry = NewNewRegionDirectory::getDirectoryEntry( *key, *sit );
               if ( dentry == NULL ) { //FIXME: maybe we need a version check to handle when the dentry exists but is old?
                  std::list< std::pair< reg_t, reg_t > > missingSubReg;
                  //_allocatedRegion.key->registerRegion( subReg.id, missingSubReg, ver );
                  _allocatedRegion.key->registerRegionReturnSameVersionSubparts( subReg.id, missingSubReg, ver );
                  for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missingSubReg.begin(); lit != missingSubReg.end(); lit++ ) {
                     global_reg_t region_shape( lit->first, key );
                     global_reg_t new_data_source( lit->second, key );
                     if ( VERBOSE_INVAL ) { std::cerr << " DIR CHECK WITH FRAGMENT: "<< lit->first << " - " << lit->second << " " << std::endl; }

                     NewNewDirectoryEntryData *subEntry = NewNewRegionDirectory::getDirectoryEntry( *key, lit->first );
                     if ( !subEntry ) {
                        std::cerr << "FIXME: Invalidation, and found a region shape (" << lit->first << ") with no entry, a new Entry may be needed." << std::endl;
                     } else if ( VERBOSE_INVAL ) {
                        std::cerr << " Fragment " << lit->first << " has entry! " << subEntry << std::endl;
                     }
                     if ( new_data_source.id == data_source.id || NewNewRegionDirectory::isOnlyLocated( key, new_data_source.id, _owner.getMemorySpaceId() ) ) {
                        subChunkInval = true;
                        if ( VERBOSE_INVAL ) { std::cerr << " COPY subReg " << lit->first << " comes from subreg "<< subReg.id << " new DS " << new_data_source.id << std::endl; }
                        invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), region_shape, entry->getVersion(), thisChunkOps, this, wd, copyIdx );
                     }
                  }
               } else {
                  DeviceOps *subRegOps = subReg.getDeviceOps();
                  hard = true;
                  if ( subRegOps->addCacheOp( /* debug: */ &wd, 5 ) ) {
                     regionsToRemoveAccess.insert( subReg );
                     invalOps.insertOwnOp( subRegOps, data_source, entry->getVersion(), 0 );
                     invalOps.addOp( &sys.getSeparateMemory( _owner.getMemorySpaceId() ), subReg, entry->getVersion(), subRegOps, this, wd, copyIdx );
                  } else {
                     invalOps.getOtherOps().insert( subRegOps );
                     std::cerr << "FIXME " << __FUNCTION__ << std::endl;
                  }
               }
            }

            if ( subChunkInval ) {
               //FIXME I think this is wrong, can potentially affect regions that are not there, 
               hard = true;
               if ( thisChunkOps->addCacheOp( /* debug: */ &wd, 6 ) ) { // FIXME: others may believe there's an ongoing op for the full region!
                 invalOps.insertOwnOp( thisChunkOps, data_source, entry->getVersion(), 0 );
               } else {
                 std::cerr << "ERROR, could not add an inval cache op " << std::endl;
               }
            }
            entry->resetVersion();
         }
      }

      if ( alloc_entry != NULL ) {
         if ( _allocatedRegion.isLocatedIn( _owner.getMemorySpaceId() ) ) {
            //if ( NewNewRegionDirectory::isOnlyLocated( _allocatedRegion.key, _allocatedRegion.id, _owner.getMemorySpaceId() ) )
            if ( ! _allocatedRegion.isLocatedIn( 0 ) ) { // FIXME: not optimal, but we write metadata to "allocatedRegion" entry so we must copy all to node 0 if its not there to keep it consistent!
               hard = true;
               if ( thisChunkOps->addCacheOp( /* debug: */ &wd, 7 ) ) {
                  invalOps.insertOwnOp( thisChunkOps, _allocatedRegion, alloc_entry->getVersion(), 0 );
                  alloc_entry->resetVersion();
               }
            } else {
               std::cerr << " ERROR: could not add a cache op to my ops!"<<std::endl;
            }
         }
      }
   }

   //*(myThread->_file) << numCall++ << "=============> " << "Cache " << _owner.getMemorySpaceId() << ( hard ? " hard":" soft" ) <<" Invalidate region "<< (void*) key << ":" << _allocatedRegion.id << " reg: "; _allocatedRegion.key->printRegion(*(myThread->_file), _allocatedRegion.id ); *(myThread->_file) << std::endl;
   return hard;

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
      if ( _VERBOSE_CACHE ) { fprintf(stderr, "[%s] Thd %d Im cache with id %d, I've found a chunk to free, %p (locked? %d) region %d addr=%p size=%zu\n",  __FUNCTION__, myThread->getId(), _memorySpaceId, *allocChunkPtrPtr, ((*allocChunkPtrPtr)->locked()?1:0), (*allocChunkPtrPtr)->getAllocatedRegion().id, (void*)it->first.getAddress(), it->first.getLength()); }
      (*allocChunkPtrPtr)->lock();
   } else {
      // if ( VERBOSE_INVAL ) {
      //    count = 0;
      //    for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
      //       if ( it->second == NULL ) std::cerr << "["<< count << "] this chunk: null chunk" << std::endl;
      //       else std::cerr << "["<< count << "] this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << " size is " << it->second->getSize() << " vs " << allocSize << " " << " dirty? " << it->second->isDirty() << std::endl;
      //       count++;
      //    }
      // }
      //fatal("IVE _not_ FOUND A CHUNK TO FREE");
      allocChunkPtrPtr = NULL;
   }
   return allocChunkPtrPtr;
}

void RegionCache::selectChunksToInvalidate( std::size_t allocSize, std::set< AllocatedChunk ** > &chunksToInvalidate, WD const &wd, unsigned int &otherReferencedChunks ) {
   //for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
   //   std::cerr << "["<< count << "] this chunk: " << ((void *) it->second) << " refs: " << (int)( (it->second != NULL) ? it->second->getReferenceCount() : -1 ) << " dirty? " << (int)( (it->second != NULL) ? it->second->isDirty() : -1 )<< std::endl;
   //   count++;
   //}
   //count = 0;
   otherReferencedChunks = 0;
   if ( VERBOSE_INVAL ) {
      std::cerr << __FUNCTION__ << " with size " << allocSize << std::endl;
   }
   if ( /*_device.supportsFreeSpaceInfo() */ true ) {
      MemoryMap<AllocatedChunk>::iterator it;
      bool done = false;
      MemoryMap< uint64_t > device_mem;

      for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
         // if ( it->second != NULL ) {
         //    global_reg_t reg = it->second->getAllocatedRegion();
         //    std::cerr << "["<< count << "] mmm this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << " size is " << it->second->getSize() << " vs " << allocSize << " ";
         //    reg.key->printRegion( reg.id );
         //    std::cerr << std::endl;
         // }
         if ( it->second != NULL ) {
            AllocatedChunk &c = *(it->second);
            AllocatedChunk **chunk_at_map_ptr = &(it->second);
            if ( it->second->getReferenceCount() == 0 ) {
               device_mem.addChunk( c.getAddress(), c.getSize(), (uint64_t) chunk_at_map_ptr );
            } else {
               bool mine = false;
               for (unsigned int idx = 0; idx < wd.getNumCopies() && !mine ; idx += 1) {
                  mine = ( wd._mcontrol._memCacheCopies[ idx ]._chunk == &c );
               }
               otherReferencedChunks += mine ? 0 : 1;
            }
         }
      }
      
      /* add the device free chunks */
      SimpleAllocator::ChunkList free_device_chunks;
      _device._getFreeMemoryChunksList( sys.getSeparateMemory( _memorySpaceId ), free_device_chunks );
      for ( SimpleAllocator::ChunkList::iterator lit = free_device_chunks.begin(); lit != free_device_chunks.end(); lit++ ) {
         device_mem.addChunk( lit->first, lit->second, (uint64_t) 0 );
      }

      MemoryMap< uint64_t >::iterator devIt, devItAhead;
      if ( VERBOSE_INVAL ) {
         std::cerr << "I can invalidate a set of these:" << std::endl;
         for ( devIt = device_mem.begin(); devIt != device_mem.end(); devIt++ ) {
            std::cerr << "Addr: " << (void *) devIt->first.getAddress() << " size: " << devIt->first.getLength() ;
            if ( devIt->second == 0 ) {
               std::cerr << " [free chunk] "<< std::endl;
            } else {
               std::cerr << " " << (void *) *((AllocatedChunk **) devIt->second) << std::endl;
            }
         }
      }
      std::map< std::size_t, std::list< MemoryMap< uint64_t >::iterator > > candidates;

      for ( devIt = device_mem.begin(); devIt != device_mem.end(); devIt++ ) {
         std::size_t len = devIt->first.getLength();
         uint64_t addr = devIt->first.getAddress();
         std::size_t num_chunks = 0;
         if ( devIt->second != 0 ) {
            num_chunks += (*((AllocatedChunk **)(devIt->second)))->isDirty() ? devIt->first.getLength() : 0;
         }
         devItAhead = devIt;
         devItAhead++;
         bool fail = false;
         while ( len < allocSize && !fail && devItAhead != device_mem.end() ) {
            if ( addr + len == devItAhead->first.getAddress() ) {
               len += devItAhead->first.getLength();
               if ( devItAhead->second != 0 ) {
                  num_chunks += (*((AllocatedChunk **)(devItAhead->second)))->isDirty() ? devItAhead->first.getLength() : 0;
               }
               devItAhead++;
            } else {
               fail = true;
            }
         }
         if ( len >= allocSize && !fail ) {
            candidates[ num_chunks ].push_back( devIt );
         } 
      } 
      if ( !candidates.empty() ) {
         MemoryMap< uint64_t >::iterator selectedIt = candidates.begin()->second.front();
         AllocatedChunk **selected_chunk = (AllocatedChunk **) selectedIt->second;
         if ( VERBOSE_INVAL ) {
            std::cerr << "Im going to invalidaet from " << (void *) *selected_chunk << std::endl;
         }
         
         for ( std::size_t len = selectedIt->first.getLength(); len < allocSize; selectedIt++ ) {
            if ( selectedIt->second != 0 ) {
               chunksToInvalidate.insert( (AllocatedChunk **) selectedIt->second );
            }
            len += selectedIt->first.getLength();
         }
      }
   }
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

AllocatedChunk *RegionCache::tryGetAddress( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   ChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   global_reg_t allocatedRegion;

   std::size_t allocSize = 0;
   uint64_t targetHostAddr = 0;

   getAllocatableRegion( reg, allocatedRegion );

   targetHostAddr = allocatedRegion.getRealFirstAddress();
   allocSize      = allocatedRegion.getDataSize();

   _chunks.getOrAddChunk2( targetHostAddr, allocSize, results );
   if ( results.size() != 1 ) {
      message0( "Got results.size()="<< results.size() << " for addr " << ((void*) targetHostAddr) << " with allocSize " << allocSize <<" I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ChunkList::iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
      if ( &wd != NULL ) {
         std::cerr << "Realloc needed. Caused by wd " << (wd.getDescription() ? wd.getDescription() : "n/a") << " copy index " << copyIdx << std::endl;
      } else {
         std::cerr << "Realloc needed. Unknown WD, probably comes from a taskwait or any other synchronization point." << std::endl;
      }
      fatal("Can not continue.");
   } else {
      if ( *(results.front().second) == NULL ) {

         void *deviceMem = _device.memAllocate( allocSize, sys.getSeparateMemory( _memorySpaceId ), targetHostAddr );
         if ( deviceMem != NULL ) {
            *(results.front().second) = NEW AllocatedChunk( *this, (uint64_t) deviceMem, results.front().first->getAddress(), results.front().first->getLength(), allocatedRegion, reg.isRooted() );
            allocChunkPtr = *(results.front().second);
            if ( reg.isRooted() ) {
               allocChunkPtr->addReference(wd.getId(), 2);
            }
            //*(results.front().second) = allocChunkPtr;
         } else {
            // I have not been able to allocate a chunk, just return NULL;
         }
      } else {
         //std::cerr << " CHUNK 1 AND NOT NULL! asekd for "<< (void *)targetHostAddr << " with size " << (unsigned int) allocSize << " got addr " << (void *) results.front().first->getAddress() << " with size " << (unsigned int) results.front().first->getLength() << " entry is " << (void *) *(results.front().second)<< std::endl;
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
      //if ( !allocChunkPtr->locked() ) {
      //   allocChunkPtr->lock();
      //   allocChunkPtr->addReference();
      //} else {
      //   allocChunkPtr = NULL;
      //}
      if ( allocChunkPtr->trylock() ) {
         allocChunkPtr->addReference( wd.getId() , 4);
      } else {
         allocChunkPtr = NULL;
      }
   }
   return allocChunkPtr;
}

AllocatedChunk *RegionCache::invalidate( global_reg_t const &allocatedRegion, WD const &wd, unsigned int copyIdx ) {
   AllocatedChunk *allocChunkPtr = NULL;
   AllocatedChunk **allocChunkPtrPtr = NULL;
   SeparateAddressSpaceOutOps inval_ops( true, true );
   std::set< global_reg_t > regions_to_remove_access;
   std::set< NewNewRegionDirectory::RegionDirectoryKey > locked_objects;

   //reg.key->invalLock();
   std::set< AllocatedChunk ** > chunks_to_invalidate;

   allocChunkPtrPtr = selectChunkToInvalidate( allocatedRegion.getDataSize() );
   if ( allocChunkPtrPtr != NULL ) {
      chunks_to_invalidate.insert( allocChunkPtrPtr );
      allocChunkPtr = *allocChunkPtrPtr;
      if ( allocChunkPtr->invalidate( this, wd, copyIdx, inval_ops, regions_to_remove_access, locked_objects ) ) {
         _hardInvalidationCount++;
      } else {
         _softInvalidationCount++;
      }
   } else {
      //try to invalidate a set of chunks
      unsigned int other_referenced_chunks = 0;
      selectChunksToInvalidate( allocatedRegion.getDataSize(), chunks_to_invalidate, wd, other_referenced_chunks );
      if ( chunks_to_invalidate.empty() ) {
         if ( other_referenced_chunks == 0 ) {
         fatal("Unable to free enough space to allocate task data, probably a fragmentation issue. Try increasing the available device memory.");
         } else {
            *(myThread->_file) << "Unable to invalidate using selectChunksToInvalidate, wd: " << wd.getId() <<std::endl;
            printReferencedChunksAndWDs();
            return NULL;
         }
      }
      for ( std::set< AllocatedChunk ** >::iterator it = chunks_to_invalidate.begin(); it != chunks_to_invalidate.end(); it++ ) {
         AllocatedChunk **chunkPtr = *it;
         AllocatedChunk *chunk = *chunkPtr;
         if ( chunk->invalidate( this, wd, copyIdx, inval_ops, regions_to_remove_access, locked_objects ) ) {
            _hardInvalidationCount++;
         } else {
            _softInvalidationCount++;
         }
         _device.memFree( chunk->getAddress(), sys.getSeparateMemory( _memorySpaceId ) );
      }
   }

   inval_ops.issue( wd );
   while ( !inval_ops.isDataReady( wd, true ) ) { myThread->idle(); }
   if ( _VERBOSE_CACHE ) { std::cerr << "===> Invalidation complete at " << _memorySpaceId << " remove access for regs: "; }
   for ( std::set< global_reg_t >::iterator it = regions_to_remove_access.begin(); it != regions_to_remove_access.end(); it++ ) {
   if ( _VERBOSE_CACHE ) { std::cerr << it->id << " "; }
      NewNewRegionDirectory::delAccess( it->key, it->id, getMemorySpaceId() );
   }
   if ( _VERBOSE_CACHE ) { std::cerr << std::endl ; }

   for ( std::set< NewNewRegionDirectory::RegionDirectoryKey >::iterator locked_object_it = locked_objects.begin(); locked_object_it != locked_objects.end(); locked_object_it++ ) {
      (*locked_object_it)->unlock();
   }

   for ( std::set< AllocatedChunk ** >::iterator it = chunks_to_invalidate.begin(); it != chunks_to_invalidate.end(); it++ ) {
      *(*it) = NULL;
   }

   if ( allocChunkPtr ) { /* FIXME ugly code */
      allocChunkPtr->increaseLruStamp();
      allocChunkPtr->clearNewRegions( allocatedRegion );
   }

   return allocChunkPtr;
   //for ( std::set< DeviceOps * >::iterator opIt = ops.begin(); opIt != ops.end(); opIt++ ) {
   //   (*opIt)->resumeInvalidations();
   //}
}

AllocatedChunk *RegionCache::getOrCreateChunk( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   ChunkList results;
   bool lock_chunk = true;
   AllocatedChunk *allocChunkPtr = NULL;
   global_reg_t allocatedRegion;

   std::size_t allocSize = 0;
   uint64_t targetHostAddr = 0;
   //std::cerr << __FUNCTION__ << " num dimensions " << cd.getNumDimensions() << std::endl;

   getAllocatableRegion( reg, allocatedRegion );

   targetHostAddr = allocatedRegion.getRealFirstAddress();
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

         void *deviceMem = _device.memAllocate( allocSize, sys.getSeparateMemory( _memorySpaceId ), targetHostAddr );
         //std::cerr << "malloc returns " << (void *)deviceMem << std::endl;
         if ( deviceMem == NULL ) {
            /* Invalidate */
            AllocatedChunk *invalidated_chunk = invalidate( allocatedRegion, wd, copyIdx );
            if ( invalidated_chunk != NULL ) {
               allocChunkPtr = invalidated_chunk;
               allocChunkPtr->setHostAddress( results.front().first->getAddress() );
               lock_chunk = false;
               *(results.front().second) = allocChunkPtr;
            } else {
               /* allocate mem */
               deviceMem = _device.memAllocate( allocSize, sys.getSeparateMemory( _memorySpaceId ), targetHostAddr );
               if ( deviceMem == NULL ) {
                  //fatal("Unable to allocate memory on the device.");
                  // let it return NULL 
               } else {
                  *(results.front().second) = NEW AllocatedChunk( *this, (uint64_t) deviceMem, results.front().first->getAddress(), results.front().first->getLength(), allocatedRegion, reg.isRooted() );
                  allocChunkPtr = *(results.front().second);
               }
            }
            //reg.key->invalUnlock();
         } else {
            *(results.front().second) = NEW AllocatedChunk( *this, (uint64_t) deviceMem, results.front().first->getAddress(), results.front().first->getLength(), allocatedRegion, reg.isRooted() );
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
      //std::cerr << "WARNING: null RegionCache::getAddress()" << std::endl;
   } else {
     //std::cerr << __FUNCTION__ << " returns dev address " << (void *) allocChunkPtr->getAddress() << std::endl;
     //if ( !allocChunkPtr->locked() ) 
      if ( lock_chunk ) allocChunkPtr->lock();
     allocChunkPtr->addReference( wd.getId(), 4 );
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

AllocatedChunk *RegionCache::getAllocatedChunk( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) const {
   return _getAllocatedChunk( reg, true, true, wd, copyIdx );
}

AllocatedChunk *RegionCache::getAllocatedChunk( global_reg_t const &reg, bool complain, WD const &wd, unsigned int copyIdx ) {
   _lock.acquire();
   AllocatedChunk *chunk = _getAllocatedChunk( reg, complain, true, wd, copyIdx );
   _lock.release();
   return chunk;
}

AllocatedChunk *RegionCache::_getAllocatedChunk( global_reg_t const &reg, bool complain, bool lockChunk, WD const &wd, unsigned int copyIdx ) const {
   ConstChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   _chunks.getChunk3( reg.getRealFirstAddress(), reg.getBreadth(), results );
   if ( results.size() == 1 ) {
      if ( results.front().second )
         allocChunkPtr = *(results.front().second);
      else
         allocChunkPtr = NULL;
   } else if ( results.size() > 1 ) {
         *(myThread->_file) <<"Requested addr " << (void *) reg.getRealFirstAddress() << " size " << reg.getBreadth() << std::endl;
      message0( "I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ConstChunkList::const_iterator it = results.begin(); it != results.end(); it++ )
         std::cerr << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
      if ( &wd != NULL ) {
         *(myThread->_file) << "Realloc needed. Caused by wd " << (wd.getDescription() ? wd.getDescription() : "n/a") << " copy index " << copyIdx << std::endl;
      } else {
         *(myThread->_file) << "Realloc needed. Unknown WD, probably comes from a taskwait or any other synchronization point." << std::endl;
      }
      fatal("Can not continue.");
   }
   if ( !allocChunkPtr && complain ) {
      printBt(*(myThread->_file) ); *(myThread->_file) << "Error, null region at spaceId "<< _memorySpaceId << " "; reg.key->printRegion( *(myThread->_file), reg.id ); *(myThread->_file) << std::endl;
      ensure(allocChunkPtr != NULL, "Chunk not found!");
   }
   if ( allocChunkPtr && lockChunk ) {
      //std::cerr << "AllocChunkPtr is " << allocChunkPtr << std::endl;
      allocChunkPtr->lock(); 
   }
   return allocChunkPtr;
}

void RegionCache::NEWcopyIn( unsigned int srcLocation, global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx, DeviceOps *givenOps, AllocatedChunk *chunk ) {
   //AllocatedChunk *chunk = getAllocatedChunk( reg );
   uint64_t origDevAddr = chunk->getAddress() + ( reg.getRealFirstAddress() - chunk->getHostAddress() );
   DeviceOps *ops = ( givenOps != NULL ) ? givenOps : chunk->getDeviceOps( reg );
   //chunk->unlock();
   //std::cerr << " COPY REGION ID " << reg.id << " OPS " << (void*)ops << std::endl;
   if ( srcLocation != 0 ) {
      AllocatedChunk *origChunk = sys.getSeparateMemory( srcLocation ).getCache().getAllocatedChunk( reg, wd, copyIdx );
      origChunk->NEWaddWriteRegion( reg.id, version );// this is needed in case we are copying out a fragment of a region
      origChunk->unlock();
   }
   copyIn( reg, origDevAddr, srcLocation, ops, NULL, wd );
}

void RegionCache::NEWcopyOut( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx, DeviceOps *givenOps, bool inval ) {
   AllocatedChunk *origChunk = getAllocatedChunk( reg, wd, copyIdx );
   uint64_t origDevAddr = origChunk->getAddress() + ( reg.getRealFirstAddress() - origChunk->getHostAddress() );
   DeviceOps *ops = ( givenOps != NULL ) ? givenOps : reg.getDeviceOps();
   //origChunk->clearDirty( reg );
   if ( !inval ) origChunk->NEWaddWriteRegion( reg.id, version );// this is needed in case we are copying out a fragment of a region, ignore in case of invalidation
   origChunk->unlock();
   CompleteOpFunctor *f = NEW CompleteOpFunctor( ops, origChunk );
   copyOut( reg, origDevAddr, ops, f, wd );
}

RegionCache::RegionCache( memory_space_id_t memSpaceId, Device &cacheArch, enum CacheOptions flags ) : _chunks(), _lock(), _device( cacheArch ), _memorySpaceId( memSpaceId ),
    _flags( flags ), _lruTime( 0 ), _softInvalidationCount( 0 ), _hardInvalidationCount( 0 ), _copyInObj( *this ), _copyOutObj( *this ) {
}

unsigned int RegionCache::getMemorySpaceId() const {
   return _memorySpaceId;
}

void RegionCache::_copyIn( global_reg_t const &reg, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device._copyIn( copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<" ["<< *((double*) hostAddr) <<"]"<<", devAddr="<< (void*)devAddr <<", len=" << len << ", _pe, ops, wd="<< wd.getId() <<" );" <<std::endl;
   }
   if (!fake) _device._copyIn( devAddr, hostAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, (CompleteOpFunctor *) NULL, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   //NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyInStrided1D( global_reg_t const &reg, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device._copyInStrided1D( copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<" ["<< *((double*) hostAddr) <<"]"<<", devAddr="<< (void*)devAddr <<", len, numChunks, ld, _pe, ops="<< (void*)ops<<", wd="<< wd.getId() <<" );" <<std::endl;
   }
   if (!fake) _device._copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, (CompleteOpFunctor *) NULL, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   //NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOut( global_reg_t const &reg, uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   ensure( f != NULL, " Error, functor received is null.");
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device._copyOut( copyFrom=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len=" << len << ", _pe, ops, wd="<< (&wd != NULL ? wd.getId() : -1 ) <<" );" <<std::endl;
   }
   if (!fake) _device._copyOut( hostAddr, devAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, f, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   //NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOutStrided1D( global_reg_t const &reg, uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld,  DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f != NULL, " Error, functor received is null.");
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device._copyOutStrided1D( copyFrom=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, numChunks, ld, _pe, ops="<< (void*)ops <<", wd="<< (&wd != NULL ? wd.getId() : -1 ) <<" );" <<std::endl;
   }
   if (!fake) _device._copyOutStrided1D( hostAddr, devAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, f, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   //NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_syncAndCopyIn( global_reg_t const &reg, unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getSeparateMemory( syncFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   sys.getSeparateMemory( syncFrom ).getCache()._copyOut( reg, hostAddr, origDevAddr, len, cout, fsource, wd, fake );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   delete cout;
   this->_copyIn( reg, devAddr, hostAddr, len, ops, (CompleteOpFunctor *) NULL, wd, fake );
}

void RegionCache::_syncAndCopyInStrided1D( global_reg_t const &reg, unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   ensure( f == NULL, " Error, functor received is not null.");
   DeviceOps *cout = NEW DeviceOps();
   AllocatedChunk *origChunk = sys.getSeparateMemory( syncFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   sys.getSeparateMemory( syncFrom ).getCache()._copyOutStrided1D( reg, hostAddr, origDevAddr, len, numChunks, ld, cout, fsource, wd, fake );
   while ( !cout->allCompleted() ){ myThread->idle(); }
   delete cout;
   this->_copyInStrided1D( reg, devAddr, hostAddr, len, numChunks, ld, ops, (CompleteOpFunctor *) NULL, wd, fake );
}

bool RegionCache::_copyDevToDev( global_reg_t const &reg, memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   bool result = true;
   ensure( f == NULL, " Error, functor received is not null.");
   //AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len );
   AllocatedChunk *origChunk = sys.getSeparateMemory( copyFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device._copyDevToDev( copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len=" << len << ", _pe, sys.getSeparateMemory( copyFrom="<< copyFrom<<" ), ops, wd="<< wd.getId() << ", f="<< f <<" );" <<std::endl;
   }
   if (!fake) {
      result = _device._copyDevToDev( devAddr, origDevAddr, len, sys.getSeparateMemory( _memorySpaceId ), sys.getSeparateMemory( copyFrom ), ops, fsource, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   }
   //NANOS_INSTRUMENT( inst.close(); );
   return result;
}

bool RegionCache::_copyDevToDevStrided1D( global_reg_t const &reg, memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   bool result = true;
   //AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len );
   AllocatedChunk *origChunk = sys.getSeparateMemory( copyFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   ensure( f == NULL, " Error, functor received is not null.");
   CompleteOpFunctor *fsource = NEW CompleteOpFunctor( ops, origChunk );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device._copyDevToDevStrided1D( copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len, _pe, sys.getCaches()[ copyFrom="<< copyFrom<<" ]->_pe, ops, wd="<< wd.getId() <<", f="<< f <<" );"<<std::endl;
   }
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   if (!fake) {
      result = _device._copyDevToDevStrided1D( devAddr, origDevAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), sys.getSeparateMemory( copyFrom ), ops, fsource, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   }
   //NANOS_INSTRUMENT( inst.close(); );
   return result;
}

void RegionCache::CopyIn::doNoStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyIn( reg, devAddr, hostAddr, size, ops, f, wd, fake );
   } else {
      //If copydev2dev unsucesfull (not supported/implemented), do a copy through host
      if ( ( &sys.getSeparateMemory( dataLocation ).getCache().getDevice() != &getParent()._device ) ||
            !getParent()._copyDevToDev( reg, dataLocation, devAddr, hostAddr, size, ops, f, wd, fake )) {
         getParent()._syncAndCopyIn( reg, dataLocation, devAddr, hostAddr, size, ops, f, wd, fake );
      }
   }
}

void RegionCache::CopyIn::doStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyInStrided1D( reg, devAddr, hostAddr, size, count, ld, ops, f, wd, fake );
   } else {
       //If copydev2dev unsucesfull (not supported/implemented), do a copy through host
      if ( ( &sys.getSeparateMemory( dataLocation ).getCache().getDevice() != &getParent()._device ) ||
            !getParent()._copyDevToDevStrided1D( reg, dataLocation, devAddr, hostAddr, size, count, ld, ops, f, wd, fake ) ) {
         getParent()._syncAndCopyInStrided1D( reg, dataLocation, devAddr, hostAddr, size, count, ld, ops, f, wd, fake );
      }
   }
}

void RegionCache::CopyOut::doNoStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   getParent()._copyOut( reg, hostAddr, devAddr, size, ops, f, wd, fake );
}
void RegionCache::CopyOut::doStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) {
   getParent()._copyOutStrided1D( reg, hostAddr, devAddr, size, count, ld, ops, f, wd, fake );
}

void RegionCache::doOp( Op *opObj, global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, CompleteOpFunctor *functor, WD const &wd ) {

   class LocalFunction {
      Op *_opObj;
      global_reg_t _hostMem;
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
         LocalFunction( Op *opO,  const global_reg_t &reg, nanos_region_dimension_internal_t *r, unsigned int n, unsigned int t, unsigned int nc, std::size_t ccs, unsigned int loc, DeviceOps *operations, WD const &workdesc, uint64_t devAddr, uint64_t hostAddr, CompleteOpFunctor *f )
               : _opObj( opO ), _hostMem( reg ), _region( r ), _numDimensions( n ), _targetDimension( t ), _numChunks( nc ), _contiguousChunkSize( ccs ), _location( loc ), _ops( operations ), _wd( workdesc ), _devBaseAddr( devAddr ), _hostBaseAddr( hostAddr ), _f( f ) {
         }
         void issueOpsRecursive( unsigned int idx, std::size_t offset, std::size_t leadingDim ) {
            if ( idx == ( _numDimensions - 1 ) ) {
               //issue copy
               unsigned int L_numChunks = _numChunks; //_region[ idx ].accessed_length;
               if ( L_numChunks > 1 && sys.usePacking() ) {
                  //std::cerr << "[NEW]opObj("<<_opObj->getStr()<<")->doStrided( src="<<_location<<", dst="<< _opObj->getParent().getMemorySpaceId()<<", "<<(void*)(_devBaseAddr+offset)<<", "<<(void*)(_hostBaseAddr+offset)<<", "<<_contiguousChunkSize<<", "<<_numChunks<<", "<<leadingDim<<", _ops="<< (void*)_ops<<", _wd="<<(&_wd != NULL ? _wd.getId():-1)<<" )";
                  _opObj->doStrided( _hostMem, _location, _devBaseAddr+offset, _hostBaseAddr+offset, _contiguousChunkSize, _numChunks, leadingDim, _ops, _f, _wd, false );
                  //std::cerr <<" done"<< std::endl;
               } else {
                  for (unsigned int chunkIndex = 0; chunkIndex < L_numChunks; chunkIndex +=1 ) {
                     //std::cerr <<"[NEW]opObj("<<_opObj->getStr()<<")->doNoStrided( src="<<_location<<", dst="<< _opObj->getParent().getMemorySpaceId()<<", "<<(void*)(_devBaseAddr+offset + chunkIndex*(leadingDim))<<", "<<(void*)(_hostBaseAddr+offset + chunkIndex*(leadingDim))<<", "<<_contiguousChunkSize<<", _ops="<< (void*)_ops<< ", _wd="<<(&_wd != NULL ? _wd.getId():-1)<<" )";
                    _opObj->doNoStrided( _hostMem, _location, _devBaseAddr+offset + chunkIndex*(leadingDim), _hostBaseAddr+offset + chunkIndex*(leadingDim), _contiguousChunkSize, _ops, _f, _wd, false );
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
   LocalFunction local( opObj, hostMem, region, hostMem.getNumDimensions(), dimIdx, numChunks, contiguousChunkSize, location, ops, wd, devBaseAddr, hostMem.getRealFirstAddress(), functor /* hostMem.key->getBaseAddress()*/ );
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
   std::cerr << "Functor called!" <<std::endl;
   //_chunk->removeReference( 0 );
}

//unsigned int RegionCache::getVersionSetVersion( global_reg_t const &reg, unsigned int newVersion ) {
//   AllocatedChunk *chunk = getAllocatedChunk( reg );
//   unsigned int version = chunk->getVersionSetVersion( reg, newVersion );
//   chunk->unlock();
//   return version;
//}

unsigned int RegionCache::getVersion( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg, wd, copyIdx );
   unsigned int version = chunk->getVersion( reg );
   chunk->unlock();
   return version;
}

void RegionCache::releaseRegion( global_reg_t const &reg, WD const &wd, unsigned int copyIdx, enum CachePolicy policy ) {
   //std::cerr << "Release region for wd " << wd.getId() << ": " << std::endl;
   //reg.key->printRegion(reg.id);
   //std::cerr << std::endl;
   AllocatedChunk *chunk = _getAllocatedChunk( reg, true, false, wd, copyIdx );
   //TODO if ( policy == NO_CACHE ) {
   //TODO    chunk->removeRegion( reg );
   //TODO }
   chunk->removeReference( wd.getId() );
   //chunk->unlock();
}

uint64_t RegionCache::getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress, AllocatedChunk *chunk ) const {
   return ( chunk->getAddress() - ( chunk->getHostAddress() - baseAddress ) );
}

bool RegionCache::prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   bool result = true;
   std::size_t total_allocatable_size = 0;
   std::set< global_reg_t > regions_to_allocate;
   for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
      global_reg_t allocatable_region;
      getAllocatableRegion( memCopies[ idx ]._reg, allocatable_region );
      regions_to_allocate.insert( allocatable_region );
   }
   for ( std::set< global_reg_t >::iterator it = regions_to_allocate.begin(); it != regions_to_allocate.end(); it++ ) {
      total_allocatable_size += it->getDataSize();
   }
   if ( total_allocatable_size <= _device.getMemCapacity( sys.getSeparateMemory( _memorySpaceId ) ) ) {
      //_lock.acquire();
      while ( !_lock.tryAcquire() ) {
         myThread->idle();
      }
      //attempt to allocate regions without triggering invalidations, this will reserve any chunk used by this WD
      for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
         if ( memCopies[ idx ]._chunk == NULL ) {
            memCopies[ idx ]._chunk = tryGetAddress( memCopies[ idx ]._reg, wd, idx );
            if ( memCopies[ idx ]._chunk != NULL ) {
               //std::cerr << "Allocated region for wd " << wd.getId() << std::endl;
               //memCopies[ idx ]._reg.key->printRegion(memCopies[ idx ]._reg.id);
               //std::cerr << std::endl;
   //AllocatedChunk *chunk = _getAllocatedChunk( memCopies[ idx ]._reg, false, false, wd, idx );
               //std::cerr << "--1--> chunk is " << (void *) memCopies[ idx ]._chunk << " other chunk " << (void*) chunk<< std::endl;
               memCopies[ idx ]._chunk->unlock();
            }
         }
      }
      for ( unsigned int idx = 0; idx < numCopies && result; idx += 1 ) {
         if ( memCopies[ idx ]._chunk == NULL ) {
            memCopies[ idx ]._chunk = getOrCreateChunk( memCopies[ idx ]._reg, wd, idx );
            if ( memCopies[ idx ]._chunk == NULL ) {
               result = false;
            } else {
               //std::cerr << "Allocated region for wd " << wd.getId() << std::endl;
               //memCopies[ idx ]._reg.key->printRegion(memCopies[ idx ]._reg.id);
               //std::cerr << std::endl;
   //AllocatedChunk *chunk = _getAllocatedChunk( memCopies[ idx ]._reg, false, false, wd, idx );
               //std::cerr << "--2--> chunk is " << (void*) memCopies[ idx ]._chunk << " other chunk " << (void*) chunk << std::endl;
               memCopies[ idx ]._chunk->unlock();
            }
         }
      }
      _lock.release();
   } else {
      result = false;
      std::cerr << "This device can not hold this task, not enough memory. Needed: "<< total_allocatable_size << " max avalilable " << _device.getMemCapacity( sys.getSeparateMemory( _memorySpaceId ) ) << std::endl;
      fatal( "This device can not hold this task, not enough memory." );
   }
   return result;
}

void RegionCache::prepareRegionsToBeCopied( std::set< global_reg_t > const &regs, unsigned int version, std::set< AllocatedChunk * > &chunks, WD const &wd, unsigned int copyIdx ) {
   _lock.acquire();
   for ( std::set< global_reg_t >::iterator it = regs.begin(); it != regs.end(); it++ ) {
      this->_prepareRegionToBeCopied( *it, version, chunks, wd, copyIdx );
   }
   _lock.release();
}

void RegionCache::_prepareRegionToBeCopied( global_reg_t const &reg, unsigned int version, std::set< AllocatedChunk * > &chunks, WD const &wd, unsigned int copyIdx ) {
   AllocatedChunk *chunk = _getAllocatedChunk( reg, false, false, wd, copyIdx );
   if ( _VERBOSE_CACHE ) { std::cerr << " reg " << reg.id << " got chunk " << chunk << std::endl; }
   if ( chunk != NULL ) {
      if ( chunks.count( chunk ) == 0 ) {
         chunk->lock();
         chunk->addReference( wd.getId(), 1 );
         chunks.insert( chunk );
         chunk->unlock();
      }
   } else {
      fatal("Could not add a reference to a source chunk."); 
   }
}

void RegionCache::setRegionVersion( global_reg_t const &hostMem, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   AllocatedChunk *chunk = getAllocatedChunk( hostMem, wd, copyIdx );
   chunk->NEWaddWriteRegion( hostMem.id, version );
   chunk->unlock();
}

void RegionCache::copyInputData( BaseAddressSpaceInOps &ops, global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx ) {
   _lock.acquire();
//reg.key->printRegion( reg.id ); std::cerr << std::endl;
   //AllocatedChunk *chunk = getAllocatedChunk( reg );
   std::set< reg_t > notPresentParts;
   //      std::cerr << "locations:  ";
   //      for ( NewLocationInfoList::const_iterator it2 = locations.begin(); it2 != locations.end(); it2++ ) {
   //         std::cerr << "[ " << it2->first << "," << it2->second << " ] ";
   //      }
   //      std::cerr << std::endl;
   if ( chunk->NEWaddReadRegion2( ops, reg.id, version, notPresentParts, output, locations, wd, copyIdx ) ) {
   }
   //chunk->unlock();
   _lock.release();
}

void RegionCache::allocateOutputMemory( global_reg_t const &reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   _lock.acquire();
   AllocatedChunk *chunk = getAllocatedChunk( reg, wd, copyIdx );
   chunk->NEWaddWriteRegion( reg.id, version );
   //*(myThread->_file) << __FUNCTION__ << " set version to " << version << " for region "; reg.key->printRegion( *myThread->_file, reg.id); *myThread->_file << std::endl;
   reg.setLocationAndVersion( _memorySpaceId, version );
   chunk->unlock();
   _lock.release();
}

std::size_t RegionCache::getAllocatableSize( global_reg_t const &reg ) const {
   global_reg_t allocated_region;
   getAllocatableRegion( reg, allocated_region );
   return allocated_region.getDataSize();
}

void RegionCache::getAllocatableRegion( global_reg_t const &reg, global_reg_t &allocRegion ) const {
   allocRegion.key = reg.key;
   if ( _flags == ALLOC_WIDE ) {
      allocRegion.id = 1;
   } else if ( _flags == ALLOC_FIT ) {
      allocRegion.id = reg.getFitRegionId();
   } else {
      std::cerr <<"RegionCache ERROR: Undefined _flags value."<<std::endl;
   }
}

bool RegionCache::canAllocateMemory( MemCacheCopy *memCopies, unsigned int numCopies, bool considerInvalidations, WD const &wd ) {
   bool result = true;
   bool *present_regions = (bool *) alloca( numCopies * sizeof(bool) );
   std::size_t *sizes = (std::size_t *) alloca( numCopies * sizeof(std::size_t) );
   unsigned int needed_chunks = 0;
   if ( _lock.tryAcquire() ) {
   
   /* check if the desired region is already allocated */
   for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
      AllocatedChunk *chunk = _getAllocatedChunk( memCopies[ idx ]._reg , false, false, wd, idx );
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

      unsigned int remaining_count = 0;
      while ( remaining_count < needed_chunks && remaining_sizes[ remaining_count ] != 0 ) {
         remaining_count +=1;
      }

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
   } else {
   return false;
   }
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


void RegionCache::invalidateObject( global_reg_t const &reg ) {
   //std::cerr << "-----------------------vvvvvvvvvvvv inv reg " << reg.id << "vvvvvvvvvvvvvvvvvv--------------------" << std::endl; 
   ConstChunkList results;
   _chunks.getChunk3( reg.getRealFirstAddress(), reg.getBreadth(), results );
   if ( results.size() > 0 ) {
      for ( ConstChunkList::iterator it = results.begin(); it != results.end(); it++ ) {
         //std::cerr << "Invalidate object, chunk:: addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
         //printBt();
         if ( it->second != NULL && *(it->second) != NULL ) {
            _device.memFree( (*(it->second))->getAddress(), sys.getSeparateMemory( _memorySpaceId ) );
            delete *(it->second);
         }
      }
      _chunks.removeChunks( reg.getRealFirstAddress(), reg.getBreadth() );
   }
   //std::cerr << "-----------------------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^--------------------" << std::endl; 
}

void RegionCache::copyOutputData( SeparateAddressSpaceOutOps &ops, global_reg_t const &reg, unsigned int version, bool output, enum CachePolicy policy, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx ) {
   std::ostream &o = *(myThread->_file);
   if ( output ) {
      if ( policy != WRITE_BACK ) {
         // WRITE_THROUGH or NO_CACHE
         o << "I should copy this back "; reg.key->printRegion( o, reg.id ); o << std::endl;
         chunk->copyRegionToHost( ops, reg.id, version, wd, copyIdx );
      }
   } 

   if ( policy == NO_CACHE ) {
      o << "I should free this region "; reg.key->printRegion( o, reg.id ); o << std::endl;
   }
}

void AllocatedChunk::printReferencingWDs() const {
   *(myThread->_file) << "Referencing WDs: [";
   for ( std::map<int, unsigned int>::const_iterator it = _refWdId.begin(); it != _refWdId.end(); it++ ) {
      if ( it->second != 0 ) {
         std::map<int, std::set<int> >::const_iterator itLoc = _refLoc.find( it->first );
         *(myThread->_file) << "(wd: " << it->first << " count: " << it->second <<" loc: {";
         for (std::set<int>::const_iterator sIt = itLoc->second.begin(); sIt != itLoc->second.end(); sIt++ ) {
            *(myThread->_file) << *sIt << " ";
         }
         *(myThread->_file) << "}";
      }
   }
   *(myThread->_file) << "]" << std::endl;
}

void RegionCache::printReferencedChunksAndWDs() const {
   MemoryMap<AllocatedChunk>::const_iterator it;
   for ( it = _chunks.begin(); it != _chunks.end(); it++ ) {
      if ( it->second != NULL ) {
         AllocatedChunk &c = *(it->second);
         c.printReferencingWDs();
      }
   }
}
