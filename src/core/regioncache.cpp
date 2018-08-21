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

#include <limits>
#include <iomanip>
#include "workdescriptor_decl.hpp"
#include "debug.hpp"
#include "memorymap.hpp"
#include "copydata.hpp"
#include "atomic.hpp"
#include "lock.hpp"
#include "processingelement.hpp"
#include "system.hpp"
#include "deviceops.hpp"
#ifdef GPU_DEV
#include "gpudd.hpp"
#endif
#include "regiondirectory.hpp"
#include "regioncache.hpp"
#include "cachedregionstatus.hpp"
#include "os.hpp"
#include "regiondict.hpp"
#include "memoryops_decl.hpp"
#include "globalregt.hpp"

#define VERBOSE_DEV_OPS ( sys.getVerboseDevOps() )
#define VERBOSE_INVAL 0

#if VERBOSE_CACHE
 #define _VERBOSE_CACHE 1
#else
 #define _VERBOSE_CACHE 0
#endif

using namespace nanos;

LockedObjects::LockedObjects() : _lockedObjects() {
}

void LockedObjects::addAndLock( RegionDirectory::RegionDirectoryKey key ) {
   std::set< RegionDirectory::RegionDirectoryKey >::iterator dict_it = _lockedObjects.find( key );
   if ( dict_it == _lockedObjects.end() ) {
      //*myThread->_file <<"trying lock object " << key << std::endl;
      key->lockObject();
      //*myThread->_file << __func__ << " locked object " << key << std::endl;
      _lockedObjects.insert( key );
   } else {
      //*myThread->_file << __func__ << " not locking object !!!!!!!!!!!! " << key << std::endl;
   }
}

void LockedObjects::releaseLockedObjects() {
   for ( std::set< RegionDirectory::RegionDirectoryKey >::iterator locked_object_it = _lockedObjects.begin(); locked_object_it != _lockedObjects.end(); locked_object_it++ ) {
      //*myThread->_file << __func__ << " releasing object " << *locked_object_it << std::endl;
      (*locked_object_it)->unlockObject();
   }
   _lockedObjects.clear();
}

AllocatedChunk::AllocatedChunk( RegionCache &owner, uint64_t addr, uint64_t hostAddress, std::size_t size, global_reg_t const &allocatedRegion, bool rooted ) :
   _owner( owner ),
   _lock(),
   _address( addr ),
   _hostAddress( hostAddress ),
   _size( size ),
   _dirty( false ),
   _rooted( rooted ),
   _lruStamp( 0 ),
   _refs( 0 ),
   _refWdId(),
   _refLoc(),
   _allocatedRegion( allocatedRegion ),
   _flushable( false ) {
      //*myThread->_file << "region " << allocatedRegion.id << " addr " << (void *) addr<<" hostAddr is " << (void*)hostAddress << " key " << allocatedRegion.key << std::endl;
      _newRegions = NEW CacheRegionDictionary( *(allocatedRegion.key) );
      //*myThread->_file << "Created dictionary " << _newRegions << " w/key " << allocatedRegion.key << std::endl;
      ensure(_newRegions->getNumDimensions() > 0, "Invalid object");
}

AllocatedChunk::~AllocatedChunk() {
   //*myThread->_file << "Im being released! "<< (void *) _newRegions << std::endl;
   for ( CacheRegionDictionary::citerator it = _newRegions->begin(); it != _newRegions->end(); it++ ) {
      CachedRegionStatus *entry = (CachedRegionStatus *) it->second.getData();
      delete entry;
   }
   delete _newRegions;
}

void AllocatedChunk::makeFlushable() {
   _flushable = true;
}

bool AllocatedChunk::isFlushable() const {
   return _flushable;
}

void AllocatedChunk::clearNewRegions( global_reg_t const &reg ) {
   // *myThread->_file << "clear regions for chunk " << (void *) this << " w/hAddr " << (void*) this->getHostAddress() << " - " << (void*)(this->getHostAddress() +this->getSize() ) << std::endl;
   for ( CacheRegionDictionary::citerator it = _newRegions->begin(); it != _newRegions->end(); it++ ) {
      CachedRegionStatus *entry = (CachedRegionStatus *) it->second.getData();
      delete entry;
   }
   delete _newRegions;
   _newRegions = NEW CacheRegionDictionary( *(reg.key) );
   _allocatedRegion = reg;
}


CacheRegionDictionary *AllocatedChunk::getNewRegions() {
   return _newRegions;
}

bool AllocatedChunk::trylock() {
   bool res = _lock.tryAcquire();
   // if ( res ) {
   // *myThread->_file << "trylock " << res << " x " << myThread->getId() <<" x Locked chunk " << (void *) this << std::endl;
   // }
   return res;
}

void AllocatedChunk::lock( bool setVerbose ) {
   //*myThread->_file << ": " << myThread->getId() <<" : Locked chunk " << (void *) this << std::endl;
   //_lock.acquire();
   while ( !_lock.tryAcquire() ) {
      myThread->processTransfers();
   }
   //sys.printBt();
   //*myThread->_file << "x " << myThread->getId() <<" x Locked chunk " << (void *) this << std::endl;
}

void AllocatedChunk::unlock( bool unsetVerbose ) {
   //printBt(*myThread->_file);
   //*myThread->_file << "x " << myThread->getId() << " x Unlocked chunk " << (void *) this << std::endl;
   _lock.release();
   //*myThread->_file << ": " << myThread->getId() << " : Unlocked chunk " << (void *) this << std::endl;
}

bool AllocatedChunk::locked() const {
   return _lock.getState() != NANOS_LOCK_FREE;
}

void AllocatedChunk::copyRegionToHost( SeparateAddressSpaceOutOps &ops, reg_t reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   RegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();
   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( entry->getVersion() == version || entry->getVersion() == (version+1) ) {
      global_reg_t greg( reg, key );
      DeviceOps * dops = greg.getHomeDeviceOps( wd, copyIdx );
      DirectoryEntryData *dict_entry = RegionDirectory::getDirectoryEntry( *key, reg );
      memory_space_id_t home = (dict_entry->getRootedLocation() == (unsigned int) -1) ? 0 : dict_entry->getRootedLocation();
      if ( dops->addCacheOp( &wd, 8 ) ) {
         ops.insertOwnOp( dops, greg, version, 0 );
         ops.addOutOp( home, _owner.getMemorySpaceId(), greg, version, dops, this, wd, copyIdx );
      } else {
         ops.getOtherOps().insert( dops );
      }
   } else {
      *(myThread->_file) << "CopyOut for wd: "<< wd.getId() << " copyIdx " << copyIdx << " requested to copy version " << version << " but cache version is " << entry->getVersion() << " region: ";
      _newRegions->printRegion( *(myThread->_file), reg );
      *(myThread->_file) << std::endl;
      
      printBt( *(myThread->_file) );
   }
}
void AllocatedChunk::copyRegionFromHost( BaseAddressSpaceInOps &ops, reg_t reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   RegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();
   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !entry ) {
      entry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, entry );
   }
      global_reg_t greg( reg, key );
      //jbueno: We want to force the operation for each thread issuing the copy
      //  if we use "addCacheOp" and several threads issue the same operation,
      //  only one will be issued, and we do not want this.
      //DeviceOps * dops = greg.getDeviceOps();
      //if ( dops->addCacheOp( &wd, 8 ) ) {
         //ops.insertOwnOp( dops, greg, version, 0 );
         ops.addOpFromHost( greg, version, this, copyIdx );
      //} else {
      //   ops.getOtherOps().insert( dops );
      //}
}

bool AllocatedChunk::NEWaddReadRegion2( BaseAddressSpaceInOps &ops, reg_t reg, unsigned int version, NewLocationInfoList const &locations, WD const &wd, unsigned int copyIdx ) {
   unsigned int currentVersion = 0;
   bool opEmitted = false;
   std::list< std::pair< reg_t, reg_t > > components;
   bool skipNull = false;

   std::ostream &o = *(myThread->_file);

   //o << "==========*******========= " BEGIN " << __FUNCTION__ << " wd " << wd.getId() << " [ " << (wd.getDescription() != NULL ? wd.getDescription() : "n/a" ) << " ] copy index " << copyIdx <<" ====================" << std::endl;

//if ( sys.getNetwork()->getNodeNum() > 0 ) {
//   o << __FUNCTION__ << " reg " << reg << std::endl;
//}
   CachedRegionStatus *thisRegEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !thisRegEntry ) {
//if ( sys.getNetwork()->getNodeNum() > 0 ) {
//   o << __FUNCTION__ << " thisEntry is null " << reg << std::endl;
//}
      thisRegEntry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, thisRegEntry );
   } else {
//if ( sys.getNetwork()->getNodeNum() > 0 ) {
//   o << __FUNCTION__ << " thisEntry is not null " << reg << std::endl;
//}
   }

   DeviceOps *thisEntryOps = thisRegEntry->getDeviceOps();
   if ( thisEntryOps->addCacheOp( /* debug: */ &wd, 1 ) ) {
      opEmitted = true;

      //o << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ " << __FUNCTION__ << " " << (void*) this << " reg " << reg << " set rversion "<< version << " ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]] This chunk key: " << (void *) _newRegions->getGlobalDirectoryKey()<< std::endl;
      // lock / free needed for multithreading on the same cache.
      _newRegions->registerRegion( reg, components, currentVersion );
      RegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();

      // for ( CacheRegionDictionaryIterator it = _newRegions->begin(); it != _newRegions->end(); it++) {
      //    o << "Region: " << it->first << " "; _newRegions->printRegion( o, it->first ); o << " has entry with version " << (( (it->second).getData() ) ? (it->second).getData()->getVersion() : -1)<< std::endl;
      // }

      // o << "Asked for region " << reg << " got: " << std::endl;
      // for ( std::list< std::pair< reg_t, reg_t > >::const_iterator it = components.begin(); it != components.end(); it++ ) {
      //    CachedRegionStatus *thisEntry_f = ( CachedRegionStatus * ) _newRegions->getRegionData( it->first );
      //    CachedRegionStatus *thisEntry_s = ( CachedRegionStatus * ) _newRegions->getRegionData( it->second );
      //    o << "component: " << it->first << "(" <<
      //       (thisEntry_f != NULL ? (int)thisEntry_f->getVersion() : (-1) ) << "), "<<
      //       it->second << "(" <<
      //       (thisEntry_s != NULL ? (int)thisEntry_s->getVersion() : (-1) ) << ") ::: ";
      //       key->printRegion( o, it->first ); o << std::endl;
      // }

      if ( components.size() > 1 ) {
         std::list< std::pair< reg_t, reg_t > > componentsNotNull;
         bool imInList = false;
         reg_t myRegData = 0;
         for ( std::list< std::pair< reg_t, reg_t > >::const_iterator it = components.begin(); it != components.end() && !imInList; it++ ) {
            CachedRegionStatus *thisEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->first );
            if ( it->first == reg ) {
               imInList = true;
               myRegData = it->second;
            }
            if ( thisEntry != NULL ) {
               componentsNotNull.push_back( *it );
            }
         }
         if ( !imInList ) {
            skipNull = _newRegions->doTheseRegionsForm( reg, componentsNotNull.begin(), componentsNotNull.end(), false );
         } else {
            components.clear();
            components.push_back( std::pair< reg_t, reg_t >( reg, myRegData ) );
         }
      }

      for ( std::list< std::pair< reg_t, reg_t > >::iterator it = components.begin(); it != components.end(); it++ )
      {
         CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->first );
         if ( !entry && skipNull ) {
            continue;
         }
         if ( !entry || version > entry->getVersion() ) {
            //DirectoryEntryData *_dentry1 = RegionDirectory::getDirectoryEntry( *key, it->first );
            //DirectoryEntryData *_dentry2 = RegionDirectory::getDirectoryEntry( *key, it->second );
            //o << " 1. DENTRY " << (void *)_dentry1  << " for reg " << it->first << std::endl;
            //if ( _dentry1 ) {
            //   key->printRegion(o, it->first); o << *_dentry1 << std::endl;
            //}
            //o << " 2. DENTRY " << (void *)_dentry2  << " for reg " << it->second << std::endl;
            //if ( _dentry2 ) {
            //   key->printRegion(o, it->second); o << *_dentry2 << std::endl;
            //}
            //if ( !entry ) {
            //   o << "No entry for region " << it->first << " must copy from region " << it->second << " "; _newRegions->printRegion(o, it->second); o << " want version "<< version << " entry version is " << ( (!entry) ? -1 : entry->getVersion() )<< std::endl;
            //} else {
            //   o << "Version lower " << it->first << " "; _newRegions->printRegion( o, it->first); o << " must copy from region " << it->second << " "; _newRegions->printRegion(o, it->second); o << " want version "<< version << " entry version is " << ( (!entry) ? -1 : entry->getVersion() )<< std::endl;
            //}
            CachedRegionStatus *copyFromEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->second );
            // if ( it->first != it->second && it->second != reg ) {
            //    o << " Operating with a superset of me, set status of region " << it->second << " to 1 (upgrading some part)" << std::endl;
            //    copyFromEntry->setStatus(1);
            // }
            // if ( copyFromEntry->getStatus() != 0 ) {
            //    o << " WARNING: status != 0, upgrading!!!" << std::endl;
            // } 
            if ( !copyFromEntry || version > copyFromEntry->getVersion() ) {
               //o << "I HAVE TO COPY: I dont have this region, entry = " << entry << " " << skipNull << std::endl;

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
                  //key->printRegion(o, locIt->first); o << std::endl;

                  //if ( locIt->first == it->first || chunkReg.contains( locReg ) )
                  if ( locReg.id == chunkReg.id || locReg.key->checkIntersect( locReg.id, chunkReg.id ) )
                  {

                     reg_t target = 0;
                     if ( locReg.id == chunkReg.id ) {
                        target = locReg.id;
                     } else {
                        target = locReg.key->computeIntersect( locReg.id, chunkReg.id );
                     }

                     global_reg_t region_shape( target , key );
                     global_reg_t data_source( locIt->second, key );

                     if ( reg != region_shape.id || _newRegions->getRegionData( region_shape.id ) == NULL ) {
                        prepareRegion( region_shape.id, version );
                     }
                     //o << "shape: "<< locIt->first << " data source: " << locIt->second << std::endl;
                     //o <<" CHECKING THIS SHIT ID " << data_source.id << std::endl;
                     //if ( location == 0 || location != _owner.getMemorySpaceId() )
                     if ( !data_source.isLocatedIn( _owner.getMemorySpaceId() ) )
                     {
                        memory_space_id_t location = data_source.getPreferedSourceLocation( _owner.getMemorySpaceId() );
                        //o << "add copy from host, reg " << region_shape.id << " version " << ops.getVersionNoLock( data_source, wd, copyIdx ) << std::endl;
                        if ( _VERBOSE_CACHE ) {
                           DirectoryEntryData *dentry = RegionDirectory::getDirectoryEntry( *(data_source.key), data_source.id );
                           DirectoryEntryData *dentry2 = RegionDirectory::getDirectoryEntry( *(data_source.key), region_shape.id );
                           if ( dentry2 && dentry ) o << "I have to copy region " << region_shape.id << " from location " << location << " (data_source is " << data_source.id << ")" << *dentry << " region_shape: "<< *dentry2<< std::endl;
                        }
                        CachedRegionStatus *entryToCopy = ( CachedRegionStatus * ) _newRegions->getRegionData( region_shape.id );
                        DeviceOps *entryToCopyOps = entryToCopy->getDeviceOps();
                        if ( entryToCopy != thisRegEntry ) {
                           if ( entryToCopyOps->addCacheOp( /* debug: */ &wd, 2 ) ) {
                              ops.insertOwnOp( entryToCopyOps, global_reg_t( locIt->first, _newRegions->getGlobalDirectoryKey() ), version, _owner.getMemorySpaceId() );
                              if ( location == 0 ) {
                                 ops.addOpFromHost( region_shape, version, this, copyIdx );
                              } else if ( location != _owner.getMemorySpaceId() ) {
                                 sys.getSeparateMemory( location ).getCache().lock();
                                 AllocatedChunk *orig_chunk = sys.getSeparateMemory( location ).getCache().getAllocatedChunk( region_shape, wd, copyIdx );
                                 uint64_t orig_dev_addr = orig_chunk->getAddress() + ( region_shape.getRealFirstAddress() - orig_chunk->getHostAddress() );
                                 sys.getSeparateMemory( location ).getCache().unlock();
                                 orig_chunk->unlock();
                                 ops.addOp( &sys.getSeparateMemory( location ) , region_shape, version, this, orig_chunk, orig_dev_addr, wd, copyIdx );
                              }
                           } else {
                              ops.getOtherOps().insert( entryToCopyOps );
                           }
                        } else {
                           if ( location == 0 ) {
                              ops.addOpFromHost( region_shape, version, this, copyIdx );
                           } else if ( location != _owner.getMemorySpaceId() ) {
                              sys.getSeparateMemory( location ).getCache().lock();
                              AllocatedChunk *orig_chunk = sys.getSeparateMemory( location ).getCache().getAllocatedChunk( region_shape, wd, copyIdx );
                              uint64_t orig_dev_addr = orig_chunk->getAddress() + ( region_shape.getRealFirstAddress() - orig_chunk->getHostAddress() );
                              sys.getSeparateMemory( location ).getCache().unlock();
                              orig_chunk->unlock();
                              ops.addOp( &sys.getSeparateMemory( location ) , region_shape, version, this, orig_chunk, orig_dev_addr, wd, copyIdx );
                           }
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
            //o << "NO NEED TO COPY: I have this region already "  << std::endl;
            ops.getOtherOps().insert( entry->getDeviceOps() );
         } else {
            o << "ERROR: version in cache (" << entry->getVersion() << ") > than version requested ("<< version <<"). WD id: "<< wd.getId() << " desc: " << (wd.getDescription() ? wd.getDescription() : "n/a") << " w index " << copyIdx << std::endl;
            o << " Wanted Reg: ";
            key->printRegion( o, reg );
            o << std::endl << "First: ";
            key->printRegion( o, it->first );
            o << std::endl << "Second: ";
            key->printRegion( o, it->second );
            o << std::endl;
            printBt(o);
         }
      }
      //*(myThread->_file) << __FUNCTION__ << " set region cache entry version to " << version << " for wd " << wd.getId() << " idx " << copyIdx << std::endl;
      if ( thisRegEntry->getVersion() < version ) {
         thisRegEntry->setVersion( version );
      } else if ( thisRegEntry->getVersion() > version ) {
         //FIXME: commutative or concurrent.
         *myThread->_file << __func__ << " Warning: Copy @ WD " << wd.getId() << " desc: " << (wd.getDescription() ? wd.getDescription() : "n/a") << " w index " << copyIdx << " is commutative or concurrent. Cache version is " << thisRegEntry->getVersion() << " wanted version " << version << std::endl;
         thisRegEntry->setVersion( version );
      }
      ops.insertOwnOp( thisEntryOps, global_reg_t( reg, _newRegions->getGlobalDirectoryKey() ), version, _owner.getMemorySpaceId() );
   } else {
      ops.getOtherOps().insert( thisEntryOps );
      //*(myThread->_file) << __FUNCTION__ << " im NOT setting region cache entry version to " << version << " for wd " << wd.getId() << " idx " << copyIdx << std::endl;
   }
   //o << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[X]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"<< std::endl;
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

void AllocatedChunk::setRegionVersion( reg_t reg, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   unsigned int currentVersion = 0;
   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   RegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();
   if ( entry == NULL ) {
      key->printRegion(*myThread->_file, reg);
      *myThread->_file << " not found, this is chunk " << (void*)this << " w/hAddr " << (void*) this->getHostAddress() << " - " << (void*)(this->getHostAddress() +this->getSize() ) << " wd " << wd.getId() << " idx " << copyIdx << std::endl;
      *myThread->_file << "Regions contained: "  << std::endl;
      for ( CacheRegionDictionaryIterator it = _newRegions->begin(); it != _newRegions->end(); it++) {
         *myThread->_file << "Region: " << it->first << " "; _newRegions->printRegion( *myThread->_file, it->first );
         *myThread->_file << " has entry with version " << (( (it->second).getData() ) ? (it->second).getData()->getVersion() : (unsigned int)-1)<< std::endl;
      }
      *myThread->_file << "End of regions contained: "  << std::endl;
   }
   ensure(entry != NULL, "CacheEntry not found!");
   currentVersion = entry->getVersion();
   entry->setVersion( version );
   //*(myThread->_file) << "setRegionVersion current: " << currentVersion << " requested " << version << " wd: " << wd.getId() << " : " << (wd.getDescription() != NULL ? wd.getDescription() : "[no description]" ) << " w index " << copyIdx << std::endl;
   if ( version > currentVersion ) {
      _dirty = true;
   } else if ( version <= currentVersion ) {
      //*(myThread->_file) << "setRegionVersion and not version increase! current: " << currentVersion << " requested " << version << " wd: " << wd.getId() << " : " << (wd.getDescription() != NULL ? wd.getDescription() : "[no description]" ) << " w index " << copyIdx << std::endl;
   }
}

void AllocatedChunk::NEWaddWriteRegion( reg_t reg, unsigned int version, WD const *wd, unsigned int copyIdx ) {
   unsigned int currentVersion = 0;
   std::list< std::pair< reg_t, reg_t > > components;
   _newRegions->registerRegion( reg, components, currentVersion );

   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg );
   if ( !entry ) {
      entry = NEW CachedRegionStatus();
      _newRegions->setRegionData( reg, entry );
   }
   //entry->setDirty();
   if ( entry->getVersion() > version ) {
      *myThread->_file << __func__ << " Warning: Copy @ WD " << wd->getId() << " desc: " << (wd->getDescription() ? wd->getDescription() : "n/a") << " w index " << copyIdx << " is commutative or concurrent. Cache version is " << entry->getVersion() << " wanted version " << version <<std::endl;
   }
   entry->setVersion( version );
   if ( _VERBOSE_CACHE ) { *myThread->_file << "[[[[[[[[[[[[[[[[[[[[ " << __FUNCTION__ << " reg " << reg << " set version " << version << " entry " << (void *)entry << " components size " << components.size() <<" ]]]]]]]]]]]]]]]]]]]]"<< std::endl; }

   _dirty = true;
}

bool AllocatedChunk::invalidate( RegionCache *targetCache, LockedObjects &srcRegions, WD const &wd, unsigned int copyIdx, SeparateAddressSpaceOutOps &invalOps, std::set< global_reg_t > &regionsToRemoveAccess ) {
   bool hard=false;
   //std::ostream &o = *(myThread->_file);
   RegionDirectory::RegionDirectoryKey key = _newRegions->getGlobalDirectoryKey();

   srcRegions.addAndLock( key );

   std::list< std::pair< reg_t, reg_t > > missing;
   unsigned int ver = 0;

   // *myThread->_file << this << " hAddr: " << (void *) this->getHostAddress() << " - " << (void*)(this->getHostAddress() + this->getSize() ) << " single chunk inval of region: ";
   // _allocatedRegion.key->printRegion( *myThread->_file, _allocatedRegion.id );
   // *myThread->_file << std::endl;

   // for ( CacheRegionDictionaryIterator it = _newRegions->begin(); it != _newRegions->end(); it++) {
   //    o << "Region: " << it->first << " "; _newRegions->printRegion( o, it->first ); o << " has entry with version " << (( (it->second).getData() ) ? (it->second).getData()->getVersion() : -1)<< std::endl;
   // }

   _allocatedRegion.key->registerRegion( _allocatedRegion.id, missing, ver );

   //std::set<DeviceOps *> ops;
   //ops.insert( _allocatedRegion.getDeviceOps() );
   //for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missing.begin(); lit != missing.end(); lit++ ) {
   //   global_reg_t data_source( lit->second, key );
   //   ops.insert( data_source.getDeviceOps() );
   //}

   //DirectoryEntryData *dict_entry = RegionDirectory::getDirectoryEntry( *_allocatedRegion.key, _allocatedRegion.id );
   //memory_space_id_t home = (dict_entry->getRootedLocation() == (unsigned int) -1) ? 0 : dict_entry->getRootedLocation();
   CachedRegionStatus *alloc_entry = ( CachedRegionStatus * ) _newRegions->getRegionData( _allocatedRegion.id );
   //bool alloc_entry_not_present = false;

   if (alloc_entry != NULL) {
      regionsToRemoveAccess.insert( _allocatedRegion );
   } /*else {
       alloc_entry_not_present = true;
       }*/

   //*myThread->_file << "Missing pieces are: " << missing.size() << " got ver " << ver << std::endl;

   if ( missing.size() == 1 ) {
      ensure( _allocatedRegion.id == missing.begin()->first, "Wrong region." );
      if ( _allocatedRegion.isLocatedIn( _owner.getMemorySpaceId() ) ) {
         //    regionsToRemoveAccess.insert( _allocatedRegion );
         if ( RegionDirectory::isOnlyLocated( _allocatedRegion.key, _allocatedRegion.id, _owner.getMemorySpaceId() ) ) {
            //*myThread->_file << "AC: has to be copied!, shape = dsrc and Im the only owner!" << std::endl;
            hard = true;
            DeviceOps *thisChunkOps = _allocatedRegion.getHomeDeviceOps( wd, copyIdx );
            if ( thisChunkOps->addCacheOp( /* debug: */ &wd, 3 ) ) {
               DirectoryEntryData *dict_entry = RegionDirectory::getDirectoryEntry( *_allocatedRegion.key, _allocatedRegion.id );
               memory_space_id_t home = (dict_entry->getRootedLocation() == (unsigned int) -1) ? 0 : dict_entry->getRootedLocation();
               invalOps.insertOwnOp( thisChunkOps, _allocatedRegion, alloc_entry->getVersion(), 0 );
               invalOps.addOutOp( home, _owner.getMemorySpaceId(), _allocatedRegion, alloc_entry->getVersion(), NULL, this, wd, copyIdx );
               //alloc_entry->resetVersion();
            } else {
               *myThread->_file << " ERROR: could not add a cache op to my ops!"<<std::endl;
            }
         }
      }
   } else {
      std::map< reg_t, std::set< reg_t > > fragmented_regions;
      for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missing.begin(); lit != missing.end(); lit++ ) {
         DirectoryEntryData *dentry = RegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), lit->first );
         if ( VERBOSE_INVAL ) {
            DirectoryEntryData *dsentry = RegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), lit->second );
            *myThread->_file << (void *)_newRegions << " missing registerReg: " << lit->first << " "; _allocatedRegion.key->printRegion( *myThread->_file, lit->first ); if (!dentry ) { *myThread->_file << " nul "; } else { *myThread->_file << *dentry; } 
            *myThread->_file << "," << lit->second << " "; _allocatedRegion.key->printRegion( *myThread->_file, lit->second ); if (!dsentry ) { *myThread->_file << " nul "; } else { *myThread->_file << *dsentry; }
            *myThread->_file <<  std::endl;
         }
         global_reg_t region_shape( lit->first, key );
         global_reg_t data_source( lit->second, key );
         if ( region_shape.id == data_source.id ) {
            ensure( _allocatedRegion.id != data_source.id, "Wrong region" );
            if ( data_source.isLocatedIn( _owner.getMemorySpaceId() ) ) {
               regionsToRemoveAccess.insert( data_source );
               //if ( RegionDirectory::isOnlyLocated( data_source.key, data_source.id, _owner.getMemorySpaceId() ) )
               if ( ! data_source.isLocatedIn( 0 ) ) { // FIXME: not optimal, but we write metadata to "allocatedRegion" entry so we must copy all to node 0 if its not there to keep it consistent!
                  if ( VERBOSE_INVAL ) {
                     for ( CacheRegionDictionary::citerator pit = _newRegions->begin(); pit != _newRegions->end(); pit++ ) {
                        DirectoryEntryData *d = RegionDirectory::getDirectoryEntry( *(_allocatedRegion.key), pit->first );
                        CachedRegionStatus *c = ( CachedRegionStatus * ) _newRegions->getRegionData( pit->first );
                        *myThread->_file << " reg " << pit->first << " "; key->printRegion( *myThread->_file, pit->first);
                        *myThread->_file << " has entry " << (void *) &pit->second << " CaheVersion: "<< (int)( c!=NULL ? c->getVersion() : (unsigned int)-1) ;
                        if ( d ) *myThread->_file << *d << std::endl;
                        else *myThread->_file << " n/a " << std::endl;
                     }
                  }
                  DeviceOps * fragment_ops = data_source.getHomeDeviceOps( wd, copyIdx );
                  CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( data_source.id );
                  if ( VERBOSE_INVAL ) { *myThread->_file << data_source.id << " has to be copied!, shape = dsrc and Im the only owner! "<< (void *)entry << std::endl; }
                  unsigned int version;
                  if ( entry ) {
                     version = entry->getVersion();
                     //entry->resetVersion();
                  } else {
                     version = RegionDirectory::getVersion( data_source.key, data_source.id, false );
                  }
                  hard = true;
                  if ( fragment_ops->addCacheOp( /* debug: */ &wd, 4 ) ) {
                     invalOps.insertOwnOp( fragment_ops, data_source, version, 0 );
                     DirectoryEntryData *dict_entry = RegionDirectory::getDirectoryEntry( *data_source.key, data_source.id );
                     memory_space_id_t home = (dict_entry->getRootedLocation() == (unsigned int) -1) ? 0 : dict_entry->getRootedLocation();
                     invalOps.addOutOp( home, _owner.getMemorySpaceId(), data_source, version, NULL, this, wd, copyIdx );
                  } else {
                     invalOps.getOtherOps().insert( fragment_ops );
                     // make sure the op we are waiting for its the same that we want to do, maybe it is impossible to reach this code
                     *myThread->_file << "FIXME " << __FUNCTION__ << " this is memspace "<< _owner.getMemorySpaceId() << std::endl;
                  }
               } else {
                  // SYNC
                  DeviceOps * fragment_ops = data_source.getHomeDeviceOps( wd, copyIdx );
                  invalOps.getOtherOps().insert( fragment_ops );
               }
            } else {
               DeviceOps * fragment_ops = data_source.getHomeDeviceOps( wd, copyIdx );
               invalOps.getOtherOps().insert( fragment_ops );
            }
         } else {

            if ( dentry == NULL || data_source.getVersion() > region_shape.getVersion() ) {
               // region_shape region is not registered or old
               if ( RegionDirectory::isOnlyLocated( data_source.key, data_source.id, _owner.getMemorySpaceId() ) ) {
                  fragmented_regions[ data_source.id ].insert( region_shape.id );
               }
            } else {
               //region_shape has a valid entry and the version is equal or greater than the one provided by the data_source region
               if ( region_shape.isLocatedIn( _owner.getMemorySpaceId() ) ) {
                  regionsToRemoveAccess.insert( region_shape );
                  if ( RegionDirectory::isOnlyLocated( region_shape.key, region_shape.id, _owner.getMemorySpaceId() ) ) {
                     //emit an op for this region
                     DeviceOps * fragment_ops = region_shape.getHomeDeviceOps( wd, copyIdx );
                     if ( fragment_ops->addCacheOp( /* debug: */ &wd, 44 ) ) {
                        invalOps.insertOwnOp( fragment_ops, region_shape, region_shape.getVersion(), 0 );
                        memory_space_id_t home = (dentry->getRootedLocation() == (unsigned int) -1) ? 0 : dentry->getRootedLocation();
                        invalOps.addOutOp( home, _owner.getMemorySpaceId(), region_shape, region_shape.getVersion(), fragment_ops, this, wd, copyIdx );
                     } else {
                        invalOps.getOtherOps().insert( fragment_ops );
                     }
                  } else {
                     DeviceOps * fragment_ops = region_shape.getHomeDeviceOps( wd, copyIdx );
                     invalOps.getOtherOps().insert( fragment_ops );
                  }
               } else {
                  DeviceOps * fragment_ops = region_shape.getHomeDeviceOps( wd, copyIdx );
                  invalOps.getOtherOps().insert( fragment_ops );
               }
            }



            // CachedRegionStatus *c_ds_entry = ( CachedRegionStatus * ) _newRegions->getRegionData( data_source.id );
            // if ( c_ds_entry != NULL && 
            //       ( dentry == NULL ||
            //         ( data_source.getVersion() <= region_shape.getVersion() && RegionDirectory::isOnlyLocated( region_shape.key, region_shape.id, _owner.getMemorySpaceId() ) ) ||
            //         ( data_source.getVersion() >  region_shape.getVersion() && RegionDirectory::isOnlyLocated( data_source.key,  data_source.id,  _owner.getMemorySpaceId() ) )
            //       )
            //    ) {
            //    //*myThread->_file << "added fragmented region: rs " << region_shape.id << " , w ds " << data_source.id << " c_ds_entry " << c_ds_entry << " dentry " << dentry /*<< " data_source.getVersion() "<< data_source.getVersion() << " region_shape.getVersion() " << region_shape.getVersion() << " isLin " << RegionDirectory::isOnlyLocated( region_shape.key, region_shape.id, _owner.getMemorySpaceId() ) << " isLin2 " <<  RegionDirectory::isOnlyLocated( data_source.key,  data_source.id,  _owner.getMemorySpaceId() ) */<< std::endl;
            //    fragmented_regions[ data_source.id ].insert( region_shape.id );
            // } else {
            //    DirectoryEntryData *d_ds_entry = RegionDirectory::getDirectoryEntry( *data_source.key, data_source.id );
            //    if ( d_ds_entry != NULL && RegionDirectory::isOnlyLocated( data_source.key,  data_source.id,  _owner.getMemorySpaceId() ) ) {
            //       fragmented_regions[ data_source.id ].insert( region_shape.id );
            //    } else {
            //       //*myThread->_file << "ignored fragmented region: rs " << region_shape.id << " , w ds " << data_source.id << " c_ds_entry " << c_ds_entry << " dentry " << dentry /*<< " data_source.getVersion() "<< data_source.getVersion() << " region_shape.getVersion() " << region_shape.getVersion() << " isLin " << RegionDirectory::isOnlyLocated( region_shape.key, region_shape.id, _owner.getMemorySpaceId() ) << " isLin2 " <<  RegionDirectory::isOnlyLocated( data_source.key,  data_source.id,  _owner.getMemorySpaceId() ) */ << std::endl;
            //    }
            // }
         }
      }

      for ( std::map< reg_t, std::set< reg_t > >::iterator mit = fragmented_regions.begin(); mit != fragmented_regions.end(); mit++ ) {
         if ( VERBOSE_INVAL ) { *myThread->_file << " fragmented region " << mit->first << " has #chunks " << mit->second.size() << std::endl; }
         global_reg_t data_source( mit->first, key );
         regionsToRemoveAccess.insert( data_source );
         //if ( RegionDirectory::isOnlyLocated( key, data_source.id, _owner.getMemorySpaceId() ) )
         if ( ! data_source.isLocatedIn( 0 ) ) { // FIXME: not optimal, but we write metadata to "allocatedRegion" entry so we must copy all to node 0 if its not there to keep it consistent!
            bool subChunkInval = false;
            CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( data_source.id );
            if ( VERBOSE_INVAL ) { *myThread->_file << "data source is " << data_source.id << " with entry "<< entry << std::endl; }

            for ( std::set< reg_t >::iterator sit = mit->second.begin(); sit != mit->second.end(); sit++ ) {
               if ( VERBOSE_INVAL ) { *myThread->_file << "    this is region " << *sit << std::endl; }
               global_reg_t subReg( *sit, key );
               DirectoryEntryData *dentry = RegionDirectory::getDirectoryEntry( *key, *sit );
               if ( dentry == NULL ) { //FIXME: maybe we need a version check to handle when the dentry exists but is old?
                  std::list< std::pair< reg_t, reg_t > > missingSubReg;
                  _allocatedRegion.key->registerRegion( subReg.id, missingSubReg, ver );
                  for ( std::list< std::pair< reg_t, reg_t > >::iterator lit = missingSubReg.begin(); lit != missingSubReg.end(); lit++ ) {
                     global_reg_t region_shape( lit->first, key );
                     global_reg_t new_data_source( lit->second, key );
                     if ( VERBOSE_INVAL ) { *myThread->_file << " DIR CHECK WITH FRAGMENT: "<< lit->first << " - " << lit->second << " " << std::endl; }

                     DirectoryEntryData *subEntry = RegionDirectory::getDirectoryEntry( *key, region_shape.id );
                     DirectoryEntryData *subDSEntry = RegionDirectory::getDirectoryEntry( *key, new_data_source.id );
                     if ( !subEntry ) {
                        //*myThread->_file << "FIXME: Invalidation, and found a region shape (" << lit->first << ") with no entry, a new Entry may be needed." << std::endl;
                        //DirectoryEntryData *subEntryData = RegionDirectory::getDirectoryEntry( *key, lit->second );
                        //this->prepareRegion( lit->first, subEntryData->getVersion() );
                     } else if ( VERBOSE_INVAL ) {
                        *myThread->_file << " Fragment " << lit->first << " has entry! " << subEntry << std::endl;
                     }
                     if ( new_data_source.id == data_source.id || RegionDirectory::isOnlyLocated( key, new_data_source.id, _owner.getMemorySpaceId() ) ) {
                        subChunkInval = true;
                        if ( VERBOSE_INVAL ) { *myThread->_file << " COPY subReg " << lit->first << " comes from subreg "<< subReg.id << " new DS " << new_data_source.id << std::endl; }
                        DeviceOps *thisChunkOps = _allocatedRegion.getHomeDeviceOps( wd, copyIdx );
                        memory_space_id_t home = (subDSEntry->getRootedLocation() == (unsigned int) -1) ? 0 : subDSEntry->getRootedLocation();
                        invalOps.addOutOp( home, _owner.getMemorySpaceId(), region_shape, subDSEntry->getVersion(), thisChunkOps, this, wd, copyIdx );
                     }
                  }
               } else {
                  DeviceOps *subRegOps = subReg.getHomeDeviceOps( wd, copyIdx );
                  hard = true;
                  if ( subRegOps->addCacheOp( /* debug: */ &wd, 5 ) ) {
                     DirectoryEntryData *dsentry = RegionDirectory::getDirectoryEntry( *key, data_source.id );
                     regionsToRemoveAccess.insert( subReg );
                     invalOps.insertOwnOp( subRegOps, data_source, dsentry->getVersion(), 0 );
                     memory_space_id_t home = (dsentry->getRootedLocation() == (unsigned int) -1) ? 0 : dsentry->getRootedLocation();
                     invalOps.addOutOp( home, _owner.getMemorySpaceId(), subReg, dsentry->getVersion(), subRegOps, this, wd, copyIdx );
                  } else {
                     invalOps.getOtherOps().insert( subRegOps );
                     *myThread->_file << "FIXME " << __FUNCTION__ << std::endl;
                  }
               }
            }

            if ( subChunkInval ) {
               //FIXME I think this is wrong, can potentially affect regions that are not there, 
               unsigned int version;
               if ( entry ) {
                  version = entry->getVersion();
                  //entry->resetVersion();
               } else {
                  version = RegionDirectory::getVersion( data_source.key, data_source.id, false );
               }
               hard = true;
               DeviceOps *thisChunkOps = _allocatedRegion.getHomeDeviceOps( wd, copyIdx );
               if ( thisChunkOps->addCacheOp( /* debug: */ &wd, 6 ) ) { // FIXME: others may believe there's an ongoing op for the full region!
                  invalOps.insertOwnOp( thisChunkOps, data_source, version, 0 );
               } else {
                  //it could have been added on a previous iteration
                  *myThread->_file << "ERROR, could not add an inval cache op " << std::endl;
               }
            }
            //entry->resetVersion();
         } else {
            // SYNC
            DeviceOps * fragment_ops = data_source.getHomeDeviceOps( wd, copyIdx );
            invalOps.getOtherOps().insert( fragment_ops );
         }
      }

      if ( alloc_entry != NULL ) {
         if ( _allocatedRegion.isLocatedIn( _owner.getMemorySpaceId() ) ) {
            //if ( RegionDirectory::isOnlyLocated( _allocatedRegion.key, _allocatedRegion.id, _owner.getMemorySpaceId() ) )
            if ( ! _allocatedRegion.isLocatedIn( 0 ) ) { // FIXME: not optimal, but we write metadata to "allocatedRegion" entry so we must copy all to node 0 if its not there to keep it consistent!
               hard = true;
               DeviceOps *thisChunkOps = _allocatedRegion.getHomeDeviceOps( wd, copyIdx );
               if ( thisChunkOps->addCacheOp( /* debug: */ &wd, 7 ) ) {
                  invalOps.insertOwnOp( thisChunkOps, _allocatedRegion, alloc_entry->getVersion(), 0 );
                  //alloc_entry->resetVersion();
               }
            } else {
               *myThread->_file << " ERROR: could not add a cache op to my ops!"<<std::endl;
            }
         }
      }
   }

   //*(myThread->_file) << "=============> " << "Cache " << _owner.getMemorySpaceId() << ( hard ? " hard":" soft" ) <<" Invalidate region "<< (void*) key << ":" << _allocatedRegion.id << " reg: "; _allocatedRegion.key->printRegion(*(myThread->_file), _allocatedRegion.id ); *(myThread->_file) << std::endl;
   return hard;

}

AllocatedChunk **RegionCache::selectChunkToInvalidate( std::size_t allocSize ) {
   AllocatedChunk **allocChunkPtrPtr = NULL;
   MemoryMap<AllocatedChunk>::iterator it;
   bool done = false;
   int count = 0;
   //for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
   //   *myThread->_file << "["<< count << "] this chunk: " << ((void *) it->second) << " refs: " << (int)( (it->second != NULL) ? it->second->getReferenceCount() : -1 ) << " dirty? " << (int)( (it->second != NULL) ? it->second->isDirty() : -1 )<< std::endl;
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
      //    *myThread->_file << "["<< count << "] mmm this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << " size is " << it->second->getSize() << " vs " << allocSize << " ";
      //    reg.key->printRegion( reg.id );
      //    *myThread->_file << std::endl;
      // }
      if ( it->second != NULL && (it->second != (AllocatedChunk *) -1)
            && (it->second != (AllocatedChunk *) -2)
            && it->second->getReferenceCount() == 0
            && !(it->second->isRooted())
            && it->second->getSize() == allocSize ) {
         if ( !it->second->isDirty() ) {
            if ( _lruTime == it->second->getLruStamp() ) {
               //*myThread->_file << "["<< count << "] this chunk: " << ((void *) it->second) << std::endl;
               chunkToReusePtr = &(it->second);
               done = true;
               break;
            } else if ( chunkToReuseNoLruPtr == NULL ) {
               chunkToReuseNoLruPtr = &(it->second);
               itNoLru = it;
            }
         } else {
            if ( _lruTime == it->second->getLruStamp() ) {
               //*myThread->_file << "["<< count << "] this chunk: " << ((void *) it->second) << std::endl;
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
         //*myThread->_file << "LRU clean chunk"<< std::endl;
         chunkToReusePtr = chunkToReuseNoLruPtr;
         done = true;
         it = itNoLru;
         increaseLruTime();
      } else if ( chunkToReuseDirtyPtr != NULL ) {
         //*myThread->_file << "Dirty chunk"<< std::endl;
         chunkToReusePtr = chunkToReuseDirtyPtr;
         it = itDirty;
         done = true;
      } else if ( chunkToReuseDirtyNoLruPtr != NULL ) {
         //*myThread->_file << "LRU Dirty chunk"<< std::endl;
         chunkToReusePtr = chunkToReuseDirtyNoLruPtr;
         it = itDirtyNoLru;
         done = true;
         increaseLruTime();
      }
   } else {
      //*myThread->_file << "clean chunk"<< std::endl;
   }
   if ( done ) {
      allocChunkPtrPtr = chunkToReusePtr;
      if ( _VERBOSE_CACHE ) { fprintf(stderr, "[%s] Thd %d Im cache with id %d, I've found a chunk to free, %p (locked? %d) region %d addr=%p size=%zu\n",  __FUNCTION__, myThread->getId(), _memorySpaceId, *allocChunkPtrPtr, ((*allocChunkPtrPtr)->locked()?1:0), (*allocChunkPtrPtr)->getAllocatedRegion().id, (void*)it->first.getAddress(), it->first.getLength()); }
      //(*allocChunkPtrPtr)->lock();
   } else {
      // if ( VERBOSE_INVAL ) {
      //    count = 0;
      //    for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
      //       if ( it->second == NULL ) *myThread->_file << "["<< count << "] this chunk: null chunk" << std::endl;
      //       else *myThread->_file << "["<< count << "] this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << " size is " << it->second->getSize() << " vs " << allocSize << " " << " dirty? " << it->second->isDirty() << std::endl;
      //       count++;
      //    }
      // }
      //fatal("IVE _not_ FOUND A CHUNK TO FREE");
      allocChunkPtrPtr = NULL;
   }
   return allocChunkPtrPtr;
}

void RegionCache::selectChunksToInvalidate( std::size_t allocSize, std::set< std::pair< AllocatedChunk **, AllocatedChunk * > > &chunksToInvalidate, WD const &wd, unsigned int &otherReferencedChunks ) {
   //for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
   //   *myThread->_file << "["<< count << "] this chunk: " << ((void *) it->second) << " refs: " << (int)( (it->second != NULL) ? it->second->getReferenceCount() : -1 ) << " dirty? " << (int)( (it->second != NULL) ? it->second->isDirty() : -1 )<< std::endl;
   //   count++;
   //}
   //count = 0;
   otherReferencedChunks = 0;
   if ( VERBOSE_INVAL ) {
      *myThread->_file << __FUNCTION__ << " with size " << allocSize << std::endl;
   }
   if ( /*_device.supportsFreeSpaceInfo() */ true ) {
      MemoryMap<AllocatedChunk>::iterator it;
      bool done = false;
      MemoryMap< uint64_t > device_mem;

      for ( it = _chunks.begin(); it != _chunks.end() && !done; it++ ) {
         // if ( it->second != NULL ) {
         //    global_reg_t reg = it->second->getAllocatedRegion();
         //    *myThread->_file << "### this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << " size is " << it->second->getSize() << " vs " << allocSize << " ";
         //    reg.key->printRegion( *myThread->_file, reg.id );
         //    *myThread->_file << std::endl;
         // }
         if ( it->second != NULL && it->second != (AllocatedChunk *) -1 && (it->second != (AllocatedChunk *) -2) ) {
            AllocatedChunk &c = *(it->second);
            AllocatedChunk **chunk_at_map_ptr = &(it->second);
            if ( it->second->getReferenceCount() == 0 && !(it->second->isRooted()) ) {
               device_mem.addChunk( c.getAddress(), c.getSize(), (uint64_t) chunk_at_map_ptr );
            } else {
               bool mine = false;
               for (unsigned int idx = 0; idx < wd.getNumCopies() && !mine ; idx += 1) {
                  mine = ( wd._mcontrol._memCacheCopies[ idx ]._chunk == &c );
               }
               otherReferencedChunks += mine ? 0 : 1;
            }
         } else if ( it->second == (AllocatedChunk *) -1 ) {
            otherReferencedChunks += 1;
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
         *myThread->_file << "I can invalidate a set of these:" << std::endl;
         for ( devIt = device_mem.begin(); devIt != device_mem.end(); devIt++ ) {
            *myThread->_file << "Addr: " << (void *) devIt->first.getAddress() << " size: " << devIt->first.getLength() ;
            if ( devIt->second == 0 ) {
               *myThread->_file << " [free chunk] "<< std::endl;
            } else {
               *myThread->_file << " " << (void *) *((AllocatedChunk **) devIt->second) << std::endl;
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
         if ( VERBOSE_INVAL ) {
            *myThread->_file << "Im going to invalidaet from " << (void *) selectedIt->first.getAddress() << std::endl;
         }
         
         for ( std::size_t len = selectedIt->first.getLength(); len < allocSize; selectedIt++ ) {
            if ( selectedIt->second != 0 ) {
               AllocatedChunk **this_chunk = (AllocatedChunk **) selectedIt->second;
               chunksToInvalidate.insert( std::make_pair( this_chunk, *this_chunk ) );
            }
            len += selectedIt->first.getLength();
         }
      } else {
         // *myThread->_file << " failed to invalidate " << std::endl;
         // printReferencedChunksAndWDs();
      }
   }
}

unsigned int AllocatedChunk::getVersion( global_reg_t const &reg ) {
   unsigned int version = 0;
   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg.id );
   if ( entry ) {
      version = entry->getVersion();
   }
   return version;
}

DeviceOps *AllocatedChunk::getDeviceOps( global_reg_t const &reg, WD const *wd, unsigned int idx ) {
   CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg.id );
   std::ostream &o = *(myThread->_file);
   if ( entry == NULL ) {
      *myThread->_file << "wd " << wd->getId() << " w/cIdx " << idx << ": Not found entry for chunk " << (void *) this << " w/hAddr " << (void *)this->getHostAddress() << " - " << (void *) (this->getHostAddress() + this->getSize()) << " ";
      reg.key->printRegion(*myThread->_file, reg.id); *myThread->_file << std::endl;
      printBt(*(myThread->_file) );
      for ( CacheRegionDictionaryIterator it = _newRegions->begin(); it != _newRegions->end(); it++) {
         o << "Region: " << it->first << " "; _newRegions->printRegion( o, it->first );
         o << " has entry with version " << (( (it->second).getData() ) ? (it->second).getData()->getVersion() : (unsigned int)-1)<< std::endl;
      }
   }
   ensure(entry != NULL, "CacheEntry not found!");
   return entry->getDeviceOps();
}

void AllocatedChunk::printReferencingWDs() const {
   *(myThread->_file) << "Addr: " << (void*)_hostAddress << " Size: " << _size << " Referencing WDs: [";
   for ( std::map<WD const *, unsigned int>::const_iterator it = _refWdId.begin(); it != _refWdId.end(); it++ ) {
      if ( it->second != 0 ) {
         WD const &wd = *it->first;
         std::map<int, std::set<int> >::const_iterator itLoc = _refLoc.find( wd.getId() );
         *(myThread->_file) << "(wd: " << wd.getId() << " desc: " << ( wd.getDescription() != NULL ? wd.getDescription() : "n/a" )  << " count: " << it->second <<" loc: {";
         for (std::set<int>::const_iterator sIt = itLoc->second.begin(); sIt != itLoc->second.end(); sIt++ ) {
            *(myThread->_file) << *sIt << " ";
         }
         *(myThread->_file) << "}";
      }
   }
   *(myThread->_file) << "]" << std::endl;
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

   _chunks.getOrAddChunkDoNotFragment( targetHostAddr, allocSize, results );
   if ( results.size() != 1 ) {
      message0( "Got results.size()="<< results.size() << " for addr " << ((void*) targetHostAddr) << " with allocSize " << allocSize <<" I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ChunkList::iterator it = results.begin(); it != results.end(); it++ )
         *myThread->_file << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
      *myThread->_file << "Realloc needed. Caused by wd " << (wd.getDescription() ? wd.getDescription() : "n/a") << " copy index " << copyIdx << std::endl;
      fatal("Can not continue.");
   } else {
      if ( *(results.front().second) == NULL ) {

         if ( VERBOSE_DEV_OPS ) {
            *(myThread->_file) << "[" << myThread->getId() << "] "<< __FUNCTION__ << " _device(" << _device.getName() << ")._memAllocate( memspace=" << _memorySpaceId <<", allocSize="<< allocSize << ", wd="<< wd.getId() << " ["<< (wd.getDescription() != NULL ? wd.getDescription() : "no description") << "], copyIdx="<< copyIdx << " );";
         }
         void *deviceMem = _device.memAllocate( allocSize, sys.getSeparateMemory( _memorySpaceId ), &wd, copyIdx );
         if ( VERBOSE_DEV_OPS ) {
            *(myThread->_file) << " returns " << (void *) deviceMem << std::endl;
         }
         if ( deviceMem != NULL ) {
            _allocatedBytes += allocSize;
            *(results.front().second) = NEW AllocatedChunk( *this, (uint64_t) deviceMem, results.front().first->getAddress(), results.front().first->getLength(), allocatedRegion, reg.getRootedLocation() == this->getMemorySpaceId() );
            allocChunkPtr = *(results.front().second);
            this->addToAllocatedRegionMap( allocatedRegion );
            //*(results.front().second) = allocChunkPtr;
         } else {
            // I have not been able to allocate a chunk, just return NULL;
         }
      } else if ( *(results.front().second) == (AllocatedChunk *) -1 || (*(results.front().second) == (AllocatedChunk *) -2) ) {
         //being invalidated.
      } else {
         //*(myThread->_file) << " CHUNK 1 AND NOT NULL! wd: " << wd.getId() << " copy " << copyIdx << " asked for "<< (void *)targetHostAddr << " with size " << (unsigned int) allocSize << " got addr " << (void *) results.front().first->getAddress() << " with size " << (unsigned int) results.front().first->getLength() << " entry is " << (void *) *(results.front().second)<< std::endl;
         if ( results.front().first->getAddress() <= targetHostAddr ) {
            if ( results.front().first->getLength() + results.front().first->getAddress() >= (targetHostAddr + allocSize) ) {
               allocChunkPtr = *(results.front().second);
            } else {
               *myThread->_file << "I need a realloc of an allocated chunk!" << std::endl;
            }
         }
      }
   }
   if ( allocChunkPtr != NULL ) {
      if ( allocChunkPtr->trylock() ) {
         allocChunkPtr->addReference( wd , 6); //tryGetAddress
      } else {
         allocChunkPtr = NULL;
      }
   }
   return allocChunkPtr;
}

AllocatedChunk *RegionCache::invalidate( LockedObjects &srcRegions, InvalidationController &invalControl, global_reg_t const &allocatedRegion, WD const &wd, unsigned int copyIdx ) {
   AllocatedChunk *allocChunkPtr = NULL;
   AllocatedChunk **allocChunkPtrPtr = NULL;
   NANOS_INSTRUMENT( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT( static nanos_event_key_t key = ID->getEventKey("cache-evict"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) wd.getId() ); )
   invalControl._allocatedRegion = allocatedRegion;
   //std::set< global_reg_t > regions_to_remove_access;

   //reg.key->invalLock();
   std::set< AllocatedChunk ** > chunks_to_invalidate;

   allocChunkPtrPtr = selectChunkToInvalidate( allocatedRegion.getDataSize() );
   if ( allocChunkPtrPtr != NULL ) {
      allocChunkPtr = *allocChunkPtrPtr;
      invalControl._invalOps = NEW SeparateAddressSpaceOutOps( myThread->runningOn(), true, true );
      invalControl._chunksToInval.insert( std::make_pair( allocChunkPtrPtr, allocChunkPtr ) );
      allocChunkPtr->addReference( wd, 22 ); //invalidate, single chunk
      bool hard_inval = allocChunkPtr->invalidate( this, srcRegions, wd, copyIdx, *invalControl._invalOps, invalControl._regions_to_remove_access );
      if ( hard_inval ) {
         invalControl._hardInvalidationCount++;
      } else {
         invalControl._softInvalidationCount++;
      }
      // *(myThread->_file) << "[" << myThread->getId() << "] single chunk invalidation ( "<<
      //    (hard_inval ? "hard" : "soft" ) <<" ):  memspace=" << _memorySpaceId <<
      //    ", neededSize="<< allocatedRegion.getDataSize() << ", selectedDevAddr=" <<
      //    (void*)allocChunkPtr->getAddress() << ", chunkSize=" << allocChunkPtr->getSize() <<
      //    ", wd="<< wd.getId() << " ["<< (wd.getDescription() != NULL ? wd.getDescription() : "no description") <<
      //    "], copyIdx="<< copyIdx << ", reg " << allocatedRegion.id << " chunk " <<
      //    (void*)(*allocChunkPtrPtr) << " will use for allocRegion "<<
      //    (void*)allocatedRegion.getRealFirstAddress() << " - " << (void *)(allocatedRegion.getRealFirstAddress() + allocatedRegion.getBreadth()) << std::endl;
      if ( VERBOSE_DEV_OPS ) {
         *(myThread->_file) << "[" << myThread->getId() << "] single chunk invalidation ( "<< (hard_inval ? "hard" : "soft" ) <<" ):  memspace=" << _memorySpaceId <<", neededSize="<< allocatedRegion.getDataSize() << ", selectedDevAddr=" << (void*)allocChunkPtr->getAddress() << ", chunkSize=" << allocChunkPtr->getSize() << ", wd="<< wd.getId() << " ["<< (wd.getDescription() != NULL ? wd.getDescription() : "no description") << "], copyIdx="<< copyIdx << ", reg " << allocatedRegion.id << " chunk " << (void*)(*allocChunkPtrPtr) << std::endl;
      }
   } else {
         //*(myThread->_file) << "[" << myThread->getId() << "] multi chunk invalidation test:  memspace=" << _memorySpaceId <<", neededSize="<< allocatedRegion.getDataSize() << ", wd="<< wd.getId() << " ["<< (wd.getDescription() != NULL ? wd.getDescription() : "no description") << "], copyIdx="<< copyIdx << std::endl;
      //try to invalidate a set of chunks
      unsigned int other_referenced_chunks = 0;
      selectChunksToInvalidate( allocatedRegion.getDataSize(), invalControl._chunksToInval, wd, other_referenced_chunks );
      if ( invalControl._chunksToInval.empty() ) {
         if ( other_referenced_chunks == 0 ) {
            printReferencedChunksAndWDs();
            *(myThread->_file) << "---" << std::endl;
      _chunks.print( *myThread->_file );
         fatal("Unable to free enough space to allocate task data, probably a fragmentation issue. Try increasing the available device memory.");
         } else {
            // *(myThread->_file) << "Unable to invalidate using selectChunksToInvalidate, wd: " << wd.getId() << " other_referenced_chunks: " << other_referenced_chunks << std::endl;
            // printReferencedChunksAndWDs();
            // *(myThread->_file) << "---" << std::endl;
            // _chunks.print( *myThread->_file );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 ); )
            return NULL;
         }
      }
      if ( VERBOSE_INVAL ) {
         *(myThread->_file) << "[" << myThread->getId() << "] multi chunk invalidation:  memspace=" << _memorySpaceId <<", neededSize="<< allocatedRegion.getDataSize() << ", wd="<< wd.getId() << " ["<< (wd.getDescription() != NULL ? wd.getDescription() : "no description") << "], copyIdx="<< copyIdx << std::endl;
      }
      invalControl._invalOps = NEW SeparateAddressSpaceOutOps( myThread->runningOn(), true, true );
      for ( std::set< std::pair< AllocatedChunk ** , AllocatedChunk * > >::iterator it = invalControl._chunksToInval.begin(); it != invalControl._chunksToInval.end(); it++ ) {
         AllocatedChunk **chunkPtr = it->first;
         AllocatedChunk *chunk = *chunkPtr;
         chunk->addReference( wd, 23 ); //invalidate, multi chunk
         if ( chunk->invalidate( this, srcRegions, wd, copyIdx, *invalControl._invalOps, invalControl._regions_to_remove_access ) ) {
            invalControl._hardInvalidationCount++;
         } else {
            invalControl._softInvalidationCount++;
         }
         //if ( VERBOSE_DEV_OPS ) {
         //   *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ").memFree(  memspace=" << _memorySpaceId <<", devAddr="<< (void *)chunk->getAddress() << ", wd="<< wd.getId() << " ["<< (wd.getDescription() != NULL ? wd.getDescription() : "no description") << "], copyIdx="<< copyIdx << " );" << std::endl;
         //}
         //_device.memFree( chunk->getAddress(), sys.getSeparateMemory( _memorySpaceId ) );
         invalControl._chunksToFree.insert( chunk );
      }
   }

   invalControl._invalChunk = allocChunkPtr;

   // should be done after inval
#if 1
   for ( std::set< std::pair< AllocatedChunk **, AllocatedChunk * > >::iterator it = invalControl._chunksToInval.begin(); it != invalControl._chunksToInval.end(); it++ ) {
      //FIXME do we need to call removeChunk? I think so
      // AllocatedChunk * chunk = *(*it);
      // *myThread->_file << "inval due to wd " << wd.getId() << ", idx " << copyIdx << " chunk entry holding chunk " << (void *) chunk << " w/hAddr " << (void*)chunk->getHostAddress() << " - " << (void*)(chunk->getHostAddress() + chunk->getSize()) << " now set to null, map entry addr " << (void *) *it << std::endl;
      *(it->first) = (AllocatedChunk *) -1;
   }
#endif

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, 0 ); )

   return allocChunkPtr;
}

AllocatedChunk *RegionCache::getOrCreateChunk( LockedObjects &srcRegions, global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   ChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   global_reg_t allocatedRegion;

   std::size_t allocSize = 0;
   uint64_t targetHostAddr = 0;

   getAllocatableRegion( reg, allocatedRegion );

   targetHostAddr = allocatedRegion.getRealFirstAddress();
   allocSize      = allocatedRegion.getDataSize();

  //*myThread->_file << "-----------------------------------------" << std::endl;
  //*myThread->_file << " Max " << cd.getMaxSize() << std::endl;
  //*myThread->_file << "WIDE targetHostAddr: "<< ((void *)cd.getBaseAddress()) << std::endl;
  //*myThread->_file << "WIDE allocSize     : "<< cd.getMaxSize() << std::endl;
  //*myThread->_file << "FIT  targetHostAddr: "<< ((void *)cd.getFitAddress()) << std::endl;
  //*myThread->_file << "FIT  allocSize     : "<< cd.getFitSize() << std::endl;
  //*myThread->_file << "-----------------------------------------" << std::endl;
  //
  //*myThread->_file << "Requesting a chunk with targetHostAddr=" << (void *) targetHostAddr << " and size=" << allocSize << " allocRegionId " << allocatedRegion.id << " "; allocatedRegion.key->printRegion( allocatedRegion.id ); *myThread->_file << std::endl;

   _chunks.getOrAddChunkDoNotFragment( targetHostAddr, allocSize, results );
   if ( results.size() != 1 ) {
      message0( "Got results.size()="<< results.size() << " I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ChunkList::iterator it = results.begin(); it != results.end(); it++ )
         *myThread->_file << " addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
   } else {
      //if ( results.front().first->getAddress() != targetHostAddr || results.front().first->getLength() < allocSize ) {
      //   *myThread->_file << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ERROR, realloc needed>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      //}
      if ( *(results.front().second) == NULL ) {

         if ( VERBOSE_DEV_OPS ) {
            *(myThread->_file) << "[" << myThread->getId() << "] "<< __FUNCTION__ << " _device(" << _device.getName() << ")._memAllocate( memspace=" << _memorySpaceId <<", allocSize="<< allocSize << ", wd="<< wd.getId() << " ["<< (wd.getDescription() != NULL ? wd.getDescription() : "no description") << "], copyIdx="<< copyIdx << " );";
         }
         void *deviceMem = _device.memAllocate( allocSize, sys.getSeparateMemory( _memorySpaceId ), &wd, copyIdx );
         if ( VERBOSE_DEV_OPS ) {
            *(myThread->_file) << " returns " << (void *) deviceMem << std::endl;
         }
         if ( deviceMem == NULL ) {
            /* Invalidate */
            //AllocatedChunk *invalidated_chunk = invalidate( wd._mcontrol._memCacheCopies[copyIdx], allocatedRegion, wd, copyIdx );
            invalidate( srcRegions, wd._mcontrol._memCacheCopies[copyIdx]._invalControl, allocatedRegion, wd, copyIdx );
            //*(myThread->_file) << " allocatedRegion " ; allocatedRegion.key->printRegion(*(myThread->_file), allocatedRegion.id); *(myThread->_file) << " set chunkPtr to "  << results.front().second << std::endl;
            if ( wd._mcontrol._memCacheCopies[copyIdx]._invalControl.isInvalidating() ) {
               *(results.front().second) = (AllocatedChunk *) -2;
               wd._mcontrol._memCacheCopies[copyIdx]._invalControl._invalChunkPtr = results.front().second;
            } else {
               //*myThread->_file << " failed to invalidate: " << std::endl;
               //printReferencedChunksAndWDs();
            }
         } else {
            _allocatedBytes += allocSize;
            *(results.front().second) = NEW AllocatedChunk( *this, (uint64_t) deviceMem, results.front().first->getAddress(), results.front().first->getLength(), allocatedRegion, reg.getRootedLocation() == this->getMemorySpaceId() );
            allocChunkPtr = *(results.front().second);
            this->addToAllocatedRegionMap( allocatedRegion );
            allocChunkPtr->addReference( wd, 4 ); //getOrCreateChunk, invalidated
            allocChunkPtr->lock();
            //*(results.front().second) = allocChunkPtr;
         }
      } else if ( *(results.front().second) == (AllocatedChunk *) -1 || (*(results.front().second) == (AllocatedChunk *) -2) ) {
         //being invalidated
      } else {
         if ( results.front().first->getAddress() <= targetHostAddr ) {
            if ( results.front().first->getLength() + results.front().first->getAddress() >= (targetHostAddr + allocSize) ) {
               allocChunkPtr = *(results.front().second);
               allocChunkPtr->addReference( wd, 5 ); //getOrCreateChunk, hit
               allocChunkPtr->lock();
            } else {
               *myThread->_file << "I need a realloc of an allocated chunk!" << std::endl;
            }
         }
      }
   }
   return allocChunkPtr;
}

AllocatedChunk *RegionCache::getAddress( uint64_t hostAddr, std::size_t len ) {
   ConstChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   _chunks.getChunk( hostAddr, len, results );
   if ( results.size() != 1 ) {
         *myThread->_file <<"Requested addr " << (void *) hostAddr << " size " <<len << std::endl;
      message0( "I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ConstChunkList::iterator it = results.begin(); it != results.end(); it++ )
         *myThread->_file << " addr: " << (void *) it->first.getAddress() << " size " << it->first.getLength() << std::endl; 
   } else {
      if ( results.front().second == NULL ) {
         message0("Address not found in cache, Error!! ");
      } else {
         allocChunkPtr = results.front().second;
      }
   }
   if ( allocChunkPtr == NULL ) *myThread->_file << "WARNING: null RegionCache::getAddress()" << std::endl; 
   allocChunkPtr->lock();
   return allocChunkPtr;
}

//AllocatedChunk *RegionCache::getAndReferenceAllocatedChunk( global_reg_t const &reg, WD const *wd, unsigned int copyIdx ) const {
//   this->lock();
//   AllocatedChunk *chunk = _getAllocatedChunk( reg, true, true, wd, copyIdx );
//   chunk->addReference( wd, 22 );
//   chunk->unlock();
//   this->unlock();
//   return chunk;
//}

AllocatedChunk *RegionCache::getAllocatedChunk( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) const {
   return _getAllocatedChunk( reg, true, true, wd, copyIdx );
}

AllocatedChunk *RegionCache::_getAllocatedChunk( global_reg_t const &reg, bool complain, bool lockChunk, WD const &wd, unsigned int copyIdx ) const {
   ConstChunkList results;
   AllocatedChunk *allocChunkPtr = NULL;
   _chunks.getChunk( reg.getRealFirstAddress(), reg.getBreadth(), results );
   if ( results.size() == 1 ) {
      allocChunkPtr = results.front().second;
      //if ( ! allocChunkPtr ) {
      //   *myThread->_file << __func__ << " allocChunkPtr " << (void *) allocChunkPtr <<
      //      " results.front().second " << (void *) results.front().second <<
      //      " w/key " << (void*) reg.getRealFirstAddress() << " - " << (void*)(reg.getRealFirstAddress()+reg.getBreadth()) <<
      //      " results key: " << (void *)results.front().first.getAddress() << " - " << (void *)(results.front().first.getAddress() + results.front().first.getLength()) << std::endl;
      //}
   } else if ( results.size() > 1 ) {
         *(myThread->_file) <<"Requested addr " << (void *) reg.getRealFirstAddress() << " size " << reg.getBreadth() << std::endl;
      message0( "I think we need to realloc " << __FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__ );
      for ( ConstChunkList::const_iterator it = results.begin(); it != results.end(); it++ )
         *myThread->_file << " addr: " << (void *) it->first.getAddress() << " size " << it->first.getLength() << std::endl; 
      *(myThread->_file) << "Realloc needed. Caused by wd " << (wd.getDescription() ? wd.getDescription() : "n/a") << " copy index " << copyIdx << std::endl;
      fatal("Can not continue.");
   }
   if ( !allocChunkPtr && complain ) {
      printBt(*(myThread->_file) ); *(myThread->_file) << "Error, null region at spaceId "<< _memorySpaceId << " "; reg.key->printRegion( *(myThread->_file), reg.id ); *(myThread->_file) << " results.size= " << results.size() << " results.front().second " << results.front().second << std::endl;
      _chunks.print( *myThread->_file );

      for ( MemoryMap<AllocatedChunk>::const_iterator it = _chunks.begin(); it != _chunks.end(); it++ ) {
         if ( it->second != NULL && it->second != (AllocatedChunk *) -1 && (it->second != (AllocatedChunk *) -2) ) {
            AllocatedChunk &c = *(it->second);
            c.getAllocatedRegion().key->printRegion( *myThread->_file, c.getAllocatedRegion().id );
            *myThread->_file << std::endl;
         }
      }
      ensure(allocChunkPtr != NULL, "Chunk not found!");
   }
   if ( allocChunkPtr && lockChunk ) {
      //*myThread->_file << "AllocChunkPtr is " << allocChunkPtr << std::endl;
      allocChunkPtr->lock(); 
   }
   return allocChunkPtr;
}

void RegionCache::NEWcopyIn( unsigned int srcLocation, global_reg_t const &reg, unsigned int version, WD const *wd, unsigned int copyIdx, DeviceOps *givenOps, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk ) {
   //AllocatedChunk *chunk = getAllocatedChunk( reg );
   uint64_t origDevAddr = destinationChunk->getAddress() + ( reg.getRealFirstAddress() - destinationChunk->getHostAddress() );
   DeviceOps *ops = ( givenOps != NULL ) ? givenOps : destinationChunk->getDeviceOps( reg, wd, copyIdx );
   //chunk->unlock();
   //*myThread->_file << " COPY REGION ID " << reg.id << " OPS " << (void*)ops << std::endl;
   if ( srcLocation != 0 ) {
      ensure( sourceChunk != NULL, "invalid argument." );
      //AllocatedChunk *origChunk = sys.getSeparateMemory( srcLocation ).getCache().getAllocatedChunk( reg, wd, copyIdx );
      //origChunk->NEWaddWriteRegion( reg.id, version, wd, copyIdx );// this is needed in case we are copying out a fragment of a region
      //origChunk->unlock();
      sourceChunk->NEWaddWriteRegion( reg.id, version, wd, copyIdx );// this is needed in case we are copying out a fragment of a region
   }
   increaseTransferredInData(reg.getDataSize());
   copyIn( reg, origDevAddr, srcLocation, ops, destinationChunk, sourceChunk, wd );
}

void RegionCache::NEWcopyOut( global_reg_t const &reg, unsigned int version, WD const *wd, unsigned int copyIdx, DeviceOps *givenOps, bool inval, AllocatedChunk *providedOrigChunk ) {
   ensure (providedOrigChunk != NULL, "copyOut but no providedOrigChunk");
   AllocatedChunk *origChunk = providedOrigChunk;
   //if ( providedOrigChunk == NULL ) {
   //   while ( !_lock.tryAcquire() ) {
   //      myThread->idle();
   //   }
   //   origChunk = getAllocatedChunk( reg, *wd, copyIdx );
   //   _lock.release();
   //} else {
   //   origChunk = providedOrigChunk;
   //   origChunk->lock();
   //}

   uint64_t origDevAddr = origChunk->getAddress() + ( reg.getRealFirstAddress() - origChunk->getHostAddress() );
   DeviceOps *ops = ( givenOps != NULL ) ? givenOps : reg.getDeviceOps();
   //origChunk->clearDirty( reg );
   increaseTransferredOutData(reg.getDataSize());
   if ( !inval ) {
      origChunk->NEWaddWriteRegion( reg.id, version, wd, copyIdx );// this is needed in case we are copying out a fragment of a region, ignore in case of invalidation
   } else {
      increaseTransferredReplacedOutData(reg.getDataSize());
   }
   //if ( providedOrigChunk == NULL ) {
   //   origChunk->unlock();
   //}
//(*myThread->_file) << std::setprecision(std::numeric_limits<double>::digits10) << OS::getMonotonicTime() << " issuing copyOut (from " << this->getMemorySpaceId() << " using chunk " << origChunk << " w/addr " << origChunk->getHostAddress() << " provided chunk was " << providedOrigChunk << std::endl;
   copyOut( reg, origDevAddr, ops, wd );
}

RegionCache::RegionCache( memory_space_id_t memSpaceId, Device &cacheArch, enum CacheOptions flags, std::size_t slabSize ) :
   _chunks(),
   _lock(),
   _MAPlock(),
   _device( cacheArch ),
   _memorySpaceId( memSpaceId ),
   _flags( flags ),
   _slabSize( slabSize ),
   _lruTime( 0 ),
   _softInvalidationCount( 0 ),
   _hardInvalidationCount( 0 ),
   _inBytes( 0 ),
   _outBytes( 0 ),
   _outRepalcementBytes( 0 ),
   _allocatedRegionMap(),
   _allocatedRegionMapCopy(),
   _mapVersion( 0 ),
   _mapVersionRequested( 0 ),
   _currentAllocations( 0 ),
   _allocatedBytes( 0 ),
    _copyInObj( *this ), _copyOutObj( *this ) 
   {
   // FIXME : improve flags propagation from system/plugins to cache.
   if ( _slabSize > 0 ) {
      _flags = ALLOC_SLAB;
   }
}

unsigned int RegionCache::getMemorySpaceId() const {
   return _memorySpaceId;
}

void RegionCache::_copyIn( global_reg_t const &reg, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const *wd, bool fake ) {
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   if ( VERBOSE_DEV_OPS ) {
      char value[128];
      double *dptr = (double *) hostAddr;
      snprintf( value, 64, "%a %a", dptr[0], dptr[len/sizeof(double) - 1] );
      *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ", #" << _device.increaseNumOps() << ")._copyIn( reg=["; reg.key->printRegionGeom( *myThread->_file, reg.id ); *myThread->_file << "] copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<" ["<< value <<"]"<<", devAddr="<< (void*)devAddr <<", len=" << len << ", _pe, ops, wd="<< wd->getId() << " ["<< (wd->getDescription() != NULL ? wd->getDescription() : "no description") << "] );" <<std::endl;
   }
   if ( sys._watchAddr != NULL ) {
   if ( (uint64_t) sys._watchAddr >= hostAddr && ((uint64_t) sys._watchAddr ) < hostAddr + len ) {
      *myThread->_file << "WATCH " ;
      *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ", #" << _device.increaseNumOps() << ")._copyIn( reg=["; reg.key->printRegionGeom( *myThread->_file, reg.id ); *myThread->_file << "] copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<" ["<< *((double*) hostAddr) <<"]"<<", devAddr="<< (void*)devAddr <<", len=" << len << ", _pe, ops, wd="<< wd->getId() << " ["<< (wd->getDescription() != NULL ? wd->getDescription() : "no description") << "] );" <<std::endl;
   }
   }
   if (!fake) _device._copyIn( devAddr, hostAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   //NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyInStrided1D( global_reg_t const &reg, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const *wd, bool fake ) {
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ", #" << _device.increaseNumOps() <<")._copyInStrided1D( reg=["; reg.key->printRegionGeom( *myThread->_file, reg.id ); *myThread->_file << "] copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<" ["<< *((double*) hostAddr) <<"]"<<", devAddr="<< (void*)devAddr <<", len="<< len << ", numChunks=" << numChunks <<", ld=" << ld << ", _pe, ops="<< (void*)ops<<", wd="<< wd->getId() << " ["<< (wd->getDescription() != NULL ? wd->getDescription() : "no description") <<"] );" <<std::endl;
   }
   if (!fake) _device._copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   //NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOut( global_reg_t const &reg, uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, WD const *wd, bool fake ) {
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ", #" << _device.increaseNumOps() <<")._copyOut( reg=["; reg.key->printRegionGeom( *myThread->_file, reg.id ); *myThread->_file << "] copyFrom=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len=" << len << ", _pe, ops, wd="<< (wd != NULL ? wd->getId() : -1 ) << " ["<< ( wd != NULL && wd->getDescription() != NULL ? wd->getDescription() : "no description") <<"] );" <<std::endl;
   }
   if ( sys._watchAddr != NULL ) {
      if ( (uint64_t) sys._watchAddr >= hostAddr && ((uint64_t) sys._watchAddr ) < hostAddr + len ) {
         *myThread->_file << "WATCH " ;
         *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ", #" << _device.increaseNumOps() <<")._copyOut( reg=["; reg.key->printRegionGeom( *myThread->_file, reg.id ); *myThread->_file << "] copyFrom=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len=" << len << ", _pe, ops, wd="<< (wd != NULL ? wd->getId() : -1 ) << " ["<< ( wd != NULL && wd->getDescription() != NULL ? wd->getDescription() : "no description") <<"] );" <<std::endl;
      }
   }
   if (!fake) _device._copyOut( hostAddr, devAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   //NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOutStrided1D( global_reg_t const &reg, uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld,  DeviceOps *ops, WD const *wd, bool fake ) {
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ", #" << _device.increaseNumOps() <<")._copyOutStrided1D( reg=["; reg.key->printRegionGeom( *myThread->_file, reg.id ); *myThread->_file << "] copyFrom=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len="<< len <<", numChunks="<< numChunks <<", ld=" << ld << ", _pe, ops="<< (void*)ops <<", wd="<< (wd != NULL ? wd->getId() : -1 )  << " ["<< (wd != NULL && wd->getDescription() != NULL ? wd->getDescription() : "no description") << "] );" <<std::endl;
   }
   if (!fake) _device._copyOutStrided1D( hostAddr, devAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   //NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_syncAndCopyIn( global_reg_t const &reg, unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, AllocatedChunk *sourceChunk, WD const *wd, bool fake ) {
   DeviceOps *cout = NEW DeviceOps();
   uint64_t origDevAddr = sourceChunk->getAddress() + ( hostAddr - sourceChunk->getHostAddress() );
   sys.getSeparateMemory( syncFrom ).getCache()._copyOut( reg, hostAddr, origDevAddr, len, cout, wd, fake );
   while ( !cout->allCompleted() ){ myThread->processTransfers(); }
   delete cout;
   this->_copyIn( reg, devAddr, hostAddr, len, ops, wd, fake );
}

void RegionCache::_syncAndCopyInStrided1D( global_reg_t const &reg, unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, AllocatedChunk *sourceChunk, WD const *wd, bool fake ) {
   DeviceOps *cout = NEW DeviceOps();
   uint64_t origDevAddr = sourceChunk->getAddress() + ( hostAddr - sourceChunk->getHostAddress() );
   sys.getSeparateMemory( syncFrom ).getCache()._copyOutStrided1D( reg, hostAddr, origDevAddr, len, numChunks, ld, cout, wd, fake );
   while ( !cout->allCompleted() ){ myThread->processTransfers(); }
   delete cout;
   this->_copyInStrided1D( reg, devAddr, hostAddr, len, numChunks, ld, ops, wd, fake );
}

bool RegionCache::_copyDevToDev( global_reg_t const &reg, memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, AllocatedChunk *sourceChunk, WD const *wd, bool fake ) {
   bool result = true;
   uint64_t origDevAddr = sourceChunk->getAddress() + ( hostAddr - sourceChunk->getHostAddress() );
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ", #" << _device.increaseNumOps() <<")._copyDevToDev( reg=["; reg.key->printRegionGeom( *myThread->_file, reg.id ); *myThread->_file << "] copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len=" << len << ", _pe, sys.getSeparateMemory( copyFrom="<< copyFrom<<" ), ops, wd="<< wd->getId() << " ["<< (wd->getDescription() != NULL ? wd->getDescription() : "no description") << "] );" <<std::endl;
   }
   if ( sys._watchAddr != NULL ) {
   if ( (uint64_t) sys._watchAddr >= hostAddr && ((uint64_t) sys._watchAddr ) < hostAddr + len ) {
      *myThread->_file << "WATCH " ;
      *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ", #" << _device.increaseNumOps() <<")._copyDevToDev( reg=["; reg.key->printRegionGeom( *myThread->_file, reg.id ); *myThread->_file << "] copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len=" << len << ", _pe, sys.getSeparateMemory( copyFrom="<< copyFrom<<" ), ops, wd="<< wd->getId() << " ["<< (wd->getDescription() != NULL ? wd->getDescription() : "no description") << "] );" <<std::endl;
   }
   }
   if (!fake) {
      result = _device._copyDevToDev( devAddr, origDevAddr, len, sys.getSeparateMemory( _memorySpaceId ), sys.getSeparateMemory( copyFrom ), ops, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   }
   //NANOS_INSTRUMENT( inst.close(); );
   return result;
}

bool RegionCache::_copyDevToDevStrided1D( global_reg_t const &reg, memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, AllocatedChunk *sourceChunk, WD const *wd, bool fake ) {
   bool result = true;
   uint64_t origDevAddr = sourceChunk->getAddress() + ( hostAddr - sourceChunk->getHostAddress() );
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ", #" << _device.increaseNumOps() <<")._copyDevToDevStrided1D( reg=["; reg.key->printRegionGeom( *myThread->_file, reg.id ); *myThread->_file << "] copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len, _pe, sys.getCaches()[ copyFrom="<< copyFrom<<" ]->_pe, ops, wd="<< wd->getId() << " ["<< (wd->getDescription() != NULL ? wd->getDescription() : "no description") << "] );"<<std::endl;
   }
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   if (!fake) {
      result = _device._copyDevToDevStrided1D( devAddr, origDevAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), sys.getSeparateMemory( copyFrom ), ops, wd, (void *) reg.key->getKeyBaseAddress(), reg.id );
   }
   //NANOS_INSTRUMENT( inst.close(); );
   return result;
}

void RegionCache::CopyIn::doNoStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, WD const *wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyIn( reg, devAddr, hostAddr, size, ops, wd, fake );
   } else {
      //If copydev2dev unsucesfull (not supported/implemented), do a copy through host
      if ( ( &sys.getSeparateMemory( dataLocation ).getCache().getDevice() != &getParent()._device ) ||
            !getParent()._copyDevToDev( reg, dataLocation, devAddr, hostAddr, size, ops, sourceChunk, wd, fake )) {
         getParent().increaseTransferredOutData(reg.getDataSize());
         getParent()._syncAndCopyIn( reg, dataLocation, devAddr, hostAddr, size, ops, sourceChunk, wd, fake );
      }
   }
}

void RegionCache::CopyIn::doStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, WD const *wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyInStrided1D( reg, devAddr, hostAddr, size, count, ld, ops, wd, fake );
   } else {
       //If copydev2dev unsucesfull (not supported/implemented), do a copy through host
      if ( ( &sys.getSeparateMemory( dataLocation ).getCache().getDevice() != &getParent()._device ) ||
            !getParent()._copyDevToDevStrided1D( reg, dataLocation, devAddr, hostAddr, size, count, ld, ops, sourceChunk, wd, fake ) ) {
         getParent().increaseTransferredOutData(reg.getDataSize());
         getParent()._syncAndCopyInStrided1D( reg, dataLocation, devAddr, hostAddr, size, count, ld, ops, sourceChunk, wd, fake );
      }
   }
}

void RegionCache::CopyOut::doNoStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, WD const *wd, bool fake ) {
   ensure( destinationChunk == NULL, "Invalid argument");
   getParent()._copyOut( reg, hostAddr, devAddr, size, ops, wd, fake );
}
void RegionCache::CopyOut::doStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, WD const *wd, bool fake ) {
   ensure( destinationChunk == NULL, "Invalid argument");
   getParent()._copyOutStrided1D( reg, hostAddr, devAddr, size, count, ld, ops, wd, fake );
}

void RegionCache::doOp( Op *opObj, global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, WD const *wd ) {

   class LocalFunction {
      Op *_opObj;
      global_reg_t _hostMem;
      nanos_region_dimension_internal_t *_region;
      unsigned int _cutoff;
      std::size_t _contiguousChunkSize;
      unsigned int _location;
      DeviceOps *_ops;
      WD const *_wd;
      uint64_t _devBaseAddr;
      uint64_t _hostBaseAddr;
      AllocatedChunk *_destinationChunk;
      AllocatedChunk *_sourceChunk;
      public:
         LocalFunction( Op *op,  const global_reg_t &reg, nanos_region_dimension_internal_t *r,
               unsigned int cutoff, std::size_t ccs, unsigned int loc, DeviceOps *devops,
               WD const *w, uint64_t devAddr, uint64_t hostAddr,
               AllocatedChunk *destChunk, AllocatedChunk *srcChunk ) :
            _opObj( op ), _hostMem( reg ), _region( r ), _cutoff( cutoff ),
            _contiguousChunkSize( ccs ), _location( loc ), _ops( devops ), _wd( w ),
            _devBaseAddr( devAddr ), _hostBaseAddr( hostAddr ),
            _destinationChunk( destChunk ), _sourceChunk( srcChunk ) {
         }

         void issueOpsRecursive( std::size_t offset, unsigned int current_dim, std::size_t current_top_ld )  {
            std::size_t current_ld = current_top_ld / _region[current_dim].size;
            std::size_t this_offset = offset + _region[current_dim].lower_bound * current_ld;
            if ( current_dim == _cutoff + 1 && _region[current_dim].accessed_length > 1 && sys.usePacking() ) {
               std::size_t extra_offset = _region[current_dim-1].lower_bound * ( current_ld / _region[current_dim-1].size );
               uint64_t dev_addr = _devBaseAddr + this_offset + extra_offset;
               uint64_t host_addr = _hostBaseAddr + this_offset + extra_offset;
               std::size_t len = _contiguousChunkSize * _region[current_dim-1].accessed_length;
               std::size_t count = _region[current_dim].accessed_length;
               //printf("[op: % 4d] memcpy2D( dst=%p, orig=%p, size=%zu, count=%zu, ld=%zu )\n", (*total_ops)++, _dst, _orig , _len, _count, current_ld );
               _opObj->doStrided( _hostMem, _location, dev_addr, host_addr, len, count, current_ld, _ops, _destinationChunk, _sourceChunk, _wd, false );
            } else if ( current_dim <= _cutoff ) {
               uint64_t dev_addr = _devBaseAddr + this_offset;
               uint64_t host_addr = _hostBaseAddr + this_offset;
               size_t len = current_dim < _cutoff ? _contiguousChunkSize : 
                  _contiguousChunkSize * _region[current_dim].accessed_length;
               _opObj->doNoStrided( _hostMem, _location, dev_addr, host_addr, len, _ops, _destinationChunk, _sourceChunk, _wd, false );
            } else {
               for ( unsigned int i = 0; i < _region[current_dim].accessed_length; i +=1 ) {
                  issueOpsRecursive( this_offset + i * current_ld, current_dim-1, current_ld );
               }
            }
         }

   };
   nanos_region_dimension_internal_t region[ hostMem.getNumDimensions() ];
   hostMem.fillDimensionData( region );

   size_t top_ld = 1;
   size_t contiguous_chunk_size = 1;
   unsigned int dim_idx = 0;
   while ( ( region[ dim_idx ].accessed_length == region[ dim_idx ].size ) &&
         dim_idx < hostMem.getNumDimensions() ) {
      contiguous_chunk_size *= region[dim_idx].size;
      dim_idx += 1;
   }
   
   for ( unsigned int idx = 0; idx < hostMem.getNumDimensions(); idx += 1 ) {
      top_ld *= region[idx].size;
   }

   /* this function expects the base address of the region, not the address
      of the first element of the region, we have to substract the offset */
   uint64_t offset = hostMem.getFirstAddress( 0 );
   uint64_t dev_base_addr = devBaseAddr - offset;
   uint64_t host_base_addr = hostMem.getRealFirstAddress() - offset;

   LocalFunction local( opObj, hostMem, region, dim_idx, contiguous_chunk_size,
         location, ops, wd, dev_base_addr, host_base_addr,
         destinationChunk, sourceChunk );

   local.issueOpsRecursive( 0, hostMem.getNumDimensions() - 1, top_ld );

}

void RegionCache::copyIn( global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, WD const *wd ) {
   doOp( &_copyInObj, hostMem, devBaseAddr, location, ops, destinationChunk, sourceChunk, wd );
}

void RegionCache::copyOut( global_reg_t const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, WD const *wd ) {
   doOp( &_copyOutObj, hostMem, devBaseAddr, /* locations unused, copyOut is always to 0 */ 0, ops, NULL, NULL, wd );
}

void RegionCache::lock() {
   while ( !_lock.tryAcquire() ) {
      myThread->processTransfers();
   }
}
void RegionCache::unlock() {
   _lock.release();
}
bool RegionCache::tryLock() {
   bool result;
   result = _lock.tryAcquire();
   return result;
}
void RegionCache::MAPlock() {
   while ( !_MAPlock.tryAcquire() ) {
      myThread->processTransfers();
   }
}
void RegionCache::MAPunlock() {
   _MAPlock.release();
}

unsigned int RegionCache::getVersion( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg, wd, copyIdx );
   unsigned int version = chunk->getVersion( reg );
   chunk->unlock();
   return version;
}

void RegionCache::releaseRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   while ( !_lock.tryAcquire() ) {
      //myThread->idle();
   }

   for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
      AllocatedChunk *chunk = _getAllocatedChunk( memCopies[ idx ]._reg, true, false, wd, idx );
      chunk->removeReference( wd ); //RegionCache::releaseRegions
      if ( chunk->getReferenceCount() == 0 && ( memCopies[ idx ]._policy == NO_CACHE || memCopies[ idx ]._policy == FPGA ) ) {
         _chunks.removeChunks( chunk->getHostAddress(), chunk->getSize() );
         //*myThread->_file << "Delete chunk for idx " << idx << std::endl;
         if ( VERBOSE_DEV_OPS ) {
            *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ").memFree(  memspace=" << _memorySpaceId <<", devAddr="<< (void *)chunk->getAddress() << ", wd="<< wd.getId() << " ["<< (wd.getDescription() != NULL ? wd.getDescription() : "no description") << "], copyIdx="<< idx << " );" << std::endl;
         }
         _device.memFree( chunk->getAddress(), sys.getSeparateMemory( _memorySpaceId ) );
         _allocatedBytes -= chunk->getSize();
         RegionDirectory::delAccess( memCopies[ idx ]._reg.key, memCopies[ idx ]._reg.id, getMemorySpaceId() );
         delete chunk;
      }
   }

   _lock.release();
}

uint64_t RegionCache::getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress, AllocatedChunk *chunk ) const {
   return ( chunk->getAddress() - ( chunk->getHostAddress() - baseAddress ) );
}

bool RegionCache::prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd ) {
   _currentAllocations++;
   bool result = true;
   std::size_t total_allocatable_size = 0;
   std::set< global_reg_t > regions_to_allocate;
   std::pair<unsigned int, global_reg_t> regions_to_allocate_w_idx[numCopies];
   for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
      MemCacheCopy &mcopy = memCopies[ idx ];
      global_reg_t allocatable_region;
      getAllocatableRegion( mcopy._reg, allocatable_region );
      if ( regions_to_allocate.count( allocatable_region ) == 0 ) {
         regions_to_allocate_w_idx[ regions_to_allocate.size() ].first = idx;
         regions_to_allocate_w_idx[ regions_to_allocate.size() ].second = allocatable_region;
         regions_to_allocate.insert( allocatable_region );
         mcopy._allocFrom = -1;
      } else {
         unsigned int alloc_idx = 0;
         for (; alloc_idx < regions_to_allocate.size() && regions_to_allocate_w_idx[alloc_idx].second != allocatable_region; alloc_idx += 1 );
         mcopy._allocFrom = alloc_idx;
      }
   }
   for ( std::set< global_reg_t >::iterator it = regions_to_allocate.begin(); it != regions_to_allocate.end(); it++ ) {
      total_allocatable_size += it->getDataSize();
   }
   if ( total_allocatable_size <= _device.getMemCapacity( sys.getSeparateMemory( _memorySpaceId ) ) ) {
      //_lock.acquire();
      //while ( !_lock.tryAcquire() ) {
      //   myThread->idle();
      //}
      if ( _lock.tryAcquire() ) {
         if ( sys.useFineAllocLock() ) {
            sys.allocLock();
         }
         LockedObjects srcRegions;
         //          printReferencedChunksAndWDs();
         //       _chunks.print( *myThread->_file );
         // *(myThread->_file) << "EOT chunks prepare" << std::endl;
         //*(myThread->_file) << "prepareRegions wd " << wd.getId() << " total mem " << total_allocatable_size << std::endl;
         //attempt to allocate regions without triggering invalidations, this will reserve any chunk used by this WD
         for ( unsigned int allocIdx = 0; allocIdx < regions_to_allocate.size(); allocIdx += 1 ) {
            unsigned int idx = regions_to_allocate_w_idx[allocIdx].first;
            MemCacheCopy &mcopy = memCopies[ idx ];
            if ( mcopy._chunk == NULL || mcopy._chunk == (AllocatedChunk *) -2 ) {
               //*(myThread->_file) << "prepareRegions wd " << wd.getId() << " total mem " << total_allocatable_size << " alloc using tryGetAddress " << std::endl;
               //*(myThread->_file) << "prepareRegions wd " << wd.getId() << " idx " << idx << " total mem " << total_allocatable_size << " current chunk is " << mcopy._chunk << std::endl;
               mcopy._chunk = tryGetAddress( mcopy._reg, wd, idx );
               //*(myThread->_file) << "prepareRegions wd " << wd.getId() << " idx " << idx << " total mem " << total_allocatable_size << " alloc using tryGetAddress returns " << mcopy._chunk << std::endl;
               if ( mcopy._chunk != NULL && mcopy._chunk != (AllocatedChunk *) -2 ) {
                  //*myThread->_file << "Allocated region for wd " << wd.getId() << std::endl;
                  //mcopy._reg.key->printRegion(mcopy._reg.id);
                  //*myThread->_file << std::endl;
                  //AllocatedChunk *chunk = _getAllocatedChunk( mcopy._reg, false, false, wd, idx );
                  //*myThread->_file << "--1--> chunk is " << (void *) mcopy._chunk << " other chunk " << (void*) chunk<< std::endl;
                  mcopy._chunk->unlock();
               }
            }
         }
         for ( unsigned int allocIdx = 0; allocIdx < regions_to_allocate.size(); allocIdx += 1 ) {
            unsigned int idx = regions_to_allocate_w_idx[allocIdx].first;
            MemCacheCopy &mcopy = memCopies[ idx ];
            if ( mcopy._chunk == NULL ) {
               //*(myThread->_file) << "prepareRegions wd " << wd.getId() << " idx " << idx << " total mem " << total_allocatable_size << " alloc using getOrCreateChunk " << std::endl;
               mcopy._chunk = getOrCreateChunk( srcRegions, mcopy._reg, wd, idx );
               //*(myThread->_file) << "prepareRegions wd " << wd.getId() << " idx " << idx << " total mem " << total_allocatable_size << " alloc using getOrCreateChunk " << mcopy._chunk << std::endl;
               if ( ( mcopy._chunk == NULL && !mcopy._invalControl.isInvalidating() ) || mcopy._chunk == (AllocatedChunk *) -1 ) {
                  result = false;
               } else if ( mcopy._chunk == (AllocatedChunk *)-2 ) {
                  //invalidating...
               } else {
                  if ( mcopy._chunk != NULL ) {
                     //*myThread->_file << "Allocated region for wd " << wd.getId() << std::endl;
                     //mcopy._reg.key->printRegion(mcopy._reg.id);
                     //*myThread->_file << std::endl;
                     //AllocatedChunk *chunk = _getAllocatedChunk( mcopy._reg, false, false, wd, idx );
                     //*myThread->_file << "--2--> chunk is " << (void*) mcopy._chunk << " other chunk " << (void*) chunk << std::endl;
                     mcopy._chunk->unlock();
                  } else {
                     //chunk being invalidated..
                  }
               }
            }
         }
         //*(myThread->_file) << "prepareRegions wd " << wd.getId() << " end of allocation process, result " << result << std::endl;
         //release the allocated chunks if the allocation fails, this avoids
         //deadlocks if other threads are trying to allocate in the same cache.
         if ( !result ) {
            //for ( unsigned int idx = 0; idx < numCopies; idx += 1 )
            for ( unsigned int allocIdx = 0; allocIdx < regions_to_allocate.size(); allocIdx += 1 ) {
               unsigned int idx = regions_to_allocate_w_idx[allocIdx].first;
               MemCacheCopy &mcopy = memCopies[ idx ];
               if ( mcopy._chunk != NULL ) {
                  mcopy._chunk->removeReference( wd ); //prepareRegions (rollback)
                  mcopy._chunk = NULL;
               } else {
                  //*myThread->_file << "abort inval!!!" << std::endl;
                  if ( mcopy._invalControl._invalChunkPtr != NULL ) {
                     *mcopy._invalControl._invalChunkPtr = (AllocatedChunk *) 0;
                  }
                  mcopy._invalControl.abort( wd );
               }
            }
         } else {
            if ( result ) {
               for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
                  if ( memCopies[idx]._invalControl._invalOps != NULL ) {
                     memCopies[idx]._invalControl.preIssueActions( this->getMemorySpaceId(), wd );
                  }
               }
            }
         }
         
         // We need to do this here, and not release the regions until we
         // are done with the operations, otherwise another operations,
         // that would expect this invalidation to be completed, could be issued
         if ( result ) {
            for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
               if ( memCopies[idx]._invalControl._invalOps != NULL ) {
                  memCopies[idx]._invalControl._invalOps->issue( &wd );
               }
            }
            for ( unsigned int idx = 0; idx < numCopies; idx += 1 ) {
               if ( memCopies[idx]._invalControl._invalOps != NULL ) {
                  memCopies[idx]._invalControl.postIssueActions( this->getMemorySpaceId() );
               }
            }
         }

         srcRegions.releaseLockedObjects();
         //*(myThread->_file) << "prepareRegions wd " << wd.getId() << " total mem " << total_allocatable_size << " finished. result: " << result << " count chunks: " << countxx << std::endl;
         if ( sys.useFineAllocLock() ) {
            sys.allocUnlock();
         }
         _lock.release();

      } else {
         result = false;
      }
   } else {
      result = false;
      *myThread->_file << "This device can not hold this task, not enough memory. Needed: "<< total_allocatable_size << " max avalilable " << _device.getMemCapacity( sys.getSeparateMemory( _memorySpaceId ) ) << " wd " << wd.getId() << " allocWide " << ( _flags == ALLOC_WIDE )  << std::endl;
      fatal( "This device can not hold this task, not enough memory." );
   }
   _currentAllocations--;
   return result;
}

void RegionCache::prepareRegionsToBeCopied( std::set< global_reg_t > const &regs, unsigned int version, std::set< AllocatedChunk * > &chunks, WD const &wd, unsigned int copyIdx ) {
   while ( !_lock.tryAcquire() ) {
      //myThread->idle();
   }
   for ( std::set< global_reg_t >::iterator it = regs.begin(); it != regs.end(); it++ ) {
      this->_prepareRegionToBeCopied( *it, version, chunks, wd, copyIdx );
   }
   _lock.release();
}

void RegionCache::_prepareRegionToBeCopied( global_reg_t const &reg, unsigned int version, std::set< AllocatedChunk * > &chunks, WD const &wd, unsigned int copyIdx ) {
   AllocatedChunk *chunk = _getAllocatedChunk( reg, false, false, wd, copyIdx );
   if ( VERBOSE_CACHE ) { *myThread->_file <<"I'm " << myThread->runningOn()->getMemorySpaceId() << ", this is cache " << this->getMemorySpaceId() << " reg " << reg.id << " got chunk " << chunk << " " << wd.getDescription() <<" copyIdx " << copyIdx << std::endl; }
   if ( chunk != NULL ) {
      if ( chunks.count( chunk ) == 0 ) {
         chunk->lock();
         chunk->addReference( wd, 1 ); //_prepareRegionToBeCopied
         chunks.insert( chunk );
         chunk->unlock();
      }
   } else {
      fatal("Could not add a reference to a source chunk."); 
   }
}

void RegionCache::setRegionVersion( global_reg_t const &hostMem, AllocatedChunk *chunk, unsigned int version, WD const &wd, unsigned int copyIdx ) {
   chunk->lock();
   chunk->setRegionVersion( hostMem.id, version, wd, copyIdx );
   chunk->unlock();
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
   } else if ( _flags == ALLOC_SLAB ) {
      //*myThread->_file << "####################################################" << std::endl;
      //*myThread->_file << "# WHOLE: "; reg.key->printRegion(*myThread->_file, 1); *myThread->_file << std::endl;
      //*myThread->_file << "# REG: "; reg.key->printRegion(*myThread->_file, reg.id); *myThread->_file << std::endl;
      //if ( reg.id == 1 ) {
      //   allocRegion.id = 1;
      //} else {
         allocRegion.id = reg.getSlabRegionId( _slabSize );
      //}
      //*myThread->_file << "# Return: "; reg.key->printRegion(*myThread->_file, allocRegion.id); *myThread->_file << std::endl;
      //*myThread->_file << "####################################################" << std::endl;
   } else {
      *myThread->_file <<"RegionCache ERROR: Undefined _flags value."<<std::endl;
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
   //*myThread->_file << __FUNCTION__ << " needed chunks is " << needed_chunks << std::endl;

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
      //    *myThread->_file << "["<< count++ << "] mmm this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << " size is " << it->second->getSize() << " ";
      //    thisreg.key->printRegion( thisreg.id );
      //    *myThread->_file << std::endl;
      // }
      if ( it->second != NULL && it->second->getReferenceCount() == 0 && !(it->second->isRooted()) ) {
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
   // *myThread->_file << "-----------------------vvvvvvvvvvvv inv reg " << reg.id << "vvvvvvvvvvvvvvvvvv--------------------" << std::endl; 
   // reg.key->printRegion( *myThread->_file, reg.id );
   // *myThread->_file << std::endl;
   ConstChunkList results;
   _chunks.getChunk( reg.getRealFirstAddress(), reg.getBreadth(), results );
   std::set< AllocatedChunk * > removedChunks; //this is done for debugging purposes, there should not be any duplicates

   if ( results.size() > 0 ) {
      //unsigned int count = 0;
      for ( ConstChunkList::iterator it = results.begin(); it != results.end(); it++ ) {
         // *(myThread->_file) << count++ << " Invalidate object, chunk:: addr: " << (void *) it->first->getAddress() << " size " << it->first->getLength() << std::endl; 
         //printBt();
         if ( it->second != NULL ) {
            if ( removedChunks.find( it->second ) != removedChunks.end() ) {
               *(myThread->_file) << "WARNING: already removed chunk!!!" << std::endl;
            }
            if ( VERBOSE_DEV_OPS ) {
               *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ").memFree(  memspace=" << _memorySpaceId <<", devAddr="<< (void *)(it->second)->getAddress() << " );" << std::endl;
            }
            _device.memFree( it->second->getAddress(), sys.getSeparateMemory( _memorySpaceId ) );
            _allocatedBytes -= it->second->getSize();
            removedChunks.insert( it->second );
            delete it->second;
         }
      }
      _chunks.removeChunks( reg.getRealFirstAddress(), reg.getBreadth() );
   }
   // *myThread->_file << "-----------------------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^--------------------" << std::endl; 
}

void RegionCache::copyOutputData( SeparateAddressSpaceOutOps &ops, global_reg_t const &reg, unsigned int version, bool output, enum CachePolicy policy, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx ) {
   if ( policy == FPGA ) { //emit copy for all data
      if ( output ) {
        chunk->copyRegionToHost( ops, reg.id, version + (output ? 1 : 0), wd, copyIdx );
      }
   } else {
      if ( output ) {
         if ( policy != WRITE_BACK ) {
            chunk->copyRegionToHost( ops, reg.id, version + 1, wd, copyIdx );
         }
      } 
   }
}

void RegionCache::printReferencedChunksAndWDs() const {
   MemoryMap<AllocatedChunk>::const_iterator it;
   for ( it = _chunks.begin(); it != _chunks.end(); it++ ) {
      if ( it->second != NULL && it->second != (AllocatedChunk *) -1 && (it->second != (AllocatedChunk *) -2) ) {
         AllocatedChunk &c = *(it->second);
         c.printReferencingWDs();
      }
   }
}

bool RegionCache::shouldWriteThrough() const {
   std::size_t dirty_bytes = 0;
   std::size_t total_bytes = 0;
   std::size_t flushable_bytes = 0;
   std::size_t cache_capacity = _device.getMemCapacity( sys.getSeparateMemory( _memorySpaceId ) );
   MemoryMap<AllocatedChunk>::const_iterator it;
   for ( it = _chunks.begin(); it != _chunks.end(); it++ ) {
      if ( it->second != NULL ) {
         AllocatedChunk &c = *(it->second);
         total_bytes += c.getSize();
         if ( c.isDirty() ) {
            //Estimation, not all chunk bytes could be dirty
            dirty_bytes += c.getSize();
         }
         if ( c.isFlushable() ) {
            flushable_bytes += c.getSize();
         }
      }
   }
   //return ( dirty_bytes * 2 > cache_capacity && flushable_bytes*2 < cache_capacity );
   return flushable_bytes*2 < cache_capacity;
}

void RegionCache::freeChunk(AllocatedChunk *chunk, WD const &wd) {
   if ( VERBOSE_DEV_OPS ) {
      *(myThread->_file) << "[" << myThread->getId() << "] _device(" << _device.getName() << ").memFree(  memspace=" << _memorySpaceId <<", devAddr="<< (void *)chunk->getAddress() << ", wd="<< wd.getId() << " ["<< (wd.getDescription() != NULL ? wd.getDescription() : "no description") << "] );" << std::endl;
   }
   _device.memFree( chunk->getAddress(), sys.getSeparateMemory( _memorySpaceId ) );
   _allocatedBytes -= chunk->getSize();
   delete chunk;
}


void RegionCache::addToAllocatedRegionMap( global_reg_t const &reg ) {
   //*myThread->_file << "add to allocatedMap "; reg.key->printRegion( *myThread->_file, reg.id ); *myThread->_file << std::endl;
   this->MAPlock();
   _allocatedRegionMap[ reg.key ].insert( reg.id );
   _mapVersion++;
   this->MAPunlock();
}

void RegionCache::removeFromAllocatedRegionMap( global_reg_t const &reg ) {
   //*myThread->_file << "remove from allocatedMap "; reg.key->printRegion( *myThread->_file, reg.id ); *myThread->_file << std::endl;
   _allocatedRegionMap[ reg.key ].erase( reg.id );
   _mapVersion++;
}

std::map<GlobalRegionDictionary *, std::set<reg_t> > const &RegionCache::getAllocatedRegionMap() {
   if ( _mapVersionRequested != _mapVersion ) {
      this->MAPlock();
      _allocatedRegionMapCopy = _allocatedRegionMap;
      _mapVersionRequested = _mapVersion;
      this->MAPunlock();
   }
   return _allocatedRegionMapCopy;
}
