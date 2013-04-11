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
   if ( sys.usingNewCache() ) {
      _newRegions = NEW CacheRegionDictionary( *(allocatedRegion.key) );
      //std::cerr << "Allocated chunk: with dict " << (void *)_newRegions << std::endl;
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

void AllocatedChunk::clearNewRegions( global_reg_t const &reg ) {
   //delete _newRegions;
   _newRegions = NEW CacheRegionDictionary( *(reg.key) );
   _allocatedRegion = reg;
}


RegionTree< CachedRegionStatus > *AllocatedChunk::getRegions() {
   return _regions;
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
   _newRegions->registerRegion( reg, components, currentVersion );

   std::cerr << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ " << __FUNCTION__ << " reg " << reg << " set rversion "<< version << " ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"<< std::endl;
   

   if ( components.size() == 1 ) {
      ensure( components.begin()->first == reg, "Error, wrong region");
   }

   for ( std::list< std::pair< reg_t, reg_t > >::iterator it = components.begin(); it != components.end(); it++ )
   {
      CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->first );
      if ( ( !entry || version > entry->getVersion() ) ) {
         std::cerr << "No entry for region " << it->first << " must copy from region " << it->second << " want version "<< version << std::endl;
         CachedRegionStatus *copyFromEntry = ( CachedRegionStatus * ) _newRegions->getRegionData( it->second );
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
         //std::cerr << "!!!!!!!!!!!! REGION READY " << it->first << std::endl;
         } else {
            /* in cache but comming */
        // std::cerr << "???????????? ON ITS WAY " << it->first << std::endl;
            currentOps.insert( entry->getDeviceOps() );
         }
      } else {
         std::cerr << "ERROR: version in cache > than version requested." << std::endl;
      }
      entry->setVersion( version + ( alsoWriteRegion ? 1 : 0) );
   }

#if 0
   if ( components.size() == 1 ) {
      CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( components.begin()->first );
      if ( ( !entry || version > entry->getVersion() ) ) {
         std::cerr << "No entry for region " << components.begin()->first << " must copy from region " << components.begin()->second << " want version "<< version << std::endl;
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
#endif
      
  // std::cerr << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[X]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"<< std::endl;
   _dirty = _dirty || alsoWriteRegion;
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

      #if 0
      bool transfer = false;
      if ( NewNewRegionDirectory::delAccess( _allocatedRegion.key, _allocatedRegion.id, targetCache->getMemorySpaceId() ) ) {
         uint64_t origDevAddr = getAddress() + ( _allocatedRegion.getFirstAddress() - getHostAddress() );
         NewNewRegionDirectory::invalidate( _allocatedRegion.key, _allocatedRegion.id );
         targetCache->copyOut( _allocatedRegion, origDevAddr, &localOps, *((WD *) NULL) );
         transfer = true;


       //  _newRegions->getGlobalDirectoryKey()->unlock();
       //  while( !localOps.allCompleted() ) { myThread->idle(); }
       //  _newRegions->getGlobalDirectoryKey()->lock();
         NewNewRegionDirectory::addAccess( _allocatedRegion.key, _allocatedRegion.id, 0, NewNewRegionDirectory::getVersion( _allocatedRegion.key, _allocatedRegion.id, false ) );
         //NewNewRegionDirectory::setOps( _allocatedRegion.key, _allocatedRegion.id, &localOps );

       //  _newRegions->getGlobalDirectoryKey()->unlock();

      }
      for ( std::map<reg_t, RegionVectorEntry>::const_iterator it = _newRegions->begin(); it != _newRegions->end(); it++ ) {
         if ( it->first == _allocatedRegion.id ) continue;
         if ( it->second.getData() != NULL ) {
            if ( NewNewRegionDirectory::delAccess( _allocatedRegion.key, it->first, targetCache->getMemorySpaceId() ) ) {
               NewNewRegionDirectory::invalidate( _allocatedRegion.key, it->first );
               NewNewRegionDirectory::addAccess( _allocatedRegion.key, it->first, 0, NewNewRegionDirectory::getVersion( _allocatedRegion.key, it->first, false ) );
            }
         }
      }
      #endif
      std::cerr << call << "========== whole reg "<< _newRegions->getRegionNodeCount() <<"===========> Invalidate region "<< (void*) key << ":" << _allocatedRegion.id << " reg: "; _allocatedRegion.key->printRegion( _allocatedRegion.id ); std::cerr << std::endl;
      //return;
#if 0
   } else {




      //std::list<global_reg_t> add_list;
      for ( std::map<reg_t, RegionVectorEntry>::const_iterator it = _newRegions->begin(); it != _newRegions->end(); it++ ) {
         global_reg_t reg( it->first, key );
         bool transfer = false;
         CachedRegionStatus *entry = ( CachedRegionStatus * ) _newRegions->getRegionData( reg.id );
         if ( NewNewRegionDirectory::getVersion( reg.key, reg.id, false ) == entry->getVersion() ) {
            std::cerr << " chunk, dir v is " << NewNewRegionDirectory::getVersion( reg.key, reg.id, false ) << ", entry is " << entry->getVersion() << std::endl;
            if ( NewNewRegionDirectory::delAccess( reg.key, reg.id, targetCache->getMemorySpaceId() ) ) {
               uint64_t origDevAddr = getAddress() + ( reg.getFirstAddress() - getHostAddress() );
               NewNewRegionDirectory::invalidate( reg.key, reg.id );
               //add_list.push_back( reg );
               //NewNewRegionDirectory::addAccess( reg.key, reg.id, 0, NewNewRegionDirectory::getVersion( reg.key, reg.id ) );
               NewNewRegionDirectory::addAccess( reg.key, reg.id, 0, NewNewRegionDirectory::getVersion( reg.key, reg.id, false ) );
               NewNewRegionDirectory::setOps( reg.key, reg.id, &localOps );
               targetCache->copyOut( reg, origDevAddr, &localOps, *((WD *) NULL) );
               transfer = true;
            }
         } else { 
     NewNewDirectoryEntryData *dentry = NewNewRegionDirectory::getDirectoryEntry( *(reg.key), reg.id );
            std::cerr << "+++ +++ +++ not equal version region +++ +++ +++: dir " << *dentry << " entry v " << entry->getVersion() << std::endl;
         }
     NewNewDirectoryEntryData *daentry = NewNewRegionDirectory::getDirectoryEntry( *(reg.key), _allocatedRegion.id );
         std::cerr << call << "================================> Invalidate region "<< (void*) key << ":" << it->first << " reg: "; reg.key->printRegion( reg.id ); std::cerr << ((transfer) ? " yes " : " no " ) << " directory allocated entry " << *daentry << std::endl;
      }
     if ( NewNewRegionDirectory::delAccess( key, _allocatedRegion.id, targetCache->getMemorySpaceId() ) ) {
        NewNewRegionDirectory::invalidate( key, _allocatedRegion.id );
        NewNewRegionDirectory::addAccess( key, _allocatedRegion.id, 0, NewNewRegionDirectory::getVersion( key, _allocatedRegion.id, false ) );
        NewNewRegionDirectory::setOps( key, _allocatedRegion.id, &localOps );
     }
//      if ( NewNewRegionDirectory::delAccess( _newRegions->getGlobalDirectoryKey(), _allocatedRegion, targetCache->getMemorySpaceId() ) ) {
//       //         add_list.push_back( global_reg_t( _allocatedRegion, _newRegions->getGlobalDirectoryKey() ) ); 
//            NewNewRegionDirectory::addAccess( _newRegions->getGlobalDirectoryKey(), _allocatedRegion,  0, NewNewRegionDirectory::getVersion( _newRegions->getGlobalDirectoryKey(), _allocatedRegion) );
//      }
 //     _newRegions->getGlobalDirectoryKey()->unlock();
 //     while( !localOps.allCompleted() ) { myThread->idle(); }

 //     _newRegions->getGlobalDirectoryKey()->lock();
 //     for ( std::list< global_reg_t >::iterator lit = add_list.begin(); lit != add_list.end(); lit++ ) {
 //        NewNewRegionDirectory::addAccess( (*lit).key, (*lit).id, 0, NewNewRegionDirectory::getVersion( (*lit).key, (*lit).id ) );
 //     }
 //     _newRegions->getGlobalDirectoryKey()->unlock();
   }
#endif

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
      //if ( it->second != NULL ) std::cerr << "["<< count << "] mmm this chunk: " << ((void *) it->second) << " refs " <<  it->second->getReferenceCount() << std::endl;  
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

AllocatedChunk *RegionCache::getAddress( global_reg_t const &reg, RegionTree< CachedRegionStatus > *&regsToInval, CacheRegionDictionary *&newRegsToInval, WD const &wd ) {
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
   std::list< AllocatedChunk *> chunks;
   
   for ( it = regions.begin(); it != regions.end(); it++ ) {
      global_reg_t id = it->first;
      CacheCopy *ccopy = it->second;
      AllocatedChunk *origChunk = getAddress( id.getFirstAddress(), id.getBreadth() );
      uint64_t origDevAddr = origChunk->getAddress() + ( id.getFirstAddress() - origChunk->getHostAddress() );
      chunks.push_back( origChunk );
      origChunk->addReference();
      origChunk->unlock();
      //std::cerr << "NEW SYNC copy out reg " << (void*)id.key<< ","<< id.id << "DEV ADDR: " << ((void*)origDevAddr) << " origChunk->getHostAddr()= "<< ( (void*) origChunk->getHostAddress() ) << " firstHostAddr " << ((void*) id.getFirstAddress()) << std::endl;
      copyOut( id, origDevAddr, ( ccopy != NULL ) ? ccopy->getOperations() : &localOps, wd );
   }

   while( !localOps.allCompleted() ) { myThread->idle(); }
   for ( std::list< AllocatedChunk *>::iterator cit = chunks.begin(); cit != chunks.end(); cit++ ) {
      (*cit)->removeReference();
   }
}
void RegionCache::syncRegion( global_reg_t const &id ) {
   std::list< std::pair< global_reg_t, CacheCopy * > > singleItemList;
   singleItemList.push_back( std::make_pair( id, ( CacheCopy * ) NULL ) );
   syncRegion( singleItemList, *(( WD * ) NULL) );
}

void RegionCache::NEWcopyIn( unsigned int srcLocation, global_reg_t const &reg, unsigned int version, WD const &wd ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   uint64_t origDevAddr = chunk->getAddress() + ( reg.getFirstAddress() - chunk->getHostAddress() );
   DeviceOps *ops = chunk->getDeviceOps( reg );
   chunk->unlock();
   //std::cerr << " COPY REGION ID " << reg.id << " OPS " << (void*)ops << std::endl;
   copyIn( reg, origDevAddr, srcLocation, ops, wd );
}

void RegionCache::NEWcopyOut( global_reg_t const &reg, unsigned int version, WD const &wd ) {
   AllocatedChunk *origChunk = getAllocatedChunk( reg );
   uint64_t origDevAddr = origChunk->getAddress() + ( reg.getFirstAddress() - origChunk->getHostAddress() );
   DeviceOps *ops = reg.getDeviceOps();
   origChunk->unlock();
   copyOut( reg, origDevAddr, ops, wd );
}

RegionCache::RegionCache( memory_space_id_t memSpaceId, Device &cacheArch, enum CacheOptions flags ) : _device( cacheArch ), _memorySpaceId( memSpaceId ),
    _flags( flags ), _lruTime( 0 ), _copyInObj( *this ), _copyOutObj( *this ) {
}

unsigned int RegionCache::getMemorySpaceId() {
   return _memorySpaceId;
}

void RegionCache::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   //std::cerr << "_device._copyIn( copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, _pe, ops, wd="<< wd.getId() <<" );";
   if (!fake) _device._copyIn( devAddr, hostAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_IN); );
   if (!fake) _device._copyInStrided1D( devAddr, hostAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   //std::cerr << "_device._copyOut( copyFrom=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", len, _pe, ops, wd="<< (&wd != NULL ? wd.getId() : -1 ) <<" );";
   if (!fake) _device._copyOut( hostAddr, devAddr, len, sys.getSeparateMemory( _memorySpaceId ), ops, wd );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld,  DeviceOps *ops, WD const &wd, bool fake ) {
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_OUT); );
   if (!fake) _device._copyOutStrided1D( hostAddr, devAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), ops, wd );
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

void RegionCache::_copyDevToDev( memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake ) {
   //AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len );
   AllocatedChunk *origChunk = sys.getSeparateMemory( copyFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *f = NEW CompleteOpFunctor( ops, origChunk );
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   //std::cerr << "_device._copyDevToDev( copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len, _pe, sys.getSeparateMemory( copyFrom="<< copyFrom<<" ), ops, wd="<< wd.getId() << ", f="<< f <<" );" <<std::endl;
   if (!fake) _device._copyDevToDev( devAddr, origDevAddr, len, sys.getSeparateMemory( _memorySpaceId ), sys.getSeparateMemory( copyFrom ), ops, wd, f );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::_copyDevToDevStrided1D( memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) {
   //AllocatedChunk *origChunk = sys.getCaches()[ copyFrom ]->getAddress( hostAddr, len );
   AllocatedChunk *origChunk = sys.getSeparateMemory( copyFrom ).getCache().getAddress( hostAddr, len );
   uint64_t origDevAddr = origChunk->getAddress() + ( hostAddr - origChunk->getHostAddress() );
   origChunk->unlock();
   CompleteOpFunctor *f = NEW CompleteOpFunctor( ops, origChunk );
   //std::cerr << "_device._copyDevToDevStrided1D( copyFrom=" << copyFrom << ", copyTo=" << _memorySpaceId <<", hostAddr="<< (void*)hostAddr <<", devAddr="<< (void*)devAddr <<", origDevAddr="<< (void*)origDevAddr <<", len, _pe, sys.getCaches()[ copyFrom="<< copyFrom<<" ]->_pe, ops, wd="<< wd.getId() <<", f="<< f <<" );"<<std::endl;
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_CC_COPY_DEV_TO_DEV); );
   if (!fake) _device._copyDevToDevStrided1D( devAddr, origDevAddr, len, numChunks, ld, sys.getSeparateMemory( _memorySpaceId ), sys.getSeparateMemory( copyFrom ), ops, wd, f );
   NANOS_INSTRUMENT( inst.close(); );
}

void RegionCache::CopyIn::doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, WD const &wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyIn( devAddr, hostAddr, size, ops, wd, fake );
   //} else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
   } else if ( sys.canCopy( dataLocation, getParent().getMemorySpaceId() ) ) { 
      getParent()._copyDevToDev( dataLocation, devAddr, hostAddr, size, ops, wd, fake );
   } else {
      getParent()._syncAndCopyIn( dataLocation, devAddr, hostAddr, size, ops, wd, fake );
   }
}

void RegionCache::CopyIn::doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) {
   if  ( dataLocation == 0 ) {
      getParent()._copyInStrided1D( devAddr, hostAddr, size, count, ld, ops, wd, fake );
   //} else if ( getParent().canCopyFrom( *sys.getCaches()[ dataLocation ] ) ) { 
   } else if ( sys.canCopy( dataLocation, getParent().getMemorySpaceId() ) ) { 
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
                  //std::cerr << "[NEW]opObj("<<_opObj->getStr()<<")->doStrided( src="<<_location<<", dst="<< _opObj->getParent().getMemorySpaceId()<<", "<<(void*)(_devBaseAddr+offset)<<", "<<(void*)(_hostBaseAddr+offset)<<", "<<_contiguousChunkSize<<", "<<_numChunks<<", "<<leadingDim<<", _ops="<< (void*)_ops<<", _wd="<<(&_wd != NULL ? _wd.getId():-1)<<" )";
                  _opObj->doStrided( _location, _devBaseAddr+offset, _hostBaseAddr+offset, _contiguousChunkSize, _numChunks, leadingDim, _ops, _wd, false );
                  //std::cerr <<" done"<< std::endl;
               } else {
                  for (unsigned int chunkIndex = 0; chunkIndex < L_numChunks; chunkIndex +=1 ) {
                     //std::cerr <<"[NEW]opObj("<<_opObj->getStr()<<")->doNoStrided( src="<<_location<<", dst="<< _opObj->getParent().getMemorySpaceId()<<", "<<(void*)(_devBaseAddr+offset + chunkIndex*(leadingDim))<<", "<<(void*)(_hostBaseAddr+offset + chunkIndex*(leadingDim))<<", "<<_contiguousChunkSize<<", _ops="<< (void*)_ops<< ", _wd="<<(&_wd != NULL ? _wd.getId():-1)<<" )";
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

#ifdef NEWINIT
CacheCopy::CacheCopy( WD const &wd , unsigned int copyIndex ) : _copy( wd.getCopies()[ copyIndex ] ), _cacheEntry( NULL ),
   _cacheDataStatus(), /*_region( NewRegionDirectory::build_region( _copy ) ),*/ _offset( 0 ),
   _version( 0 ), _newVersion(0), _locations(), _newLocations(), _operations(), _otherPendingOps() 
#else
CacheCopy::CacheCopy( WD const &wd , unsigned int copyIndex, CacheController &ccontrol ) : _copy( wd.getCopies()[ copyIndex ] ), _cacheEntry( NULL ),
   _cacheDataStatus(), /*_region( NewRegionDirectory::build_region( _copy ) ),*/ _offset( 0 ),
   _version( 0 ), _newVersion(0), _locations(), _newLocations(), _operations(), _otherPendingOps() 
#endif
{
   if ( !sys.usingNewCache() ) {
      wd.getNewDirectory()->getLocation( _region, _locations, _version, wd );
   } else {
#ifdef NEWINIT
      _reg.key = sys.getMasterRegionDirectory().getRegionDirectoryKeyRegisterIfNeeded( wd.getCopies()[ copyIndex ] );
      _reg.id = _reg.key->obtainRegionId( wd.getCopies()[ copyIndex ] );
#else
      _reg.key = sys.getMasterRegionDirectory().getRegionDirectoryKeyRegisterIfNeeded( wd.getCopies()[ copyIndex ] );
      _reg.id = 0;
      _regId = _reg.key->tryObtainRegionId( wd.getCopies()[ copyIndex ] );

      if ( _regId != 0 && ccontrol.hasVersionInfoForRegion( global_reg_t( _regId, _reg.key ) , _newVersion, _newLocations ) ) {
         _reg.id = _regId;
         //_newLocations.push_back( std::make_pair( _regId, _regId ) );
      } else {
         tryGetLocation( wd, copyIndex );
         //fprintf(stderr, " wd %d, regId is %d, index is %d TRYGET\n ", wd.getId(), _reg.id, copyIndex);
      }
#endif
   }
   //std::cerr << "Region is " << _regId << " # Components: " << _newLocations.size() << " " << std::endl;
   //for ( NewNewRegionDirectory::NewLocationInfoList::iterator it = _newLocations.begin(); it != _newLocations.end(); it++ ) {
   //   std::cerr << "\tReg " << *it << std::endl;
   //}
}

void CacheCopy::getVersionInfo( WD const &wd, unsigned int copyIndex, CacheController &ccontrol ) {
   ccontrol.hasVersionInfoForRegion( _reg , _newVersion, _newLocations );
      //_newLocations.push_back( std::make_pair( _regId, _regId ) );
   //} else {
   //   tryGetLocationNewInit( wd, copyIndex );
      //fprintf(stderr, " wd %d, regId is %d, index is %d TRYGET\n ", wd.getId(), _reg.id, copyIndex);
   //}
}

bool CacheCopy::tryGetLocationNewInit( WD const &wd, unsigned int copyIndex ) {
   //do {
   NewNewRegionDirectory::tryGetLocation( _reg.key, wd.getCopies()[ copyIndex ], _newLocations, _newVersion, wd );
   //} while ( _reg.id == 0 );
   return (_newVersion != 0);
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

	NewLocationInfoList::const_iterator loc_it;
   for ( loc_it = _newLocations.begin(); loc_it != _newLocations.end() && allReady; loc_it++ ) {
      DeviceOps *ops = NewNewRegionDirectory::getOps( _reg.key, loc_it->second );
      if ( ops ) std::cerr << " element " << loc_it->first << "," << loc_it->second << " ops are " << ops << std::endl;
      if ( ops != NULL && !ops->allCompleted() ) {
         allReady = false;
      }
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
   CacheRegionDictionary *newRegsToInvalidate = NULL;
   //std::cerr << __FUNCTION__ << " ";  _reg.key->printRegion( _reg.id ); std::cerr << " entry is "<< (void *)_cacheEntry << std::endl;
   do {
   _cacheEntry = targetCache->getAddress( _reg, regsToInvalidate, newRegsToInvalidate, *((WD const *)NULL) );
   } while ( _cacheEntry == NULL );
   //if ( sys.usingNewCache() ) {
   //   if ( newRegsToInvalidate ) {
   //      std::cerr << "New! Got to do something..." << std::endl;
   //      //sys.getMasterRegionDirectory().invalidate( newRegsToInvalidate, targetCache->getMemorySpaceId() );
   //   }
   //} else {
   //   if ( regsToInvalidate ) {
   //      std::cerr << "Got to do something..." << std::endl;
   //      dir->invalidate( regsToInvalidate );
   //   }
   //}
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
	NewLocationInfoList::const_iterator it;
   if ( targetCache ) _cacheEntry->lock();
   if ( _copy.isInput() ) {
      //if ( NewNewRegionDirectory::isLocatedIn( _reg.key, it->first, ( !targetCache ) ? 0 : targetCache->getMemorySpaceId() /*, _newVersion*/ ) ) continue; //Version matching is already done in getLocation, if a region is listed, its because its a max version fragment
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
         for ( it = _newLocations.begin(); it != _newLocations.end(); it++ ) {
            global_reg_t gr( it->first, _reg.key );
      if(sys.getNetwork()->getNodeNum() == 0) { 
         std::cerr <<"[" << sys.getNetwork()->getNodeNum()<<"] check location copy first: "<< it->first << " ("<< NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->first )<< ")[ "<<(*(NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->first )))<<" ]" << (void*) gr.getFirstAddress() << " second: " << it->second<< " (" << NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->second )<< ")[ " << (*(NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->second ))) << " do we have cache " << (void *)targetCache << " wanted version " << _newVersion << " "; gr.key->printRegion(gr.id); std::cerr << std::endl;
      }
            if ( !NewNewRegionDirectory::isLocatedIn( _reg.key, it->first, 0 ) &&
                  ( NewNewRegionDirectory::getVersion( _reg.key, it->first, false ) > NewNewRegionDirectory::getVersion( _reg.key, it->second, false ) ||
                    !( NewNewRegionDirectory::hasBeenInvalidated( _reg.key, it->second ) && NewNewRegionDirectory::isLocatedIn( _reg.key, it->second, 0 ) ) ) ) {
              unsigned int loc = NewNewRegionDirectory::getFirstLocation( _reg.key, it->first );
              if (sys.getNetwork()->getNodeNum() == 0)   std::cerr << "Region " << (void*) _reg.key << ":" << it->first << " set to be copied from " << loc << std::endl; 
 ensure( loc != 0 && targetCache == NULL, "impossible")
              opsBySourceRegions[ loc ].push_back( std::make_pair( global_reg_t( it->first, _reg.key ) , this ) );
            }
         }
      } else {
         /* check if version in cache matches desired version. */
         for ( it = _newLocations.begin(); it != _newLocations.end(); it++ ) {
            global_reg_t gr( it->first, _reg.key );
            if(sys.getNetwork()->getNodeNum() == 0) { 
               std::cerr <<"[" << sys.getNetwork()->getNodeNum()<<"] check location copy first: "<< it->first << " ("<< NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->first )<< ")[ "<<(*(NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->first )))<<" ]" << (void*) gr.getFirstAddress() << " second: " << it->second<< " (" << NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->second )<< ")[ " << (*(NewNewRegionDirectory::getDirectoryEntry( *_reg.key, it->second ))) << " do we have cache " << (void *)targetCache << " wanted version " << _newVersion << " "; gr.key->printRegion(gr.id); std::cerr << std::endl;
            }

            std::list< reg_t > components;
            _cacheEntry->NEWaddReadRegion( it->first, NewNewRegionDirectory::getVersion( _reg.key, it->first, false ), _otherPendingOps, components, &_operations, _copy.isOutput() );
            std::list< reg_t >::iterator cit;

            if ( !NewNewRegionDirectory::isLocatedIn( _reg.key, it->first, targetCache->getMemorySpaceId() ) ) {
               for ( cit = components.begin(); cit != components.end(); cit++ ) {
                  global_reg_t r( *cit, _reg.key );
                  if ( it->first != it->second ) {
                     unsigned int loc;
                     fprintf(stderr, "Processing region "); _reg.key->printRegion( _reg.id ); fprintf(stderr, " it->first " );  _reg.key->printRegion( it->first ); fprintf(stderr, " it->second " ); _reg.key->printRegion( it->second ); fprintf(stderr, " *cit " ); _reg.key->printRegion( *cit ); fprintf(stderr, " \n" );
                     if ( NewNewRegionDirectory::hasBeenInvalidated( _reg.key, it->second ) ) {
                        loc = NewNewRegionDirectory::getFirstLocation( _reg.key, it->second );
                     } else {
                        loc = NewNewRegionDirectory::getFirstLocation( _reg.key, it->first ); //FIXME
                     }
                     if ( loc != 0 ) sys.getCaches()[ loc ]->pin( global_reg_t( *cit, _reg.key ) );
                     std::list< std::pair< global_reg_t, CacheCopy * > > &thisCopyOpsRegions = opsBySourceRegions[ loc ];
                     thisCopyOpsRegions.push_back( std::make_pair( global_reg_t( *cit, _reg.key ), this ) );
                  } else { //same first and second:
                     unsigned int loc = NewNewRegionDirectory::getFirstLocation( r.key, r.id );
                     std::list< std::pair< global_reg_t, CacheCopy * > > &thisCopyOpsRegions = opsBySourceRegions[ loc ];
                     fprintf(stderr, "Processing region "); _reg.key->printRegion( _reg.id ); fprintf(stderr, " it->first,second " );  _reg.key->printRegion( it->first ); fprintf(stderr, " *cit " ); _reg.key->printRegion( *cit ); fprintf(stderr, " \n" );
                     std::cerr << "Reg " << *cit << " must be copied from loc " << loc << " comes from reg " << it->second << std::endl;
                     if ( loc != 0 ) sys.getCaches()[ loc ]->pin( global_reg_t( *cit, _reg.key ) );
                     thisCopyOpsRegions.push_back( std::make_pair( global_reg_t( *cit, _reg.key ), this ) );
                  }
               }
            } else {
               if ( !components.empty() ) {
                  std::cerr << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> There are some components that I have to copy because they have been invalidated." << std::endl;
                  for ( cit = components.begin(); cit != components.end(); cit++ ) {
                     std::cerr << "Component: " << *cit << std::endl;
                  }
                  for ( cit = components.begin(); cit != components.end(); cit++ ) {
                     NewNewRegionDirectory::updateFromInvalidated( _reg.key, it->first, it->second );
                     std::cerr << "[I] Reg " << *cit << " must be copied from loc " << NewNewRegionDirectory::getFirstLocation( _reg.key, it->first ) << " comes from reg " << it->second << std::endl;
                     std::list< std::pair< global_reg_t, CacheCopy * > > &thisCopyOpsRegions = opsBySourceRegions[ NewNewRegionDirectory::getFirstLocation( _reg.key, it->first ) ];
 //ensure( NewNewRegionDirectory::getFirstLocation( _reg.key, it->first ) != 0 && targetCache == NULL, "impossible")
                     thisCopyOpsRegions.push_back( std::make_pair( global_reg_t( *cit, _reg.key ), this ) );
                  }
               }
            }
         }
      }
   } else { //else if ( !_copy.isInput() && _copy.isOutput() && targetCache ) {
      ensure( ( !_copy.isInput() && _copy.isOutput() ) , "Wrong copy directions.");
      unsigned int currentVersion = 1;
      for ( it = _newLocations.begin(); it != _newLocations.end(); it++ ) {
         currentVersion = std::max( currentVersion, NewNewRegionDirectory::getVersion( _reg.key, it->first, false ) );
      }
      /* write only region */
      if ( targetCache ) _cacheEntry->NEWaddWriteRegion( _reg.id, currentVersion + 1 );
   }
   if ( targetCache ) _cacheEntry->unlock();
}

inline void CacheCopy::confirmCopyIn( unsigned int memorySpaceId) {
   NewNewRegionDirectory::addAccessRegisterIfNeeded( _reg.key, _reg.id, memorySpaceId, _newVersion );
   //if ( _cacheEntry )
   //_cacheEntry->confirmCopyIn( _reg.id, _newVersion );
}

void CacheCopy::copyDataOut( RegionCache *targetCache ) {
   if ( targetCache ) {
      _cacheEntry->lock();
      _cacheEntry->removeReference();
      _cacheEntry->unlock();
   }
   if ( getCopyData().isOutput() ) {

   if ( sys.usingNewCache() ) {
      NANOS_INSTRUMENT( InstrumentState inst3(NANOS_POST_OUTLINE_WORK3); );
      NewNewRegionDirectory::addAccess( _reg.key, _reg.id, ( !targetCache ) ? 0 : targetCache->getMemorySpaceId(), _newVersion + 1 );
      NANOS_INSTRUMENT( inst3.close(); );
   }

   }
   // TODO: WriteThrough code

   //Region reg = NewRegionDirectory::build_region( *ccopy._copy );
   //std::cerr << "Adding version "<< ccopy._version << " for addr " << ccopy._copy->getBaseAddress() << std::endl;
   //_directory->addAccess( reg, ccopy._copy->isInput(), ccopy._copy->isOutput(), 0, ((uint64_t)ccopy._copy->getBaseAddress()) + ccopy._copy->getOffset(), ccopy._version + 1 );
   //Region origReg = NewRegionDirectory::build_region( *ccopy._copy );
   //_targetCache->syncRegion( reg, (ccopy._cacheEntry->address + ccopy._offset) + ccopy._copy->getOffset() );
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
#ifdef NEWINIT
   unsigned int index;
   for ( index = 0; index < _numCopies; index += 1 ) {
      //std::cerr << "WD "<< _wd.getId() << " Depth: "<< _wd.getDepth() <<" Creating copy "<< index << std::endl;
      //std::cerr << _wd.getCopies()[ index ];
      new ( &_cacheCopies[ index ] ) CacheCopy( _wd, index );
      _cacheCopies[index].getVersionInfo( _wd, index, *this );
   }
   for ( index = 0; index < _numCopies; index += 1 ) {
   }
#else
   unsigned int index;
   for ( index = 0; index < _numCopies; index += 1 ) {
      //std::cerr << "WD "<< _wd.getId() << " Depth: "<< _wd.getDepth() <<" Creating copy "<< index << std::endl;
      //std::cerr << _wd.getCopies()[ index ];
      new ( &_cacheCopies[ index ] ) CacheCopy( _wd, index, *this );
   }
#endif
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
#ifdef NEWINIT
            if ( _cacheCopies[ idx ].getNewVersion() != 0 ) {
               locationInfoReady += 1;
            } else {
               if ( _cacheCopies[ idx ].tryGetLocationNewInit( _wd, idx ) ) {
                  locationInfoReady += 1;
               }
            }
#else
            if ( _cacheCopies[ idx ]._reg.id != 0 ) {
               locationInfoReady += 1;
            } else {
               if ( _cacheCopies[ idx ].tryGetLocation( _wd, idx ) ) {
                  locationInfoReady += 1;
               }
            }
#endif
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
      if(sys.getNetwork()->getNodeNum() == 0) {
         fprintf(stderr, "wd: %d thd: %d ---------------------------- GEN COPIES WD %d TO RUN IN MEMSPACE %d -------------------------\n", _wd.getId(), myThread->getId(), _wd.getId(), (_targetCache ? _targetCache->getMemorySpaceId() : 0 ));
      }
      for ( index = 0; index < _numCopies; index += 1 ) {
         if( sys.getNetwork()->getNodeNum() == 0 ) { 
            fprintf(stderr, "wd: %d thd: %d index: %d : ", _wd.getId(), myThread->getId(), index); 
            _cacheCopies[index]._reg.key->printRegion( _cacheCopies[index]._reg.id );
            std::cerr << std::endl;
         }
         if (!sys.usingNewCache()) {
            _cacheCopies[ index ].generateCopyInOps( _targetCache, opsBySourceRegions );
         } else {
            _cacheCopies[ index ].NEWgenerateCopyInOps( _targetCache, NEWopsBySourceRegions );
         }
      }
      if(sys.getNetwork()->getNodeNum() == 0) {
         fprintf(stderr, "wd: %d thd: %d -----------------------------------------------------\n", _wd.getId(), myThread->getId());
      }
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
                  std::list< std::pair< global_reg_t, CacheCopy * > > &ops = NEWmapOpsStrIt->second;
                  std::cerr << "No Cache sync " << sys.getNetwork()->getNodeNum() << " regs: ";
                  for ( std::list< std::pair< global_reg_t, CacheCopy * > >::iterator listIt = ops.begin(); listIt != ops.end(); listIt++ ) {
                     std::cerr << "[" << (void *) listIt->first.key << "," << listIt->first.id << "]";
                  }
                  std::cerr << std::endl;
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

   if ( allReady && !_registered ) {
      if ( sys.usingNewCache() ) {
         for ( index = 0; index < _numCopies; index += 1 ) {
            CacheCopy &ccopy = _cacheCopies[ index ];
            ccopy.confirmCopyIn( ( !_targetCache ) ? 0 : _targetCache->getMemorySpaceId() );
            //if (sys.getNetwork()->getNodeNum() == 0)         std::cerr<<"wd: "<< _wd.getId() <<" add access to " << (( !_targetCache ) ? 0 : _targetCache->getMemorySpaceId()) << " " << (void *)ccopy.getRegionDirectoryKey() << ":" <<  ccopy.getRegId() << std::endl;
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

      ccopy.copyDataOut( _targetCache );

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

bool CacheController::hasVersionInfoForRegion( global_reg_t reg, unsigned int &version, NewLocationInfoList &locations ) {
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

CompleteOpFunctor::CompleteOpFunctor( DeviceOps *ops, AllocatedChunk *chunk ) : _ops( ops ), _chunk( chunk ) {
}

CompleteOpFunctor::~CompleteOpFunctor() {
}

void CompleteOpFunctor::operator()() {
   //fprintf(stderr, "EXECURE FUNCTOR!!! \n");
   //_ops->completeOp();
   _chunk->removeReference();
}

//unsigned int RegionCache::getVersionAllocateChunkIfNeeded( global_reg_t const &reg, bool increaseVersion ) {
//   RegionTree< CachedRegionStatus > *regsToInvalidate = NULL;
//   CacheRegionDictionary *newRegsToInvalidate = NULL;
//   AllocatedChunk *chunk = getAddress( reg, regsToInvalidate, newRegsToInvalidate );
//   unsigned int version = chunk->getVersion( reg, increaseVersion );
//   chunk->unlock();
//   return version;
//}

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

void RegionCache::releaseRegion( global_reg_t const &reg ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
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
   RegionTree< CachedRegionStatus > *regsToInvalidate = NULL;
   CacheRegionDictionary *newRegsToInvalidate = NULL;
   AllocatedChunk *chunk = getAddress( reg, regsToInvalidate, newRegsToInvalidate, wd );
   chunk->unlock();
}

void RegionCache::setRegionVersion( global_reg_t const &hostMem, unsigned int version ) {
   AllocatedChunk *chunk = getAllocatedChunk( hostMem );
   chunk->NEWaddWriteRegion( hostMem.id, version );
   chunk->unlock();
}

void RegionCache::copyInputData( BaseAddressSpaceInOps &ops, global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   std::set< reg_t > notPresentParts;
   //      std::cerr << "locations:  ";
   //      for ( NewLocationInfoList::const_iterator it2 = locations.begin(); it2 != locations.end(); it2++ ) {
   //         std::cerr << "[ " << it2->first << "," << it2->second << " ] ";
   //      }
   //      std::cerr << std::endl;
   if ( chunk->NEWaddReadRegion2( ops, reg.id, version, ops.getOtherOps(), notPresentParts, ops.getOwnOps(), output, locations ) ) {
   }

   reg.setLocationAndVersion( _memorySpaceId, version + ( output ? 1 : 0 ) );
   chunk->unlock();
}

void RegionCache::allocateOutputMemory( global_reg_t const &reg, unsigned int version ) {
   AllocatedChunk *chunk = getAllocatedChunk( reg );
   chunk->NEWaddWriteRegion( reg.id, version );
   reg.setLocationAndVersion( _memorySpaceId, version );
   chunk->unlock();
}


