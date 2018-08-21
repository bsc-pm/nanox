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

#ifdef STANDALONE_TEST

#ifdef message
#undef message
#define message(x)
#else
#define message(x)
#endif
#ifdef ensure
#undef ensure
#define ensure(x,y)
#else
#define ensure(x,y)
#endif
#ifndef NEW
#define NEW new
#endif

#else
#include "basethread.hpp"
#include "debug.hpp"
#endif

#include "regiondirectory.hpp"
#include "hashfunction_decl.hpp"
#include "regiondict.hpp"
#include "os.hpp"
#include "globalregt.hpp"

#if VERBOSE_CACHE
 #define _VERBOSE_CACHE 1
#else
 #define _VERBOSE_CACHE 0
 //#define _VERBOSE_CACHE ( sys.getNetwork()->getNodeNum() == 0 )
#endif


namespace nanos {

std::ostream & operator<< (std::ostream &o, DirectoryEntryData const &ent)
{
   //o << "WL: " << ent._writeLocation << " V: " << ent.getVersion() << " Locs: ";
   o << " V: " << ent.getVersion() << " Locs: ";
   for ( std::set<memory_space_id_t>::iterator it = ent._location.begin(); it != ent._location.end(); it++ ) {
      o << *it << " ";
   }
   o << "R: " << ent.getRootedLocation();
   return o;
}

RegionDirectory::HashBucket::HashBucket() : _lock(), _bobjects( NULL ) { }

RegionDirectory::HashBucket::HashBucket( RegionDirectory::HashBucket const &hb ) : _lock(), _bobjects( hb._bobjects ) { }

RegionDirectory::HashBucket &RegionDirectory::HashBucket::operator=( RegionDirectory::HashBucket const &hb ) {
   _bobjects = hb._bobjects;
   return *this;
}
RegionDirectory::HashBucket::~HashBucket() { }

#define HASH_BUCKETS 256

RegionDirectory::RegionDirectory() : _keys(), _keysSeed( 1 ),
   _keysLock(), _objects( HASH_BUCKETS, HashBucket() ) {}

uint64_t RegionDirectory::_getKey( uint64_t addr, std::size_t len, WD const *wd ) {
   bool exact;
   while ( !_keysLock.tryAcquire() ) {
      myThread->processTransfers();
   }
   uint64_t keyIfNotFound = ( _keysSeed + 1 == 0 ) ? 1 : _keysSeed + 1;
   //*myThread->_file << __func__ << " with addr " << (void *) addr << " and size " << len << " wd " << ( wd != NULL ? wd->getId() : -1 ) << " [ " << ( wd != NULL ? ( ( wd->getDescription() != NULL) ? wd->getDescription() : "wd desc. not available" ) : "null WD, comming from nanos_register probably" ) << " ] " << std::endl;
   uint64_t conflict_addr = 0;
   std::size_t conflict_size = 0;
   uint64_t key = _keys.getExactOrFullyOverlappingInsertIfNotFound( addr, len, exact, keyIfNotFound, 0, conflict_addr, conflict_size );
   if ( key == 0 ) {
      printBt(*myThread->_file);
      fatal("invalid key, can not continue. Address " << (void *) addr << " w/len " << len << " [" << ( wd != NULL ? ( ( wd->getDescription() != NULL) ? wd->getDescription() : "wd desc. not available" ) : "null WD, comming from nanos_register probably" ) << "] conflicts with address: " << (void *) conflict_addr << ", size: " << conflict_size );
   } else if ( key == keyIfNotFound ) {
      _keysSeed += 1;
   }
   _keysLock.release();
   return key;
}

uint64_t RegionDirectory::_getKey( uint64_t addr ) const {
   uint64_t key = _keys.getExactByAddress( addr, 0 );
   return key;
}

GlobalRegionDictionary *RegionDirectory::getRegionDictionaryRegisterIfNeeded( CopyData const &cd, WD const *wd ) {
   uint64_t objectAddr = ( cd.getHostBaseAddress() == 0 ? ( uint64_t ) cd.getBaseAddress() : cd.getHostBaseAddress() );
   std::size_t objectSize = cd.getMaxSize();
#if 0
   unsigned int key = ( jen_hash( objectAddr ) & (HASH_BUCKETS-1) );
#else
   uint64_t key = jen_hash( this->_getKey( objectAddr, objectSize, wd ) ) & (HASH_BUCKETS-1);
#endif
   HashBucket &hb = _objects[ key ];
   GlobalRegionDictionary *dict = NULL;

   while ( !hb._lock.tryAcquire() ) {
      myThread->processTransfers();
   }

   if ( hb._bobjects == NULL ) {
      hb._bobjects = NEW MemoryMap< Object >();
   }
   bool exact = false;
   Object **o = hb._bobjects->getExactOrFullyOverlappingInsertIfNotFound( objectAddr, objectSize, exact );
   if ( o != NULL ) {
      if ( *o == NULL ) {
         *o = NEW Object( NEW GlobalRegionDictionary( cd ) );
         dict = (*o)->getGlobalRegionDictionary();
         DirectoryEntryData *entry = getDirectoryEntry( *dict, 1 );
         if ( entry == NULL ) {
            entry = NEW DirectoryEntryData();
            dict->setRegionData( 1, entry ); //getRegionDictionaryRegisterIfNeeded
            if ( sys.getVerboseCopies() ) {
               *myThread->_file << "New object: ";
               dict->printRegion(*myThread->_file, 1);
               *myThread->_file << " from WD " << wd->getId() << " [" << ( ( wd->getDescription() != NULL) ? wd->getDescription() : "n/a" ) << "]" << std::endl;
            }
         }
      } else {
         /* already registered */
         dict = (*o)->getGlobalRegionDictionary();
         if ( dict != NULL ) {
            if ( (*o)->getRegisteredObject() != NULL ) {
               /* object pre registered with CopyData */
               dict->setRegisteredObject( (*o)->getRegisteredObject() );
            }
         } else {
            dict = NEW GlobalRegionDictionary( cd );
            DirectoryEntryData *entry = getDirectoryEntry( *dict, 1 );
            if ( entry == NULL ) {
               entry = NEW DirectoryEntryData();
               dict->setRegionData( 1, entry ); //getRegionDictionaryRegisterIfNeeded
            }
            (*o)->setGlobalRegionDictionary( dict );
         }
      }
   } else {
      /* not found and could not insert a new one */
      fatal("Unable to register prorgam object: " << cd );
   }
   hb._lock.release();
   return dict;
}

GlobalRegionDictionary *RegionDirectory::getRegionDictionary( CopyData const &cd ) {
   uint64_t objectAddr = ( cd.getHostBaseAddress() == 0 ? ( uint64_t ) cd.getBaseAddress() : cd.getHostBaseAddress() );
   return getRegionDictionary( objectAddr, false );
}

GlobalRegionDictionary *RegionDirectory::getRegionDictionary( uint64_t objectAddr, bool canFail ) {
#if 0
   unsigned int key = ( jen_hash( objectAddr ) & (HASH_BUCKETS-1) );
#else
   uint64_t key = jen_hash( this->_getKey( objectAddr ) ) & (HASH_BUCKETS-1);
#endif
   HashBucket &hb = _objects[ key ];
   GlobalRegionDictionary *dict = NULL;

#if 0
   std::map< uint64_t, Object >::const_iterator it = hb._bobjects.lower_bound( objectAddr );
   if ( it == hb._bobjects.end() || hb._bobjects.key_comp()( objectAddr, it->first) ) {
     *(myThread->_file) << "Error, CopyData object not registered in the RegionDictionary " << (void *) objectAddr << std::endl;
     printBt( *(myThread->_file) );
     fatal("can not continue");
   }
#endif
   while ( !hb._lock.tryAcquire() ) {
      myThread->processTransfers();
   }
   if ( hb._bobjects == NULL ) {
      if ( !canFail ) {
         *(myThread->_file) << "Error, CopyData object not registered in the RegionDictionary " << (void *) objectAddr << std::endl;
         printBt( *(myThread->_file) );
         fatal("can not continue");
      }
   } else {
      Object *o = hb._bobjects->getExactByAddress( objectAddr );
      if ( o == NULL ) {
         if ( !canFail ) {
            *(myThread->_file) << "Error, CopyData object not registered in the RegionDictionary " << (void *) objectAddr << std::endl;
            printBt( *(myThread->_file) );
            fatal("can not continue");
         }
      } else {
         dict = o->getGlobalRegionDictionary();
      }
   }
   hb._lock.release();
   return dict;
}

void RegionDirectory::addAccess( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe, memory_space_id_t loc, unsigned int version )
{
   if (dict->getVersion() < version ) dict->setVersion( version );
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   regEntry->addAccess( pe, loc, version ); //addAccess IMPL
}

void RegionDirectory::addRootedAccess( RegionDirectoryKey dict, reg_t id, memory_space_id_t loc, unsigned int version )
{
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   regEntry->addRootedAccess( loc, version );
}

DirectoryEntryData *RegionDirectory::getDirectoryEntry( GlobalRegionDictionary &dict, reg_t id ) {
   DirectoryEntryData *entry = ( DirectoryEntryData * ) dict.getRegionData( id ); //getDirectoryEntry
   return entry;
}

bool RegionDirectory::delAccess( RegionDirectoryKey dict, reg_t id, memory_space_id_t memorySpaceId ) {
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   bool res;
   res = regEntry->delAccess( memorySpaceId );
   return res;
}

bool RegionDirectory::isOnlyLocated( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe ) {
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   bool res;
   res = ( ( regEntry->isLocatedIn( pe ) ) && ( regEntry->getNumLocations() == 1 ) );
   return res;
}

bool RegionDirectory::isOnlyLocated( RegionDirectoryKey dict, reg_t id, memory_space_id_t loc ) {
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   bool res;
   res = ( ( regEntry->isLocatedIn( loc ) ) && ( regEntry->getNumLocations() == 1 ) );
   return res;
}

void RegionDirectory::updateFromInvalidated( RegionDirectoryKey dict, reg_t id, reg_t from ) {
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   DirectoryEntryData *fromEntry = getDirectoryEntry( *dict, from );
   *regEntry = *fromEntry;
}

void RegionDirectory::print() const {
   for ( std::vector< HashBucket >::const_iterator bit = _objects.begin(); bit != _objects.end(); bit++ ) {
      HashBucket const &hb = *bit;
      if ( hb._bobjects != NULL ) {
         for ( MemoryMap<Object>::const_iterator it = hb._bobjects->begin(); it != hb._bobjects->end(); it++ ) {
            GlobalRegionDictionary *dict = it->second->getGlobalRegionDictionary();
            if ( dict == NULL ) continue;
            *myThread->_file <<"Object "<< (void*)dict << std::endl;
            for (reg_t i = 1; i < dict->getMaxRegionId(); i++ ) {
               DirectoryEntryData *entry = ( DirectoryEntryData * ) dict->getRegionData( i ); //print
               if ( !entry ) {
                  *myThread->_file << "\t" << i << " "; dict->printRegion( *myThread->_file, i ); *myThread->_file << " : null " << std::endl;
               } else {
                  *myThread->_file << "\t" << i << " "; dict->printRegion( *myThread->_file, i ); *myThread->_file << " : ("<< entry <<") "<< *entry << std::endl;
               }
            }
         }
      }
   }
}


unsigned int RegionDirectory::getVersion( RegionDirectoryKey dict, reg_t id, bool increaseVersion ) {
   DirectoryEntryData *entry = getDirectoryEntry( *dict, id );
   return entry->getVersion( increaseVersion );
}

bool RegionDirectory::isLocatedIn( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe, unsigned int version ) {
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   return regEntry->isLocatedIn( pe, version );
}

bool RegionDirectory::isLocatedIn( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe ) {
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   //*myThread->_file << dict << " IS LOCATED " << id << " in loc " << loc <<" entry is " <<*regEntry  << std::endl;
   return (regEntry) ? regEntry->isLocatedIn( pe ) : 0;
}

bool RegionDirectory::isLocatedIn( RegionDirectoryKey dict, reg_t id, memory_space_id_t loc ) {
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   //*myThread->_file << dict << " IS LOCATED " << id << " in loc " << loc <<" entry is " <<*regEntry  << std::endl;
   return (regEntry) ? regEntry->isLocatedIn( loc ) : 0;
}

unsigned int RegionDirectory::getFirstLocation( RegionDirectoryKey dict, reg_t id ) {
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   return regEntry->getFirstLocation();
}

GlobalRegionDictionary &RegionDirectory::getDictionary( CopyData const &cd ) {
   return *getRegionDictionary( cd );
}

void RegionDirectory::_invalidateObjectsFromDevices( std::map< uint64_t, MemoryMap< Object > * > &objects ) {
   for ( std::map< uint64_t, MemoryMap< Object > * >::iterator it = objects.begin(); it != objects.end(); it++ ) {
      for ( memory_space_id_t id = 1; id <= sys.getSeparateMemoryAddressSpacesCount(); id++ ) {
         Object *o = it->second->getExactByAddress(it->first);
         sys.getSeparateMemory( id ).invalidate( global_reg_t( 1, o->getGlobalRegionDictionary() ) );
      }
   }
}

RegionDirectory::~RegionDirectory() {
   for ( std::vector< HashBucket >::iterator bit = _objects.begin(); bit != _objects.end(); bit++ ) {
      HashBucket &hb = *bit;
      delete hb._bobjects;
   }
}

void RegionDirectory::_unregisterObjects( std::map< uint64_t, MemoryMap< Object > * > &objects ) {
   for ( std::map< uint64_t, MemoryMap< Object > * >::iterator it = objects.begin(); it != objects.end(); it++ ) {
      Object *o = it->second->getExactByAddress(it->first);
      sys.getNetwork()->deleteDirectoryObject( o->getGlobalRegionDictionary() );
      it->second->eraseByAddress( it->first );
      if ( o->getRegisteredObject() != NULL ) {
         o->resetGlobalRegionDictionary();
         CopyData *cd = o->getRegisteredObject();
         Object **dict_o = it->second->getExactInsertIfNotFound( (uint64_t) cd->getBaseAddress(), cd->getMaxSize() );
         if ( dict_o != NULL ) {
            if ( *dict_o == NULL ) {
               *dict_o = o;
            } else {
               /* something went wrong, we cleared the dictionary so
                * this call must return an new object pointing to NULL
                */
               fatal("Dictionary error.");
            }
         } else {
            /* something went wrong, we cleared the dictionary so
             * this call can not return NULL at this point
             */
            fatal("Dictionary error.");
         }
      } else {
         _keys.eraseByAddress( it->first );
         delete o;
      }
   }
}

void RegionDirectory::synchronize( WD &wd, void *addr ) {
   //std::ostream &o = (*myThread->_file);
   //o << "++++ WaitOn synchronize, w addr " << addr << std::endl;
   uint64_t objectAddr = (uint64_t) addr;

   GlobalRegionDictionary *dict = getRegionDictionary( objectAddr, true );
   if ( dict == NULL ) {
      return;
   }
   std::map< uint64_t, MemoryMap< Object > * > objects_to_clear;

   std::list< std::pair< reg_t, reg_t > > missingParts;
   unsigned int version = 0;
   /*reg_t lol =*/ dict->registerRegion(1, missingParts, version);
   uint64_t key = jen_hash( this->_getKey( objectAddr ) ) & (HASH_BUCKETS-1);
   HashBucket &hb = _objects[ key ];
   ensure( hb._bobjects != NULL, "null dictionary");
   objects_to_clear.insert( std::make_pair( objectAddr, hb._bobjects ) );
   SeparateAddressSpaceOutOps outOps( myThread->runningOn(), true, false );

   for ( std::list< std::pair< reg_t, reg_t > >::iterator mit = missingParts.begin(); mit != missingParts.end(); mit++ ) {
      //*myThread->_file << "sync region " << mit->first << " : "<< ( void * ) dict->getRegionData( mit->first ) <<" with second reg " << mit->second << " : " << ( void * ) dict->getRegionData( mit->second )<< std::endl;
      if ( mit->first == mit->second ) {
         global_reg_t reg( mit->first, dict );
         if ( !reg.isRooted() ) { //ignore regions rooted to a certain location
            if ( !reg.isLocatedIn( 0 ) ) {
               DeviceOps *thisOps = reg.getDeviceOps();
               if ( thisOps->addCacheOp( /* debug: */ &wd ) ) {
                  DirectoryEntryData *entry = ( DirectoryEntryData * ) reg.key->getRegionData( reg.id  ); //synchronize
                  if ( _VERBOSE_CACHE ) {
                     *myThread->_file << "f SYNC REGION! "; reg.key->printRegion( *myThread->_file, reg.id );
                     if ( entry ) *myThread->_file << " " << *entry << std::endl;
                     else *myThread->_file << " nil " << std::endl;
                  }
                  //*myThread->_file << " reg is in: " << reg.getFirstLocation() << std::endl;
                  outOps.addOutOp( 0 /* sync only non rooted objects */, reg.getFirstLocation(), reg, reg.getVersion(), thisOps, wd, (unsigned int)0xdeadbeef ); //Out op synchronize
                  outOps.insertOwnOp( thisOps, reg, reg.getVersion()+1, 0 ); //increase version to invalidate the device copy
               } else {
                  outOps.getOtherOps().insert( thisOps );
               }
            }
            // another mechanism to inval data: else if ( reg.getNumLocations() > 1 ) {
            // another mechanism to inval data:    //*myThread->_file << " have too upgrade host region" << std::endl;
            // another mechanism to inval data:    reg.setLocationAndVersion( 0, reg.getVersion()+1 ); //increase version to invalidate the device copy
            // another mechanism to inval data: }

            // aggregate the locations, later, we will invalidate the full object from those locations
            // locations[dict].insert(reg.getLocations().begin(), reg.getLocations().end()); //this requires delayedCommit = yes in the ops object!! FIXME
         } else {
            objects_to_clear.erase( objectAddr );
         }
      } else {
         global_reg_t region_shape( mit->first, dict );
         global_reg_t data_source( mit->second, dict );
         if ( !data_source.isRooted() ) { //ignore regions rooted to a certain location
            if ( !data_source.isLocatedIn( 0 ) ) {
               //*myThread->_file << "FIXME: I should sync region! " << region_shape.id << " "; region_shape.key->printRegion( region_shape.id ); *myThread->_file << std::endl;
               //*myThread->_file << "FIXME: I should sync region! " << data_source.id << " "; data_source.key->printRegion( data_source.id ); *myThread->_file << std::endl;
               DirectoryEntryData *regEntry = getDirectoryEntry( *region_shape.key, region_shape.id );
               if ( regEntry == NULL ) {
                  regEntry = NEW DirectoryEntryData();
                  region_shape.key->setRegionData( region_shape.id, regEntry );
               }
               DeviceOps *thisOps = regEntry->getOps();
               if ( thisOps->addCacheOp( /* debug: */ &wd ) ) {
                  DirectoryEntryData *entry = ( DirectoryEntryData * ) data_source.key->getRegionData( data_source.id  ); //synchronize
                  if ( _VERBOSE_CACHE ) {
                     *myThread->_file << " SYNC REGION! "; region_shape.key->printRegion( *myThread->_file, region_shape.id );
                     if ( entry ) *myThread->_file << " " << *entry << std::endl;
                     else *myThread->_file << " nil " << std::endl;
                  }
                  //*myThread->_file << " reg is in: " << reg.getFirstLocation() << std::endl;
                  outOps.addOutOp( 0 /* sync only non rooted objects */, data_source.getFirstLocation(), region_shape, data_source.getVersion(), thisOps, wd, (unsigned int)0xdeadbeef ); //Out op synchronize
                  outOps.insertOwnOp( thisOps, region_shape, data_source.getVersion()+1, 0 ); //increase version to invalidate the device copy
               } else {
                  outOps.getOtherOps().insert( thisOps );
               }
            }
         } else {
            objects_to_clear.erase( objectAddr );
         }
      }
   }
   //bool orig_verbose_devops = sys.getVerboseDevOps();
   //sys.setVerboseDevOps( true );
   outOps.issue( &wd );
   while ( !outOps.isDataReady( wd ) ) { myThread->processTransfers(); }
   //sys.setVerboseDevOps( orig_verbose_devops );

   if ( wd.getDepth() == 0 ) {
      // invalidate data on devices
      _invalidateObjectsFromDevices( objects_to_clear );

      //clear objects from directory
      _unregisterObjects( objects_to_clear );
      if ( sys.usingCluster() ) {
         for ( std::map< uint64_t, MemoryMap< Object > * >::iterator it = objects_to_clear.begin(); it != objects_to_clear.end(); it++ ) {
            sys.getNetwork()->synchronizeDirectory( (void *) it->first );
         }
      }
   }
   //o << "++++ DONE ++++ WaitOn synchronize, w addr " << addr << std::endl;
}

void RegionDirectory::synchronize( WD &wd, std::size_t numDataAccesses, DataAccess *data )
{
   SeparateAddressSpaceOutOps outOps( myThread->runningOn(), true, false );

   std::map< uint64_t, MemoryMap< Object > * > objects_to_clear;

   for ( std::size_t idx = 0; idx < numDataAccesses; idx += 1 ) {

      uint64_t objectAddr = (uint64_t) data[idx].getDepAddress(); //NOTE: Using the region base address in the host
      GlobalRegionDictionary *dict = getRegionDictionary( objectAddr, true );
      if ( dict == NULL ) continue;

      std::list< std::pair< reg_t, reg_t > > missingParts;
      unsigned int version = 0;
      dict->registerRegion(1, missingParts, version );
      uint64_t key = jen_hash( this->_getKey( objectAddr ) ) & (HASH_BUCKETS-1);
      HashBucket &hb = _objects[ key ];
      ensure( hb._bobjects != NULL, "null dictionary");

      if ( data[idx].isOutput() )  objects_to_clear.insert( std::make_pair( objectAddr, hb._bobjects ) );

      for ( std::list< std::pair< reg_t, reg_t > >::iterator mit = missingParts.begin(); mit != missingParts.end(); mit++ ) {
         if ( mit->first == mit->second ) {
            global_reg_t reg( mit->first, dict );
            if ( !reg.isRooted() ) { //ignore regions rooted to a certain location
               if ( !reg.isLocatedIn( 0 ) ) {
                  DeviceOps *thisOps = reg.getDeviceOps();
                  if ( thisOps->addCacheOp( /* debug: */ &wd ) ) {
                     DirectoryEntryData *entry = ( DirectoryEntryData * ) reg.key->getRegionData( reg.id  ); //synchronize
                     if ( _VERBOSE_CACHE ) {
                        *myThread->_file << "f SYNC REGION! "; reg.key->printRegion( *myThread->_file, reg.id );
                        if ( entry ) *myThread->_file << " " << *entry << std::endl;
                        else *myThread->_file << " nil " << std::endl;
                     }
                     outOps.addOutOp( 0 /* sync only non rooted objects */, reg.getFirstLocation(), reg, reg.getVersion(), thisOps, wd, (unsigned int)0xdeadbeef ); //Out op synchronize
                     outOps.insertOwnOp( thisOps, reg, reg.getVersion()+1, 0 ); //increase version to invalidate the device copy
                  } else {
                     outOps.getOtherOps().insert( thisOps );
                  }
               }
            } else {
               objects_to_clear.erase( objectAddr );
            }
         } else {
            global_reg_t region_shape( mit->first, dict );
            global_reg_t data_source( mit->second, dict );
            if ( !data_source.isRooted() ) { //ignore regions rooted to a certain location
               if ( !data_source.isLocatedIn( 0 ) ) {
                  DirectoryEntryData *regEntry = getDirectoryEntry( *region_shape.key, region_shape.id );
                  if ( regEntry == NULL ) {
                     regEntry = NEW DirectoryEntryData();
                     region_shape.key->setRegionData( region_shape.id, regEntry );
                  }
                  DeviceOps *thisOps = regEntry->getOps();
                  if ( thisOps->addCacheOp( /* debug: */ &wd ) ) {
                     DirectoryEntryData *entry = ( DirectoryEntryData * ) data_source.key->getRegionData( data_source.id  ); //synchronize
                     if ( _VERBOSE_CACHE ) {
                        *myThread->_file << " SYNC REGION! "; region_shape.key->printRegion( *myThread->_file, region_shape.id );
                        if ( entry ) *myThread->_file << " " << *entry << std::endl;
                        else *myThread->_file << " nil " << std::endl;
                     }
                     outOps.addOutOp( 0 /* sync only non rooted objects */, data_source.getFirstLocation(), region_shape, data_source.getVersion(), thisOps, wd, (unsigned int)0xdeadbeef ); //Out op synchronize
                     outOps.insertOwnOp( thisOps, region_shape, data_source.getVersion()+1, 0 ); //increase version to invalidate the device copy
                  } else {
                     outOps.getOtherOps().insert( thisOps );
                  }
               }
            } else {
               objects_to_clear.erase( objectAddr );
            }
         }
      }
   } // end of iterate DataAccesses

   outOps.issue( &wd );
   while ( !outOps.isDataReady( wd ) ) { myThread->processTransfers(); }

   if ( wd.getDepth() == 0 ) {
      // invalidate data on devices
      _invalidateObjectsFromDevices( objects_to_clear );

      //clear objects from directory
      _unregisterObjects( objects_to_clear );
      if ( sys.usingCluster() ) {
         for ( std::map< uint64_t, MemoryMap< Object > * >::iterator it = objects_to_clear.begin(); it != objects_to_clear.end(); it++ ) {
            sys.getNetwork()->synchronizeDirectory( (void *) it->first );
         }
      }
   }
}

void RegionDirectory::synchronize( WD &wd ) {
   std::ostream &o = (*myThread->_file);
   //o << "++++ WaitOn synchronize ALL" << std::endl;
   //*myThread->_file << "SYNC DIR with wd " << wd.getId() << std::endl;
   //int c = 0;
   //print();
   if ( sys.getVerboseCopies() )  {
      //std::ostream &o = (*myThread->_file);
      o << "Synchronize directory from WD " << wd.getId() << " [" << ( ( wd.getDescription() != NULL) ? wd.getDescription() : "n/a" ) << "]" << std::endl;
   }
   if ( sys.getSeparateMemoryAddressSpacesCount() == 0 ) {

      std::map< uint64_t, MemoryMap< Object > * > objects_to_clear;

      for ( std::vector< HashBucket >::iterator bit = _objects.begin(); bit != _objects.end(); bit++ ) {
         HashBucket &hb = *bit;
         while ( !hb._lock.tryAcquire() ) {
            myThread->processTransfers();
         }
         if ( hb._bobjects != NULL ) {
            for ( MemoryMap<Object>::iterator it = hb._bobjects->begin(); it != hb._bobjects->end(); it++ ) {
               GlobalRegionDictionary *dict = it->second->getGlobalRegionDictionary();
               if ( dict == NULL ) continue;
               uint64_t objectAddr = it->first.getAddress();
               if ( !wd._mcontrol.hasObjectOfRegion( global_reg_t( 1, dict ) ) ) {
                  if ( sys.getVerboseCopies() ) {
                     //std::ostream &o = (*myThread->_file);
                     o << "Not synchronizing this object! "; dict->printRegion( o, 1 ); o << std::endl;
                  }
                  continue;
               }
               if ( dict->getKeepAtOrigin() ) continue;
               std::list< std::pair< reg_t, reg_t > > missingParts;
               unsigned int version = 0;
               //double tini = OS::getMonotonicTime();
               /*reg_t lol =*/ dict->registerRegion(1, missingParts, version);
               objects_to_clear.insert( std::make_pair( objectAddr, hb._bobjects ) );

               for ( std::list< std::pair< reg_t, reg_t > >::iterator mit = missingParts.begin(); mit != missingParts.end(); mit++ ) {
                  //*myThread->_file << "sync region " << mit->first << " : "<< ( void * ) dict->getRegionData( mit->first ) <<" with second reg " << mit->second << " : " << ( void * ) dict->getRegionData( mit->second )<< std::endl;
                  if ( mit->first == mit->second ) {
                     global_reg_t reg( mit->first, dict );
                     if ( reg.isRooted() ) { //ignore regions rooted to a certain location
                        objects_to_clear.erase( objectAddr );
                     }
                  } else {
                     global_reg_t region_shape( mit->first, dict );
                     global_reg_t data_source( mit->second, dict );
                     if ( data_source.isRooted() ) { //ignore regions rooted to a certain location
                        objects_to_clear.erase( objectAddr );
                     }
                  }
               }
            }
         }
         hb._lock.release();
      }

      if ( wd.getDepth() == 0 ) {
         _unregisterObjects( objects_to_clear );
      }
      return;
   }

   SeparateAddressSpaceOutOps outOps( myThread->runningOn(), true, false );
   std::map< GlobalRegionDictionary *, std::set< memory_space_id_t > > locations;
   //std::map< uint64_t, std::map< uint64_t, Object > * > objects_to_clear;
   std::map< uint64_t, MemoryMap< Object > * > objects_to_clear;

   for ( std::vector< HashBucket >::iterator bit = _objects.begin(); bit != _objects.end(); bit++ ) {
      HashBucket &hb = *bit;
      while ( !hb._lock.tryAcquire() ) {
         myThread->processTransfers();
      }
      if ( hb._bobjects != NULL ) {
         for ( MemoryMap<Object>::iterator it = hb._bobjects->begin(); it != hb._bobjects->end(); it++ ) {
            GlobalRegionDictionary *dict = it->second->getGlobalRegionDictionary();
            //*myThread->_file << "==================  start object " << ++c << "("<< dict <<") ================="<<std::endl;
            //if ( it->second->getKeepAtOrigin() ) {
            //   *myThread->_file << "Object " << it->second << " Keep " << std::endl;
            //}
            if ( dict == NULL ) continue;
            uint64_t objectAddr = it->first.getAddress();
            if ( !wd._mcontrol.hasObjectOfRegion( global_reg_t( 1, dict ) ) ) {
               if ( sys.getVerboseCopies() ) {
                  //std::ostream &o = (*myThread->_file);
                  o << "Not synchronizing this object! "; dict->printRegion( o, 1 ); o << std::endl;
               }
               continue;
            }
            if ( dict->getKeepAtOrigin() ) continue;

            std::list< std::pair< reg_t, reg_t > > missingParts;
            unsigned int version = 0;
            //double tini = OS::getMonotonicTime();
            /*reg_t lol =*/ dict->registerRegion(1, missingParts, version);
            //double tfini = OS::getMonotonicTime();
            //*myThread->_file << __FUNCTION__ << " addRegion time " << (tfini-tini) << std::endl;
            //*myThread->_file << "Missing parts are: (want version) "<< version << " got " << lol << " { ";
            //for ( std::list< std::pair< reg_t, reg_t > >::iterator mit = missingParts.begin(); mit != missingParts.end(); mit++ ) {
            //   *myThread->_file <<"("<< mit->first << "," << mit->second << ") ";
            //}
            //*myThread->_file << "}"<<std::endl;

            objects_to_clear.insert( std::make_pair( objectAddr, hb._bobjects ) );

            for ( std::list< std::pair< reg_t, reg_t > >::iterator mit = missingParts.begin(); mit != missingParts.end(); mit++ ) {
               //*myThread->_file << "sync region " << mit->first << " : "<< ( void * ) dict->getRegionData( mit->first ) <<" with second reg " << mit->second << " : " << ( void * ) dict->getRegionData( mit->second )<< std::endl;
               if ( mit->first == mit->second ) {
                  global_reg_t reg( mit->first, dict );
                  if ( !reg.isRooted() ) { //ignore regions rooted to a certain location
                     if ( !reg.isLocatedIn( 0 ) ) {
                        DeviceOps *thisOps = reg.getDeviceOps();
                        if ( thisOps->addCacheOp( /* debug: */ &wd ) ) {
                           DirectoryEntryData *entry = ( DirectoryEntryData * ) reg.key->getRegionData( reg.id  ); //synchronize
                           if ( _VERBOSE_CACHE ) {
                              *myThread->_file << "f SYNC REGION! "; reg.key->printRegion( *myThread->_file, reg.id );
                              if ( entry ) *myThread->_file << " " << *entry << std::endl;
                              else *myThread->_file << " nil " << std::endl;
                           }
                           //*myThread->_file << " reg is in: " << reg.getFirstLocation() << std::endl;
                           outOps.addOutOp( 0 /* sync only non rooted objects */, reg.getFirstLocation(), reg, reg.getVersion(), thisOps, wd, (unsigned int)0xdeadbeef ); //Out op synchronize
                           outOps.insertOwnOp( thisOps, reg, reg.getVersion()+1, 0 ); //increase version to invalidate the device copy
                        } else {
                           outOps.getOtherOps().insert( thisOps );
                        }
                     }
                     // another mechanism to inval data: else if ( reg.getNumLocations() > 1 ) {
                     // another mechanism to inval data:    //*myThread->_file << " have too upgrade host region" << std::endl;
                     // another mechanism to inval data:    reg.setLocationAndVersion( 0, reg.getVersion()+1 ); //increase version to invalidate the device copy
                     // another mechanism to inval data: }

                     // aggregate the locations, later, we will invalidate the full object from those locations
                     locations[dict].insert(reg.getLocations().begin(), reg.getLocations().end()); //this requires delayedCommit = yes in the ops object!! FIXME
                  } else {
                     //objects_to_clear.insert( std::make_pair( objectAddr, &hb._bobjects ) ); //FIXME: objects may be added later
                     objects_to_clear.erase( objectAddr );
                  }
               } else {
                  global_reg_t region_shape( mit->first, dict );
                  global_reg_t data_source( mit->second, dict );
                  if ( !data_source.isRooted() ) { //ignore regions rooted to a certain location
                     if ( !data_source.isLocatedIn( 0 ) ) {
                        //*myThread->_file << "FIXME: I should sync region! " << region_shape.id << " "; region_shape.key->printRegion( region_shape.id ); *myThread->_file << std::endl;
                        //*myThread->_file << "FIXME: I should sync region! " << data_source.id << " "; data_source.key->printRegion( data_source.id ); *myThread->_file << std::endl;
								DirectoryEntryData *regEntry = getDirectoryEntry( *region_shape.key, region_shape.id );
								if ( regEntry == NULL ) {
									regEntry = NEW DirectoryEntryData();
									region_shape.key->setRegionData( region_shape.id, regEntry );
								}
								DeviceOps *thisOps = regEntry->getOps();
                        if ( thisOps->addCacheOp( /* debug: */ &wd ) ) {
                           DirectoryEntryData *entry = ( DirectoryEntryData * ) data_source.key->getRegionData( data_source.id  ); //synchronize
                           if ( _VERBOSE_CACHE ) {
                              *myThread->_file << " SYNC REGION! "; region_shape.key->printRegion( *myThread->_file, region_shape.id );
                              if ( entry ) *myThread->_file << " " << *entry << std::endl;
                              else *myThread->_file << " nil " << std::endl;
                           }
                           //*myThread->_file << " reg is in: " << reg.getFirstLocation() << std::endl;
                           outOps.addOutOp( 0 /* sync only non rooted objects */, data_source.getFirstLocation(), region_shape, data_source.getVersion(), thisOps, wd, (unsigned int)0xdeadbeef ); //Out op synchronize
                           outOps.insertOwnOp( thisOps, region_shape, data_source.getVersion()+1, 0 ); //increase version to invalidate the device copy
                        } else {
                           outOps.getOtherOps().insert( thisOps );
                        }
                     }
                  } else {
                     objects_to_clear.erase( objectAddr );
                  }
               }
            }
            //*myThread->_file << "=============================================================="<<std::endl;
         }
      }
      hb._lock.release();
   }
   //bool orig_verbose_devops = sys.getVerboseDevOps();
   //sys.setVerboseDevOps( true );
   outOps.issue( &wd );
   while ( !outOps.isDataReady( wd ) ) { myThread->processTransfers(); }
   //sys.setVerboseDevOps( orig_verbose_devops );

   //*myThread->_file << "taskwait flush, wd (" << wd.getId() << ") depth is " << wd.getDepth() << " this node is " <<  sys.getNetwork()->getNodeNum() << std::endl;
   //printBt();
   if ( wd.getDepth() == 0 ) {
      // invalidate data on devices
      _invalidateObjectsFromDevices( objects_to_clear );

      //clear objects from directory
      _unregisterObjects( objects_to_clear );
      if ( sys.usingCluster() ) {
         sys.getNetwork()->synchronizeDirectory( NULL );
      }
   }
   //*myThread->_file << "SYNC DIR DONE" << std::endl;
   //print();
   //*myThread->_file << "SYNC DIR & PRINT DONE" << std::endl;
   //o << "++++ DONE ++++ WaitOn synchronize ALL" << std::endl;
}

DeviceOps *RegionDirectory::getOps( RegionDirectoryKey dict, reg_t id ) {
   DirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   DeviceOps *ops = NULL;
   if ( regEntry != NULL ) {
      ops = regEntry->getOps();
   }
   return ops;
}

reg_t RegionDirectory::getLocalRegionId(void * hostObject, reg_t hostRegionId ) {
   GlobalRegionDictionary *dict = getRegionDictionary( (uint64_t) hostObject, false );
   return dict->getLocalRegionIdFromMasterRegionId( hostRegionId );
}

void RegionDirectory::addMasterRegionId( RegionDirectoryKey dict, reg_t masterId, reg_t localId ) {
   dict->addMasterRegionId( masterId, localId );
}

void RegionDirectory::registerObject(nanos_copy_data_internal_t *obj) {
   //allocate dimensions
   nanos_region_dimension_internal_t *dimensions =
      NEW nanos_region_dimension_internal_t[obj->dimension_count];

   ::memcpy(dimensions, obj->dimensions,
         sizeof(nanos_region_dimension_internal_t) * obj->dimension_count);

   CopyData *cd = NEW CopyData( (uint64_t)obj->address, obj->sharing,
      obj->flags.input, obj->flags.output, obj->dimension_count, dimensions,
      obj->offset, 0, 0 );

   uint64_t objectAddr = (uint64_t)cd->getBaseAddress();
   std::size_t objectSize = cd->getMaxSize();
#if 0
   unsigned int key = ( jen_hash( objectAddr ) & (HASH_BUCKETS-1) );
#else
   uint64_t key = jen_hash( this->_getKey( objectAddr, objectSize, NULL ) ) & (HASH_BUCKETS-1);
#endif
   HashBucket &hb = _objects[ key ];

   while ( !hb._lock.tryAcquire() ) {
      myThread->processTransfers();
   }
#if 0
   std::map< uint64_t, Object >::iterator it = hb._bobjects.lower_bound( objectAddr );
   if ( it == hb._bobjects.end() || hb._bobjects.key_comp()( objectAddr, it->first) ) {
      it = hb._bobjects.insert( it, std::map< uint64_t, Object >::value_type( objectAddr, Object( cd ) ) );
      message("Registered object " << *cd );
   } else {
      fatal("Object already registered (same base addr).");
   }
#endif
   if ( hb._bobjects == NULL ) {
      hb._bobjects = NEW MemoryMap< Object >();
   }
   Object **o = hb._bobjects->getExactInsertIfNotFound( objectAddr, objectSize );
   if ( o != NULL ) {
      if ( *o == NULL ) {
         *o = NEW Object( NEW GlobalRegionDictionary( *cd ), cd );
         GlobalRegionDictionary *dict = (*o)->getGlobalRegionDictionary();
         DirectoryEntryData *entry = getDirectoryEntry( *dict, 1 );
         if ( entry == NULL ) {
            entry = NEW DirectoryEntryData();
            dict->setRegionData( 1, entry ); //registerObject
            if ( sys.getVerboseCopies() ) {
               WD const &wd = *myThread->getCurrentWD();
               *myThread->_file << "New object (w/register): ";
               dict->printRegion(*myThread->_file, 1);
               *myThread->_file << " from WD " << wd.getId() << " [" << ( ( wd.getDescription() != NULL) ? wd.getDescription() : "n/a" ) << "]" << std::endl;
               printBt(*myThread->_file);
            }
         }
      } else {
         /* already registered */
         fatal("Object already registered (same base addr).");
      }
   } else {
      /* not found and could not insert a new one */
      fatal("Unable to register prorgam object: " << cd );
   }
   hb._lock.release();
}


void RegionDirectory::unregisterObject(void *baseAddr) {
   uint64_t key = jen_hash( this->_getKey( (uint64_t)baseAddr ) ) & (HASH_BUCKETS-1);
   HashBucket &hb = _objects[ key ];
   while ( !hb._lock.tryAcquire() ) {
      myThread->processTransfers();
   }
   if ( hb._bobjects == NULL ) {
      *(myThread->_file) << "Error, unregister object: object not registered " << baseAddr << std::endl;
      printBt( *(myThread->_file) );
      fatal("can not continue");
   } else {
      Object *o = hb._bobjects->getExactByAddress( (uint64_t) baseAddr );
      if ( o == NULL ) {
         *(myThread->_file) << "Error, unregister object: object not registered " << baseAddr << std::endl;
         printBt( *(myThread->_file) );
         fatal("can not continue");
      } else {
         delete o;
         hb._bobjects->eraseByAddress( (uint64_t) baseAddr );
         _keys.eraseByAddress( (uint64_t) baseAddr );
      }
   }
   hb._lock.release();
}

} // namespace nanos
