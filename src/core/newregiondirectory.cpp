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

#include "newregiondirectory.hpp"
#include "regiondict.hpp"
#include "os.hpp"

using namespace nanos;

std::ostream & nanos::operator<< (std::ostream &o, nanos::NewNewDirectoryEntryData const &ent)
{
   o << "WL: " << ent._writeLocation << " V: " << ent.getVersion() << " Inv: " << ent._invalidated << " Locs: ";
   for ( std::set<int>::iterator it = ent._location.begin(); it != ent._location.end(); it++ ) {
      o << *it << " ";
   }
   return o;
}

NewNewRegionDirectory::NewNewRegionDirectory() : _objects() {}

GlobalRegionDictionary *NewNewRegionDirectory::getRegionDictionaryRegisterIfNeeded( CopyData const &cd ) {
   uint64_t objectAddr = ( cd.getHostBaseAddress() == 0 ? ( uint64_t ) cd.getBaseAddress() : cd.getHostBaseAddress() );
   std::map< uint64_t, GlobalRegionDictionary * >::iterator it = _objects.lower_bound( objectAddr );
   if ( it == _objects.end() || _objects.key_comp()( objectAddr, it->first) ) {
     it = _objects.insert( it, std::map< uint64_t, GlobalRegionDictionary * >::value_type( objectAddr, NEW GlobalRegionDictionary( cd ) ) );
     NewNewDirectoryEntryData *entry = getDirectoryEntry( *(it->second), 1 );
     if ( entry == NULL ) {
       entry = NEW NewNewDirectoryEntryData();
       entry->addAccess( 0, 1 );
       it->second->setRegionData( 1, entry );
     }
     
     //(void) entry;
   }
   return it->second;
}

GlobalRegionDictionary *NewNewRegionDirectory::getRegionDictionary( CopyData const &cd ) const {
   uint64_t objectAddr = ( cd.getHostBaseAddress() == 0 ? ( uint64_t ) cd.getBaseAddress() : cd.getHostBaseAddress() );
   return getRegionDictionary( objectAddr );
}

GlobalRegionDictionary *NewNewRegionDirectory::getRegionDictionary( uint64_t objectAddr ) const {
   std::map< uint64_t, GlobalRegionDictionary * >::const_iterator it = _objects.lower_bound( objectAddr );
   if ( it == _objects.end() || _objects.key_comp()( objectAddr, it->first) ) {
     std::cerr << "Error, CopyData object not registered in the RegionDictionary " << std::endl;
     sys.printBt();
     fatal("can not continue");
   }
   return it->second;
}

reg_t NewNewRegionDirectory::tryGetLocation( RegionDirectoryKey dict, CopyData const &cd, NewLocationInfoList &missingParts, unsigned int &version, WD const &wd ) {
   reg_t reg = 0;
   if ( dict->tryLock() ) {
   NANOS_INSTRUMENT( InstrumentState inst1(NANOS_POST_OUTLINE_WORK2 ); );
    reg = _getLocation( dict, cd, missingParts, version, wd );
   NANOS_INSTRUMENT( inst1.close(); );
      dict->unlock();
   }
   return reg;
}

void NewNewRegionDirectory::tryGetLocation( RegionDirectoryKey dict, reg_t reg, NewLocationInfoList &missingParts, unsigned int &version, WD const &wd ) {
   if ( dict->tryLock() ) {
   NANOS_INSTRUMENT( InstrumentState inst1(NANOS_POST_OUTLINE_WORK2 ); );
    __getLocation( dict, reg, missingParts, version, wd );
   NANOS_INSTRUMENT( inst1.close(); );
      dict->unlock();
   }
}

reg_t NewNewRegionDirectory::_getLocation( RegionDirectoryKey dict, CopyData const &cd, NewLocationInfoList &missingParts, unsigned int &version, WD const &wd )
{
   reg_t reg = 0;
   ensure( missingParts.empty(), "Non empty list provided." );
   missingParts.clear();
   //sys.getMasterRegionDirectory().print();

   reg = dict->registerRegion( cd, missingParts, version );

   for ( std::list< std::pair< reg_t, reg_t > >::iterator it = missingParts.begin(); it != missingParts.end(); it++ ) {
      if ( it->first != it->second ) {
         NewNewDirectoryEntryData *firstEntry = ( NewNewDirectoryEntryData * ) dict->getRegionData( it->first );
         NewNewDirectoryEntryData *secondEntry = ( NewNewDirectoryEntryData * ) dict->getRegionData( it->second );
         if ( firstEntry == NULL ) {
            if ( secondEntry != NULL ) {
      //if( sys.getNetwork()->getNodeNum() == 0) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] getLocation: " << it->first <<"(null),"<< it->second <<"("<< *secondEntry<<")"<<std::endl;
               firstEntry = NEW NewNewDirectoryEntryData( *secondEntry );
      //if( sys.getNetwork()->getNodeNum() ) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] case, null, !null"<<std::endl;
          } else {
      //if( sys.getNetwork()->getNodeNum() ) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] case, null, null"<<std::endl;
      //if( sys.getNetwork()->getNodeNum() == 0) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] getLocation: " << it->first <<"(null),"<< it->second <<"(null)"<<std::endl;
               firstEntry = NEW NewNewDirectoryEntryData();
               firstEntry->addAccess( 0, 1 );
               secondEntry = NEW NewNewDirectoryEntryData();
               secondEntry->addAccess( 0, 1 );
               dict->setRegionData( it->second, secondEntry );
            }
            dict->setRegionData( it->first, firstEntry );
         } else {
            if ( secondEntry != NULL ) {
      //if( sys.getNetwork()->getNodeNum() == 0) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] getLocation: " << it->first <<"("<<*firstEntry<<"),"<< it->second <<"("<< *secondEntry<<")"<<std::endl;
      //if( wd.getId() == 27) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] case, !null, !null"<<std::endl;

               *firstEntry = *secondEntry;
            } else {
               std::cerr << "Dunno what to do..."<<std::endl;
            }
         }
      } else {
      //if( wd.getId() == 27) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] case, same id"<<std::endl;
         NewNewDirectoryEntryData *entry = ( NewNewDirectoryEntryData * ) dict->getRegionData( it->first );
      //if( sys.getNetwork()->getNodeNum() == 0) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] getLocation: " << it->first <<"( " << (void*) entry << " ),"<< it->second <<"( )"<<std::endl;
         if ( entry == NULL ) {
            entry = NEW NewNewDirectoryEntryData();
            entry->addAccess( 0, 1 );
            dict->setRegionData( it->first, entry );
         }
      }
   }

   //if ( wd.getId() == 27 ) {
   //std::cerr << "Git region " << reg << std::endl;
   //for ( std::list< std::pair< reg_t, reg_t > >::iterator it = missingParts.begin(); it != missingParts.end(); it++ ) {
   //   std::cerr << "\tPart " << it->first << " comes from " << it->second << " dict " << (void *) dict << std::endl;
   //}
   //std::cerr <<" end of getLocation "<< std::endl;
   //}

   return reg;
}

void NewNewRegionDirectory::__getLocation( RegionDirectoryKey dict, reg_t reg, NewLocationInfoList &missingParts, unsigned int &version, WD const &wd )
{
   ensure( missingParts.empty(), "Non empty list provided." );
   missingParts.clear();
   //sys.getMasterRegionDirectory().print();

   dict->registerRegion( reg, missingParts, version );

   for ( std::list< std::pair< reg_t, reg_t > >::iterator it = missingParts.begin(); it != missingParts.end(); it++ ) {
      if ( it->first != it->second ) {
         NewNewDirectoryEntryData *firstEntry = ( NewNewDirectoryEntryData * ) dict->getRegionData( it->first );
         NewNewDirectoryEntryData *secondEntry = ( NewNewDirectoryEntryData * ) dict->getRegionData( it->second );
         if ( firstEntry == NULL ) {
            if ( secondEntry != NULL ) {
      //if( sys.getNetwork()->getNodeNum() == 0) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] getLocation: " << it->first <<"(null),"<< it->second <<"("<< *secondEntry<<")"<<std::endl;
               firstEntry = NEW NewNewDirectoryEntryData( *secondEntry );
      //if( sys.getNetwork()->getNodeNum() ) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] case, null, !null"<<std::endl;
          } else {
      //if( sys.getNetwork()->getNodeNum() ) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] case, null, null"<<std::endl;
      //if( sys.getNetwork()->getNodeNum() == 0) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] getLocation: " << it->first <<"(null),"<< it->second <<"(null)"<<std::endl;
               firstEntry = NEW NewNewDirectoryEntryData();
               firstEntry->addAccess( 0, 1 );
               secondEntry = NEW NewNewDirectoryEntryData();
               secondEntry->addAccess( 0, 1 );
               dict->setRegionData( it->second, secondEntry );
            }
            dict->setRegionData( it->first, firstEntry );
         } else {
            if ( secondEntry != NULL ) {
      //if( sys.getNetwork()->getNodeNum() == 0) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] getLocation: " << it->first <<"("<<*firstEntry<<"),"<< it->second <<"("<< *secondEntry<<")"<<std::endl;
      //if( wd.getId() == 27) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] case, !null, !null"<<std::endl;

               *firstEntry = *secondEntry;
            } else {
               std::cerr << "Dunno what to do..."<<std::endl;
            }
         }
      } else {
      //if( wd.getId() == 27) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] case, same id"<<std::endl;
         NewNewDirectoryEntryData *entry = ( NewNewDirectoryEntryData * ) dict->getRegionData( it->first );
      //if( sys.getNetwork()->getNodeNum() == 0) std::cerr <<"["<<sys.getNetwork()->getNodeNum()<< "] getLocation: " << it->first <<"( " << (void*) entry << " ),"<< it->second <<"( )"<<std::endl;
         if ( entry == NULL ) {
            entry = NEW NewNewDirectoryEntryData();
            entry->addAccess( 0, 1 );
            dict->setRegionData( it->first, entry );
         }
      }
   }

   //if ( wd.getId() == 27 ) {
   //std::cerr << "Git region " << reg << std::endl;
   //for ( std::list< std::pair< reg_t, reg_t > >::iterator it = missingParts.begin(); it != missingParts.end(); it++ ) {
   //   std::cerr << "\tPart " << it->first << " comes from " << it->second << " dict " << (void *) dict << std::endl;
   //}
   //std::cerr <<" end of getLocation "<< std::endl;
   //}

}

void NewNewRegionDirectory::addAccess( RegionDirectoryKey dict, reg_t id, unsigned int memorySpaceId, unsigned int version )
{
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   //if(sys.getNetwork()->getNodeNum() > 0) { std::cerr << dict << " ADDING ACCESS reg " << id << " version " << version << " TO LOC " << memorySpaceId << " entry: " << *regEntry << std::endl; }
   //if(sys.getNetwork()->getNodeNum() == 0) { std::cerr << dict << " ADDING ACCESS reg " << id << " version " << version << " TO LOC " << memorySpaceId << " entry: " << *regEntry << std::endl; }
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << "ADD ACCESS "<< (void*) dict << ":" << id << " to "<< memorySpaceId << std::endl;
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << *regEntry<<std::endl;
   regEntry->addAccess( memorySpaceId, version );
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << *regEntry<<std::endl;
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << "---------" << std::endl;
}

void NewNewRegionDirectory::addAccessRegisterIfNeeded( RegionDirectoryKey dict, reg_t id, unsigned int memorySpaceId, unsigned int version )
{
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   if ( regEntry == NULL ) {
     regEntry = NEW NewNewDirectoryEntryData();
     dict->setRegionData( id, regEntry );
   }
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << "ADD ACCESS if needed "<< (void*) dict << ":" << id << " to "<< memorySpaceId << std::endl;
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << *regEntry<<std::endl;
   //if(sys.getNetwork()->getNodeNum() > 0) { std::cerr << dict << " ADDING ACCESS reg " << id << " version " << version << " TO LOC " << memorySpaceId << " entry: " << *regEntry << std::endl; }
   regEntry->addAccess( memorySpaceId, version );
   //if(sys.getNetwork()->getNodeNum() == 0) { std::cerr << dict << " ADDING ACCESS reg " << id << " version " << version << " TO LOC " << memorySpaceId << " entry: " << *regEntry << std::endl; }
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << *regEntry<<std::endl;
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << "---------" << std::endl;
}

NewNewDirectoryEntryData *NewNewRegionDirectory::getDirectoryEntry( GlobalRegionDictionary &dict, reg_t id ) {
   NewNewDirectoryEntryData *entry = ( NewNewDirectoryEntryData * ) dict.getRegionData( id );
   //if ( !entry ) {
   //   entry = NEW NewNewDirectoryEntryData();
   //   dict.setRegionData( id, entry );
   //}
   return entry;
}

bool NewNewRegionDirectory::delAccess( RegionDirectoryKey dict, reg_t id, unsigned int memorySpaceId ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   bool res;
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << "DEL ACCESS "<< (void*) dict << ":" << id << " from "<< memorySpaceId << std::endl;
   //if (sys.getNetwork()->getNodeNum() == 0 && regEntry )std::cerr << *regEntry<<std::endl;
   res = regEntry->delAccess( memorySpaceId );
   //if (sys.getNetwork()->getNodeNum() == 0 && regEntry )std::cerr << *regEntry<<std::endl;
   //if (sys.getNetwork()->getNodeNum() == 0 )std::cerr << "---------" << std::endl;
   return res;
}

void NewNewRegionDirectory::invalidate( RegionDirectoryKey dict, reg_t id ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   regEntry->invalidate();
}

bool NewNewRegionDirectory::hasBeenInvalidated( RegionDirectoryKey dict, reg_t id ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   return regEntry->hasBeenInvalidated();
}

void NewNewRegionDirectory::updateFromInvalidated( RegionDirectoryKey dict, reg_t id, reg_t from ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   NewNewDirectoryEntryData *fromEntry = getDirectoryEntry( *dict, from );
   *regEntry = *fromEntry;
}

void NewNewRegionDirectory::print() const {
   for ( std::map< uint64_t, GlobalRegionDictionary * >::const_iterator it = _objects.begin(); it != _objects.end(); it++ ) {
      std::cerr <<"Object "<< (void*)it->second << std::endl;
      for (reg_t i = 1; i < it->second->getMaxRegionId(); i++ ) {
         NewNewDirectoryEntryData *entry = ( NewNewDirectoryEntryData * ) it->second->getRegionData( i );
         if ( !entry ) {
            std::cerr << "\t" << i << " "; it->second->printRegion( i ); std::cerr << " : null " << std::endl;
         } else {
            std::cerr << "\t" << i << " "; it->second->printRegion( i ); std::cerr << " : ("<< entry <<") "<< *entry << std::endl;
         }
      }
   }
}


unsigned int NewNewRegionDirectory::getVersion( RegionDirectoryKey dict, reg_t id, bool increaseVersion ) {
   NewNewDirectoryEntryData *entry = getDirectoryEntry( *dict, id );
   return entry->getVersion( increaseVersion );
}

bool NewNewRegionDirectory::isLocatedIn( RegionDirectoryKey dict, reg_t id, unsigned int loc, unsigned int version ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   //std::cerr << dict << " IS LOCATED " << id << " in loc " << loc <<" entry is " <<*regEntry << " version requested " << version<< std::endl;
   return regEntry->isLocatedIn( loc, version );
}

bool NewNewRegionDirectory::isLocatedIn( RegionDirectoryKey dict, reg_t id, unsigned int loc ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   //std::cerr << dict << " IS LOCATED " << id << " in loc " << loc <<" entry is " <<*regEntry << " version requested " << version<< std::endl;
   return (regEntry) ? regEntry->isLocatedIn( loc ) : 0;
}

unsigned int NewNewRegionDirectory::getFirstLocation( RegionDirectoryKey dict, reg_t id ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   return regEntry->getFirstLocation();
}

bool NewNewRegionDirectory::hasWriteLocation( RegionDirectoryKey dict, reg_t id ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   return regEntry->hasWriteLocation();
}

unsigned int NewNewRegionDirectory::getWriteLocation( RegionDirectoryKey dict, reg_t id ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   return regEntry->getWriteLocation();
}

GlobalRegionDictionary &NewNewRegionDirectory::getDictionary( CopyData const &cd ) const {
   return *getRegionDictionary( cd );
}

void NewNewRegionDirectory::synchronize( bool flushData ) {
   if ( flushData ) {
      //std::cerr << "SYNC DIR" << std::endl;
      int c = 0;
      print();
      for ( std::map< uint64_t, GlobalRegionDictionary *>::iterator it = _objects.begin(); it != _objects.end(); it++ ) {
         std::cerr << "==================  start object " << ++c << " of " << _objects.size() << "("<< it->second <<") ================="<<std::endl;
         std::list< std::pair< reg_t, reg_t > > missingParts;
         unsigned int version = 0;
   //double tini = OS::getMonotonicTime();
         /*reg_t lol =*/ it->second->registerRegion(1, missingParts, version, true);
   //double tfini = OS::getMonotonicTime();
   //std::cerr << __FUNCTION__ << " addRegion time " << (tfini-tini) << std::endl;
         //std::cerr << "Missing parts are: (want version) "<< version << " got " << lol << " { ";
         //for ( std::list< std::pair< reg_t, reg_t > >::iterator mit = missingParts.begin(); mit != missingParts.end(); mit++ ) {
         //   std::cerr <<"("<< mit->first << "," << mit->second << ") ";
         //}
         //std::cerr << "}"<<std::endl;
         for ( std::list< std::pair< reg_t, reg_t > >::iterator mit = missingParts.begin(); mit != missingParts.end(); mit++ ) {
            //std::cerr << "sync region " << mit->first << " : "<< ( void * ) it->second->getRegionData( mit->first ) <<" with second reg " << mit->second << " : " << ( void * ) it->second->getRegionData( mit->second )<< std::endl;
            if ( it->second->getRegionData( mit->first ) != NULL ) {
               if ( !isLocatedIn( it->second, mit->first, 0 ) ) {
                  NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *it->second, mit->first );
                  std::cerr << "\t" << mit->first << " " << mit->second; it->second->printRegion( mit->first );  std::cerr <<" MUST SYNC. "<< *regEntry <<std::endl;
                  sys.getCaches()[ getFirstLocation( it->second, mit->first ) ]->syncRegion( global_reg_t( mit->first, it->second ) );
                  regEntry->addAccess( 0, regEntry->getVersion() );
               } else {
                  //std::cerr << "\t" << mit->first << " " << mit->second <<" already in loc 0." <<std::endl;
                  std::cerr << "\t" << mit->first << " " << mit->second; it->second->printRegion( mit->first );  std::cerr <<" already in loc 0. " <<std::endl;
               }
            } else if ( it->second->getRegionData( mit->second ) != NULL ) {
                  NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *it->second, mit->second );
                  std::cerr << "\t" << mit->first << " " << mit->second; it->second->printRegion( mit->first );  std::cerr <<" no data for first entry. "<< *regEntry <<std::endl;
               //if ( !isLocatedIn( it->second, mit->second, 0, version ) ) {
               //   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *it->second, mit->first );
               //   std::cerr << "\t" << mit->first << " " << mit->second <<" MUST SYNC. "<< *regEntry <<std::endl;
               //   sys.getCaches()[ getFirstLocation( it->second, mit->first ) ]->syncRegion( global_reg_t( mit->first, it->second ) );
               //} else {
               //   std::cerr << "\t" << mit->first << " " << mit->second <<" already in loc 0." <<std::endl;
               //}
            } else {
               std::cerr << "FIXME" << std::endl;
            }
         }
         std::cerr << "=============================================================="<<std::endl;
      }
   }
}

void NewNewRegionDirectory::synchronize2( bool flushData ) {
   if ( flushData ) {
      //std::cerr << "SYNC DIR" << std::endl;
      //int c = 0;
      //print();
      std::set< DeviceOps * > ops;
      std::set< DeviceOps * > myOps;
      for ( std::map< uint64_t, GlobalRegionDictionary *>::iterator it = _objects.begin(); it != _objects.end(); it++ ) {
         //std::cerr << "==================  start object " << ++c << " of " << _objects.size() << "("<< it->second <<") ================="<<std::endl;
         std::list< std::pair< reg_t, reg_t > > missingParts;
         unsigned int version = 0;
   //double tini = OS::getMonotonicTime();
         /*reg_t lol =*/ it->second->registerRegion(1, missingParts, version, true);
   //double tfini = OS::getMonotonicTime();
   //std::cerr << __FUNCTION__ << " addRegion time " << (tfini-tini) << std::endl;
         //std::cerr << "Missing parts are: (want version) "<< version << " got " << lol << " { ";
         //for ( std::list< std::pair< reg_t, reg_t > >::iterator mit = missingParts.begin(); mit != missingParts.end(); mit++ ) {
         //   std::cerr <<"("<< mit->first << "," << mit->second << ") ";
         //}
         //std::cerr << "}"<<std::endl;
         for ( std::list< std::pair< reg_t, reg_t > >::iterator mit = missingParts.begin(); mit != missingParts.end(); mit++ ) {
            //std::cerr << "sync region " << mit->first << " : "<< ( void * ) it->second->getRegionData( mit->first ) <<" with second reg " << mit->second << " : " << ( void * ) it->second->getRegionData( mit->second )<< std::endl;
            if ( it->second->getRegionData( mit->first ) != NULL ) {
               if ( !isLocatedIn( it->second, mit->first, 0 ) ) {
                  NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *it->second, mit->first );
                  //std::cerr << "\t" << mit->first << " " << mit->second << " "; it->second->printRegion( mit->first );  std::cerr <<" MUST SYNC. "<< *regEntry <<std::endl;
                  DeviceOps *thisOps = regEntry->getOps();
                  if ( thisOps->addCacheOp() ) {
                     sys.getSeparateMemory( getFirstLocation( it->second, mit->first ) ).copyOut( global_reg_t( mit->first, it->second ), regEntry->getVersion() );
                     myOps.insert( thisOps );
                  } else {
                     ops.insert( thisOps );
                  }
                  //regEntry->addAccess( 0, regEntry->getVersion() );
               } else {
                  //std::cerr << "\t" << mit->first << " " << mit->second <<" already in loc 0." <<std::endl;
                  //std::cerr << "\t" << mit->first << " " << mit->second << " "; it->second->printRegion( mit->first );  std::cerr <<" already in loc 0. " <<std::endl;
               }
            } else if ( it->second->getRegionData( mit->second ) != NULL ) {
               //   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *it->second, mit->second );
                  //std::cerr << "\t" << mit->first << " " << mit->second << " "; it->second->printRegion( mit->first );  std::cerr <<" no data for first entry. "<< *regEntry <<std::endl;
               //if ( !isLocatedIn( it->second, mit->second, 0, version ) ) {
               //   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *it->second, mit->first );
               //   std::cerr << "\t" << mit->first << " " << mit->second <<" MUST SYNC. "<< *regEntry <<std::endl;
               //   sys.getCaches()[ getFirstLocation( it->second, mit->first ) ]->syncRegion( global_reg_t( mit->first, it->second ) );
               //} else {
               //   std::cerr << "\t" << mit->first << " " << mit->second <<" already in loc 0." <<std::endl;
               //}
            } else {
               std::cerr << "FIXME" << std::endl;
            }
         }
         //std::cerr << "=============================================================="<<std::endl;
      }
      bool allDone = true;
      do {
         allDone = true;
         for ( std::set< DeviceOps *>::iterator it = myOps.begin(); it != myOps.end(); it++ ) {
            allDone = allDone && (*it)->allCompleted();
         }
         myThread->idle( true);
      } while ( !allDone );
      for ( std::set< DeviceOps *>::iterator it = myOps.begin(); it != myOps.end(); it++ ) {
         (*it)->completeCacheOp();
      }

      do {
         for ( std::set< DeviceOps *>::iterator it = ops.begin(); it != ops.end(); it++ ) {
            allDone = (*it)->allCacheOpsCompleted();
         }
         myThread->idle( true);
      } while ( !allDone );
   }
}
#if 0
void NewNewRegionDirectory::invalidate( CacheRegionDictionary *regions, unsigned int from ) {
   std::cerr << " inval from " << from << std::endl;
   std::list< RegionVectorEntry * > regionEntries;

   for ( std::map<reg_t, RegionVectorEntry>::const_iterator it = regions->begin(); it != regions->end(); it++ ) {
      std::cerr << "have to inval region " << it->first << std::endl;
      NewNewDirectoryEntryData *regEntry = ( NewNewDirectoryEntryData * ) regions->getGlobalRegionData(it->first);
      if ( !regEntry->isLocatedIn( 0 ) ) {
          sys.getCaches()[ from ]->syncRegion( global_reg_t( it->first, regions->getGlobalDirectoryKey() ) );
          regEntry->addAccess( 0, regEntry->getVersion() );
      } else {
          regEntry->delAccess( from );
      }
   }
   //regions->fill( regionEntries );
   
   //NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *regions, id );
   //if(sys.getNetwork()->getNodeNum() > 0) { std::cerr << dict << " ADDING ACCESS reg " << id << " version " << version << " TO LOC " << memorySpaceId << " entry: " << *regEntry << std::endl; }
   //regEntry->addAccess( memorySpaceId, version );
   
}
#endif

#if 0
void NewNewRegionDirectory::setOps( RegionDirectoryKey dict, reg_t id, DeviceOps *ops ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   if ( regEntry != NULL ) {
      regEntry->setOps( ops );
   }
}
#endif
DeviceOps *NewNewRegionDirectory::getOps( RegionDirectoryKey dict, reg_t id ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   DeviceOps *ops = NULL;
   if ( regEntry != NULL ) {
      ops = regEntry->getOps();
   }
   return ops;
}

void NewNewRegionDirectory::initializeEntry( RegionDirectoryKey dict, reg_t reg ) {
   NewNewDirectoryEntryData *entry = NEW NewNewDirectoryEntryData();
   entry->addAccess( 0, 1 );
   dict->setRegionData( reg, entry );
}

