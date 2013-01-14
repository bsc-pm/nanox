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

using namespace nanos;

std::ostream & nanos::operator<< (std::ostream &o, nanos::NewNewDirectoryEntryData const &ent)
{
   o << "WL: " << ent._writeLocation << " V: " << ent.getVersion() << " Locs: ";
   for ( std::set<int>::iterator it = ent._location.begin(); it != ent._location.end(); it++ ) {
      o << *it << " ";
   }
   return o;
}

NewNewRegionDirectory::NewNewRegionDirectory() : _objects() {}

RegionDictionary *NewNewRegionDirectory::getRegionDictionaryRegisterIfNeeded( CopyData const &cd ) {
   uint64_t objectAddr = (uint64_t) cd.getBaseAddress();
   std::map< uint64_t, RegionDictionary * >::iterator it = _objects.lower_bound( objectAddr );
   if ( it == _objects.end() || _objects.key_comp()( objectAddr, it->first) ) {
     it = _objects.insert( it, std::map< uint64_t, RegionDictionary * >::value_type( objectAddr, NEW RegionDictionary( cd, *( NEW RegionVector() ) ) ) );
     NewNewDirectoryEntryData *entry = getDirectoryEntry( *(it->second), 1 );
     (void) entry;
   }
   return it->second;
}

RegionDictionary *NewNewRegionDirectory::getRegionDictionary( CopyData const &cd ) const {
   uint64_t objectAddr = (uint64_t) cd.getBaseAddress();
   return getRegionDictionary( objectAddr );
}

RegionDictionary *NewNewRegionDirectory::getRegionDictionary( uint64_t objectAddr ) const {
   std::map< uint64_t, RegionDictionary * >::const_iterator it = _objects.lower_bound( objectAddr );
   if ( it == _objects.end() || _objects.key_comp()( objectAddr, it->first) ) {
     std::cerr << "Error, CopyData object not registered in the RegionDictionary " << std::endl;
     sys.printBt();
     fatal("can not continue");
   }
   return it->second;
}

reg_t NewNewRegionDirectory::getLocation( RegionDirectoryKey dict, CopyData const &cd, NewLocationInfoList &missingParts, unsigned int &version, WD const &wd )
{
   reg_t reg = 0;

   ensure( missingParts.empty(), "Non empty list provided." );
   missingParts.clear();
   //sys.getMasterRegionDirectory().print();

   dict->lock();
   reg = dict->addRegion( cd, missingParts, version );

   for ( std::list< std::pair< reg_t, reg_t > >::iterator it = missingParts.begin(); it != missingParts.end(); it++ ) {
      //std::cerr << "getLocation: " << it->first << ","<< it->second <<std::endl;
      if ( it->first != it->second ) {
         NewNewDirectoryEntryData *firstEntry = ( NewNewDirectoryEntryData * ) dict->getRegionData( it->first );
         NewNewDirectoryEntryData *secondEntry = ( NewNewDirectoryEntryData * ) dict->getRegionData( it->second );
         if ( firstEntry == NULL ) {
            firstEntry = NEW NewNewDirectoryEntryData( *secondEntry );
            dict->setRegionData( it->first, firstEntry );
         } else {
            *firstEntry = *secondEntry;
         }
      } else {
         NewNewDirectoryEntryData *entry = ( NewNewDirectoryEntryData * ) dict->getRegionData( it->first );
         if ( entry == NULL ) {
            entry = NEW NewNewDirectoryEntryData();
            dict->setRegionData( it->first, entry );
         }
      }
   }
   dict->unlock();
   //std::cerr << "Git region " << reg << std::endl;
   //for ( std::list< std::pair< reg_t, reg_t > >::iterator it = missingParts.begin(); it != missingParts.end(); it++ ) {
   //   std::cerr << "\tPart " << it->first << " comes from " << it->second << " dict " << (void *) dict << std::endl;
   //}
   //std::cerr <<" end of getLocation "<< std::endl;

   return reg;
}

void NewNewRegionDirectory::addAccess( RegionDirectoryKey dict, reg_t id, unsigned int memorySpaceId, unsigned int version )
{
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   //if(sys.getNetwork()->getNodeNum() == 0) { std::cerr << dict << " ADDING ACCESS reg " << id << " version " << version << " TO LOC " << memorySpaceId << " entry: " << *regEntry << std::endl; }
   regEntry->addAccess( memorySpaceId, version );
   //if(sys.getNetwork()->getNodeNum() == 0) { std::cerr << dict << " ADDING ACCESS reg " << id << " version " << version << " TO LOC " << memorySpaceId << " entry: " << *regEntry << std::endl; }
}

NewNewDirectoryEntryData *NewNewRegionDirectory::getDirectoryEntry( RegionDictionary &dict, reg_t id ) {
   NewNewDirectoryEntryData *entry = ( NewNewDirectoryEntryData * ) dict.getRegionData( id );
   //if ( !entry ) {
   //   entry = NEW NewNewDirectoryEntryData();
   //   dict.setRegionData( id, entry );
   //}
   return entry;
}

void NewNewRegionDirectory::print() const {
   for ( std::map< uint64_t, RegionDictionary * >::const_iterator it = _objects.begin(); it != _objects.end(); it++ ) {
      std::cerr <<"Object "<< it->first << std::endl;
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


unsigned int NewNewRegionDirectory::getVersion( RegionDirectoryKey dict, reg_t id ) {
   NewNewDirectoryEntryData *entry = getDirectoryEntry( *dict, id );
   return entry->getVersion();
}

bool NewNewRegionDirectory::isLocatedIn( RegionDirectoryKey dict, reg_t id, unsigned int loc, unsigned int version ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   //std::cerr << dict << " IS LOCATED " << id << " in loc " << loc <<" entry is " <<*regEntry << " version requested " << version<< std::endl;
   return regEntry->isLocatedIn( loc, version );
}

bool NewNewRegionDirectory::isLocatedIn( RegionDirectoryKey dict, reg_t id, unsigned int loc ) {
   NewNewDirectoryEntryData *regEntry = getDirectoryEntry( *dict, id );
   //std::cerr << dict << " IS LOCATED " << id << " in loc " << loc <<" entry is " <<*regEntry << " version requested " << version<< std::endl;
   return regEntry->isLocatedIn( loc );
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

RegionDictionary &NewNewRegionDirectory::getDictionary( CopyData const &cd ) const {
   return *getRegionDictionary( cd );
}

void NewNewRegionDirectory::synchronize( bool flushData ) {
   if ( flushData ) {
      //std::cerr << "SYNC DIR" << std::endl;
      //int c = 0;
      //print();
      for ( std::map< uint64_t, RegionDictionary *>::iterator it = _objects.begin(); it != _objects.end(); it++ ) {
         //std::cerr << "==================  start object " << ++c << " of " << _objects.size() << "("<< it->first <<") ================="<<std::endl;
         std::list< std::pair< reg_t, reg_t > > missingParts;
         unsigned int version = 0;
         /*reg_t lol =*/ it->second->addRegion(1, missingParts, version);
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
                  //std::cerr << "\t" << mit->first << " " << mit->second <<" MUST SYNC. "<< *regEntry <<std::endl;
                  sys.getCaches()[ getFirstLocation( it->second, mit->first ) ]->syncRegion( global_reg_t( mit->first, it->second ) );
                  regEntry->addAccess( 0, regEntry->getVersion() );
               } else {
                  //std::cerr << "\t" << mit->first << " " << mit->second <<" already in loc 0." <<std::endl;
               }
            } else if ( it->second->getRegionData( mit->second ) != NULL ) {
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
   }
}
