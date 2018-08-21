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

#include "system_decl.hpp"
#include "regiondict.hpp"
#include "regionset_decl.hpp"
#include "regiondirectory.hpp"
#include "addressspace.hpp"
#include "globalregt.hpp"

using namespace nanos;

RegionSet::RegionSet() : _lock(), _set() {
}

void RegionSet::addRegion( global_reg_t const &reg, unsigned int version ) {
   while ( !_lock.tryAcquire() ) {
      myThread->processTransfers();
   }
   reg_set_t &regs = _set[ reg.key ];
   reg_set_t::iterator elem = regs.lower_bound( reg.id );
   //std::cerr << "Add region object " << reg.key <<", dict: " << this << std::endl;

   if ( elem == regs.end() || regs.key_comp()( reg.id, elem->first ) ) {
      /* not found */
   //std::cerr << "Not found region, adding it " << reg.id <<", dict: " << this << std::endl;
      regs.insert(elem, reg_set_t::value_type( reg.id, version ) );
   } else {
      /* found */
   //std::cerr << "Found region, not adding it " << reg.id <<", dict: " << this << std::endl;
      if ( elem->second < version ) {
         elem->second = version;
      }
   }
   _lock.release();
}


bool RegionSet::hasObjectOfRegion( global_reg_t const &reg ) {
   while ( !_lock.tryAcquire() ) {
      myThread->processTransfers();
   }
   bool i_has_it = ( _set.find( reg.key ) != _set.end() );
   // std::cerr << "asking for " << reg.key << "  My Objects ( " << this << " ) : ";
   // for (object_set_t::iterator it = _set.begin(); it != _set.end(); it++ ) {
   //    std::cerr << " " << it->first;
   // }
   // std::cerr << " result: " << (i_has_it ? "found!":"not found") << std::endl;
   _lock.release();
   return i_has_it;
}

unsigned int RegionSet::hasRegion( global_reg_t const &reg ) {
   unsigned int version = (unsigned int) -1;
   while ( !_lock.tryAcquire() ) {
      myThread->processTransfers();
   }
   _lock.release();
   return version;
}

bool RegionSet::hasVersionInfoForRegion( global_reg_t const &reg, unsigned int &version, NewLocationInfoList &locations ) {
   bool resultHIT = false;
   bool resultSUBR = false;
   bool resultSUPER = false;
   //std::ostream &o = *myThread->_file;
   object_set_t::iterator wantedDir = _set.find( reg.key );
   if ( wantedDir != _set.end() ) {
      unsigned int versionHIT = 0;
      reg_set_t::iterator wantedReg = wantedDir->second.find( reg.id );
      if ( wantedReg != wantedDir->second.end() ) {
         versionHIT = wantedReg->second;
         //double check the directory because a there may be WDs that have not been detected as predecessors
         DirectoryEntryData *entry = ( DirectoryEntryData * ) wantedDir->first->getRegionData( wantedReg->first );
         if ( entry->getVersion() > versionHIT ) {
            versionHIT = entry->getVersion();
         }
         resultHIT = true;
         wantedDir->second.erase( wantedReg );
      }
      //if ( resultHIT ) {
      //   o << " HIT got version " << versionHIT << " for region " << reg.id << std::endl;
      //}

      unsigned int versionSUPER = 0;
      reg_t superPart = wantedDir->first->isThisPartOf( reg.id, wantedDir->second.begin(), wantedDir->second.end(), versionSUPER ); 
      if ( superPart != 0 ) {
         resultSUPER = true;
      }

      unsigned int versionSUBR = 0;
      if ( wantedDir->first->doTheseRegionsForm( reg.id, wantedDir->second.begin(), wantedDir->second.end(), versionSUBR ) ) {
         if ( versionHIT < versionSUBR && versionSUPER < versionSUBR ) {
            DirectoryEntryData *dirEntry = ( DirectoryEntryData * ) wantedDir->first->getRegionData( reg.id );
            if ( dirEntry != NULL ) { /* if entry is null, do check directory, because we need to insert the region info in the intersect maps */
               for ( std::map< reg_t, unsigned int >::const_iterator it = wantedDir->second.begin(); it != wantedDir->second.end(); it++ ) {
                  global_reg_t r( it->first, wantedDir->first );
                  reg_t intersect = r.key->computeIntersect( reg.id, r.id );
                  if ( it->first == intersect ) {
                     locations.push_back( std::make_pair( it->first, it->first ) );
                  }
               }
               version = versionSUBR;
            } else {
               sys.getHostMemory().getVersionInfo( reg, version, locations );
            }
            resultSUBR = true;
            //o << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! VERSION INFO !!! CHUNKS FORM THIS REG!!! and version computed is " << version << std::endl;
         }
      }
      if ( !resultSUBR && ( resultSUPER || resultHIT ) ) {
         if ( versionHIT >= versionSUPER ) {
            version = versionHIT;
            //o << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! VERSION INFO !!! CHUNKS HIT!!! and version computed is " << version << std::endl;
            locations.push_back( std::make_pair( reg.id, reg.id ) );
         } else {
            version = versionSUPER;
            DirectoryEntryData *firstEntry = ( DirectoryEntryData * ) wantedDir->first->getRegionData( reg.id );
            //o << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! VERSION INFO !!! CHUNKS COMES FROM A BIGGER!!! and version computed is " << version << " entry " << firstEntry << std::endl;
            if ( firstEntry != NULL ) {
               locations.push_back( std::make_pair( reg.id, superPart ) );
               DirectoryEntryData *secondEntry = ( DirectoryEntryData * ) wantedDir->first->getRegionData( superPart );
               if (secondEntry == NULL) std::cerr << "LOLWTF!"<< std::endl;
               *firstEntry = *secondEntry;
            } else {
               unsigned int tmpversion = 0;
               sys.getHostMemory().getVersionInfo( reg, tmpversion, locations );
               if ( tmpversion != 0 ) {
                  version = tmpversion;
               }
            }
         }
      }
   }
   return (resultSUBR || resultSUPER || resultHIT) ;
}
