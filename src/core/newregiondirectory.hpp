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

#ifndef NANOS_NEWNEWDIRECTORY_H
#define NANOS_NEWNEWDIRECTORY_H

#include "regiondirectory_decl.hpp"
#include "regionbuilder.hpp"
#include "version.hpp"


inline NewNewDirectoryEntryData::NewNewDirectoryEntryData(): Version( 1 ), _writeLocation(0), _location() {
   _location.insert(0);
}

inline NewNewDirectoryEntryData::NewNewDirectoryEntryData( const NewNewDirectoryEntryData &de ): Version( de ), _writeLocation( de._writeLocation ),
   _location( de._location ) {
}

inline NewNewDirectoryEntryData::~NewNewDirectoryEntryData() {
}

inline const NewNewDirectoryEntryData & NewNewDirectoryEntryData::operator= ( const NewNewDirectoryEntryData &de ) {
   Version::operator=( de );
   _writeLocation = de._writeLocation;
   _location.clear();
   _location.insert( de._location.begin(), de._location.end() );
   return *this;
}

inline bool NewNewDirectoryEntryData::hasWriteLocation() const {
   return ( _writeLocation != -1 );
}

inline int NewNewDirectoryEntryData::getWriteLocation() const {
   return _writeLocation;
}

inline void NewNewDirectoryEntryData::setWriteLocation( int id ) {
   _writeLocation = id;
}

inline void NewNewDirectoryEntryData::addAccess( int id, unsigned int version ) {
   if ( version > this->getVersion() ) {
      _location.clear();
      _writeLocation = id;
      this->setVersion( version );
      _location.insert( id );
   } else if ( version == this->getVersion() ) {
      // entry is going to be replicated, so it must be that multiple copies are used as inputs only
      _location.insert( id );
      if ( _location.size() > 1 )
      {
         _writeLocation = -1;
      }
   } else {
     //std::cerr << "FIXME: wrong case" << std::endl;
   }
}

inline bool NewNewDirectoryEntryData::isLocatedIn( int id, unsigned int version ) const {
   return ( version <= this->getVersion() && _location.count( id ) > 0 );
}

inline bool NewNewDirectoryEntryData::isLocatedIn( int id ) const {
   return ( _location.count( id ) > 0 );
}

inline void NewNewDirectoryEntryData::merge( const NewNewDirectoryEntryData &de ) {
   //if ( hasWriteLocation() && de.hasWriteLocation() ) {
   //   if ( getWriteLocation() != de.getWriteLocation() && this->getVersion() == de.getVersion() ) std::cerr << "write loc mismatch WARNING !!! two write locations!, missing dependencies?" << std::endl;
   //} 
   /*else if ( de.hasWriteLocation() ) {
      setWriteLocation( de.getWriteLocation() );
   } else setWriteLocation( -1 );*/

   if ( this->getVersion() == de.getVersion() ) {
      _location.insert( de._location.begin(), de._location.end() );
   }
   else if ( this->getVersion() < de.getVersion() ){
      setWriteLocation( de.getWriteLocation() );
      _location.clear();
      _location.insert( de._location.begin(), de._location.end() );
      this->setVersion( de.getVersion() );
   } /*else {
      std::cerr << "version mismatch! WARNING !!! two write locations!, missing dependencies? current " << this->getVersion() << " inc " << de.getVersion() << std::endl;
   }*/
}
inline void NewNewDirectoryEntryData::print() const {
   std::cerr << "WL: " << _writeLocation << " V: " << this->getVersion() << " Locs: ";
   for ( std::set< int >::iterator it = _location.begin(); it != _location.end(); it++ ) {
      std::cerr << *it << " ";
   }
   std::cerr << std::endl;
}

inline bool NewNewDirectoryEntryData::equal( const NewNewDirectoryEntryData &d ) const {
   bool soFarOk = ( this->getVersion() == d.getVersion() && _writeLocation == d._writeLocation );
   for ( std::set< int >::iterator it = _location.begin(); it != _location.end() && soFarOk; it++ ) {
      soFarOk = ( soFarOk && d._location.count( *it ) == 1 );
   }
   for ( std::set< int >::iterator it = d._location.begin(); it != d._location.end() && soFarOk; it++ ) {
      soFarOk = ( soFarOk && _location.count( *it ) == 1 );
   }
   return soFarOk;
}

inline bool NewNewDirectoryEntryData::contains( const NewNewDirectoryEntryData &d ) const {
   bool soFarOk = ( this->getVersion() == d.getVersion() && _writeLocation == d._writeLocation );
   for ( std::set< int >::iterator it = d._location.begin(); it != d._location.end() && soFarOk; it++ ) {
      soFarOk = ( soFarOk && _location.count( *it ) == 1 );
   }
   return soFarOk;
}

inline int NewNewDirectoryEntryData::getFirstLocation() const {
   return *(_location.begin());
}

inline int NewNewDirectoryEntryData::getNumLocations() const {
   return _location.size();
}

inline NewNewRegionDirectory::RegionDirectoryKey NewNewRegionDirectory::getRegionDirectoryKeyRegisterIfNeeded( CopyData const &cd ) {
   return getRegionDictionaryRegisterIfNeeded( cd );
}

inline NewNewRegionDirectory::RegionDirectoryKey NewNewRegionDirectory::getRegionDirectoryKey( CopyData const &cd ) const {
   return getRegionDictionary( cd );
}

inline NewNewRegionDirectory::RegionDirectoryKey NewNewRegionDirectory::getRegionDirectoryKey( uint64_t addr ) const {
   return getRegionDictionary( addr );
}

#endif
