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

#include "deviceops.hpp"
#include "version.hpp"
#include "printbt_decl.hpp"


inline NewNewDirectoryEntryData::NewNewDirectoryEntryData(): Version( 1 ), _writeLocation( -1 ), _ops(), _location(), _rooted( false ), _setLock() {
}

inline NewNewDirectoryEntryData::NewNewDirectoryEntryData( const NewNewDirectoryEntryData &de ): Version( de ), _writeLocation( de._writeLocation ),
   _ops(), _location( de._location ), _rooted( de._rooted ), _setLock() {
}

inline NewNewDirectoryEntryData::~NewNewDirectoryEntryData() {
}

inline NewNewDirectoryEntryData & NewNewDirectoryEntryData::operator= ( NewNewDirectoryEntryData &de ) {
   Version::operator=( de );
   _writeLocation = de._writeLocation;
   _setLock.acquire();
   _location.clear();
   de._setLock.acquire();
   _location.insert( de._location.begin(), de._location.end() );
   _rooted = de._rooted;
   de._setLock.release();
   _setLock.release();
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
   _setLock.acquire();
   //std::cerr << "+++++++++++++++++v entry " << (void *) this << " v++++++++++++++++++++++" << std::endl;
   if ( version > this->getVersion() ) {
      //std::cerr << "Upgrading version to " << version << " @location " << id << std::endl;
      _location.clear();
      _writeLocation = id;
      this->setVersion( version );
      _location.insert( id );
   } else if ( version == this->getVersion() ) {
      //std::cerr << "Equal version (" << version << ") @location " << id << std::endl;
      // entry is going to be replicated, so it must be that multiple copies are used as inputs only
      _location.insert( id );
      if ( _location.size() > 1 )
      {
         _writeLocation = -1;
      }
   } else {
     //std::cerr << "FIXME: wrong case, current version is " << this->getVersion() << " and requested is " << version << " @location " << id <<std::endl;
   }
   //printBt();
   //std::cerr << "+++++++++++++++++^ entry " << (void *) this << " ^++++++++++++++++++++++" << std::endl;
   _setLock.release();
}

inline void NewNewDirectoryEntryData::addRootedAccess( int id, unsigned int version ) {
   _setLock.acquire();
   ensure(version == this->getVersion(), "addRootedAccess of already accessed entry." );
   _location.clear();
   _writeLocation = id;
   this->setVersion( version );
   _location.insert( id );
   _rooted = true;
   _setLock.release();
}

inline bool NewNewDirectoryEntryData::delAccess( int from ) {
   bool result;
   _setLock.acquire();
   _location.erase( from );
   result = _location.empty();
   _setLock.release();
   return result;
}

inline bool NewNewDirectoryEntryData::isLocatedIn( int id, unsigned int version ) {
   bool result;
   _setLock.acquire();
   result = ( version <= this->getVersion() && _location.count( id ) > 0 );
   _setLock.release();
   return result;
}

//inline void NewNewDirectoryEntryData::invalidate() {
//   _invalidated = 1;
//}
//
//inline bool NewNewDirectoryEntryData::hasBeenInvalidated() const {
//   return _invalidated == 1;
//}

inline bool NewNewDirectoryEntryData::isLocatedIn( int id ) {
   bool result;
   _setLock.acquire();
   result = ( _location.count( id ) > 0 );
   _setLock.release();
   return result;
}

//inline void NewNewDirectoryEntryData::merge( const NewNewDirectoryEntryData &de ) {
//   //if ( hasWriteLocation() && de.hasWriteLocation() ) {
//   //   if ( getWriteLocation() != de.getWriteLocation() && this->getVersion() == de.getVersion() ) std::cerr << "write loc mismatch WARNING !!! two write locations!, missing dependencies?" << std::endl;
//   //} 
//   /*else if ( de.hasWriteLocation() ) {
//      setWriteLocation( de.getWriteLocation() );
//   } else setWriteLocation( -1 );*/
//
//   if ( this->getVersion() == de.getVersion() ) {
//      _location.insert( de._location.begin(), de._location.end() );
//   }
//   else if ( this->getVersion() < de.getVersion() ){
//      setWriteLocation( de.getWriteLocation() );
//      _location.clear();
//      _location.insert( de._location.begin(), de._location.end() );
//      this->setVersion( de.getVersion() );
//   } /*else {
//      std::cerr << "version mismatch! WARNING !!! two write locations!, missing dependencies? current " << this->getVersion() << " inc " << de.getVersion() << std::endl;
//   }*/
//}
inline void NewNewDirectoryEntryData::print() const {
   std::cerr << "WL: " << _writeLocation << " V: " << this->getVersion() << " Locs: ";
   for ( std::set< memory_space_id_t >::iterator it = _location.begin(); it != _location.end(); it++ ) {
      std::cerr << *it << " ";
   }
   std::cerr << std::endl;
}

//inline bool NewNewDirectoryEntryData::equal( const NewNewDirectoryEntryData &d ) const {
//   bool soFarOk = ( this->getVersion() == d.getVersion() && _writeLocation == d._writeLocation );
//   for ( std::set< int >::iterator it = _location.begin(); it != _location.end() && soFarOk; it++ ) {
//      soFarOk = ( soFarOk && d._location.count( *it ) == 1 );
//   }
//   for ( std::set< int >::iterator it = d._location.begin(); it != d._location.end() && soFarOk; it++ ) {
//      soFarOk = ( soFarOk && _location.count( *it ) == 1 );
//   }
//   return soFarOk;
//}
//
//inline bool NewNewDirectoryEntryData::contains( const NewNewDirectoryEntryData &d ) const {
//   bool soFarOk = ( this->getVersion() == d.getVersion() && _writeLocation == d._writeLocation );
//   for ( std::set< int >::iterator it = d._location.begin(); it != d._location.end() && soFarOk; it++ ) {
//      soFarOk = ( soFarOk && _location.count( *it ) == 1 );
//   }
//   return soFarOk;
//}

inline int NewNewDirectoryEntryData::getFirstLocation() {
   int result;
   _setLock.acquire();
   result = *(_location.begin());
   _setLock.release();
   return result;
}

inline int NewNewDirectoryEntryData::getNumLocations() {
   int result;
   _setLock.acquire();
   result = _location.size();
   _setLock.release();
   return result;
}

inline DeviceOps *NewNewDirectoryEntryData::getOps() {
   return &_ops;
}

inline std::set< memory_space_id_t > const &NewNewDirectoryEntryData::getLocations() const {
   return _location;
}

inline void NewNewDirectoryEntryData::setRooted() {
   _rooted = true;
}

inline bool NewNewDirectoryEntryData::isRooted() const{
   return _rooted;
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
