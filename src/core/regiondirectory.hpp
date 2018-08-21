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

#ifndef NANOS_NEWNEWDIRECTORY_H
#define NANOS_NEWNEWDIRECTORY_H

#include "deviceops.hpp"
#include "version.hpp"

namespace nanos {

inline DirectoryEntryData::DirectoryEntryData() : Version( 1 )
   //, _writeLocation( -1 )
   , _ops()
   , _location()
   , _pes()
   , _rooted( (memory_space_id_t) -1 )
   , _home( (memory_space_id_t) -1 )
   , _setLock() 
   , _firstWriterPE( NULL )
   , _baseAddress( 0 )
{
   _location.insert(0);
}

inline DirectoryEntryData::DirectoryEntryData( memory_space_id_t home ) : Version( 1 )
   //, _writeLocation( -1 )
   , _ops()
   , _location()
   , _pes()
   , _rooted( (memory_space_id_t) -1 )
   , _home( home )
   , _setLock() 
   , _firstWriterPE( NULL )
   , _baseAddress( 0 )
{
   _location.insert( home );
}

inline DirectoryEntryData::DirectoryEntryData( const DirectoryEntryData &de ) : Version( de )
   //, _writeLocation( de._writeLocation )
   , _ops()
   , _location( de._location )
   , _pes( de._pes )
   , _rooted( de._rooted )
   , _home( de._home )
   , _setLock()
   , _firstWriterPE( de._firstWriterPE )
   , _baseAddress( de._baseAddress )
{
}

inline DirectoryEntryData::~DirectoryEntryData() {
}

inline DirectoryEntryData & DirectoryEntryData::operator= ( DirectoryEntryData &de ) {
   Version::operator=( de );
   //_writeLocation = de._writeLocation;
   while ( !_setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
   while ( !de._setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
   _location.clear();
   _pes.clear();
   _location.insert( de._location.begin(), de._location.end() );
   _pes.insert( de._pes.begin(), de._pes.end() );
   _rooted = de._rooted;
   _home = de._home;
   _firstWriterPE = de._firstWriterPE;
   _baseAddress = de._baseAddress;
   de._setLock.release();
   _setLock.release();
   return *this;
}

//inline bool DirectoryEntryData::hasWriteLocation() const {
//   return ( _writeLocation != -1 );
//}

// inline int DirectoryEntryData::getWriteLocation() const {
//    return _writeLocation;
// }
// 
// inline void DirectoryEntryData::setWriteLocation( int id ) {
//    _writeLocation = id;
// }

inline void DirectoryEntryData::addAccess( ProcessingElement *pe, memory_space_id_t loc, unsigned int version ) {
   while ( !_setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
   //*myThread->_file << "+++++++++++++++++v entry " << (void *) this << " v++++++++++++++++++++++" << std::endl;
   if ( version > this->getVersion() ) {
      //*myThread->_file << "Upgrading version to " << version << " @location " << id << std::endl;
      _location.clear();
      //_writeLocation = id;
      this->setVersion( version );
      _location.insert( loc );
      if ( pe != NULL && loc == pe->getMemorySpaceId() ) {
         _pes.insert( pe );
      }
      if ( version == 2 ) {
         _firstWriterPE = pe;
      }
   } else if ( version == this->getVersion() ) {
      //*myThread->_file << "Equal version (" << version << ") @location " << id << std::endl;
      // entry is going to be replicated, so it must be that multiple copies are used as inputs only
      _location.insert( loc );
      if ( pe != NULL && loc == pe->getMemorySpaceId() ) {
         _pes.insert( pe );
      }
      // if ( _location.size() > 1 )
      // {
      //    _writeLocation = -1;
      // }
   } else {
     //*myThread->_file << "FIXME: wrong case, current version is " << this->getVersion() << " and requested is " << version << " @location " << id <<std::endl;
   }
   //*myThread->_file << "+++++++++++++++++^ entry " << (void *) this << " ^++++++++++++++++++++++" << std::endl;
   _setLock.release();
}

inline void DirectoryEntryData::addRootedAccess( memory_space_id_t loc, unsigned int version ) {
   while ( !_setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
   ensure(version == this->getVersion(), "addRootedAccess of already accessed entry." );
   _location.clear();
   //_writeLocation = id;
   this->setVersion( version );
   _location.insert( loc );
   _rooted = loc;
   _setLock.release();
}

inline bool DirectoryEntryData::delAccess( memory_space_id_t from ) {
   bool result;
   while ( !_setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
   _location.erase( from );
   std::set< ProcessingElement * >::iterator it = _pes.begin();
   while ( it != _pes.end() ) {
      if ( (*it)->getMemorySpaceId() == from ) {
         std::set< ProcessingElement * >::iterator toBeErased = it;
         ++it;
         _pes.erase( toBeErased );
      } else {
         ++it;
      }
   }
   result = _location.empty();
   _setLock.release();
   return result;
}

inline bool DirectoryEntryData::isLocatedIn( ProcessingElement *pe, unsigned int version ) {
   bool result;
   while ( !_setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
   if ( _location.empty() ) {
      *myThread->_file << " Warning: empty _location set, it is likely that an invalidation is ongoing for this region. " << std::endl;
   }
   result = ( version <= this->getVersion() && _location.count( pe->getMemorySpaceId() ) > 0 );
   _setLock.release();
   return result;
}

inline bool DirectoryEntryData::isLocatedIn( ProcessingElement *pe ) {
   return this->isLocatedIn( pe->getMemorySpaceId() );
}

inline bool DirectoryEntryData::isLocatedIn( memory_space_id_t loc ) {
   bool result;
   while ( !_setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
   result = ( _location.count( loc ) > 0 );
   if ( !result && _location.size() == 0 ) { //locations.size = 0 means we are invalidating
      result = (loc == 0);
   }
   _setLock.release();
   return result;
}

inline void DirectoryEntryData::print(std::ostream &o) const {
   o << " V: " << this->getVersion() << " Locs: ";
   for ( std::set< memory_space_id_t >::iterator it = _location.begin(); it != _location.end(); it++ ) {
      o << *it << " ";
   }
   o << std::endl;
}

inline int DirectoryEntryData::getFirstLocation() {
   int result;
   while ( !_setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
   result = *(_location.begin());
   _setLock.release();
   return result;
}

inline int DirectoryEntryData::getNumLocations() {
   int result;
   while ( !_setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
   result = _location.size();
   _setLock.release();
   return result;
}

inline DeviceOps *DirectoryEntryData::getOps() {
   return &_ops;
}

inline std::set< memory_space_id_t > const &DirectoryEntryData::getLocations() const {
   return _location;
}

inline memory_space_id_t DirectoryEntryData::getRootedLocation() const{
   return _rooted;
}


inline bool DirectoryEntryData::isRooted() const{
   return _rooted != (memory_space_id_t) -1;
}

inline ProcessingElement *DirectoryEntryData::getFirstWriterPE() const {
   return _firstWriterPE;
}

inline void DirectoryEntryData::setBaseAddress(uint64_t addr) {
   _baseAddress = addr;
}

inline uint64_t DirectoryEntryData::getBaseAddress() const {
   return _baseAddress;
}

inline memory_space_id_t DirectoryEntryData::getHome() const {
   return _home;
}

inline void DirectoryEntryData::lock() {
   while ( !_setLock.tryAcquire() ) {
      //myThread->processTransfers();
   }
}

inline void DirectoryEntryData::unlock() {
   _setLock.release();
}

inline RegionDirectory::RegionDirectoryKey RegionDirectory::getRegionDirectoryKeyRegisterIfNeeded( CopyData const &cd, WD const *wd ) {
   return getRegionDictionaryRegisterIfNeeded( cd, wd );
}

inline RegionDirectory::RegionDirectoryKey RegionDirectory::getRegionDirectoryKey( CopyData const &cd ) {
   return getRegionDictionary( cd );
}

inline RegionDirectory::RegionDirectoryKey RegionDirectory::getRegionDirectoryKey( uint64_t addr ) {
   return getRegionDictionary( addr );
}

inline void RegionDirectory::__getLocation( RegionDirectoryKey dict, reg_t reg, NewLocationInfoList &missingParts, unsigned int &version )
{
   dict->lockObject();
   dict->registerRegion( reg, missingParts, version );
   dict->unlockObject();
}

} // namespace nanos

#endif
