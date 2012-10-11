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

#ifndef _NANOS_NEWDIRECTORY_H
#define _NANOS_NEWDIRECTORY_H

#include "regiondirectory_decl.hpp"
#include "regionbuilder.hpp"


inline NewDirectoryEntryData::NewDirectoryEntryData(): _writeLocation(0), _version(1), _location() {
}

inline NewDirectoryEntryData::NewDirectoryEntryData( const NewDirectoryEntryData &de ): _writeLocation( de._writeLocation ),
   _version( de._version ), _location( de._location ) {
}

inline NewDirectoryEntryData::~NewDirectoryEntryData() {
}

inline const NewDirectoryEntryData & NewDirectoryEntryData::operator= ( const NewDirectoryEntryData &de ) {
   _writeLocation = de._writeLocation;
   _version = de._version;
   _location.clear();
   _location.insert( de._location.begin(), de._location.end() );
   return *this;
}

inline bool NewDirectoryEntryData::hasWriteLocation() const {
   return ( _writeLocation != -1 );
}

inline int NewDirectoryEntryData::getWriteLocation() const {
   return _writeLocation;
}

inline void NewDirectoryEntryData::setWriteLocation( int id ) {
   _writeLocation = id;
}

inline void NewDirectoryEntryData::addAccess( int id, uint64_t address, unsigned int version ) {
   if ( version > _version ) {
      _location.clear();
      _writeLocation = id;
      _version = version;
   } else if ( version == _version ) {
      // entry is going to be replicated, so it must be that multiple copies are used as inputs only
      _writeLocation = -1;
   } else {
     std::cerr << "FIXME: wrong case" << std::endl;
   }
   _location.insert( id );
}

inline bool NewDirectoryEntryData::isLocatedIn( int id ) const {
   return ( _location.count( id ) > 0 );
}

inline unsigned int NewDirectoryEntryData::getVersion() const {
   return _version;
}

inline void NewDirectoryEntryData::merge( const NewDirectoryEntryData &de ) {
   //if ( hasWriteLocation() && de.hasWriteLocation() ) {
   //   if ( getWriteLocation() != de.getWriteLocation() && _version == de._version ) std::cerr << "write loc mismatch WARNING !!! two write locations!, missing dependencies?" << std::endl;
   //} 
   /*else if ( de.hasWriteLocation() ) {
      setWriteLocation( de.getWriteLocation() );
   } else setWriteLocation( -1 );*/

   if ( _version == de._version ) {
      _location.insert( de._location.begin(), de._location.end() );
   }
   else if ( _version < de._version ){
      setWriteLocation( de.getWriteLocation() );
      _location.clear();
      _location.insert( de._location.begin(), de._location.end() );
      _version = de._version;
   } /*else {
      std::cerr << "version mismatch! WARNING !!! two write locations!, missing dependencies? current " << _version << " inc " << de._version << std::endl;
   }*/
}
inline void NewDirectoryEntryData::print() const {
   std::cerr << "WL: " << _writeLocation << " V: " << _version << " Locs: ";
   for ( std::set< int >::iterator it = _location.begin(); it != _location.end(); it++ ) {
      std::cerr << *it << " ";
   }
   std::cerr << std::endl;
}

inline bool NewDirectoryEntryData::equal( const NewDirectoryEntryData &d ) const {
   bool soFarOk = ( _version == d._version && _writeLocation == d._writeLocation );
   for ( std::set< int >::iterator it = _location.begin(); it != _location.end() && soFarOk; it++ ) {
      soFarOk = ( soFarOk && d._location.count( *it ) == 1 );
   }
   for ( std::set< int >::iterator it = d._location.begin(); it != d._location.end() && soFarOk; it++ ) {
      soFarOk = ( soFarOk && _location.count( *it ) == 1 );
   }
   return soFarOk;
}

inline bool NewDirectoryEntryData::contains( const NewDirectoryEntryData &d ) const {
   bool soFarOk = ( _version == d._version && _writeLocation == d._writeLocation );
   for ( std::set< int >::iterator it = d._location.begin(); it != d._location.end() && soFarOk; it++ ) {
      soFarOk = ( soFarOk && _location.count( *it ) == 1 );
   }
   return soFarOk;
}

inline int NewDirectoryEntryData::getFirstLocation() const {
   return *(_location.begin());
}

inline int NewDirectoryEntryData::getNumLocations() const {
   return _location.size();
}

template <class RegionDesc>
Region NewRegionDirectory::build_region( RegionDesc const &dataAccess ) {
   // Find out the displacement due to the lower bounds and correct it in the address
   size_t base = 1UL;
   size_t displacement = 0L;
   for (short dimension = 0; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];
      displacement = displacement + dimensionData.lower_bound * base;
      base = base * dimensionData.size;
   }
   size_t address = (size_t)dataAccess.address + displacement;

   // Build the Region

   // First dimension is base 1
   size_t additionalContribution = 0UL; // Contribution of the previous dimensions (necessary due to alignment issues)
   //std::cerr << "build region 0 len is " << dataAccess.dimensions[0].accessed_length << std::endl;
   Region region = RegionBuilder::build(address, 1UL, dataAccess.dimensions[0].accessed_length, additionalContribution);

   // Add the bits corresponding to the rest of the dimensions (base the previous one)
   base = 1 * dataAccess.dimensions[0].size;
   for (short dimension = 1; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];

      //std::cerr << "build region " << dimension << " len is " << dimensionData.accessed_length << std::endl;
      region |= RegionBuilder::build(address, base, dimensionData.accessed_length, additionalContribution);
      base = base * dimensionData.size;
   }
   //std::cerr << "end build region n" << std::endl;

   return region;
}

template <class RegionDesc>
Region NewRegionDirectory::build_region_with_given_base_address( RegionDesc const &dataAccess, uint64_t newBaseAddress ) {
   // Find out the displacement due to the lower bounds and correct it in the address
   size_t base = 1UL;
   size_t displacement = 0L;
   for (short dimension = 0; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];
      displacement = displacement + dimensionData.lower_bound * base;
      base = base * dimensionData.size;
   }
   size_t address = (size_t)newBaseAddress + displacement;

   // Build the Region

   // First dimension is base 1
   size_t additionalContribution = 0UL; // Contribution of the previous dimensions (necessary due to alignment issues)
   //std::cerr << "build region 0 len is " << dataAccess.dimensions[0].accessed_length << std::endl;
   Region region = RegionBuilder::build(address, 1UL, dataAccess.dimensions[0].accessed_length, additionalContribution);

   // Add the bits corresponding to the rest of the dimensions (base the previous one)
   base = 1 * dataAccess.dimensions[0].size;
   for (short dimension = 1; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];

      //std::cerr << "build region " << dimension << " len is " << dimensionData.accessed_length << std::endl;
      region |= RegionBuilder::build(address, base, dimensionData.accessed_length, additionalContribution);
      base = base * dimensionData.size;
   }
   //std::cerr << "end build region n" << std::endl;

   return region;
}
#endif
