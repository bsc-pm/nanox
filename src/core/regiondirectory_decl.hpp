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

#ifndef NANOS_NEW2DIRECTORY_DECL_H
#define NANOS_NEW2DIRECTORY_DECL_H

#include <cstddef>
#include <list>
#include <vector>
#include <set>
#include <iostream>
#include "compatibility.hpp"
#include "regiontree_decl.hpp"
#include "atomic_decl.hpp"

namespace nanos
{
   class NewDirectoryEntryData {

      private:
         class LocationEntry {
            private:
               int _memorySpaceId;
               uint64_t _address;
            public:
               LocationEntry(): _memorySpaceId( 0 ), _address( 0xdeadbeef ) {}
               LocationEntry( int id, uint64_t address ): _memorySpaceId( id ), _address( address ) {}
               LocationEntry( LocationEntry const &le ): _memorySpaceId( le._memorySpaceId ), _address( le._address ) {}
               LocationEntry &operator=( LocationEntry const &le ) { _memorySpaceId = le._memorySpaceId; _address = le._address; return *this; }
               bool operator<( LocationEntry const &le ) const { return _memorySpaceId < le._memorySpaceId; }
               int getMemorySpaceId() const { return _memorySpaceId; }
               uint64_t getAddress() const { return _address; }
               void setMemorySpaceId( int id ) { _memorySpaceId = id; }
               void setAddress( uint64_t addr ) { _address = addr; }
         };
         int _writeLocation;
         unsigned int _version;
         std::set< LocationEntry > _location;
      public:
         NewDirectoryEntryData(): _writeLocation(0), _version(1), _location() { }
         NewDirectoryEntryData( const NewDirectoryEntryData &de ): _writeLocation( de._writeLocation ),
            _version( de._version ), _location( de._location ) { }
         ~NewDirectoryEntryData() { }
         const NewDirectoryEntryData & operator= ( const NewDirectoryEntryData &de ) {
            _writeLocation = de._writeLocation;
            _version = de._version;
            _location.clear();
            _location.insert( de._location.begin(), de._location.end() );
            return *this;
         }
         bool hasWriteLocation() const { return ( _writeLocation != -1 ); }
         int getWriteLocation() const { return _writeLocation; }
         void setWriteLocation( int id ) { _writeLocation = id; }
         void addAccess( int id, uint64_t address ) { _location.insert( LocationEntry( id, address ) ); }
         bool isLocatedIn( int id ) const { return ( _location.count( LocationEntry( id, -1UL ) ) > 0 ); }
         uint64_t getAddressOfLocation( int id ) const { std::set< LocationEntry >::iterator it = _location.find( LocationEntry( id, -1UL ) ); return it->getAddress(); }
         void increaseVersion() { _version += 1; }
         void setVersion( unsigned int ver ) { _version = ver; }
         unsigned int getVersion() const { return _version; }
         void merge( const NewDirectoryEntryData &de ) {
            if ( hasWriteLocation() && de.hasWriteLocation() ) {
               if ( getWriteLocation() != de.getWriteLocation() && _version >= de._version ) std::cerr << "write loc mismatch WARNING !!! two write locations!, missing dependencies?" << std::endl;
            } else if ( de.hasWriteLocation() ) {
               setWriteLocation( de.getWriteLocation() );
            } else setWriteLocation( -1 );

            if ( _version == de._version ) {
               _location.insert( de._location.begin(), de._location.end() );
            }
            else if ( _version < de._version ){
               _location.clear();
               _location.insert( de._location.begin(), de._location.end() );
               _version = de._version;
            } /*else {
               std::cerr << "version mismatch! WARNING !!! two write locations!, missing dependencies? current " << _version << " inc " << de._version << std::endl;
            }*/
         }
         void print() {
            std::cerr << "WL: " << _writeLocation << " V: " << _version << " Locs: ";
            for ( std::set< LocationEntry >::iterator it = _location.begin(); it != _location.end(); it++ ) {
               std::cerr << it->getMemorySpaceId() << ":" << it->getAddress() << " ";
            }
            std::cerr << std::endl;
         }
         bool equal( const NewDirectoryEntryData &d ) const {
            bool soFarOk = ( _version == d._version && _writeLocation == d._writeLocation );
            for ( std::set< LocationEntry >::iterator it = _location.begin(); it != _location.end() && soFarOk; it++ ) {
               soFarOk = ( soFarOk && d._location.count( *it ) == 1 );
            }
            for ( std::set< LocationEntry >::iterator it = d._location.begin(); it != d._location.end() && soFarOk; it++ ) {
               soFarOk = ( soFarOk && _location.count( *it ) == 1 );
            }
            return soFarOk;
         }
         bool contains( const NewDirectoryEntryData &d ) const {
            bool soFarOk = ( _version == d._version && _writeLocation == d._writeLocation );
            for ( std::set< LocationEntry >::iterator it = d._location.begin(); it != d._location.end() && soFarOk; it++ ) {
               soFarOk = ( soFarOk && _location.count( *it ) == 1 );
            }
            return soFarOk;
         }
         int getFirstLocation() const { return (_location.begin())->getMemorySpaceId(); }
         friend std::ostream & operator<< (std::ostream &o, NewDirectoryEntryData const &entry);
   };

  /*! \class NewDirectory
   *  \brief Stores copy accesses controls their versions and if they are dirty in any cache
   */
   class NewRegionDirectory
   {
      private:
         typedef RegionTree<NewDirectoryEntryData> NewRegionDirectoryMap; /**< Directorie's HashMap*/
         NewRegionDirectoryMap _directory; /**< The map will store the entries indexed by they tag */
         NewRegionDirectoryMap _inputDirectory; /**< The map will store the entries indexed by they tag */
         NewRegionDirectory *_parent;
         Lock _mergeLock;
         Lock _outputMergeLock;

         static NewRegionDirectory *_root;

      private:

         /*! \brief NewDirectory copy constructor (private) 
          */
         NewRegionDirectory( const NewRegionDirectory &dir );

         /*! \brief NewDirectory copy assignment operator (private) 
          */
         const NewRegionDirectory & operator= ( const NewRegionDirectory &dir );

      public:
         typedef std::pair< Region, NewDirectoryEntryData const *> LocationInfo;
         typedef std::list<LocationInfo> LocationInfoList; 

         /*! \brief NewDirectory default constructor
          */
         NewRegionDirectory();

         /*! \brief NewDirectory destructor
          */
         ~NewRegionDirectory() {};

        /*! \brief Set the parent NewDirectory to 'parent'
         */
         void setParent( NewRegionDirectory *parent );

        /*! \brief Register an access to a copy by the host
         *  \param tag Identifier key of the access (address)
         *  \param size Size of the read or written block
         *  \param input Whether the access is a read
         *  \param output Whether the access is a write
         */
         void registerAccess( Region reg, bool input, bool output, unsigned int memorySpaceId, uint64_t devAddr, LocationInfoList &loc );
         void addAccess(Region reg, bool input, bool output, unsigned int memorySpaceId, unsigned int version, uint64_t devAddr );

         void merge( const NewRegionDirectory &input );
         void mergeOutput( const NewRegionDirectory &input );
         void setRoot();
         bool isRoot() const;
         void consolidate( bool flushData );
         void print() const;
         bool checkConsistency( uint64_t tag, std::size_t size, unsigned int memorySpaceId );

         static void insertRegionIntoTree( RegionTree<NewDirectoryEntryData> &dir, Region const &r, unsigned int memorySpaceId, uint64_t devAddr, bool setLoc, NewDirectoryEntryData const &ent, unsigned int version);
         template <class RegionDesc> static Region build_region( RegionDesc const &cd );
         template <class RegionDesc> static Region build_region_with_given_base_address( RegionDesc const &dataAccess, uint64_t newBaseAddress );

      private:
         static void _internal_merge( RegionTree<NewDirectoryEntryData> const &inputDir, RegionTree<NewDirectoryEntryData> &targetDir );
   };


}


#endif
