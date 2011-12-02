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

#ifndef _NANOS_NEWDIRECTORY_DECL_H
#define _NANOS_NEWDIRECTORY_DECL_H

#include <cstddef>
#include <list>
#include <vector>
#include <set>
#include <iostream>
#include "compatibility.hpp"
#include "memorymap_decl.hpp"
#include "atomic_decl.hpp"

namespace nanos
{
   class NewDirectoryEntryData {
      private:
         int _writeLocation;
         unsigned int _version;
         std::set<int> _location;
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
         void addAccess( int id ) { _location.insert( id ); }
         bool isLocatedIn( int id ) const { return ( _location.count( id ) > 0 ); }
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
            } else {
               std::cerr << "version mismatch! WARNING !!! two write locations!, missing dependencies? current " << _version << " inc " << de._version << std::endl;
            }
         }
         void print() {
            std::cerr << "WL: " << _writeLocation << " V: " << _version << " Locs: ";
            for ( std::set<int>::iterator it = _location.begin(); it != _location.end(); it++ ) {
               std::cerr << *it << " ";
            }
            std::cerr << std::endl;
         }
         bool equal( const NewDirectoryEntryData &d ) const {
            bool soFarOk = ( _version == d._version && _writeLocation == d._writeLocation );
            for ( std::set<int>::iterator it = _location.begin(); it != _location.end() && soFarOk; it++ ) {
               soFarOk = ( soFarOk && d._location.count( *it ) == 1 );
            }
            for ( std::set<int>::iterator it = d._location.begin(); it != d._location.end() && soFarOk; it++ ) {
               soFarOk = ( soFarOk && _location.count( *it ) == 1 );
            }
            return soFarOk;
         }
   };

  /*! \class NewDirectory
   *  \brief Stores copy accesses controls their versions and if they are dirty in any cache
   */
   class NewDirectory
   {
      private:
         typedef MemoryMap<NewDirectoryEntryData> NewDirectoryMap; /**< Directorie's HashMap*/
         NewDirectoryMap _directory; /**< The map will store the entries indexed by they tag */
         NewDirectoryMap _inputDirectory; /**< The map will store the entries indexed by they tag */
         NewDirectory *_parent;
         Lock _mergeLock;
         Lock _outputMergeLock;

         static NewDirectory *_root;

      private:

         /*! \brief NewDirectory copy constructor (private) 
          */
         NewDirectory( const NewDirectory &dir );

         /*! \brief NewDirectory copy assignment operator (private) 
          */
         const NewDirectory & operator= ( const NewDirectory &dir );

      public:

         /*! \brief NewDirectory default constructor
          */
         NewDirectory();

         /*! \brief NewDirectory destructor
          */
         ~NewDirectory() {};

        /*! \brief Set the parent NewDirectory to 'parent'
         */
         void setParent( NewDirectory *parent );

        /*! \brief Register an access to a copy by the host
         *  \param tag Identifier key of the access (address)
         *  \param size Size of the read or written block
         *  \param input Whether the access is a read
         *  \param output Whether the access is a write
         */
         void registerAccess( uint64_t tag, std::size_t size, bool input, bool output, unsigned int memorySpaceId );

         void merge( const NewDirectory &input );
         void mergeOutput( const NewDirectory &input );
         void setRoot();
         bool isRoot() const;
         void consolidate();
         void print() const;
   };
};

#endif
