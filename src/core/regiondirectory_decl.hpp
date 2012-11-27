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
#include "workdescriptor_fwd.hpp"
#include "cachedregionstatus_decl.hpp"

namespace nanos
{
   class NewDirectoryEntryData {

      private:
         int _writeLocation;
         unsigned int _version;
         std::set< int > _location;
      public:
         NewDirectoryEntryData();
         NewDirectoryEntryData( const NewDirectoryEntryData &de );
         ~NewDirectoryEntryData();
         const NewDirectoryEntryData & operator= ( const NewDirectoryEntryData &de );
         bool hasWriteLocation() const ;
         int getWriteLocation() const ;
         void setWriteLocation( int id ) ;
         void addAccess( int id, uint64_t address, unsigned int version ); 
         bool isLocatedIn( int id ) const ;
         unsigned int getVersion() const ;
         void merge( const NewDirectoryEntryData &de ) ;
         void print() const ;
         bool equal( const NewDirectoryEntryData &d ) const ;
         bool contains( const NewDirectoryEntryData &d ) const ;
         int getFirstLocation() const ;
         int getNumLocations() const ;
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
         //typedef std::pair< Region, NewDirectoryEntryData const *> LocationInfo;
         typedef std::pair< Region const, NewDirectoryEntryData > LocationInfo;
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
         void getLocation( Region const &reg, LocationInfoList &loc, unsigned int &version, WD const &wd );
         void masterGetLocation( Region const &reg, LocationInfoList &loc, unsigned int &version );
         void addAccess(Region const &reg, unsigned int memorySpaceId, unsigned int version );

         void merge( const NewRegionDirectory &input );
         void mergeOutput( const NewRegionDirectory &input );
         void setRoot();
         bool isRoot() const;
         void consolidate( bool flushData );
         void print() const;
         void lock();
         void unlock();
         void invalidate( RegionTree<CachedRegionStatus> *regions );

         static void insertRegionIntoTree( RegionTree<NewDirectoryEntryData> &dir, Region const &r, unsigned int memorySpaceId, bool setLoc, NewDirectoryEntryData const &ent, unsigned int version);
         template <class RegionDesc> static Region build_region( RegionDesc const &cd );
         template <class RegionDesc> static Region build_region_with_given_base_address( RegionDesc const &dataAccess, uint64_t newBaseAddress );

      private:
         static void _internal_merge( RegionTree<NewDirectoryEntryData> const &inputDir, RegionTree<NewDirectoryEntryData> &targetDir, bool print = false );
   };


}


#endif
