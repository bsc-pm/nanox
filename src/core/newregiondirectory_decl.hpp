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

#ifndef NEW_NANOS_NEW2DIRECTORY_DECL_H
#define NEW_NANOS_NEW2DIRECTORY_DECL_H

#include "regiondict_decl.hpp"

namespace nanos
{
   class NewNewDirectoryEntryData : public Version {
      private:
         int _writeLocation;
         std::set< int > _location;
      public:
         NewNewDirectoryEntryData();
         NewNewDirectoryEntryData( const NewNewDirectoryEntryData &de );
         ~NewNewDirectoryEntryData();
         const NewNewDirectoryEntryData & operator= ( const NewNewDirectoryEntryData &de );
         bool hasWriteLocation() const ;
         int getWriteLocation() const ;
         void setWriteLocation( int id ) ;
         void addAccess( int id, unsigned int version ); 
         bool isLocatedIn( int id, unsigned int version ) const ;
         bool isLocatedIn( int id ) const ;
         void merge( const NewNewDirectoryEntryData &de ) ;
         void print() const ;
         bool equal( const NewNewDirectoryEntryData &d ) const ;
         bool contains( const NewNewDirectoryEntryData &d ) const ;
         int getFirstLocation() const ;
         int getNumLocations() const ;
         friend std::ostream & operator<< (std::ostream &o, NewNewDirectoryEntryData const &entry);
   };

  /*! \class NewDirectory
   *  \brief Stores copy accesses controls their versions and if they are dirty in any cache
   */
   class NewNewRegionDirectory
   {
      private:
         std::map< uint64_t, RegionDictionary * > _objects;
         Lock _lock;

      private:

         /*! \brief NewDirectory copy constructor (private) 
          */
         NewNewRegionDirectory( const NewNewRegionDirectory &dir );

         /*! \brief NewDirectory copy assignment operator (private) 
          */
         const NewNewRegionDirectory & operator= ( const NewNewRegionDirectory &dir );

         RegionDictionary *getRegionDictionaryRegisterIfNeeded( CopyData const &cd );
         RegionDictionary *getRegionDictionary( CopyData const &cd ) const;
         RegionDictionary *getRegionDictionary( uint64_t addr ) const;
         static void addSubRegion( RegionDictionary &dict, std::list< std::pair< reg_t, reg_t > > &partsList, reg_t regionToInsert );

      public:
         typedef RegionDictionary *RegionDirectoryKey;
         //typedef std::pair< Region, NewNewDirectoryEntryData const *> LocationInfo;
         typedef std::list< std::pair< reg_t, reg_t > > NewLocationInfoList; 
         RegionDirectoryKey getRegionDirectoryKey( CopyData const &cd ) const;
         RegionDirectoryKey getRegionDirectoryKey( uint64_t addr ) const;
         RegionDirectoryKey getRegionDirectoryKeyRegisterIfNeeded( CopyData const &cd );
         void synchronize( bool flushData );

         /*! \brief NewDirectory default constructor
          */
         NewNewRegionDirectory();

         /*! \brief NewDirectory destructor
          */
         ~NewNewRegionDirectory() {};

         //void invalidate( RegionTree<CachedRegionStatus> *regions );

         void print() const;
         void lock();
         void tryLock();
         RegionDictionary &getDictionary( CopyData const &cd ) const;

         static NewNewDirectoryEntryData *getDirectoryEntry( RegionDictionary &dict, reg_t id );

         static reg_t _getLocation( RegionDirectoryKey dict, CopyData const &cd, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static reg_t tryGetLocation( RegionDirectoryKey dict, CopyData const &cd, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, unsigned int loc, unsigned int version );
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, unsigned int loc );
         static bool hasWriteLocation( RegionDirectoryKey dict, reg_t id );
         static unsigned int getWriteLocation( RegionDirectoryKey dict, reg_t id );
         static unsigned int getVersion( RegionDirectoryKey dict, reg_t id );
         static void addAccess( RegionDirectoryKey dict, reg_t id, unsigned int memorySpaceId, unsigned int version );
         static unsigned int getFirstLocation( RegionDirectoryKey dict, reg_t id );
   };
}

#endif
