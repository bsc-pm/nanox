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
#include "globalregt_decl.hpp"
#include "deviceops_decl.hpp"
#include "workdescriptor_fwd.hpp"

namespace nanos
{
   class NewNewDirectoryEntryData : public Version {
      private:
         int _writeLocation;
         int _invalidated;
         DeviceOpsPtr _opsPtr;
         DeviceOps _ops;
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
         bool delAccess( int id ); 
         void invalidate(); 
         bool hasBeenInvalidated() const; 
         bool isLocatedIn( int id, unsigned int version ) const ;
         bool isLocatedIn( int id ) const ;
         void merge( const NewNewDirectoryEntryData &de ) ;
         void print() const ;
         bool equal( const NewNewDirectoryEntryData &d ) const ;
         bool contains( const NewNewDirectoryEntryData &d ) const ;
         int getFirstLocation() const ;
         int getNumLocations() const ;
         void setOps( DeviceOps *ops );
         DeviceOps *getOps() ;
         friend std::ostream & operator<< (std::ostream &o, NewNewDirectoryEntryData const &entry);
   };

  /*! \class NewDirectory
   *  \brief Stores copy accesses controls their versions and if they are dirty in any cache
   */
   typedef std::list< std::pair< reg_t, reg_t > > NewLocationInfoList; 
   class NewNewRegionDirectory
   {
      private:
         std::map< uint64_t, GlobalRegionDictionary * > _objects;
         Lock _lock;

      private:

         /*! \brief NewDirectory copy constructor (private) 
          */
         NewNewRegionDirectory( const NewNewRegionDirectory &dir );

         /*! \brief NewDirectory copy assignment operator (private) 
          */
         const NewNewRegionDirectory & operator= ( const NewNewRegionDirectory &dir );

         GlobalRegionDictionary *getRegionDictionaryRegisterIfNeeded( CopyData const &cd );
         GlobalRegionDictionary *getRegionDictionary( CopyData const &cd ) const;
         GlobalRegionDictionary *getRegionDictionary( uint64_t addr ) const;
         static void addSubRegion( GlobalRegionDictionary &dict, std::list< std::pair< reg_t, reg_t > > &partsList, reg_t regionToInsert );

      public:
         typedef GlobalRegionDictionary *RegionDirectoryKey;
         //typedef std::pair< Region, NewNewDirectoryEntryData const *> LocationInfo;
         RegionDirectoryKey getRegionDirectoryKey( CopyData const &cd ) const;
         RegionDirectoryKey getRegionDirectoryKey( uint64_t addr ) const;
         RegionDirectoryKey getRegionDirectoryKeyRegisterIfNeeded( CopyData const &cd );
         void synchronize2( bool flushData );

         /*! \brief NewDirectory default constructor
          */
         NewNewRegionDirectory();

         /*! \brief NewDirectory destructor
          */
         ~NewNewRegionDirectory() {};

         void invalidate( CacheRegionDictionary *regions, unsigned int from );

         void print() const;
         void lock();
         void tryLock();
         GlobalRegionDictionary &getDictionary( CopyData const &cd ) const;

         static NewNewDirectoryEntryData *getDirectoryEntry( GlobalRegionDictionary &dict, reg_t id );

         static reg_t _getLocation( RegionDirectoryKey dict, CopyData const &cd, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static reg_t tryGetLocation( RegionDirectoryKey dict, CopyData const &cd, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, unsigned int loc, unsigned int version );
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, unsigned int loc );
         static bool hasWriteLocation( RegionDirectoryKey dict, reg_t id );
         static unsigned int getWriteLocation( RegionDirectoryKey dict, reg_t id );
         static unsigned int getVersion( RegionDirectoryKey dict, reg_t id, bool increaseVersion );
         static void addAccess( RegionDirectoryKey dict, reg_t id, unsigned int memorySpaceId, unsigned int version );
         static void addAccessRegisterIfNeeded( RegionDirectoryKey dict, reg_t id, unsigned int memorySpaceId, unsigned int version );
         static bool delAccess( RegionDirectoryKey dict, reg_t id, unsigned int memorySpaceId );
         static void invalidate( RegionDirectoryKey dict, reg_t id );
         static bool hasBeenInvalidated( RegionDirectoryKey dict, reg_t id );
         static void updateFromInvalidated( RegionDirectoryKey dict, reg_t id, reg_t from );
         static unsigned int getFirstLocation( RegionDirectoryKey dict, reg_t id );
         static DeviceOps *getOps( RegionDirectoryKey dict, reg_t id );
         static void setOps( RegionDirectoryKey dict, reg_t id, DeviceOps *ops );


         static void tryGetLocation( RegionDirectoryKey dict, reg_t reg, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static void __getLocation( RegionDirectoryKey dict, reg_t reg, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static void initializeEntry( RegionDirectoryKey dict, reg_t reg );

   };
}

#endif
