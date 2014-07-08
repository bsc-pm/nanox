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
#include "processingelement_fwd.hpp"

namespace nanos
{
   class NewNewDirectoryEntryData : public Version {
      private:
         //int _writeLocation;
         //int _invalidated;
         DeviceOps _ops;
         std::set< memory_space_id_t > _location;
         std::set< ProcessingElement *> _pes;
         bool _rooted;
         Lock _setLock;
         ProcessingElement * _firstWriterPE;
      public:
         NewNewDirectoryEntryData();
         NewNewDirectoryEntryData( const NewNewDirectoryEntryData &de );
         ~NewNewDirectoryEntryData();
         NewNewDirectoryEntryData & operator= ( NewNewDirectoryEntryData &de );
         // bool hasWriteLocation() const ;
         // int getWriteLocation() const ;
         // void setWriteLocation( int id ) ;
         void addAccess( ProcessingElement *pe, memory_space_id_t loc, unsigned int version );
         void addRootedAccess( memory_space_id_t loc, unsigned int version );
         bool delAccess( memory_space_id_t id );
         //void invalidate(); 
         //bool hasBeenInvalidated() const; 
         bool isLocatedIn( ProcessingElement *pe, unsigned int version );
         bool isLocatedIn( ProcessingElement *pe );
         bool isLocatedIn( memory_space_id_t loc );
         void setRooted();
         bool isRooted() const;
         //void merge( const NewNewDirectoryEntryData &de ) ;
         void print() const ;
         //bool equal( const NewNewDirectoryEntryData &d ) const ;
         //bool contains( const NewNewDirectoryEntryData &d ) const ;
         int getFirstLocation();
         ProcessingElement *getFirstWriterPE() const;
         int getNumLocations();
         void setOps( DeviceOps *ops );
         std::set< memory_space_id_t > const &getLocations() const;
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
         //struct __attribute__((aligned(64))) HashBucket {
         struct HashBucket {
            Lock _lock;
            std::map< uint64_t, GlobalRegionDictionary * > _bobjects;
            HashBucket();
            HashBucket( HashBucket const & hb );
            HashBucket &operator=( HashBucket const &hb );
         };

         std::vector< HashBucket > _objects;

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
         void synchronize( WD const &wd );
         reg_t getLocalRegionId( void *hostObject, reg_t hostRegionId ) const;

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
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe, unsigned int version );
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe );
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, memory_space_id_t loc );
         //static bool hasWriteLocation( RegionDirectoryKey dict, reg_t id );
         //static unsigned int getWriteLocation( RegionDirectoryKey dict, reg_t id );
         static unsigned int getVersion( RegionDirectoryKey dict, reg_t id, bool increaseVersion );
         static void addAccess( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe, memory_space_id_t loc, unsigned int version );
         static void addRootedAccess( RegionDirectoryKey dict, reg_t id, memory_space_id_t loc, unsigned int version );
         //static void addAccessRegisterIfNeeded( RegionDirectoryKey dict, reg_t id, unsigned int memorySpaceId, unsigned int version );
         static bool delAccess( RegionDirectoryKey dict, reg_t id, memory_space_id_t memorySpaceId );
         static bool isOnlyLocated( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe );
         static bool isOnlyLocated( RegionDirectoryKey dict, reg_t id, memory_space_id_t loc );
         static void invalidate( RegionDirectoryKey dict, reg_t id );
         static bool hasBeenInvalidated( RegionDirectoryKey dict, reg_t id );
         static void updateFromInvalidated( RegionDirectoryKey dict, reg_t id, reg_t from );
         static unsigned int getFirstLocation( RegionDirectoryKey dict, reg_t id );
         static DeviceOps *getOps( RegionDirectoryKey dict, reg_t id );
         static void setOps( RegionDirectoryKey dict, reg_t id, DeviceOps *ops );


         static void tryGetLocation( RegionDirectoryKey dict, reg_t reg, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static void __getLocation( RegionDirectoryKey dict, reg_t reg, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static void initializeEntry( RegionDirectoryKey dict, reg_t reg );
         static void addRegionId( RegionDirectoryKey dict, reg_t masterId, reg_t localId );
         static reg_t getLocalRegionIdFromMasterRegionId( RegionDirectoryKey dict, reg_t localId );
         static void addMasterRegionId( RegionDirectoryKey dict, reg_t masterId, reg_t localId );
   };
}

#endif
