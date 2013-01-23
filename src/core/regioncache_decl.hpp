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

#ifndef _NANOS_REGION_CACHE_H
#define _NANOS_REGION_CACHE_H

#include "memorymap_decl.hpp"
#include "region_decl.hpp"
#include "copydata_decl.hpp"
#include "atomic_decl.hpp"
#include "workdescriptor_fwd.hpp"
#include "processingelement_fwd.hpp"
#include "deviceops_decl.hpp"
#include "regiondirectory_decl.hpp"
#include "newregiondirectory_decl.hpp"

namespace nanos {

   class AllocatedChunk {
      private:
         Lock                              _lock;
         uint64_t                          _address;
         uint64_t                          _hostAddress;
         std::size_t                       _size;
         bool                              _dirty;
         std::size_t                       _roBytes;
         std::size_t                       _rwBytes;
         
         RegionTree< CachedRegionStatus > *_regions;

         RegionDictionary *_newRegions;

      public:
         //AllocatedChunk( );
         AllocatedChunk( uint64_t addr, uint64_t hostAddr, std::size_t size, CopyData const &cd );
         AllocatedChunk( AllocatedChunk const &chunk );
         AllocatedChunk &operator=( AllocatedChunk const &chunk );
         ~AllocatedChunk();

         uint64_t getAddress() const;
         uint64_t getHostAddress() const;
         std::size_t getSize() const;
         bool isDirty() const;
         void setHostAddress( uint64_t addr );

         void addReadRegion( Region const &reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::list< Region > &notPresentRegions, DeviceOps *ops, bool alsoWriteReg );
         void addWriteRegion( Region const &reg, unsigned int version );
         void clearRegions();
         RegionTree< CachedRegionStatus > *getRegions();
         bool isReady( Region reg );

         void lock();
         void unlock();
         void NEWaddReadRegion( reg_t reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::list< reg_t > &notPresentRegions, DeviceOps *ops, bool alsoWriteReg );
         void NEWaddWriteRegion( reg_t reg, unsigned int version );
   };

   class CacheCopy;
   
   class RegionCache {
      public:
         enum CacheOptions {
            ALLOC_FIT,
            ALLOC_WIDE
         };
      private:
         MemoryMap<AllocatedChunk>  _chunks;
         Lock                       _lock;
         Device                    &_device;
         ProcessingElement         &_pe;
         CacheOptions               _flags;

         typedef MemoryMap<AllocatedChunk>::MemChunkList ChunkList;
         typedef MemoryMap<AllocatedChunk>::ConstMemChunkList ConstChunkList;

         class Op {
               RegionCache &_parent;
               std::string _name;
            public:
               Op( RegionCache &parent, std::string name ) : _parent ( parent ), _name ( name ) { }
               RegionCache &getParent() const { return _parent; }
               std::string const &getStr() { return _name; }
               virtual void doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, WD const &wd, bool fake ) = 0;
               virtual void doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) = 0;
         };

         class CopyIn : public Op {
            public:
               CopyIn( RegionCache &parent ) : Op( parent, "CopyIn" ) {}
               void doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, WD const &wd, bool fake ) ;
               void doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) ;
         } _copyInObj;

         class CopyOut : public Op {
            public:
               CopyOut( RegionCache &parent ) : Op( parent, "CopyOut" ) {}
               void doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, WD const &wd, bool fake ) ;
               void doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake ) ;
         } _copyOutObj;

         void doOp( Op *opObj, Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ); 
         void doOp( Op *opObj, global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ); 
         //void _generateRegionOps( Region const &reg, std::map< uintptr_t, MemoryMap< uint64_t > * > &opMap );

      public:
         RegionCache( ProcessingElement &pe, Device &cacheArch, enum CacheOptions flags );
         AllocatedChunk *getAddress( CopyData const &d, RegionTree< CachedRegionStatus > *&regsToInvalidate );
         AllocatedChunk *getAddress( uint64_t hostAddr, std::size_t len );
         void syncRegion( Region const &r ) ;
         void syncRegion( std::list< std::pair< Region, CacheCopy * > > const &regions, WD const &wd ) ;
         void syncRegion( global_reg_t const &r ) ;
         void syncRegion( std::list< std::pair< global_reg_t, CacheCopy * > > const &regions, WD const &wd ) ;
         unsigned int getMemorySpaceId();
         /* device stubs */
         void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake );
         void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake );
         void _syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake );
         void _copyDevToDev( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake );
         void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake );
         void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake );
         void _syncAndCopyInStrided1D( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake );
         void _copyDevToDevStrided1D( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake );
         /* *********** */
         void copyIn( Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ); 
         void copyOut( Region const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, WD const &wd ); 
         void copyIn( global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ); 
         void copyOut( global_reg_t const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, WD const &wd ); 
         void lock();
         void unlock();
         bool tryLock();
         bool canCopyFrom( RegionCache const &from ) const;
         Device const &getDevice() const;
         unsigned int getNodeNumber() const;
         ProcessingElement &getPE() const;
   };

   class CacheController;
   class CacheCopy {

      private:
         CopyData const &_copy;
         AllocatedChunk *_cacheEntry;
         std::list< std::pair<Region, CachedRegionStatus const &> > _cacheDataStatus;
         Region _region;
         uint64_t _offset;
         unsigned int _version;
         unsigned int _newVersion;
         NewRegionDirectory::LocationInfoList _locations;
         NewNewRegionDirectory::NewLocationInfoList _newLocations;
         DeviceOps _operations;
         std::set< DeviceOps * > _otherPendingOps;
         reg_t _regId;

      public:
         global_reg_t _reg;
         CacheCopy();
         CacheCopy( WD const &wd, unsigned int index, CacheController &ccontrol );
         
         bool isReady();
         void setUpDeviceAddress( RegionCache *targetCache, NewRegionDirectory *dir );
         void generateCopyInOps( RegionCache *targetCache, std::map<unsigned int, std::list< std::pair< Region, CacheCopy * > > > &opsBySourceRegions ) ;
         void NEWgenerateCopyInOps( RegionCache *targetCache, std::map<unsigned int, std::list< std::pair< global_reg_t, CacheCopy * > > > &opsBySourceRegions ) ;
         bool tryGetLocation( WD const &wd, unsigned int index );

         NewRegionDirectory::LocationInfoList const &getLocations() const;
         NewNewRegionDirectory::NewLocationInfoList const &getNewLocations() const;
         uint64_t getDeviceAddress() const;
         DeviceOps *getOperations();
         Region const &getRegion() const;
         unsigned int getVersion() const;
         unsigned int getNewVersion() const;
         CopyData const & getCopyData() const;
         reg_t getRegId() const;
         NewNewRegionDirectory::RegionDirectoryKey getRegionDirectoryKey() const;
   };


   class CacheController {

      private:
         WD const &_wd;
         unsigned int _numCopies;
         CacheCopy *_cacheCopies;
         RegionCache *_targetCache;  
         bool _registered;
         Lock _provideLock;
         std::map< NewNewRegionDirectory::RegionDirectoryKey, std::map< reg_t, unsigned int > > _providedRegions;

      public:
         CacheController();
         CacheController( WD const &wd );
         ~CacheController();
         bool isCreated() const;
         void preInit( );
         void copyDataIn( RegionCache *targetCache );
         bool dataIsReady() ;
         uint64_t getAddress( unsigned int copyIndex ) const;
         void copyDataOut();
         void getInfoFromPredecessor( CacheController const &predecessorController );
         bool hasVersionInfoForRegion( global_reg_t reg, unsigned int &version, NewNewRegionDirectory::NewLocationInfoList &locations ) ;

         CacheCopy *getCacheCopies() const;
         RegionCache *getTargetCache() const;
   };
}

#endif
