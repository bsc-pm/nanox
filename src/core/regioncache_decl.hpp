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

#include "functor_decl.hpp"
#include "memorymap_decl.hpp"
#include "region_decl.hpp"
#include "copydata_decl.hpp"
#include "atomic_decl.hpp"
#include "workdescriptor_fwd.hpp"
#include "processingelement_fwd.hpp"
#include "deviceops_decl.hpp"
#include "newregiondirectory_decl.hpp"
#include "memoryops_fwd.hpp"

namespace nanos {

   class RegionCache;

   class AllocatedChunk {
      private:
         RegionCache                      &_owner;
         Lock                              _lock;
         uint64_t                          _address;
         uint64_t                          _hostAddress;
         std::size_t                       _size;
         bool                              _dirty;
         bool                              _invalidated;
         unsigned int                      _lruStamp;
         std::size_t                       _roBytes;
         std::size_t                       _rwBytes;
         Atomic<unsigned int>              _refs;
         global_reg_t                      _allocatedRegion;
         CacheRegionDictionary *_newRegions;

      public:
         static Atomic<int> numCall;
         AllocatedChunk( RegionCache &owner, uint64_t addr, uint64_t hostAddr, std::size_t size, global_reg_t const &allocatedRegion );
         AllocatedChunk( AllocatedChunk const &chunk );
         AllocatedChunk &operator=( AllocatedChunk const &chunk );
         ~AllocatedChunk();

         uint64_t getAddress() const;
         uint64_t getHostAddress() const;
         std::size_t getSize() const;
         bool isDirty() const;
         unsigned int getLruStamp() const;
         void increaseLruStamp();
         void setHostAddress( uint64_t addr );

         void addReadRegion( Region const &reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::list< Region > &notPresentRegions, DeviceOps *ops, bool alsoWriteReg );
         void addWriteRegion( Region const &reg, unsigned int version );
         void clearRegions();
         void clearNewRegions( global_reg_t const &newAllocatedRegion );
         CacheRegionDictionary *getNewRegions();
         bool isReady( Region reg );
         bool isInvalidated() const;
         void invalidate(RegionCache *targetCache, WD const &wd );

         void lock();
         void unlock();
         void NEWaddReadRegion( reg_t reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::list< reg_t > &notPresentRegions, DeviceOps *ops, bool alsoWriteReg );
         bool NEWaddReadRegion2( BaseAddressSpaceInOps &ops, reg_t reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::set< reg_t > &notPresentRegions, std::set<DeviceOps *> &thisRegOps, bool output, NewLocationInfoList const &locations );
         void NEWaddWriteRegion( reg_t reg, unsigned int version );
         void addReference();
         void removeReference();
         unsigned int getReferenceCount() const;
         void confirmCopyIn( reg_t id, unsigned int version );
         unsigned int getVersion( global_reg_t const &reg );

         DeviceOps *getDeviceOps( global_reg_t const &reg );
         void prepareRegion( reg_t reg, unsigned int version );
   };

   class CompleteOpFunctor : public Functor {
      private:
         DeviceOps *_ops;
         AllocatedChunk *_chunk;
      public:
         CompleteOpFunctor( DeviceOps *ops, AllocatedChunk *_chunk );
         virtual ~CompleteOpFunctor();
         virtual void operator()();
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
         memory_space_id_t          _memorySpaceId;
         CacheOptions               _flags;
         unsigned int               _lruTime;

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

         void doOp( Op *opObj, global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ); 

      public:
         RegionCache( memory_space_id_t memorySpaceId, Device &cacheArch, enum CacheOptions flags );
         AllocatedChunk *getAddress( global_reg_t const &reg, CacheRegionDictionary *&newRegsToInvalidate, WD const &wd );
         AllocatedChunk *getAllocatedChunk( global_reg_t const &reg ) const;
         AllocatedChunk *getAddress( uint64_t hostAddr, std::size_t len );
         AllocatedChunk **selectChunkToInvalidate( std::size_t allocSize );
         void syncRegion( Region const &r ) ;
         void syncRegion( std::list< std::pair< Region, CacheCopy * > > const &regions, WD const &wd ) ;
         void syncRegion( global_reg_t const &r ) ;
         void syncRegion( std::list< std::pair< global_reg_t, CacheCopy * > > const &regions, WD const &wd ) ;
         memory_space_id_t getMemorySpaceId() const;
         /* device stubs */
         void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake );
         void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake );
         void _syncAndCopyIn( memory_space_id_t syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake );
         void _copyDevToDev( memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, WD const &wd, bool fake );
         void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake );
         void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake );
         void _syncAndCopyInStrided1D( memory_space_id_t syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake );
         void _copyDevToDevStrided1D( memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, WD const &wd, bool fake );
         /* *********** */
         void copyIn( Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ); 
         void copyOut( Region const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, WD const &wd ); 
         void copyIn( global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, WD const &wd ); 
         void copyOut( global_reg_t const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, WD const &wd ); 
         void NEWcopyIn( unsigned int location, global_reg_t const &hostMem, unsigned int version, WD const &wd ); 
         void NEWcopyOut( global_reg_t const &hostMem, unsigned int version, WD const &wd ); 
         uint64_t getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress ) const;
         void lock();
         void unlock();
         bool tryLock();
         bool canCopyFrom( RegionCache const &from ) const;
         Device const &getDevice() const;
         unsigned int getNodeNumber() const;
         unsigned int getLruTime() const;
         void increaseLruTime();
         bool pin( global_reg_t const &hostMem );
         void unpin( global_reg_t const &hostMem );

         unsigned int getVersion( global_reg_t const &hostMem );
         void releaseRegion( global_reg_t const &hostMem );
         void prepareRegion( global_reg_t const &hostMem, WD const &wd );

         void copyInputData( BaseAddressSpaceInOps &ops, global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations );
         void allocateOutputMemory( global_reg_t const &reg, unsigned int version );
   };
}

#endif
