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
#include "copydata_decl.hpp"
#include "atomic_decl.hpp"
#include "workdescriptor_fwd.hpp"
#include "processingelement_fwd.hpp"
#include "deviceops_decl.hpp"
#include "newregiondirectory_decl.hpp"
#include "memoryops_fwd.hpp"
#include "cachedregionstatus_decl.hpp"
#include "memcachecopy_fwd.hpp"

#define VERBOSE_CACHE 0

namespace nanos {

   class RegionCache;

   class AllocatedChunk {
      private:
         RegionCache                      &_owner;
         RecursiveLock                     _lock;
         uint64_t                          _address;
         uint64_t                          _hostAddress;
         std::size_t                       _size;
         bool                              _dirty;
         bool                              _rooted;
         unsigned int                      _lruStamp;
         std::size_t                       _roBytes;
         std::size_t                       _rwBytes;
         Atomic<unsigned int>              _refs;
         std::map<int, unsigned int>       _refWdId;
         std::map<int, std::set<int> >     _refLoc;
         global_reg_t                      _allocatedRegion;
         
         CacheRegionDictionary *_newRegions;

      public:
         static Atomic<int> numCall;
         AllocatedChunk( RegionCache &owner, uint64_t addr, uint64_t hostAddr, std::size_t size, global_reg_t const &allocatedRegion, bool rooted );
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

         void clearRegions();
         void clearNewRegions( global_reg_t const &newAllocatedRegion );
         CacheRegionDictionary *getNewRegions();
         bool invalidate( RegionCache *targetCache, WD const &wd, unsigned int copyIdx, SeparateAddressSpaceOutOps &invalOps, std::set< global_reg_t > &regionsToRemoveAccess, std::set< NewNewRegionDirectory::RegionDirectoryKey > &alreadyLockedObjects );

         bool trylock();
         void lock( bool setVerbose=false );
         void unlock( bool unsetVerbose=false );
         bool locked() const;
         bool NEWaddReadRegion2( BaseAddressSpaceInOps &ops, reg_t reg, unsigned int version, std::set< reg_t > &notPresentRegions, NewLocationInfoList const &locations, WD const &wd, unsigned int copyIdx );
         void NEWaddWriteRegion( reg_t reg, unsigned int version, WD const &wd, unsigned int copyIdx );
         void setRegionVersion( reg_t reg, unsigned int version, WD const &wd, unsigned int copyIdx );
         void addReference(int wdId, unsigned int loc);
         void removeReference(int wdId);
         unsigned int getReferenceCount() const;
         //void confirmCopyIn( reg_t id, unsigned int version );
         unsigned int getVersion( global_reg_t const &reg );
         //unsigned int getVersionSetVersion( global_reg_t const &reg, unsigned int newVersion );
         //void removeRegionAndMarkForChunkDeallocation( reg_t reg, WD const &wd, unsigned int copyIdx );

         DeviceOps *getDeviceOps( global_reg_t const &reg, WD const &wd, unsigned int idx);
         void prepareRegion( reg_t reg, unsigned int version );
         global_reg_t getAllocatedRegion() const;
         bool isRooted() const;


         void copyRegionToHost( SeparateAddressSpaceOutOps &ops, reg_t reg, unsigned int version, WD const &wd, unsigned int copyIdx );
         //void clearDirty( global_reg_t const &reg );
         void printReferencingWDs() const;
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

   class RegionCache {
      public:
         enum CachePolicy { WRITE_BACK, WRITE_THROUGH, NO_CACHE };
         enum CacheOptions {
            ALLOC_FIT,
            ALLOC_WIDE,
            ALLOC_SLAB
         };
      private:
         MemoryMap<AllocatedChunk>  _chunks;
         RecursiveLock              _lock;
         Device                    &_device;
         //ProcessingElement         &_pe;
         memory_space_id_t          _memorySpaceId;
         CacheOptions               _flags;
         std::size_t                _slabSize;
         unsigned int               _lruTime;
         Atomic<unsigned int>       _softInvalidationCount;
         Atomic<unsigned int>       _hardInvalidationCount;

         typedef MemoryMap<AllocatedChunk>::MemChunkList ChunkList;
         typedef MemoryMap<AllocatedChunk>::ConstMemChunkList ConstChunkList;

         class Op {
               RegionCache &_parent;
               std::string _name;
            public:
               Op( RegionCache &parent, std::string name ) : _parent ( parent ), _name ( name ) { }
               RegionCache &getParent() const { return _parent; }
               std::string const &getStr() { return _name; }
               virtual void doNoStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) = 0;
               virtual void doStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) = 0;
         };

         class CopyIn : public Op {
            public:
               CopyIn( RegionCache &parent ) : Op( parent, "CopyIn" ) {}
               void doNoStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) ;
               void doStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) ;
         } _copyInObj;

         class CopyOut : public Op {
            public:
               CopyOut( RegionCache &parent ) : Op( parent, "CopyOut" ) {}
               void doNoStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) ;
               void doStrided( global_reg_t const &reg, int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake ) ;
         } _copyOutObj;

         void doOp( Op *opObj, global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd ); 

      public:
         RegionCache( memory_space_id_t memorySpaceId, Device &cacheArch, enum CacheOptions flags, std::size_t slabSize );
         AllocatedChunk *tryGetAddress( global_reg_t const &reg, WD const &wd, unsigned int copyIdx );
         AllocatedChunk *getOrCreateChunk( global_reg_t const &reg, WD const &wd, unsigned int copyIdx );
         AllocatedChunk *getAllocatedChunk( global_reg_t const &reg, WD const &wd, unsigned int copyIdx ) const;
         AllocatedChunk *getAllocatedChunk( global_reg_t const &reg, bool complain, WD const &wd, unsigned int copyIdx );
         AllocatedChunk *_getAllocatedChunk( global_reg_t const &reg, bool complain, bool lock, WD const &wd, unsigned int copyIdx ) const;
         AllocatedChunk *getAddress( uint64_t hostAddr, std::size_t len );
         AllocatedChunk **selectChunkToInvalidate( std::size_t allocSize );
         AllocatedChunk *invalidate( global_reg_t const &allocatedRegion, WD const &wd, unsigned int copyIdx );
         void invalidateObject( global_reg_t const &reg );
         void selectChunksToInvalidate( std::size_t allocSize, std::set< AllocatedChunk ** > &chunksToInvalidate, WD const &wd, unsigned int &otherReferencedChunks );
         void syncRegion( global_reg_t const &r ) ;
         //void syncRegion( std::list< std::pair< global_reg_t, CacheCopy * > > const &regions, WD const &wd ) ;
         unsigned int getMemorySpaceId() const;
         /* device stubs */
         void _copyIn( global_reg_t const &reg, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake );
         void _copyOut( global_reg_t const &reg, uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake );
         void _syncAndCopyIn( global_reg_t const &reg, memory_space_id_t syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake );
         bool _copyDevToDev( global_reg_t const &reg, memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake );
         void _copyInStrided1D( global_reg_t const &reg, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake );
         void _copyOutStrided1D( global_reg_t const &reg, uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake );
         void _syncAndCopyInStrided1D( global_reg_t const &reg, memory_space_id_t syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake );
         bool _copyDevToDevStrided1D( global_reg_t const &reg, memory_space_id_t copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd, bool fake );
         /* *********** */
         void copyIn( global_reg_t const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd ); 
         void copyOut( global_reg_t const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, CompleteOpFunctor *f, WD const &wd ); 
         void NEWcopyIn( unsigned int location, global_reg_t const &hostMem, unsigned int version, WD const &wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *chunk ); 
         void NEWcopyOut( global_reg_t const &hostMem, unsigned int version, WD const &wd, unsigned int copyIdx, DeviceOps *ops, bool inval ); 
         uint64_t getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress, AllocatedChunk *chunk ) const;
         void lock();
         void unlock();
         bool tryLock();
         bool canCopyFrom( RegionCache const &from ) const;
         Device const &getDevice() const;
         unsigned int getNodeNumber() const;
         unsigned int getLruTime() const;
         void increaseLruTime();

         unsigned int getVersion( global_reg_t const &hostMem, WD const &wd, unsigned int copyIdx );
         //void releaseRegion( global_reg_t const &hostMem, WD const &wd, unsigned int copyIdx, enum CachePolicy policy );

         void releaseRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd );
         bool prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd );
         void setRegionVersion( global_reg_t const &hostMem, unsigned int version, WD const &wd, unsigned int copyIdx );

         void copyInputData( BaseAddressSpaceInOps &ops, global_reg_t const &reg, unsigned int version, NewLocationInfoList const &locations, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx );
         void allocateOutputMemory( global_reg_t const &reg, ProcessingElement *pe, unsigned int version, WD const &wd, unsigned int copyIdx );

         unsigned int getSoftInvalidationCount() const;
         unsigned int getHardInvalidationCount() const;
         bool canAllocateMemory( MemCacheCopy *memCopies, unsigned int numCopies, bool considerInvalidations, WD const &wd );
         bool canInvalidateToFit( std::size_t *sizes, unsigned int numChunks ) const;
         std::size_t getAllocatableSize( global_reg_t const &reg ) const;
         void getAllocatableRegion( global_reg_t const &reg, global_reg_t &allocRegion ) const;
         void prepareRegionsToBeCopied( std::set< global_reg_t > const &regs, unsigned int version, std::set< AllocatedChunk * > &chunks, WD const &wd, unsigned int copyIdx ) ;
         void _prepareRegionToBeCopied( global_reg_t const &regs, unsigned int version, std::set< AllocatedChunk * > &chunks, WD const &wd, unsigned int copyIdx ) ;
         void registerOwnedMemory(void *addr, std::size_t len);

         void copyOutputData( SeparateAddressSpaceOutOps &ops, global_reg_t const &reg, unsigned int version, bool output, enum CachePolicy policy, AllocatedChunk *chunk, WD const &wd, unsigned int copyIdx );
         void printReferencedChunksAndWDs() const;
   };
}

#endif
