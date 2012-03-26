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
#include "regiondirectory_decl.hpp"

namespace nanos {

   class DeviceOps {
      private:
         Atomic<unsigned int> _pendingDeviceOps;
      public:
         DeviceOps();
         void completeOp();
         void addOp();
         bool allCompleted();
   };

   class CachedRegionStatus {
      private:
         enum entryStatus { READY, COPYING };
         enum entryStatus _status;
         unsigned int _version;
      public:
         CachedRegionStatus();
         CachedRegionStatus( CachedRegionStatus const &rs );
         CachedRegionStatus &operator=( CachedRegionStatus const &rs );
         unsigned int getVersion();
         //unsigned int getStatus();
         void setVersion( unsigned int version );
         //void setStatus( unsigned int status );
         void setCopying();
         bool isReady();
   };

   class AllocatedChunk {
      public:
      uint64_t address;
      RegionTree<CachedRegionStatus> _regions;
      void addRegion( Region reg, std::list< std::pair<Region, CachedRegionStatus const &> > &outs );
      bool isReady( Region reg );
      void setCopying( Region reg );
   };
   
   class RegionCache {
      
      //CacheAllocator             _allocator;
      MemoryMap<AllocatedChunk> _chunks;
      Lock                       _lock;
      typedef MemoryMap<AllocatedChunk>::MemChunkList ChunkList;
      typedef MemoryMap<AllocatedChunk>::ConstMemChunkList ConstChunkList;
      Device *_device;
      ProcessingElement *_pe;

      public:
         AllocatedChunk *getAddress( CopyData const &d, uint64_t &offset );
         AllocatedChunk *getAddress( uint64_t hostAddr, std::size_t len, uint64_t &offset );
         void putRegion( CopyData const &d, Region const &r );
         void syncRegion( Region const &r, uint64_t devAddr );
         void discardRegion( CopyData const &d, Region const &r );
         void setDevice( Device *d );
         void setPE( ProcessingElement *pe );
         unsigned int getMemorySpaceId();
         void copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId );
         void copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops );
         void syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId );
         void copyDevToDev( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId );
         void lock();
         void unlock();
         bool tryLock();
         bool canCopyFrom( RegionCache const &from ) const;
   };

   class CacheCopy {
      public:
      CopyData *_copy;
      AllocatedChunk *_cacheEntry;
      std::list< std::pair<Region, CachedRegionStatus const &> > _cacheDataStatus;
      Region _devRegion;
      uint64_t _offset;
      unsigned int _version;
      NewRegionDirectory::LocationInfoList _locations;
      DeviceOps _operations;
   };


   class CacheControler {
      private:
         unsigned int _wdId;
         unsigned int _numCopies;
         CacheCopy *_cacheCopies;
         RegionCache *_targetCache;  
         NewRegionDirectory *_directory;
      public:
         CacheControler();
         bool isCreated() const;
         void create( RegionCache *targetCache, NewRegionDirectory *dir, std::size_t numCopies, CopyData *copies, unsigned int wdId );
         bool dataIsReady() const;
         uint64_t getAddress( unsigned int copyIndex ) const;
         void copyDataOut();
   };
}

#endif
