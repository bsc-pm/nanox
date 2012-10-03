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

   class DeviceOpsPtr;
   class DeviceOps {
      private:
         Atomic<unsigned int> _pendingDeviceOps;
         Lock _lock;
         std::set<DeviceOpsPtr *> _refs;
      public:
         std::string _desc;
         DeviceOps();
         DeviceOps( std::string &s);
         ~DeviceOps();
         void completeOp();
         void addOp();
         unsigned int getNumOps() ;
         bool allCompleted() ;
         bool addRef( DeviceOpsPtr *opsPtr, DeviceOpsPtr &p );
         void delRef( DeviceOpsPtr *opsPtr );
         void addFirstRef( DeviceOpsPtr *opsPtr );
   };
   
   class DeviceOpsPtr {
      private:
         DeviceOps *_value;
      public:
         DeviceOpsPtr() : _value( NULL ) {}
         DeviceOpsPtr( DeviceOpsPtr const &p ) {
            DeviceOps *tmpValue = p._value;
            _value = NULL;
            if ( tmpValue != NULL ) {
               if ( tmpValue->addRef( this, const_cast<DeviceOpsPtr &>( p ) ) )
                  _value = tmpValue;
            }
         }
         ~DeviceOpsPtr();
         DeviceOpsPtr & operator=( DeviceOpsPtr const &p ) {
            DeviceOps *tmpValue = p._value;
            _value = NULL;
            if ( tmpValue != NULL ) {
               if ( tmpValue->addRef( this, const_cast<DeviceOpsPtr &>( p ) ) )
                  _value = tmpValue;
            }
            return *this;
         }
         DeviceOpsPtr( DeviceOpsPtr &p ) {
            DeviceOps *tmpValue = p._value;
            _value = NULL;
            if ( tmpValue != NULL ) {
               if ( tmpValue->addRef( this, p ) )
                  _value = tmpValue;
            }
         }
         DeviceOpsPtr & operator=( DeviceOpsPtr &p ) {
            DeviceOps *tmpValue = p._value;
            _value = NULL;
            if ( tmpValue != NULL ) {
               if ( tmpValue->addRef( this, p ) )
                  _value = tmpValue;
            }
            return *this;
         }
         DeviceOps & operator*() const {
            return *_value;
         }
         DeviceOps * operator->() const {
            return _value;
         }
         void set( DeviceOps *ops ) {
            _value = ops;
            _value->addFirstRef( this );
         }
         DeviceOps *get() {
            return _value;
         }
         void clear() {
            _value = NULL;
         }
         bool isNotSet() const {
            return _value == NULL;
         }
   };

   class CachedRegionStatus {
      private:
         enum entryStatus { READY, COPYING };
         enum entryStatus _status;
         unsigned int _version;
         DeviceOpsPtr _waitObject;
      public:
         CachedRegionStatus();
         CachedRegionStatus( CachedRegionStatus const &rs );
         CachedRegionStatus &operator=( CachedRegionStatus const &rs );
         CachedRegionStatus( CachedRegionStatus &rs );
         CachedRegionStatus &operator=( CachedRegionStatus &rs );
         unsigned int getVersion();
         //unsigned int getStatus();
         void setVersion( unsigned int version );
         //void setStatus( unsigned int status );
         void setCopying( DeviceOps *ops );
         DeviceOps * getDeviceOps();
         bool isReady();
   };

   class AllocatedChunk {
      Lock _lock;
      public:
      AllocatedChunk();
      AllocatedChunk( AllocatedChunk const &chunk );
      AllocatedChunk &operator=( AllocatedChunk const &chunk );

      uint64_t address;
      RegionTree<CachedRegionStatus> _regions;
      void addReadRegion( Region reg, unsigned int version, std::set< DeviceOps * > &currentOps, std::list< Region > &notPresentRegions, DeviceOps *ops, bool alsoWriteReg );
      void addWriteRegion( Region reg, unsigned int version );
      bool isReady( Region reg );
      void lock();
      void unlock();
   };
   
   class RegionCache {
      
      MemoryMap<AllocatedChunk> _chunks;
      Lock                       _lock;
      typedef MemoryMap<AllocatedChunk>::MemChunkList ChunkList;
      typedef MemoryMap<AllocatedChunk>::ConstMemChunkList ConstChunkList;
      Device *_device;
      ProcessingElement *_pe;

      private:
         void _generateRegionOps( Region const &reg, std::map< uintptr_t, MemoryMap< uint64_t > * > &opMap );

         class Op {
            RegionCache &_parent;
            public:
            Op( RegionCache &parent ) : _parent ( parent ) { }
            RegionCache &getParent() const { return _parent; }
            virtual void doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, unsigned int wdId, WD *wd ) = 0;
            virtual void doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd ) = 0;
         };
         class CopyIn : public Op {
            public:
            CopyIn( RegionCache &parent ) : Op( parent ) {}
            void doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, unsigned int wdId, WD *wd ) ;
            void doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd ) ;
         } _copyInObj;
         class CopyOut : public Op {
            public:
            CopyOut( RegionCache &parent ) : Op( parent ) {}
            void doNoStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, DeviceOps *ops, unsigned int wdId, WD *wd ) ;
            void doStrided( int dataLocation, uint64_t devAddr, uint64_t hostAddr, std::size_t size, std::size_t count, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd ) ;
         } _copyOutObj;
         void doOp( Op *opObj, Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, unsigned int wdId, WD *wd ); 

      public:
         RegionCache() : _copyInObj( *this ), _copyOutObj( *this ) { }
         AllocatedChunk *getAddress( CopyData const &d, uint64_t &offset );
         AllocatedChunk *getAddress( uint64_t hostAddr, std::size_t len, uint64_t &offset );
         void putRegion( CopyData const &d, Region const &r );
         void syncRegion( Region const &r ) ;
         void syncRegion( std::list<Region> const &regions ) ;
         void syncRegion( std::list<Region> const &regions, DeviceOps *ops, WD *wd ) ;
         void discardRegion( CopyData const &d, Region const &r );
         void setDevice( Device *d );
         void setPE( ProcessingElement *pe );
         unsigned int getMemorySpaceId();
         /* device stubs */
         void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId, WD *wd );
         void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, DeviceOps *ops, unsigned int wdId, WD *wd );
         void _syncAndCopyIn( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId, WD *wd );
         void _copyDevToDev( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, DeviceOps *ops, unsigned int wdId, WD *wd );
         void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd );
         void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd );
         void _syncAndCopyInStrided1D( unsigned int syncFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd );
         void _copyDevToDevStrided1D( unsigned int copyFrom, uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, DeviceOps *ops, unsigned int wdId, WD *wd );
         /* *********** */
         void copyIn( Region const &hostMem, uint64_t devBaseAddr, unsigned int location, DeviceOps *ops, unsigned int wdId, WD *wd ); 
         void copyOut( Region const &hostMem, uint64_t devBaseAddr, DeviceOps *ops, unsigned int wdId, WD *wd ); 
         void lock();
         void unlock();
         bool tryLock();
         bool canCopyFrom( RegionCache const &from ) const;
         Device const *getDevice() const;
         unsigned int getNodeNumber() const;
         ProcessingElement *getPE() { return _pe; }
   };

   class CacheCopy {
      public:
      CopyData *_copy;
      AllocatedChunk *_cacheEntry;
      std::list< std::pair<Region, CachedRegionStatus const &> > _cacheDataStatus;
      Region _devRegion;
      uint64_t _devBaseAddr;
      Region _region;
      uint64_t _offset;
      unsigned int _version;
      NewRegionDirectory::LocationInfoList _locations;
      DeviceOps _operations;
      std::set< DeviceOps * > _otherPendingOps;
   };


   class CacheControler {
      // affinity private:
      public:
         unsigned int _wdId;
         WD *_wd;
         unsigned int _numCopies;
         CacheCopy *_cacheCopies;
         RegionCache *_targetCache;  
         NewRegionDirectory *_directory;
      DeviceOps _operations;
      std::set< DeviceOps * > _otherPendingOps;
      public:
         CacheControler();
         bool isCreated() const;
         void preInit( NewRegionDirectory *dir, std::size_t numCopies, CopyData *copies, unsigned int wdId, WD *wd );
         void copyDataIn( RegionCache *targetCache );
         void copyDataInNoCache();
         bool dataIsReady() const;
         uint64_t getAddress( unsigned int copyIndex ) const;
         void copyDataOut();
   };
}

#endif
