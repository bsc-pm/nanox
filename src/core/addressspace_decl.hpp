/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef ADDRESSSPACE_DECL
#define ADDRESSSPACE_DECL

#include "regiondirectory_decl.hpp"
#include "regioncache_decl.hpp"

#include "addressspace_fwd.hpp"
#include "memoryops_fwd.hpp"

namespace nanos {

class TransferListEntry {
   global_reg_t    _reg;
   unsigned int    _version;
   DeviceOps      *_ops;
   AllocatedChunk *_destinationChunk;
   AllocatedChunk *_sourceChunk;
   uint64_t        _srcAddress;
   unsigned int    _copyIndex;
   public:
   TransferListEntry( global_reg_t reg, unsigned int version, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, uint64_t srcAddr, unsigned int copyIdx );
   TransferListEntry( TransferListEntry const &t );
   TransferListEntry &operator=( TransferListEntry const &t );
   global_reg_t getRegion() const;
   unsigned int getVersion() const;
   DeviceOps *getDeviceOps() const;
   AllocatedChunk *getDestinationChunk() const;
   AllocatedChunk *getSourceChunk() const;
   unsigned int getCopyIndex() const;
   uint64_t getSrcAddress() const;
};

inline TransferListEntry::TransferListEntry( global_reg_t reg, unsigned int version, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, uint64_t srcAddr, unsigned int copyIdx ) :
   _reg( reg ), _version( version ), _ops( ops ), _destinationChunk( destinationChunk ), _sourceChunk( sourceChunk ), _srcAddress( srcAddr ), _copyIndex( copyIdx ) {
}

inline TransferListEntry::TransferListEntry( TransferListEntry const &t ) : 
   _reg( t._reg ), _version( t._version ), _ops( t._ops ), _destinationChunk( t._destinationChunk ), _sourceChunk( t._sourceChunk ), _srcAddress( t._srcAddress ), _copyIndex( t._copyIndex ) {
}

inline TransferListEntry &TransferListEntry::operator=( TransferListEntry const &t ) {
   _reg = t._reg;
   _version = t._version;
   _ops = t._ops;
   _destinationChunk = t._destinationChunk;
   _sourceChunk = t._sourceChunk;
   _srcAddress = t._srcAddress;
   _copyIndex = t._copyIndex;
   return *this;
}

inline global_reg_t TransferListEntry::getRegion() const {
   return _reg;
}

inline unsigned int TransferListEntry::getVersion() const {
   return _version;
}

inline DeviceOps *TransferListEntry::getDeviceOps() const {
   return _ops;
}

inline AllocatedChunk *TransferListEntry::getDestinationChunk() const {
   return _destinationChunk;
}

inline AllocatedChunk *TransferListEntry::getSourceChunk() const {
   return _sourceChunk;
}

inline unsigned int TransferListEntry::getCopyIndex() const {
   return _copyIndex;
}

inline uint64_t TransferListEntry::getSrcAddress() const {
   return _srcAddress;
}

typedef std::list< TransferListEntry > TransferList;

class HostAddressSpace {
   RegionDirectory _directory;

   public:
   HostAddressSpace( Device &arch );

   void doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const *wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, bool inval );
   void getVersionInfo( global_reg_t const &reg, unsigned int &version, NewLocationInfoList &locations );
   void getRegionId( CopyData const &cd, global_reg_t &reg, WD const *wd, unsigned int idx );
   void failToLock( MemSpace< SeparateAddressSpace > &from, global_reg_t const &reg, unsigned int version );

   void synchronize( WD &wd );
   void synchronize( WD &wd, void *addr );
   void synchronize( WD &wd, std::size_t numDataAccesses, DataAccess *data );
   memory_space_id_t getMemorySpaceId() const;
   reg_t getLocalRegionId( void *hostObject, reg_t hostRegionId );
   RegionDirectory::RegionDirectoryKey getRegionDirectoryKey( uint64_t addr );
   void registerObject( nanos_copy_data_internal_t *obj );
   void unregisterObject( void *baseAddr );
   RegionDirectory const &getDirectory() const;
};


class SeparateAddressSpace {
   RegionCache  _cache;
   unsigned int _nodeNumber;
   unsigned int _acceleratorNumber;
   bool         _isAccelerator;
   void        *_sdata;
   
   public:
   SeparateAddressSpace( memory_space_id_t memorySpaceId, Device &arch, bool allocWide, std::size_t slabSize );

   void copyOut( global_reg_t const &reg, unsigned int version, DeviceOps *ops, WD const *wd, unsigned int copyIdx, bool inval, AllocatedChunk *origChunk );
   void doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const *wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, bool inval );
   void doOp( MemSpace<HostAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const *wd, unsigned int copyIdx, DeviceOps *ops, AllocatedChunk *destinationChunk, AllocatedChunk *sourceChunk, bool inval );
   void failToLock( MemSpace< SeparateAddressSpace > &from, global_reg_t const &reg, unsigned int version );
   void failToLock( MemSpace< HostAddressSpace > &from, global_reg_t const &reg, unsigned int version );
   void copyFromHost( TransferList &list, WD const *wd );

   void releaseRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd );
   //void releaseRegion( global_reg_t const &reg, WD const &wd, unsigned int copyIdx, enum RegionCache::CachePolicy policy );
   uint64_t getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress, AllocatedChunk *chunk ) const;
   
   bool prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd );
   void setRegionVersion( global_reg_t const &reg, AllocatedChunk *chunk, unsigned int version, WD const &wd, unsigned int copyIdx );
   unsigned int getCurrentVersion( global_reg_t const &reg, WD const &wd, unsigned int copyIdx );

   unsigned int getNodeNumber() const;
   unsigned int getAcceleratorNumber() const;
   void setNodeNumber( unsigned int n );
   void setAcceleratorNumber( unsigned int n );
   bool isAccelerator() const;
   void *getSpecificData() const;
   void setSpecificData( void *data );

   RegionCache &getCache();
   ProcessingElement &getPE();
   ProcessingElement const &getConstPE() const;
   memory_space_id_t getMemorySpaceId() const;

   void invalidate( global_reg_t const &reg );

   unsigned int getSoftInvalidationCount() const;
   unsigned int getHardInvalidationCount() const;
   bool canAllocateMemory( MemCacheCopy *memCopies, unsigned int numCopies, bool considerInvalidations, WD const &wd );
   void registerOwnedMemory(global_reg_t reg);
   Device const &getDevice() const;
   AllocatedChunk *getAndReferenceAllocatedChunk( global_reg_t reg, WD const *wd, unsigned int copyIdx );
};

template <class T>
class MemSpace : public T {
   public:
   MemSpace<T>( Device &d );
   MemSpace<T>( memory_space_id_t memSpaceId, Device &d, bool allocWide, std::size_t slabSize );
   void copy( MemSpace< SeparateAddressSpace > &from, TransferList &list, WD const *wd, bool inval = false );
};

typedef MemSpace<SeparateAddressSpace> SeparateMemoryAddressSpace;
typedef MemSpace<HostAddressSpace> HostMemoryAddressSpace;

} // namespace nanos

#endif /* ADDRESSSPACE_DECL */
