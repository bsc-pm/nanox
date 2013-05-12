#ifndef ADDRESSSPACE_DECL
#define ADDRESSSPACE_DECL

#include "newregiondirectory_decl.hpp"
#include "regioncache_decl.hpp"

#include "addressspace_fwd.hpp"
#include "memoryops_fwd.hpp"

namespace nanos {

class TransferListEntry {
   global_reg_t  _reg;
   unsigned int  _version;
   DeviceOps    *_ops;
   public:
   TransferListEntry( global_reg_t reg, unsigned int version, DeviceOps *ops );
   TransferListEntry( TransferListEntry const &t );
   TransferListEntry &operator=( TransferListEntry const &t );
   global_reg_t getRegion() const;
   unsigned int getVersion() const;
   DeviceOps *getDeviceOps() const;
};

inline TransferListEntry::TransferListEntry( global_reg_t reg, unsigned int version, DeviceOps *ops ) :
   _reg( reg ), _version( version ), _ops( ops ) {
}

inline TransferListEntry::TransferListEntry( TransferListEntry const &t ) : 
   _reg( t._reg ), _version( t._version ), _ops( t._ops ) {
}

inline TransferListEntry &TransferListEntry::operator=( TransferListEntry const &t ) {
   _reg = t._reg;
   _version = t._version;
   _ops = t._ops;
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

typedef std::list< TransferListEntry > TransferList;

class HostAddressSpace {
   NewNewRegionDirectory _directory;

   public:
   HostAddressSpace( Device &arch );

   bool lockForTransfer( global_reg_t const &reg, unsigned int version );
   void releaseForTransfer( global_reg_t const &reg, unsigned int version );
   void doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const &wd, DeviceOps *ops );
   void getVersionInfo( global_reg_t const &reg, unsigned int &version, NewLocationInfoList &locations );
   void getRegionId( CopyData const &cd, global_reg_t &reg );
   void failToLock( MemSpace< SeparateAddressSpace > &from, global_reg_t const &reg, unsigned int version );

   void synchronize( bool flushData );
   memory_space_id_t getMemorySpaceId() const;
};


class SeparateAddressSpace {
   RegionCache  _cache;
   unsigned int _nodeNumber;
   void        *_sdata;
   
   public:
   SeparateAddressSpace( memory_space_id_t memorySpaceId, Device &arch );

   bool lockForTransfer( global_reg_t const &reg, unsigned int version );
   void releaseForTransfer( global_reg_t const &reg, unsigned int version );
   void copyOut( global_reg_t const &reg, unsigned int version, DeviceOps *ops );
   void doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const &wd, DeviceOps *ops );
   void doOp( MemSpace<HostAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const &wd, DeviceOps *ops );
   void failToLock( MemSpace< SeparateAddressSpace > &from, global_reg_t const &reg, unsigned int version );
   void failToLock( MemSpace< HostAddressSpace > &from, global_reg_t const &reg, unsigned int version );
   void copyFromHost( TransferList list, WD const &wd );


   
   //unsigned int lockRegionAndGetCurrentVersion( global_reg_t const &reg, bool increaseVersion = false );
   void releaseRegion( global_reg_t const &reg, WD const &wd );
   uint64_t getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress ) const;
   
   void prepareRegions( MemCacheCopy *memCopies, unsigned int numCopies, WD const &wd );
   //void prepareRegion( global_reg_t const &reg, WD const &wd );
   void setRegionVersion( global_reg_t const &reg, unsigned int version );
   unsigned int getCurrentVersionSetVersion( global_reg_t const &reg, unsigned int version );
   unsigned int getCurrentVersion( global_reg_t const &reg );

   unsigned int getNodeNumber() const;
   void setNodeNumber( unsigned int n );
   void *getSpecificData() const;
   void setSpecificData( void *data );


   void allocateOutputMemory( global_reg_t const &reg, unsigned int version );
   void copyInputData( BaseAddressSpaceInOps &ops, global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations );

   RegionCache &getCache();
   ProcessingElement &getPE();
   memory_space_id_t getMemorySpaceId() const;

   unsigned int getInvalidationCount() const;
};

template <class T>
class MemSpace : public T {
   public:
   MemSpace<T>( Device &d );
   MemSpace<T>( memory_space_id_t memSpaceId, Device &d );
   void copy( MemSpace< SeparateAddressSpace > &from, TransferList list, WD const &wd );
   void releaseRegions( MemSpace< SeparateAddressSpace > &from, TransferList list, WD const &wd );
};

typedef MemSpace<SeparateAddressSpace> SeparateMemoryAddressSpace;
typedef MemSpace<HostAddressSpace> HostMemoryAddressSpace;

typedef unsigned int memory_space_id_t;

}

#endif /* ADDRESSSPACE_DECL */
