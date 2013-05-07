#ifndef ADDRESSSPACE_DECL
#define ADDRESSSPACE_DECL

#include "newregiondirectory_decl.hpp"
#include "regioncache_decl.hpp"

#include "addressspace_fwd.hpp"
#include "memoryops_fwd.hpp"

namespace nanos {

typedef std::list< std::pair< global_reg_t, unsigned int > > TransferListType;

class HostAddressSpace {
   NewNewRegionDirectory _directory;

   public:
   HostAddressSpace( Device &arch );

   bool lockForTransfer( global_reg_t const &reg, unsigned int version );
   void releaseForTransfer( global_reg_t const &reg, unsigned int version );
   void doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const &wd );
   void getVersionInfo( global_reg_t const &reg, unsigned int &version, NewLocationInfoList &locations );
   void getRegionId( CopyData const &cd, global_reg_t &reg );
   void failToLock( MemSpace< SeparateAddressSpace > &from, global_reg_t const &reg, unsigned int version );

   void synchronize( bool flushData );
};


class SeparateAddressSpace {
   RegionCache  _cache;
   unsigned int _nodeNumber;
   void        *_sdata;
   
   public:
   SeparateAddressSpace( memory_space_id_t memorySpaceId, Device &arch );

   bool lockForTransfer( global_reg_t const &reg, unsigned int version );
   void releaseForTransfer( global_reg_t const &reg, unsigned int version );
   void copyOut( global_reg_t const &reg, unsigned int version );
   void doOp( MemSpace<SeparateAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const &wd );
   void doOp( MemSpace<HostAddressSpace> &from, global_reg_t const &reg, unsigned int version, WD const &wd );
   void failToLock( MemSpace< SeparateAddressSpace > &from, global_reg_t const &reg, unsigned int version );
   void failToLock( MemSpace< HostAddressSpace > &from, global_reg_t const &reg, unsigned int version );
   void copyFromHost( TransferListType list, WD const &wd );


   
   //unsigned int lockRegionAndGetCurrentVersion( global_reg_t const &reg, bool increaseVersion = false );
   void releaseRegion( global_reg_t const &reg, WD const &wd );
   uint64_t getDeviceAddress( global_reg_t const &reg, uint64_t baseAddress ) const;
   
   void prepareRegion( global_reg_t const &reg, WD const &wd );
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

};

template <class T>
class MemSpace : public T {
   public:
   MemSpace<T>( Device &d );
   MemSpace<T>( memory_space_id_t memSpaceId, Device &d );
   //virtual void copy( MemSpace from, TransferListType list ) = 0;
   void copy( MemSpace< SeparateAddressSpace > &from, TransferListType list, WD const &wd );
   void releaseRegions( MemSpace< SeparateAddressSpace > &from, TransferListType list, WD const &wd );
};

typedef MemSpace<SeparateAddressSpace> SeparateMemoryAddressSpace;
typedef MemSpace<HostAddressSpace> HostMemoryAddressSpace;

typedef unsigned int memory_space_id_t;

}

#endif /* ADDRESSSPACE_DECL */
