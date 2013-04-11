#ifndef MEMORYOPS_DECL
#define MEMORYOPS_DECL

#include "addressspace_decl.hpp"
#include "memcachecopy_fwd.hpp"
namespace nanos {
class BaseAddressSpaceInOps {
   protected:
   typedef std::map< SeparateMemoryAddressSpace *, TransferListType > MapType;
   MapType _separateTransfers;
   std::set< DeviceOps * > _ownDeviceOps;
   std::set< DeviceOps * > _otherDeviceOps;

   public:
   BaseAddressSpaceInOps();
   ~BaseAddressSpaceInOps();

   void addOp( SeparateMemoryAddressSpace *from, global_reg_t const &reg, unsigned int version );
   bool isDataReady();

   std::set< DeviceOps * > &getOwnOps();
   std::set< DeviceOps * > &getOtherOps();

   virtual void addOpFromHost( global_reg_t const &reg, unsigned int version );
   virtual void issue( WD const &wd );

   virtual unsigned int getVersionSetVersion( global_reg_t const &reg, unsigned int newVersion );
   virtual void prepareRegion( global_reg_t const &reg, WD const &wd );
   virtual unsigned int getVersionNoLock( global_reg_t const &reg );
   virtual void setRegionVersion( global_reg_t const &reg, unsigned int version );
   //virtual unsigned int increaseVersion( global_reg_t const &reg );

   virtual void copyInputData( global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations );
   virtual void allocateOutputMemory( global_reg_t const &reg, unsigned int version );
};

typedef BaseAddressSpaceInOps HostAddressSpaceInOps;

class SeparateAddressSpaceInOps : public BaseAddressSpaceInOps {
   protected:
   SeparateMemoryAddressSpace &_destination;
   TransferListType _hostTransfers;

   public:
   SeparateAddressSpaceInOps( SeparateMemoryAddressSpace &destination );
   ~SeparateAddressSpaceInOps();

   virtual void addOpFromHost( global_reg_t const &reg, unsigned int version );
   virtual void issue( WD const &wd );

   virtual unsigned int getVersionSetVersion( global_reg_t const &reg, unsigned int newVersion );
   virtual void prepareRegion( global_reg_t const &reg, WD const &wd );
   virtual unsigned int getVersionNoLock( global_reg_t const &reg );
   virtual void setRegionVersion( global_reg_t const &reg, unsigned int version );
   //virtual unsigned int increaseVersion( global_reg_t const &reg );

   virtual void copyInputData( global_reg_t const &reg, unsigned int version, bool output, NewLocationInfoList const &locations );
   virtual void allocateOutputMemory( global_reg_t const &reg, unsigned int version );
};

class SeparateAddressSpaceOutOps {
   SeparateMemoryAddressSpace &_source;
   TransferListType _hostTransfers;

   public:
   SeparateAddressSpaceOutOps( SeparateMemoryAddressSpace &source );
   ~SeparateAddressSpaceOutOps();

   void addOp( global_reg_t const &reg, unsigned int version );
   void issue( WD const &wd, MemCacheCopy *memCacheCopies );
};

}

#endif /* MEMORYOPS_DECL */
