#ifndef MEMCACHECOPY_DECL
#define MEMCACHECOPY_DECL
#include "memoryops_decl.hpp"
namespace nanos {
   class MemCacheCopy {
      public:
         global_reg_t _reg;
         unsigned int _version;
         NewLocationInfoList _locations;
         bool                _locationDataReady;
         MemCacheCopy();
         MemCacheCopy( WD const &wd, unsigned int index );

         void getVersionInfo();
         void generateInOps( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd );
         void generateOutOps( SeparateAddressSpaceOutOps &ops, bool input, bool output );
   };
}
#endif /* MEMCACHECOPY_DECL */
