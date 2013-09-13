#ifndef MEMCACHECOPY_DECL
#define MEMCACHECOPY_DECL
#include "memoryops_decl.hpp"
#include "regioncache_fwd.hpp"
namespace nanos {
   class MemCacheCopy {
         unsigned int         _version;
      public:
         global_reg_t         _reg;
         NewLocationInfoList  _locations;
         bool                 _locationDataReady;
         AllocatedChunk      *_chunk;
         MemCacheCopy();
         MemCacheCopy( WD const &wd, unsigned int index );

         void getVersionInfo();

         void generateInOps( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd );
         void generateInOps2( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd );
         void generateOutOps( SeparateAddressSpaceOutOps &ops, bool input, bool output );
         unsigned int getVersion() const;
         void setVersion( unsigned int version );
   };
}
#endif /* MEMCACHECOPY_DECL */
