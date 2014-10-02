#ifndef MEMCACHECOPY_DECL
#define MEMCACHECOPY_DECL
#include "memoryops_decl.hpp"
#include "regioncache_decl.hpp"
#include <fstream>
namespace nanos {
   class MemCacheCopy {
         unsigned int         _version;
         unsigned int         _childrenProducedVersion;
      public:
         global_reg_t         _reg;
         NewLocationInfoList  _locations;
         bool                 _locationDataReady;
         AllocatedChunk      *_chunk;
         enum RegionCache::CachePolicy _policy;
         MemCacheCopy();
         MemCacheCopy( WD const &wd, unsigned int index );

         void getVersionInfo();

         void generateInOps( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd, unsigned int copyIdx );
         void generateOutOps( SeparateMemoryAddressSpace *from, SeparateAddressSpaceOutOps &ops, bool input, bool output, WD const &wd, unsigned int copyIdx );
         unsigned int getVersion() const;
         void setVersion( unsigned int version );
         bool isRooted( memory_space_id_t &loc ) const;
         void printLocations( std::ostream &o) const;
         unsigned int getChildrenProducedVersion() const;
         void setChildrenProducedVersion( unsigned int version );
   };
}
#endif /* MEMCACHECOPY_DECL */
