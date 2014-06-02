#ifndef MEMCONTROLLER_DECL
#define MEMCONTROLLER_DECL
#include <map>
#include "workdescriptor_fwd.hpp"
#include "atomic_decl.hpp"
#include "newregiondirectory_decl.hpp"
#include "addressspace_decl.hpp"
#include "memoryops_decl.hpp"
#include "memcachecopy_decl.hpp"

namespace nanos {

class MemController {
   bool                        _initialized;
   bool                        _preinitialized;
   bool                        _inputDataReady;
   bool                        _outputDataReady;
   bool                        _memoryAllocated;
   bool                        _mainWd;
   WD const                   &_wd;
   memory_space_id_t           _memorySpaceId;
   Lock                        _provideLock;
   std::map< NewNewRegionDirectory::RegionDirectoryKey, std::map< reg_t, unsigned int > > _providedRegions;
   BaseAddressSpaceInOps      *_inOps;
   SeparateAddressSpaceOutOps *_outOps;
   std::size_t                 _affinityScore;
   std::size_t                 _maxAffinityScore;

public:
   enum MemControllerPolicy {
      WRITE_BACK,
      WRITE_THROUGH,
      NO_CACHE
   };
   MemCacheCopy *_memCacheCopies;
   MemController( WD const &wd );
   bool hasVersionInfoForRegion( global_reg_t reg, unsigned int &version, NewLocationInfoList &locations );
   void getInfoFromPredecessor( MemController const &predecessorController );
   void preInit();
   void initialize( ProcessingElement &pe );
   bool allocateInputMemory();
   void copyDataIn();
   void copyDataOut( MemControllerPolicy policy );
   bool isDataReady( WD const &wd );
   bool isOutputDataReady( WD const &wd );
   uint64_t getAddress( unsigned int index ) const;
   bool canAllocateMemory( memory_space_id_t memId, bool considerInvalidations ) const;
   void setAffinityScore( std::size_t score );
   std::size_t getAffinityScore() const;
   void setMaxAffinityScore( std::size_t score );
   std::size_t getMaxAffinityScore() const;
   std::size_t getAmountOfTransferredData() const;
   std::size_t getTotalAmountOfData() const;
   bool isRooted( memory_space_id_t &loc ) const ;
   void setMainWD();
   void synchronize();
   bool isMemoryAllocated() const;
};

}
#endif /* MEMCONTROLLER_DECL */