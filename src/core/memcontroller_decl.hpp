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
   WD const                   &_wd;
   memory_space_id_t           _memorySpaceId;
   Lock                        _provideLock;
   std::map< NewNewRegionDirectory::RegionDirectoryKey, std::map< reg_t, unsigned int > > _providedRegions;
   BaseAddressSpaceInOps      *_inOps;
   SeparateAddressSpaceOutOps *_outOps;
   std::size_t _affinityScore;
   std::size_t _maxAffinityScore;

public:
   MemCacheCopy * _memCacheCopies;
   MemController( WD const &wd );
   bool hasVersionInfoForRegion( global_reg_t reg, unsigned int &version, NewLocationInfoList &locations );
   void getInfoFromPredecessor( MemController const &predecessorController );
   void preInit();
   void initialize( ProcessingElement &pe );
   bool allocateInputMemory();
   void copyDataIn();
   void copyDataOut();
   bool isDataReady( WD const &wd );
   uint64_t getAddress( unsigned int index ) const;
   bool canAllocateMemory( memory_space_id_t memId, bool considerInvalidations ) const;
   void setAffinityScore( std::size_t score );
   std::size_t getAffinityScore() const;
   void setMaxAffinityScore( std::size_t score );
   std::size_t getMaxAffinityScore() const;
   std::size_t getAmountOfTransferredData() const;
   std::size_t getTotalAmountOfData() const;
};

}
#endif /* MEMCONTROLLER_DECL */
