#ifndef INVALIDATIONCONTROLLER_DECL_H
#define INVALIDATIONCONTROLLER_DECL_H

#include "memoryops_decl.hpp"

namespace nanos {

   class InvalidationController {
      public:
         SeparateAddressSpaceOutOps   *_invalOps;
         AllocatedChunk               *_invalChunk;
         AllocatedChunk              **_invalChunkPtr;
         std::set< global_reg_t >      _regions_to_remove_access;
         std::set< AllocatedChunk * >  _chunksToFree;
         std::set< std::pair< AllocatedChunk **, AllocatedChunk * > > _chunksToInval;
         global_reg_t                  _allocatedRegion;
         unsigned int                  _softInvalidationCount;
         unsigned int                  _hardInvalidationCount;

         InvalidationController();
         ~InvalidationController();
         void abort(WD const &wd);
         bool isInvalidating() const;
         void waitOps( memory_space_id_t id, WD const &wd );
         void postCompleteActions( memory_space_id_t id, WD const &wd );
         void preIssueActions( memory_space_id_t id, WD const &wd );
         void postIssueActions( memory_space_id_t id );
   };

} // namespace nanos
#endif /* INVALIDATIONCONTROLLER_DECL_H */
