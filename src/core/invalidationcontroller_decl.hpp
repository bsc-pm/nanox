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
