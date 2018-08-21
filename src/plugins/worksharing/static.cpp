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

#include "nanos-int.h"
#include "loop.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "worksharing_decl.hpp"

namespace nanos {
namespace ext {

class WorkSharingStaticFor : public WorkSharing {

      //! \brief create a loop descriptor
      //! \return only one thread per loop will get 'true' (single like behaviour)
      bool create( nanos_ws_desc_t **wsd, nanos_ws_info_t *info )
      {
         nanos_ws_info_loop_t *loop_info = (nanos_ws_info_loop_t *) info;

         *wsd = myThread->getLocalWorkSharingDescriptor();

         (*wsd)->data = NEW WorkSharingLoopInfo();

         ((WorkSharingLoopInfo *)(*wsd)->data)->lowerBound = loop_info->lower_bound;
         ((WorkSharingLoopInfo *)(*wsd)->data)->upperBound = loop_info->upper_bound;
         ((WorkSharingLoopInfo *)(*wsd)->data)->loopStep   = loop_info->loop_step;
         ((WorkSharingLoopInfo *)(*wsd)->data)->chunkSize  = loop_info->chunk_size;

         (*wsd)->ws = this;

         debug("Loop create -> lower: " << loop_info->lower_bound << " upper: " << loop_info->upper_bound << " step: " << loop_info->loop_step << " chunk size: " << loop_info->chunk_size );

         return myThread->singleGuard();
      }

      //! \brief Duplicate related data
      void duplicateWS ( nanos_ws_desc_t *orig, nanos_ws_desc_t **copy)
      {
         *copy = NEW nanos_ws_desc_t;

         (*copy)->data = NEW WorkSharingLoopInfo();

         ((WorkSharingLoopInfo *)(*copy)->data)->lowerBound = ((WorkSharingLoopInfo *)orig->data)->lowerBound;
         ((WorkSharingLoopInfo *)(*copy)->data)->upperBound = ((WorkSharingLoopInfo *)orig->data)->upperBound;
         ((WorkSharingLoopInfo *)(*copy)->data)->loopStep   = ((WorkSharingLoopInfo *)orig->data)->loopStep;
         ((WorkSharingLoopInfo *)(*copy)->data)->chunkSize  = ((WorkSharingLoopInfo *)orig->data)->chunkSize;

         (*copy)->ws = orig->ws;
      }

      //! \brief Get next chunk of iterations
      void nextItem( nanos_ws_desc_t *wsd, nanos_ws_item_t *item )
      {
         nanos_ws_item_loop_t *loop_item = ( nanos_ws_item_loop_t *) item;
         WorkSharingLoopInfo  *loop_data = ( WorkSharingLoopInfo  *) wsd->data;
         loop_item->last = false;

         int sign = (( loop_data->loopStep < 0 ) ? -1 : +1);
         if ( (loop_data->lowerBound * sign) > (loop_data->upperBound * sign) ) {
            loop_item->execute = false;
            return;
         }

         ThreadTeam *team = myThread->getTeam();

         int num_threads = team->getFinalSize();
         int thid = myThread->getTeamId();

         int64_t niters = (((loop_data->upperBound - loop_data->lowerBound) / loop_data->loopStep ) + 1 );
         int64_t adjust = niters % num_threads;
         int64_t schunk  = ((niters / num_threads) -1 ) * loop_data->loopStep;

         if ( loop_data->chunkSize == 0){
            // static distribution
            loop_item->lower = loop_data->lowerBound
                             + (schunk + loop_data->loopStep) * thid
                             + ( (adjust > thid) ? thid * loop_data->loopStep : adjust * loop_data->loopStep );
            loop_item->upper = loop_item->lower + schunk + ((adjust > thid ) ? loop_data->loopStep : 0);
            loop_data->lowerBound = loop_data->upperBound + loop_data->loopStep;
            if ( loop_item->upper == loop_data->upperBound ) loop_item->last = true;
         } else {
            // interleaved distribution
            loop_item->lower = loop_data->lowerBound + (loop_data->chunkSize * loop_data->loopStep ) * thid;
            loop_item->upper = loop_item->lower + loop_data->chunkSize * loop_data->loopStep - loop_data->loopStep;
            if ( (loop_item->upper * sign) >= (loop_data->upperBound * sign) ) {
               loop_item->upper = loop_data->upperBound;
               loop_item->last = true;
            }
            loop_data->lowerBound = loop_data->lowerBound + (loop_data->chunkSize * loop_data->loopStep) * num_threads;
         }

         loop_item->execute = (loop_item->lower * sign) <= (loop_item->upper * sign);
         debug("Loop next item -> lower: " << loop_item->lower << " upper: " << loop_item->upper );
      }

      int64_t getItemsLeft( nanos_ws_desc_t *wsd )
      {
         return 0;
      }

      bool instanceOnCreation()
      {
         return true;
      }
};

class WorkSharingStaticForPlugin : public Plugin {
   public:
      WorkSharingStaticForPlugin () : Plugin("Worksharing plugin for loops using a static policy",1) {}
     ~WorkSharingStaticForPlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerWorkSharing("static_for", NEW WorkSharingStaticFor() );
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN( "worksharing-static", nanos::ext::WorkSharingStaticForPlugin );
