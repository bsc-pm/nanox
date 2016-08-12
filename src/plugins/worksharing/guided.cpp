/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "nanos-int.h"
#include "atomic.hpp"
#include "loop.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "worksharing_decl.hpp"

namespace nanos {
namespace ext {

class WorkSharingGuidedFor : public WorkSharing {

      //! \brief create a loop descriptor
      //! \return only one thread per loop will get 'true' (single like behaviour)
      bool create ( nanos_ws_desc_t **wsd, nanos_ws_info_t *info )
      {
         nanos_ws_info_loop_t *loop_info = (nanos_ws_info_loop_t *) info;
         bool single = false;

         *wsd = myThread->getTeamWorkSharingDescriptor( &single );
         if ( single ) {

            (*wsd)->data = NEW WorkSharingLoopInfo();

            int64_t num_threads = myThread->getTeam()->getFinalSize();

            ((WorkSharingLoopInfo *)(*wsd)->data)->lowerBound = loop_info->lower_bound;
            ((WorkSharingLoopInfo *)(*wsd)->data)->upperBound = loop_info->upper_bound;
            ((WorkSharingLoopInfo *)(*wsd)->data)->loopStep   = loop_info->loop_step;

            int64_t chunk_size = (1 < loop_info->chunk_size) ? loop_info->chunk_size : 1;
            ((WorkSharingLoopInfo *)(*wsd)->data)->chunkSize  = chunk_size;

            int64_t niters = (((loop_info->upper_bound - loop_info->lower_bound) / loop_info->loop_step ) + 1 );
            int64_t chunks = niters / chunk_size;
            if ( niters % chunk_size != 0 ) chunks++;
            ((WorkSharingLoopInfo *)(*wsd)->data)->numOfChunks = 0;

            while ( niters > 0 ) {
               niters = niters - ((niters/(2*num_threads) < chunk_size) ? chunk_size : niters/(2*num_threads));
               ((WorkSharingLoopInfo *)(*wsd)->data)->numOfChunks++;
            }

            ((WorkSharingLoopInfo *)(*wsd)->data)->currentChunk  = 0;

            memoryFence();     // Split initialization phase (before) from make it visible (after)

            (*wsd)->ws = this; // Once 'ws' field has a value, any other thread can use the structure

         }

         // wait until worksharing descriptor is initialized
         while ( (*wsd)->ws == NULL ) {;}

         return single;
      }

      //! \brief Get next chunk of iterations
      void nextItem( nanos_ws_desc_t *wsd, nanos_ws_item_t *item )
      {
         nanos_ws_item_loop_t *loop_item = ( nanos_ws_item_loop_t *) item;
         WorkSharingLoopInfo  *loop_data = ( WorkSharingLoopInfo  *) wsd->data;

         int64_t mychunk = loop_data->currentChunk++;
         if ( mychunk > loop_data->numOfChunks)
         {
            loop_item->execute = false;
            return;
         }

         int64_t num_threads = myThread->getTeam()->getFinalSize();
         int sign = (( loop_data->loopStep < 0 ) ? -1 : +1);

         loop_item->lower = loop_data->lowerBound;
         int64_t i = 0;
         int64_t niters = (((loop_data->upperBound - loop_data->lowerBound) / loop_data->loopStep ) + 1 );

         int64_t current;
         while ( i < mychunk ) {
            current = (niters/(2*num_threads) < loop_data->chunkSize) ? loop_data->chunkSize : niters/(2*num_threads);
            niters -= current;
            loop_item->lower += (current * loop_data->loopStep);
            i++;
         }
         loop_item->upper = loop_item->lower 
                          + std::max( niters/(2*num_threads), loop_data->chunkSize) * loop_data->loopStep 
                          - loop_data->loopStep;

         if ( ( loop_data->upperBound * sign ) < ( loop_item->upper * sign ) ) loop_item->upper = loop_data->upperBound;
         loop_item->last = mychunk == (loop_data->numOfChunks - 1);
         loop_item->execute = (loop_item->lower * sign) <= (loop_item->upper * sign);
      }

      void duplicateWS ( nanos_ws_desc_t *orig, nanos_ws_desc_t **copy) {}

   
};

class WorkSharingGuidedForPlugin : public Plugin {
   public:
      WorkSharingGuidedForPlugin () : Plugin("Worksharing plugin for loops using a guided policy",1) {}
     ~WorkSharingGuidedForPlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerWorkSharing("guided_for", NEW WorkSharingGuidedFor() );	
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN( "placeholder-name", nanos::ext::WorkSharingGuidedForPlugin );
