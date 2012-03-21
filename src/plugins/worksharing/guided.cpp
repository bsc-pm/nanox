#include "nanos-int.h"
#include "loop.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "worksharing_decl.hpp"

namespace nanos {
namespace ext {

class WorkSharingGuidedFor : public WorkSharing {

     /*! \brief create a loop descriptor
      *  
      *  \return only one thread per loop will get 'true' (single like behaviour)
      */
      bool create ( nanos_ws_desc_t **wsd, nanos_ws_info_t *info )
{
         nanos_ws_info_loop_t *loop_info = (nanos_ws_info_loop_t *) info;
         bool single = false;

         *wsd = myThread->getTeamWorkSharingDescriptor( &single );
         if ( single ) {
            (*wsd)->data = NEW WorkSharingLoopInfo();
             int num_threads = myThread->getTeam()->size();
            ((WorkSharingLoopInfo *)(*wsd)->data)->lowerBound = loop_info->lower_bound;
            ((WorkSharingLoopInfo *)(*wsd)->data)->upperBound = loop_info->upper_bound;
            ((WorkSharingLoopInfo *)(*wsd)->data)->loopStep   = loop_info->loop_step;
            int chunk_size = std::max(1,loop_info->chunk_size);
            ((WorkSharingLoopInfo *)(*wsd)->data)->chunkSize  = chunk_size;
            int niters = (((loop_info->upper_bound - loop_info->lower_bound) / loop_info->loop_step ) + 1 );
            int chunks = niters / chunk_size;
            if ( niters % chunk_size != 0 ) chunks++;
            ((WorkSharingLoopInfo *)(*wsd)->data)->numOfChunks = 0;
            while ( niters > 0 ) {
               niters = niters - std::max( niters/(2*num_threads), chunk_size);
               ((WorkSharingLoopInfo *)(*wsd)->data)->numOfChunks++;
            }
            ((WorkSharingLoopInfo *)(*wsd)->data)->currentChunk  = 0;

            memoryFence();

            (*wsd)->ws = this; // Once 'ws' field has a value, any other thread can use the structure
         }

         // wait until worksharing descriptor is initialized
         while ( (*wsd)->ws == NULL ) {;}

         return single;
      }

     /*! \brief Get next chunk of iterations
      *
      */
      void nextItem( nanos_ws_desc_t *wsd, nanos_ws_item_t *item )
      {
         nanos_ws_item_loop_t *loop_item = ( nanos_ws_item_loop_t *) item;
         WorkSharingLoopInfo  *loop_data = ( WorkSharingLoopInfo  *) wsd->data;

         int mychunk = loop_data->currentChunk++;
         if ( mychunk > loop_data->numOfChunks)
         {
            loop_item->execute = false;
            return;
         }

         int num_threads = myThread->getTeam()->size();
         int sign = (( loop_data->loopStep < 0 ) ? -1 : +1);

         loop_item->lower = loop_data->lowerBound;
         int i = 0, niters = (((loop_data->upperBound - loop_data->lowerBound) / loop_data->loopStep ) + 1 );

         int current;
         while ( i < mychunk ) {
            current = std::max( niters/(2*num_threads), loop_data->chunkSize);
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

nanos::ext::WorkSharingGuidedForPlugin NanosXPlugin;
