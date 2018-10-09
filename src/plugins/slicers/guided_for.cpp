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

#include "plugin.hpp"
#include "slicer.hpp"
#include "smpdd.hpp"
#include "system.hpp"

namespace nanos {
namespace ext {

class SlicerGuidedFor: public Slicer
{
   private:
   public:
      // constructor
      SlicerGuidedFor ( ) { }

      // destructor
      ~SlicerGuidedFor ( ) { }

      // headers (implemented below)
      void submit ( WorkDescriptor & work ) ;
      bool dequeue ( WorkDescriptor *wd, WorkDescriptor **slice );
};

void SlicerGuidedFor::submit ( WorkDescriptor &work )
{
   debug0 ( "Using sliced work descriptor: Guided For" );

   nanos_loop_info_t *nli = (nanos_loop_info_t *) work.getData();

   //! Normalize Chunk size,
   nli->chunk = (1 > nli->chunk)? 1 : nli->chunk;

   //! get team size,
   ThreadTeam *team = myThread->getTeam();
   int i, num_threads = team->getFinalSize();

   //! and determine the number of valid threads
   nli->threads = 0;
   for ( i = 0; i < num_threads; i++) {
     if ( (*team)[i].runningOn()->canRun( work ) )  nli->threads++;
   }
   ensure(nli->threads > 0, "Slicer has computed an invalid number of threads");

   //! in order to submit the work.
   Scheduler::submit ( work );
}

bool SlicerGuidedFor::dequeue(nanos::WorkDescriptor* wd, nanos::WorkDescriptor** slice)
{
   bool retval = false;

   //! nli represents the chunk of iterations pending to be executed
   nanos_loop_info_t *nli = ( nanos_loop_info_t * ) wd->getData();

   int64_t _niters = ((( nli->upper - nli->lower) / nli->step ) + 1 );
   int64_t _chunk = std::max( _niters / ( 2 * nli->threads ), nli->chunk );
   int64_t _upper = nli->lower + _chunk * nli->step ;

   //! Computing empty iteration spaces to avoid infinite task generation
   bool empty = (( nli->step > 0 ) && (nli->lower > nli->upper )) ||
                (( nli->step < 0 ) && (nli->lower < nli->upper ));
   if (empty ||
         (_upper >= nli->upper && nli->step > 0) ||
         (_upper <= nli->upper && nli->step < 0)) {
      *slice = wd; retval = true;
   } else {
      WorkDescriptor *nwd = NULL;
      sys.duplicateWD( &nwd, wd );
      // Advance the lower bound of the chunk of iterations pending to be executed
      nli->lower = _upper + nli->step;

      nanos_loop_info_t *current_nli = ( nanos_loop_info_t * ) nwd->getData();
      current_nli->upper = _upper;
      sys.setupWD(*nwd, wd );

      *slice = nwd;
   }

   return retval;
}

class SlicerGuidedForPlugin : public Plugin {
   public:
      SlicerGuidedForPlugin () : Plugin("Slicer for Loops using a guided policy",1) {}
      ~SlicerGuidedForPlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerSlicer("guided_for", NEW SlicerGuidedFor() );
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN("slicer-guided_for",nanos::ext::SlicerGuidedForPlugin);
