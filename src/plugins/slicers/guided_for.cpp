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
      void submit ( SlicedWD & work ) ;
      bool dequeue ( SlicedWD *wd, WorkDescriptor **slice );
};

void SlicerGuidedFor::submit ( SlicedWD &work )
{
   debug0 ( "Using sliced work descriptor: Guided For" );

   nanos_loop_info_t *nli = (nanos_loop_info_t *) work.getData();

   //! Normalize Chunk size,
   nli->chunk = std::max(1, nli->chunk);

   //! get team size,
   ThreadTeam *team = myThread->getTeam();
   int i, num_threads = team->size();

   //! and determine the number of valid threads
   nli->threads = 0;
   for ( i = 0; i < num_threads; i++) {
     if (  work.canRunIn( *((*team)[i].runningOn()) ) )  nli->threads++;
   }

   //! in order to submit the work. 
   Scheduler::submit ( work );
}

bool SlicerGuidedFor::dequeue(nanos::SlicedWD* wd, nanos::WorkDescriptor** slice)
{
   bool retval = false;

   nanos_loop_info_t *nli = ( nanos_loop_info_t * ) wd->getData();

   int _niters = ((( nli->upper - nli->lower) / nli->step ) + 1 );
   int _chunk = std::max( _niters / ( 2 * nli->threads ), nli->chunk );
   int _upper = nli->lower + _chunk * nli->step ;

   //! Computing empty iteration spaces to avoid infinite task generation
   bool empty = (( nli->step > 0 ) && (nli->lower > nli->upper )) ? true : false;
   empty = empty || (( nli->step < 0 ) && (nli->lower < nli->upper )) ? true : false;

   if ( (_upper >= nli->upper) || empty ) {
      *slice = wd; retval = true;
   } else {
      WorkDescriptor *nwd = NULL;
      sys.duplicateWD( &nwd, wd );
      nli->lower = _upper + nli->step;

      nli = ( nanos_loop_info_t * ) nwd->getData();
      nli->upper = _upper;
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
