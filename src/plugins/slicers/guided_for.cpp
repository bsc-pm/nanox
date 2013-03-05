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

   // Normalize Chunk size
   nanos_loop_info_t *nli = (nanos_loop_info_t *) work.getData();
   nli->chunk = std::max(1, nli->chunk);

   // Get Team size
   ThreadTeam *team = myThread->getTeam();
   int i, num_threads = team->size();

   // Determine the number of valid threads
   nli->threads = 0;
   for ( i = 0; i < num_threads; i++) {
     if (  work.canRunIn( *((*team)[i].runningOn()) ) )  nli->threads++;
   }

   Scheduler::submit ( work );
}

bool SlicerGuidedFor::dequeue(nanos::SlicedWD* wd, nanos::WorkDescriptor** slice)
{
   bool retval = false;

   nanos_loop_info_t *nli = ( nanos_loop_info_t * ) wd->getData();

   int _niters = ((( nli->upper - nli->lower) / nli->step ) + 1 );
   int _chunk = std::max( _niters / ( 2 * nli->threads ), nli->chunk );
   int _upper = nli->lower + _chunk * nli->step ;

   if ( _upper >= nli->upper ) {
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
