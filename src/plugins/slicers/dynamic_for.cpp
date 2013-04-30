#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"
#include "smpdd.hpp"

namespace nanos {
namespace ext {

class SlicerDynamicFor: public Slicer
{
   private:
   public:
      // constructor
      SlicerDynamicFor ( ) { }

      // destructor
      ~SlicerDynamicFor ( ) { }

      // headers (implemented below)
      void submit ( SlicedWD & work ) ;
      bool dequeue(nanos::SlicedWD* wd, nanos::WorkDescriptor** slice);
};

void SlicerDynamicFor::submit ( SlicedWD &work )
{
   debug0 ( "Using sliced work descriptor: Dynamic For" );

   // Normalize Chunk size
   nanos_loop_info_t *nli = (nanos_loop_info_t *) work.getData();
   nli->chunk = std::max(1, nli->chunk);

   Scheduler::submit ( work );
}

bool SlicerDynamicFor::dequeue(nanos::SlicedWD* wd, nanos::WorkDescriptor** slice)
{
   bool retval = false;

   nanos_loop_info_t *nli = ( nanos_loop_info_t * ) wd->getData();

   int _upper = nli->lower + nli->chunk * nli->step ;

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

class SlicerDynamicForPlugin : public Plugin {
   public:
      SlicerDynamicForPlugin () : Plugin("Slicer for Loops using a dynamic policy",1) {}
      ~SlicerDynamicForPlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerSlicer("dynamic_for", NEW SlicerDynamicFor() );	
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN("slicer-dynamic_for",nanos::ext::SlicerDynamicForPlugin);
