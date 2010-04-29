#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"

namespace nanos {

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
      bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) ;
};

void SlicerDynamicFor::submit ( SlicedWD &work )
{
   debug0 ( "Using sliced work descriptor: Dynamic For" );

   // compute sign value
   int sign = ((SlicerDataFor *)work.getSlicerData())->getStep();
   sign = ( sign < 0 ) ? -1 : +1;
   (( SlicerDataFor *)work.getSlicerData())->setSign( sign );

   // submit wd
   Scheduler::submit ( work );
}

bool SlicerDynamicFor::dequeue ( SlicedWD *wd, WorkDescriptor **slice )
{
   int lower, upper;
   bool last = false;

   // TODO: (#107) performance evaluation on this algorithm

   // copying slicer data values
   int _lower = ((SlicerDataFor *)wd->getSlicerData())->getLower();
   int _upper = ((SlicerDataFor *)wd->getSlicerData())->getUpper();
   int _step = ((SlicerDataFor *)wd->getSlicerData())->getStep();
   int _sign = ((SlicerDataFor *)wd->getSlicerData())->getSign();
   int _chunk = ((SlicerDataFor *)wd->getSlicerData())->getChunk();

   // computing initial bounds
   lower = _lower;
   upper = _lower + ( _chunk * _step );

   // checking boundaries
   if ( ( upper * _sign ) >= ( _upper * _sign ) ) {
      upper = _upper;
      last = true;
   }

   (( SlicerDataFor *)wd->getSlicerData())->setLower( upper + _step );

   if ( last ) *slice = wd;
   else {
      *slice = NULL;
      sys.duplicateWD( slice, wd );
   }

   ((nanos_loop_info_t *)((*slice)->getData()))->lower = lower;
   ((nanos_loop_info_t *)((*slice)->getData()))->upper = upper;
   ((nanos_loop_info_t *)((*slice)->getData()))->step = _step;

   return last;
}

namespace ext {

class SlicerDynamicForPlugin : public Plugin {
   public:
      SlicerDynamicForPlugin () : Plugin("Slicer for Loops using a dynamic policy",1) {}
      ~SlicerDynamicForPlugin () {}

      virtual void config( Config& config ) {}

      void init ()
      {
         sys.registerSlicer("dynamic_for", new SlicerDynamicFor() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::SlicerDynamicForPlugin NanosXPlugin;
