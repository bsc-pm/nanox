#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"

namespace nanos {

class SlicerStaticFor: public Slicer
{
   private:
   public:
      // constructor
      SlicerStaticFor ( ) { }

      // destructor
      ~SlicerStaticFor ( ) { }

      // headers (implemented below)
      void submit ( SlicedWD & work ) ;
      bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) ;
};

void SlicerStaticFor::submit ( SlicedWD &work )
{
   debug ( "Using sliced work descriptor: Static For" );

   int lower, upper, i, num_threads = myThread->getTeam()->size();
   WorkDescriptor **slice = NULL;

   // compute sign value
   int _sign = ((SlicerDataFor *)work.getSlicerData())->getStep();
   _sign = ( _sign < 0 ) ? -1 : +1;
   (( SlicerDataFor *)work.getSlicerData())->setSign( _sign );

   // copying rest of slicer data values
   int _lower = ((SlicerDataFor *) work.getSlicerData())->getLower();
   int _upper = ((SlicerDataFor *) work.getSlicerData())->getUpper();
   int _step  = ((SlicerDataFor *) work.getSlicerData())->getStep();
   int _chunk = ((SlicerDataFor *) work.getSlicerData())->getChunk();

   // if chunk == 0: generate a WD for each thread
   if ( _chunk == 0 ) {
      // compute chunk and adjustment
      _chunk = (_upper - _lower) / num_threads * _step;
      int _adjust = (_upper - _lower) % num_threads * _step;

      // Init WorkDescriptor 'work'
      // computing initial bounds
      lower = _lower;
      upper = _lower + ( _chunk * _step ) + ( ((0 < _adjust) ? 1 : 0) * _step );

      // checking boundaries
      if ( ( upper * _sign ) >= ( _upper * _sign ) ) upper = _upper;

      // computing specific loop boundaries for current slice
      ((nanos_loop_info_t *)(work.getData()))->lower = lower;
      ((nanos_loop_info_t *)(work.getData()))->upper = upper; 
      ((nanos_loop_info_t *)(work.getData()))->step = _step;

      // next slice init
      _lower = upper;

      // Init and Submit WorkDescriptors: 1..N
      for ( i = 1; i < num_threads; i++ ) {
         // computing initial bounds
         lower = _lower;
         upper = _lower + ( _chunk * _step ) + ( ((i < _adjust) ? 1 : 0) * _step );

         // checking boundaries
         if ( ( upper * _sign ) >= ( _upper * _sign ) ) upper = _upper;

         // duplicating slice
         sys.duplicateWD( slice, &work );

         // computing specific loop boundaries for current slice
         ((nanos_loop_info_t *)((*slice)->getData()))->lower = lower;
         ((nanos_loop_info_t *)((*slice)->getData()))->upper = upper;
         ((nanos_loop_info_t *)((*slice)->getData()))->step = _step;

         // xteruel: FIXME: slice has to run in a specific thread, (threadId == i)
         Scheduler::submit ( **slice );

         // next slice init
         _lower = upper;
         slice = NULL;
      }

      // Submit: work
      Scheduler::submit ( work );

   }
   // if chunk != 0: generate a SlicedWD for each thread (interleaved)
   else {

   // xteruel: FIXME: interleaved case ???

   }
}

bool SlicerStaticFor::dequeue ( SlicedWD *wd, WorkDescriptor **slice )
{
   // int lower, upper;
   bool last = false;

   // TODO: (#107) performance evaluation on this algorithm

   // copying slicer data values
   int _chunk = ((SlicerDataFor *)wd->getSlicerData())->getChunk();

   // if chunk == 0: do nothing (fields are already computed)
   if ( _chunk == 0 ) {
      *slice = wd;
      last = true;
   }
   // if chunk != 0: generate a SlicedWD for each thread (interleaved)
   else {
#if 0
      // copying slicer data values
      int _lower = ((SlicerDataFor *)wd->getSlicerData())->getLower();
      int _upper = ((SlicerDataFor *)wd->getSlicerData())->getUpper();
      int _step = ((SlicerDataFor *)wd->getSlicerData())->getStep();
      int _sign = ((SlicerDataFor *)wd->getSlicerData())->getSign();
#endif
   // xteruel: FIXME: interleaved case ???
   }

   return last;
}

namespace ext {

class SlicerStaticForPlugin : public Plugin {
   public:
      SlicerStaticForPlugin () : Plugin("Slicer for Loops using a static policy",1) {}
      ~SlicerStaticForPlugin () {}

      void init ()
      {
         sys.registerSlicer("static_for", new SlicerStaticFor() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::SlicerStaticForPlugin NanosXPlugin;
