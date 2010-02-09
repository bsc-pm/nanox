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
   WorkDescriptor *slice = NULL;
   SlicedWD *wd = NULL;

   // copying rest of slicer data values
   int _lower = ((SlicerDataFor *) work.getSlicerData())->getLower();
   int _upper = ((SlicerDataFor *) work.getSlicerData())->getUpper();
   int _step  = ((SlicerDataFor *) work.getSlicerData())->getStep();
   int _chunk = ((SlicerDataFor *) work.getSlicerData())->getChunk();

   // compute sign value
   int _sign = ( _step < 0 ) ? -1 : +1;
   (( SlicerDataFor *)work.getSlicerData())->setSign( _sign );

   // if chunk == 0: generate a WD for each thread
   if ( _chunk == 0 ) {
      // compute chunk and adjustment
      _chunk = (((_upper - _lower) / _step ) + 1 ) / num_threads;
      int _adjust = (((_upper - _lower)/_step) + 1 ) % num_threads;

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
         sys.duplicateWD( &slice, &work );

         // computing specific loop boundaries for current slice
         ((nanos_loop_info_t *)(slice->getData()))->lower = lower;
         ((nanos_loop_info_t *)(slice->getData()))->upper = upper;
         ((nanos_loop_info_t *)(slice->getData()))->step = _step;

         slice->tieTo( (*myThread->getTeam())[i] );
         Scheduler::submit ( *slice );

         // next slice init
         _lower = upper;
         slice = NULL;
      }

      // Submit: work
      work.tieTo( (*myThread->getTeam())[0] );
      Scheduler::submit ( work );
   }
   // if chunk != 0: generate a SlicedWD for each thread (interleaved)
   else {

      // Init and Submit WorkDescriptors: 1..N
      for ( i = 1; i < num_threads; i++ ) {
         // duplicating slice
         sys.duplicateSlicedWD( &wd, &work );

         (( SlicerDataFor *)wd->getSlicerData())->setLower( _lower + ( i * _chunk * _step ));
         (( SlicerDataFor *)wd->getSlicerData())->setUpper( _upper );
         (( SlicerDataFor *)wd->getSlicerData())->setStep( _step );
         (( SlicerDataFor *)wd->getSlicerData())->setChunk( _chunk );
         (( SlicerDataFor *)wd->getSlicerData())->setSign( _sign );

         wd->tieTo( (*myThread->getTeam())[i] );
         Scheduler::submit ( *wd );

         // next wd init
         wd = NULL;

      }
      // (( SlicerDataFor *)work.getSlicerData())->setLower( upper );

      // Submit: work
      work.tieTo( (*myThread->getTeam())[0] );
      Scheduler::submit ( work );
   }
}

bool SlicerStaticFor::dequeue ( SlicedWD *wd, WorkDescriptor **slice )
{
   int lower, upper, num_threads = myThread->getTeam()->size();
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
      // copying slicer data values
      int _lower = ((SlicerDataFor *)wd->getSlicerData())->getLower();
      int _upper = ((SlicerDataFor *)wd->getSlicerData())->getUpper();
      int _step = ((SlicerDataFor *)wd->getSlicerData())->getStep();
      int _sign = ((SlicerDataFor *)wd->getSlicerData())->getSign();

      // computing initial bounds
      lower = _lower;
      upper = _lower + ( _chunk * _step ) - _sign;

      // computing next lower
      _lower = _lower + ( _chunk * _step * num_threads );

      // checking boundaries
      if ( ( upper * _sign ) >= ( _upper * _sign ) ) {
         upper = _upper;
         last = true;
      }
      if ( (_lower * _sign) > (_upper * _sign)) {
         last = true;
      }

      if ( last ) *slice = wd;
      else {
         sys.duplicateWD( slice, wd );
         (( SlicerDataFor *)wd->getSlicerData())->setLower( _lower );
      }

      ((nanos_loop_info_t *)((*slice)->getData()))->lower = lower;
      ((nanos_loop_info_t *)((*slice)->getData()))->upper = upper;
      ((nanos_loop_info_t *)((*slice)->getData()))->step = _step;
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
