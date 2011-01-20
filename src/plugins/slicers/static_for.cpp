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

   SlicedWD *wd = NULL;
   WorkDescriptor *slice = NULL;
   ThreadTeam *team = myThread->getTeam();
   int upper, first_valid_thread = 0, i, j, valid_threads = 0, num_threads = team->size();

   /* Determine which threads are compatible with the work descriptor:
    *   - number of valid threads
    *   - first valid thread (i.e. master thread)
    *   - a map of compatible threads with a normalized id (or '-1' if not compatible):
    *     e.g. 6 threads in total with just 4 valid threads (1,2,4 & 5) and 2 non-valid
    *     threads (0 & 3)
    *
    *     - valid_threads = 4
    *     - first_valid_thread = 1
    *
    *                       0    1    2    3    4    5
    *                    +----+----+----+----+----+----+
    *     - thread_map = | -1 |  0 |  1 | -1 |  2 |  3 |
    *                    +----+----+----+----+----+----+
    */
   int *thread_map = (int *) alloca ( sizeof(int) * num_threads );
   for ( i = 0; i < num_threads; i++) {
     if (  work.canRunIn( *((*team)[i].runningOn()) ) ) {
       if ( valid_threads == 0 ) first_valid_thread = i;
       thread_map[i] = valid_threads++;
     }
     else thread_map[i] = -1;
   }

   // copying rest of slicer data values and computing sign value
   SlicerDataFor *sdf = (SlicerDataFor *) work.getSlicerData();
   int _lower = sdf->getLower();
   int _upper = sdf->getUpper();
   int _step  = sdf->getStep();
   int _chunk = sdf->getChunk();
   int _sign = ( _step < 0 ) ? -1 : +1;
   sdf->setSign( _sign );

   // if chunk == 0: generate a WD for each thread (STATIC)
   if ( _chunk == 0 ) {
      // compute chunk and adjustment
      int _niters = (((_upper - _lower) / _step ) + 1 );
      int _adjust =  _niters % valid_threads;
      _chunk = _niters / valid_threads;

      // computing upper bound
      upper = _lower + ( (_chunk-1) * _step ) + (( _adjust > 0) ? _step : 0);
      if ( ( upper * _sign ) > ( _upper * _sign ) ) upper = _upper;

      // computing specific loop boundaries for current slice
      nanos_loop_info_t *nli = ( nanos_loop_info_t * ) work.getData();
      nli->lower = _lower;
      nli->upper = upper; 
      nli->step = _step;
      nli->last = (valid_threads == 1 );

      j = first_valid_thread;
      // Init and Submit WorkDescriptors: 1..N
      for ( i = 1; i < valid_threads; i++ ) {
         // next slice lower bound
         _lower = upper + _step;

         // finding 'j', as the next valid thread 
         while ( (j < num_threads) && (thread_map[j] != i) ) j++;

         // debug code
         ensure ( thread_map[j] == i, "Slicer for (static) doesn't found target thread");

         // computing upper bound
         upper = _lower + ( (_chunk-1) * _step ) + (( _adjust > i ) ? _step : 0);
         if ( ( upper * _sign ) > ( _upper * _sign ) ) upper = _upper;

         // duplicating slice
         slice = NULL;
         sys.duplicateWD( &slice, &work );

         // computing specific loop boundaries for current slice
         nli = ( nanos_loop_info_t * ) slice->getData();
         nli->lower = _lower;
         nli->upper = upper;
         nli->step = _step;
         nli->last = ( i == (valid_threads - 1) );

         slice->tieTo( (*team)[j] );
         Scheduler::submit ( *slice );
      }

      // Submit: work
      work.tieTo( (*team)[first_valid_thread] );
      Scheduler::submit ( work );
   }
   // if chunk != 0: generate a SlicedWD for each thread (INTERLEAVED)
   else {

      BaseThread *thread = getMyThreadSafe();

      j = 0;
      // Init and Submit WorkDescriptors: 1..N
      for ( i = 1; i < valid_threads; i++ ) {
         // j is the next valid thread 
         while ( (j < num_threads) && (thread_map [j] != i) ) j++;

         ensure (thread_map[j] == i, "Slicer for (interleaved) doesn't found target thread");

         // duplicating slice into wd
         sys.duplicateSlicedWD( &wd, &work );

         (( SlicerDataFor *)wd->getSlicerData())->setLower( _lower + ( i * _chunk * _step ));
         (( SlicerDataFor *)wd->getSlicerData())->setUpper( _upper );

         // if chunk == 1 then, adjust chunk and step to minimize wd's creation
         if ( _chunk == 1 ) {
            int _chunk2 = (((_upper - _lower) / _step ) / valid_threads) +1;
            int _step2 = _step * valid_threads;
            (( SlicerDataFor *)wd->getSlicerData())->setStep( _step2 );
            (( SlicerDataFor *)wd->getSlicerData())->setChunk( _chunk2 );
         } else {
            (( SlicerDataFor *)wd->getSlicerData())->setStep( _step );
            (( SlicerDataFor *)wd->getSlicerData())->setChunk( _chunk );
        }
         (( SlicerDataFor *)wd->getSlicerData())->setSign( _sign );

         // submit: wd (tied to 'j' thread)
         wd->tieTo( (*thread->getTeam())[j] );
         Scheduler::submit ( *wd );

         /* Some schedulers change to execute submited wd. We must
          * ensure to get new myThread variable */
         thread = getMyThreadSafe();

         // next wd init
         wd = NULL;

      }

      // if chunk == 1 then, adjust chunk and step to minimize wd's creation
      if ( _chunk == 1 ) {
         _chunk = (((_upper - _lower) / _step ) / valid_threads) +1;
         _step = _step * valid_threads;
         ((SlicerDataFor *) work.getSlicerData())->setStep(_step);
         ((SlicerDataFor *) work.getSlicerData())->setChunk(_chunk);
      }

      // Submit: work (tied to first valid thread)
      work.tieTo( (*thread->getTeam())[first_valid_thread] );
      Scheduler::submit ( work );
   }
}

bool SlicerStaticFor::dequeue ( SlicedWD *wd, WorkDescriptor **slice )
{
   int lower, i, upper, valid_threads = 0, num_threads = myThread->getTeam()->size();
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

      /* Determine the number of valid threads */
      for ( i = 0; i < num_threads; i++) {
        if (  wd->canRunIn( *(((*myThread->getTeam())[i]).runningOn()) ) ) valid_threads++;
      }

      // copying slicer data values
      int _lower = ((SlicerDataFor *)wd->getSlicerData())->getLower();
      int _upper = ((SlicerDataFor *)wd->getSlicerData())->getUpper();
      int _step = ((SlicerDataFor *)wd->getSlicerData())->getStep();
      int _sign = ((SlicerDataFor *)wd->getSlicerData())->getSign();

      // computing initial bounds
      lower = _lower;
      upper = _lower + ( _chunk * _step ) - _sign;

      // computing next lower
      _lower = _lower + ( _chunk * _step * valid_threads );

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
         *slice = NULL;
         sys.duplicateWD( slice, wd );
         (( SlicerDataFor *)wd->getSlicerData())->setLower( _lower );
      }

      ((nanos_loop_info_t *)((*slice)->getData()))->lower = lower;
      ((nanos_loop_info_t *)((*slice)->getData()))->upper = upper;
      ((nanos_loop_info_t *)((*slice)->getData()))->step = _step;

      // If it is 'actually' a chunk of iterations and it is the last one...
      if ( ((_lower * _sign) < (_upper * _sign)) && ( upper == _upper ) ) {
         ((nanos_loop_info_t *)((*slice)->getData()))->last = true;
      }
      else ((nanos_loop_info_t *)((*slice)->getData()))->last = false;
   }

   return last;
}

namespace ext {

class SlicerStaticForPlugin : public Plugin {
   public:
      SlicerStaticForPlugin () : Plugin("Slicer for Loops using a static policy",1) {}
      ~SlicerStaticForPlugin () {}

      virtual void config( Config& config ) {}

      void init ()
      {
         sys.registerSlicer("static_for", NEW SlicerStaticFor() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::SlicerStaticForPlugin NanosXPlugin;
