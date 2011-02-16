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
      // Compute chunk and adjustment
      int _niters = (((_upper - _lower) / _step ) + 1 );
      int _adjust =  _niters % valid_threads;
      _chunk = _niters / valid_threads;

      // Computing upper bound
      upper = _lower + ( (_chunk-1) * _step ) + (( _adjust > 0) ? _step : 0);
      if ( ( upper * _sign ) > ( _upper * _sign ) ) upper = _upper;

      // Computing specific loop boundaries for WorkDescriptor 0
      nanos_loop_info_t *nli = ( nanos_loop_info_t * ) work.getData();
      nli->lower = _lower;
      nli->upper = upper; 
      nli->step = _step;
      nli->last = (valid_threads == 1 );

      j = first_valid_thread;
      // Creating additional WorkDescriptors: 1..N
      for ( i = 1; i < valid_threads; i++ ) {

         // Next slice lower bound
         _lower = upper + _step;

         // Finding 'j', as the next valid thread 
         while ( (j < num_threads) && (thread_map[j] != i) ) j++;

         // Debug code
         ensure ( thread_map[j] == i, "Slicer for (static) doesn't found target thread");

         // Computing upper bound
         upper = _lower + ( (_chunk-1) * _step ) + (( _adjust > i ) ? _step : 0);
         if ( ( upper * _sign ) > ( _upper * _sign ) ) upper = _upper;

         // Duplicating slice
         slice = NULL;
         sys.duplicateWD( &slice, &work );

         // Computing specific loop boundaries for current slice
         nli = ( nanos_loop_info_t * ) slice->getData();
         nli->lower = _lower;
         nli->upper = upper;
         nli->step = _step;
         nli->last = ( i == (valid_threads - 1) );

         // Submit: slice (WorkDescriptor i, running on Thread j)
         slice->tieTo( (*team)[j] );
         Scheduler::submit ( *slice );
      }

      // Submit: work (WorkDescriptor 0, running on thread 'first')
      work.tieTo( (*team)[first_valid_thread] );
      Scheduler::submit ( work );
   }
   // if chunk != 0: generate a SlicedWD for each thread (INTERLEAVED)
   else {

      // Duplicated slicer data for for each thread
      SlicerDataFor *dsdf;

      // Computing offset between threads
      int _offset = _chunk * _step;

      // if chunk == 1 then, adjust chunk and step to minimize wd's creation
      if ( _chunk == 1 ) {
         _chunk = (((_upper - _lower) / _step ) / valid_threads) +1;
         _step = _step * valid_threads;
         sdf->setChunk( _chunk );
         sdf->setStep( _step );
      }

      j = first_valid_thread;
      // Init and Submit WorkDescriptors: 1..N
      for ( i = 1; i < valid_threads; i++ ) {

         // Avoiding to create 'empty' WorkDescriptors
         if ( ((_lower + (i * _offset)) * _sign) > ( _upper * _sign ) ) break;

         // Finding 'j', as the next valid thread 
         while ( (j < num_threads) && (thread_map [j] != i) ) j++;

         // Debug code
         ensure (thread_map[j] == i, "Slicer for (interleaved) doesn't found target thread");

         // Duplicating slice into wd
         wd = NULL;
         sys.duplicateSlicedWD( &wd, &work );

         dsdf = (SlicerDataFor *) wd->getSlicerData();

         dsdf->setLower( _lower + ( i * _offset ) );
         dsdf->setUpper( _upper );
         dsdf->setStep( _step );
         dsdf->setChunk( _chunk );
         dsdf->setSign( _sign );

         // submit: wd (tied to 'j' thread)
         wd->tieTo( (*team)[j] );
         Scheduler::submit ( *wd );
      }

      // Submit: work (tied to first valid thread)
      work.tieTo( (*team)[first_valid_thread] );
      Scheduler::submit ( work );
   } // close chunk selector
}

bool SlicerStaticFor::dequeue ( SlicedWD *wd, WorkDescriptor **slice )
{
   // TODO: (#107) performance evaluation on this algorithm
   ThreadTeam *team = myThread->getTeam();
   int lower, i, upper, valid_threads = 0, num_threads = team->size();
   bool last = false;

   // copying chunk slicer data value
   SlicerDataFor *sdf = (SlicerDataFor *) wd->getSlicerData();
   int _chunk = sdf->getChunk();

   // if chunk == 0: do nothing (fields are already computed)
   if ( _chunk == 0 ) {
     *slice = wd;
     last = true;
   }
   // if chunk != 0: generate a SlicedWD for each thread (interleaved)
   else {

      /* Determine the number of valid threads */
      for ( i = 0; i < num_threads; i++) {
         if (  wd->canRunIn( *(((*team)[i]).runningOn()) ) ) valid_threads++;
      }

      // copying slicer data values
      int _lower = sdf->getLower();
      int _upper = sdf->getUpper();
      int _step = sdf->getStep();
      int _sign = sdf->getSign();

      // Computing current bounds
      lower = _lower;
      upper = _lower + ( _chunk * _step ) - _sign;

      // Checking boundaries for current chunk
      if ( ( upper * _sign ) >= ( _upper * _sign ) ) {
         upper = _upper;
         last = true;
      }

      // Computing next lower and checking boundaries for next chunk
      // avoiding to create an 'empty' WorkDescriptor
      _lower = _lower + ( _chunk * _step * valid_threads );
      if ( (_lower * _sign) > (_upper * _sign)) last = true;
      else (( SlicerDataFor *)wd->getSlicerData())->setLower( _lower );

      // Duplicate WorkDescriptor (if needed)
      if ( last ) *slice = wd;
      else {
         *slice = NULL;
         sys.duplicateWD( slice, wd );
      }

      nanos_loop_info_t *nli = ( nanos_loop_info_t * ) (*slice)->getData();
      nli->lower = lower;
      nli->upper = upper;
      nli->step = _step;

      // If it is the last iteration ( last is a field in the loop_info_t struct )
      if ( upper == _upper ) nli->last = true;
      else nli->last = false;
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
