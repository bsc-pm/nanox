#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"

namespace nanos {

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
      bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) ;
};

void SlicerGuidedFor::submit ( SlicedWD &work )
{
   debug0 ( "Using sliced work descriptor: Guided For" );

   // compute sign value
   int sign = ((SlicerDataFor *)work.getSlicerData())->getStep();
   sign = ( sign < 0 ) ? -1 : +1;
   (( SlicerDataFor *)work.getSlicerData())->setSign( sign );

   // submit wd
   Scheduler::submit ( work );
}

bool SlicerGuidedFor::dequeue ( SlicedWD *wd, WorkDescriptor **slice )
{
   // TODO: (#107) performance evaluation on this algorithm
   int lower, upper, current_chunk, num_threads = myThread->getTeam()->size();
   bool last = false;

   // copying slicer data values
   int _lower = ((SlicerDataFor *)wd->getSlicerData())->getLower();
   int _upper = ((SlicerDataFor *)wd->getSlicerData())->getUpper();
   int _step = ((SlicerDataFor *)wd->getSlicerData())->getStep();
   int _sign = ((SlicerDataFor *)wd->getSlicerData())->getSign();
   int _chunk = ((SlicerDataFor *)wd->getSlicerData())->getChunk();

   // computing current chunk
   if ( _sign == 1 ) current_chunk = ((_upper-_lower)/(_step*_sign)) / ( 2*num_threads);
   else current_chunk = ((_lower-_upper)/(_step*_sign)) / ( 2*num_threads);
   if ( current_chunk < _chunk ) current_chunk = _chunk;

   // computing initial bounds
   lower = _lower;
   upper = _lower + ( current_chunk * _step );

   // checking boundaries
   if ( ( upper * _sign ) >= ( _upper * _sign ) ) {
      upper = _upper;
      last = true;
   }

   (( SlicerDataFor *)wd->getSlicerData())->setLower( upper + _step);

   if ( last ) *slice = wd;
   else {
      *slice = NULL;
      sys.duplicateWD( slice, wd );
   }

   ((nanos_loop_info_t *)((*slice)->getData()))->lower = lower;
   ((nanos_loop_info_t *)((*slice)->getData()))->upper = upper;
   ((nanos_loop_info_t *)((*slice)->getData()))->step = _step;
   ((nanos_loop_info_t *)((*slice)->getData()))->last = last;

   /* If not last, scheduler will enqueue this workdescriptor */
   if (!last) sys.getSchedulerStats()._readyTasks++;

   return last;
}

namespace ext {

class SlicerGuidedForPlugin : public Plugin {
   public:
      SlicerGuidedForPlugin () : Plugin("Slicer for Loops using a guided policy",1) {}
      ~SlicerGuidedForPlugin () {}

      virtual void config( Config& config ) {}

      void init ()
      {
         sys.registerSlicer("guided_for", new SlicerGuidedFor() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::SlicerGuidedForPlugin NanosXPlugin;
