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
      bool dequeue(nanos::SlicedWD* wd, nanos::WorkDescriptor** slice) { *slice = wd; return true; }
};

struct DynamicData {
   SMPDD::work_fct _realWork;
   Atomic<int>     _current;
  
   int             _nchunks;
   int             _lower;
   int             _upper;
   int             _chunk;
   int             _step;

   Atomic<int>     _nrefs;
};

static void dynamicLoop ( void *arg )
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t loop_lower = ID->getEventKey("loop-lower"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t loop_upper = ID->getEventKey("loop-upper"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t loop_step  = ID->getEventKey("loop-step"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t chunk_size = ID->getEventKey("chunk-size"); )
   NANOS_INSTRUMENT ( nanos_event_key_t Keys[4]; )
   NANOS_INSTRUMENT ( Keys[0] = loop_lower; )
   NANOS_INSTRUMENT ( Keys[1] = loop_upper; )
   NANOS_INSTRUMENT ( Keys[2] = loop_step; )
   NANOS_INSTRUMENT ( Keys[3] = chunk_size; )
   
   nanos_loop_info_t * nli = (nanos_loop_info_t *) arg;
   DynamicData * dsd = (DynamicData *) nli->args;

   debug0 ( "Executing dynamic loop wrapper chunks=" << dsd->_nchunks );

   int _lower = dsd->_lower;
   int _upper = dsd->_upper;
   int _chunk = dsd->_chunk;
   int _step = dsd->_step; 
   int _sign = ( _step < 0 ) ? -1 : +1;

   int mychunk = dsd->_current++;
   nli->step = _step; /* step will be constant among chunks */
   
   for ( ; mychunk < dsd->_nchunks; mychunk = dsd->_current++ )
   {
      nli->lower = _lower + mychunk * _chunk * _step;
      nli->upper = nli->lower + _chunk * _step - _sign;
      if ( ( nli->upper * _sign ) > ( _upper * _sign ) ) nli->upper = _upper;
      nli->last = mychunk == dsd->_nchunks-1;

      NANOS_INSTRUMENT ( nanos_event_value_t Values[4]; )
      NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) nli->lower; )
      NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) nli->upper; )
      NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) nli->step; )
      NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) (nli->upper - nli->lower) / nli->step; )
  
      NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents (4, Keys, Values); )

      dsd->_realWork(arg);
   }

   if ( --dsd->_nrefs == 0 ) {
     // Arena::deallocate(dsd); // TODO
     delete dsd;
   }
}

void SlicerDynamicFor::submit ( SlicedWD &work )
{
   debug0 ( "Using sliced work descriptor: Dynamic For" );

   BaseThread *mythread = myThread;

   ThreadTeam *team = mythread->getTeam();
   int i, num_threads = team->size();

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
   int valid_threads = 0, first_valid_thread = 0;
   int *thread_map = (int *) alloca ( sizeof(int) * num_threads );
   for ( i = 0; i < num_threads; i++) {
     if (  work.canRunIn( *((*team)[i].runningOn()) ) ) {
       if ( valid_threads == 0 ) first_valid_thread = i;
       thread_map[i] = valid_threads++;
     }
     else thread_map[i] = -1;
   }

   nanos_loop_info_t *nlip = ( nanos_loop_info_t * ) work.getData();
   int _lower = nlip->lower;
   int _upper = nlip->upper;
   int _step  = nlip->step;
   int _chunk = std::max(1, nlip->chunk);

   int _niters = (((_upper - _lower) / _step ) + 1 );
   int _nchunks = _niters / _chunk;
   if ( _niters % _chunk != 0 ) _nchunks++;

   // DynamicData:
   DynamicData *dsd = NEW DynamicData;

   dsd->_nrefs = valid_threads;
   dsd->_nchunks = _nchunks;
   dsd->_current = 0;

   dsd->_lower = _lower;
   dsd->_upper = _upper;
   dsd->_step = _step;
   dsd->_chunk = _chunk;

   // record original work function and changing to our wrapper
   SMPDD &dd = ( SMPDD & ) work.getActiveDevice();
   dsd->_realWork = dd.getWorkFct();
   dd = SMPDD(dynamicLoop);

   // Linking loop_info with DynamicData (through args field)
   nanos_loop_info_t *nli = (nanos_loop_info_t *) work.getData();
   nli->args = dsd;

   int j = first_valid_thread; /* initializing thread id */
   for ( i = 1; i < valid_threads; i++ )
   {
      WorkDescriptor *wd = NULL;

      // Finding 'j', as the next valid thread 
      while ( (j < num_threads) && (thread_map[j] != i) ) j++;
      // Duplicating slice into wd
      sys.duplicateWD( &wd, &work );

      sys.setupWD(*wd, work.getParent() );
      wd->tieTo((*team)[j]);
      if ( (*team)[j].setNextWD(wd) == false ) Scheduler::submit ( *wd );
   }
    
   work.tieTo(*mythread);
   if ( (*team)[first_valid_thread].setNextWD(&work) == false ) Scheduler::submit ( work );
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
