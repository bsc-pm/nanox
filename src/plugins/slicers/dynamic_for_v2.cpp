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
//   int     _current;
  
   int             _nchunks;
   int             _lower;
   int             _upper;
   int             _chunk;
   int             _step;

   Atomic<int>     _nrefs;
};

static void dynamicLoop ( void *arg )
{
   nanos_loop_info_t * nli = (nanos_loop_info_t *) arg;
   DynamicData * dsd = (DynamicData *) nli->args;

   debug0 ( "Executing dynamic loop wrapper chunks=" << dsd->_nchunks );

   #if 0
   BaseThread *mythread = myThread;
   ThreadTeam *team = mythread->getTeam();
   WorkDescriptor &work = *mythread->getCurrentWD();
   int myid = mythread->getTeamId();
   int size = team->size();

   WorkDescriptor *wd;

   int left_child = 2*myid + 1;
   int right_child = 2*myid + 2;

   if ( left_child < size ) {
     wd = NULL;
     sys.duplicateWD( &wd, &work );
     wd->tieTo((*team)[left_child]);
      (*team)[left_child].setNextWD(wd);
    // Scheduler::submit ( *wd );
   }

   if ( right_child < size ) {
     wd = NULL;
     sys.duplicateWD( &wd, &work );
     wd->tieTo((*team)[right_child]);
      (*team)[right_child].setNextWD(wd);
     //Scheduler::submit ( *wd );      
   }

   #endif
   
   int _lower = dsd->_lower;
   int _upper = dsd->_upper;
   int _chunk = dsd->_chunk;
   int _step = dsd->_step; 
   int _sign = ( _step < 0 ) ? -1 : +1;

   int mychunk = dsd->_current++;
  
   for ( ; mychunk < dsd->_nchunks; mychunk = dsd->_current++ )
   {
      nli->lower = _lower + mychunk * _chunk * _step;
      nli->upper = nli->lower + _chunk * _step - _sign;
      if ( ( nli->upper * _sign ) > ( _upper * _sign ) ) nli->upper = _upper;
      nli->step = _step;
      nli->last = mychunk == dsd->_nchunks-1;
      dsd->_realWork(arg);
   }

   if ( --dsd->_nrefs == 0 ) {
     Arena::deallocate(dsd);
   }
}

void SlicerDynamicFor::submit ( SlicedWD &work )
{
   debug0 ( "Using sliced work descriptor: Dynamic For" );

   BaseThread *mythread = myThread;

   ThreadTeam *team = mythread->getTeam();
   SlicerDataFor * sdf = (SlicerDataFor *) work.getSlicerData();
   
   int _lower = sdf->getLower();
   int _upper = sdf->getUpper();
   int _step  = sdf->getStep();
   int _chunk = std::max( 1, sdf->getChunk());

   int _niters = (((_upper - _lower) / _step ) + 1 );
   int _nchunks = _niters / _chunk;
   if ( _niters % _chunk != 0 ) _nchunks++;

   DynamicData *dsd = (DynamicData *)mythread->_arena.allocate(sizeof(DynamicData));
   dsd->_nrefs = team->size();
   dsd->_nchunks = _nchunks;
   dsd->_current = 0;

   dsd->_lower = _lower;
   dsd->_upper = _upper;
   dsd->_step = _step;
   dsd->_chunk = _chunk;

   // record original work function
   SMPDD &dd = ( SMPDD & )work.getActiveDevice();
   dsd->_realWork = dd.getWorkFct();

   nanos_loop_info_t *nli = (nanos_loop_info_t *) work.getData();
   nli->args = dsd;

   // change to our wrapper
   dd = SMPDD(dynamicLoop);

#if 1
   for ( size_t i = 1; i < team->size() ; i++ )
   {
      WorkDescriptor *wd = NULL;
      // Duplicating slice into wd
      wd = NULL;
      sys.duplicateWD( &wd, &work );

      wd->tieTo((*team)[i]);
      (*team)[i].setNextWD(wd);
//      Scheduler::submit ( *wd );
   }
#endif   
    
   work.tieTo(*mythread);
   Scheduler::switchTo ( &work );
}


class SlicerDynamicForPlugin : public Plugin {
   public:
      SlicerDynamicForPlugin () : Plugin("Slicer for Loops using a dynamic policy",1) {}
      ~SlicerDynamicForPlugin () {}

      virtual void config( Config& config ) {}

      void init ()
      {
         sys.registerSlicer("dynamic_for", NEW SlicerDynamicFor() );	
      }
};

} // namespace ext
} // namespace nanos

nanos::ext::SlicerDynamicForPlugin NanosXPlugin;
