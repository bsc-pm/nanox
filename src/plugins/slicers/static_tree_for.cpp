/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"
#include "smpdd.hpp"

namespace nanos {
namespace ext {

class SlicerStaticFor: public Slicer
{
   private:
   public:
      // constructor
      SlicerStaticFor ( ) { }

      // destructor
      ~SlicerStaticFor ( ) { }

      // headers (implemented below)
      void submit ( WorkDescriptor & work ) ;
      bool dequeue ( WorkDescriptor *wd, WorkDescriptor **slice ) { *slice = wd; return true; }
};

// FIXME: Temporary defined to enable/disable hierarchical slicer creation
#define NANOS_TREE_CREATION
#ifdef NANOS_TREE_CREATION
static void staticLoop ( void *arg )
{
   debug ( "Executing static loop wrapper");

   WorkDescriptor *slice = NULL;
   BaseThread *mythread = myThread;
   ThreadTeam *team = mythread->getTeam();
   int num_threads = team->getFinalSize();
   WorkDescriptor *work = mythread->getCurrentWD();

   nanos_loop_info_t * nli = (nanos_loop_info_t *) arg;
   nanos_loop_info_t * nli_1, *nli_2;

   if ( (nli->thid * 2 + 1) < num_threads ) {
      // Duplicating slice into wd
      slice = NULL;
      sys.duplicateWD( &slice, work );

      debug ( "Creating task " << slice << ":" << slice->getId() << " from sliced one " << work << ":" << work->getId() );

      // Computing specific loop boundaries for 1st child slice
      nli_1 = ( nanos_loop_info_t * ) slice->getData();
      nli_1->thid  = nli->thid * 2 + 1;
      nli_1->lower = nli->lower;
      nli_1->upper = nli->upper;
      nli_1->step  = nli->step;
      nli_1->args  = nli->args;

      sys.setupWD(*slice, work->getParent());
      // Submit: slice (WorkDescriptor i, running on Thread j)
      slice->tieTo( (*team)[nli_1->thid] );
      (*team)[nli_1->thid].addNextWD(slice);
   }

   if ( (nli->thid * 2 + 2) < num_threads ) {
      // Duplicating slice into wd
      slice = NULL;
      sys.duplicateWD( &slice, work );

      debug ( "Creating task " << slice << ":" << slice->getId() << " from sliced one " << work << ":" << work->getId() );

      // Computing specific loop boundaries for 1st child slice
      nli_2 = ( nanos_loop_info_t * ) slice->getData();
      nli_2->thid = nli->thid * 2 + 2;
      nli_2->lower = nli->lower;
      nli_2->upper = nli->upper;
      nli_2->step = nli->step;
      nli_2->args  = nli->args;

      sys.setupWD(*slice, work->getParent());
      // Submit: slice (WorkDescriptor i, running on Thread j)
      slice->tieTo( (*team)[nli_2->thid] );
      (*team)[nli_2->thid].addNextWD(slice);
   }

   int _niters = (((nli->upper - nli->lower) / nli->step ) + 1 );
   int _adjust = _niters % num_threads;
   int _chunk  = ((_niters / num_threads) -1 ) * nli->step;
   int i;

   for ( i = 0; i < nli->thid; i++) {
      nli->lower = nli->lower + _chunk + ((_adjust > i )? nli->step : 0) + nli->step;
   }
   nli->upper = nli->lower + _chunk + (( _adjust > nli->thid ) ? nli->step : 0);
   if ( nli->thid == (num_threads - 1) ) nli->last = true;

//fprintf(stderr, "lower=%d, upper=%d, step=%d\n",nli->lower, nli->upper, nli->step); //FIXME

   ((SMPDD::work_fct)(nli->args))(arg);
}
#endif

static void interleavedLoop ( void *arg )
{
   debug ( "Executing static loop wrapper");
   int _upper, _stride, _chunk;

   nanos_loop_info_t * nli = (nanos_loop_info_t *) arg;

   // forcing last to be false
   nli->last = false;
   // getting initial parameters
   _upper = nli->upper;
   _stride = nli->stride;

   // loop replication (according to step) related with performance issues
   if ( nli->step < 0 ) {
      _chunk = nli->chunk + 1;
      for ( ; nli->lower >= _upper; nli->lower += _stride )
      {
         // computing current parameters
         nli->upper = nli->lower + _chunk;
         if ( nli->upper  <= _upper ) {
            nli->upper = _upper;
            nli->last = true;
         }
         // calling realwork
         ((SMPDD::work_fct)(nli->args))(arg);
      }
   } else {
      _chunk = nli->chunk - 1;
      for ( ; nli->lower <= _upper; nli->lower += _stride )
      {
         // computing current parameters
         nli->upper = nli->lower + _chunk;
         if ( nli->upper  >= _upper ) {
            nli->upper = _upper;
            nli->last = true;
         }
         // calling realwork
         ((SMPDD::work_fct)(nli->args))(arg);
      }
   }
}

void SlicerStaticFor::submit ( WorkDescriptor &work )
#ifdef NANOS_TREE_CREATION
{
   debug ( "Submitting sliced task " << &work << ":" << work.getId() );
   
   BaseThread *mythread = myThread;
   ThreadTeam *team = mythread->getTeam();
   int i, num_threads = team->getFinalSize();
   WorkDescriptor *slice = NULL;
   nanos_loop_info_t *nli;

#if 0
   // copying rest of slicer data values and computing sign value
   // getting user defined chunk, lower, upper and step
   // SlicerDataFor * sdf = (SlicerDataFor *) work.getSlicerData(); /* FIXME: deprecated*/
   int _chunk = sdf->getChunk();
   int _lower = sdf->getLower();
   int _upper = sdf->getUpper();
   int _step  = sdf->getStep();
//fprintf(stderr, "whole loop lower=%d, upper=%d, step=%d (among %d threads)\n",_lower,_upper,_step, num_threads); // FIXME
#endif


// XXX: static
   if ( _chunk == 0 ) {
#if 0
      nli = ( nanos_loop_info_t * ) work.getData();
      nli->thid  = 0;
      nli->lower = _lower;
      nli->upper = _upper; 
      nli->step  = _step;
#endif
      SMPDD &dd = ( SMPDD & ) work.getActiveDevice();
      nli->args = ( void * ) dd.getWorkFct();
      dd = SMPDD(staticLoop);
// XXX: static
   } else {
      // Computing offset between threads
      int _sign = ( _step < 0 ) ? -1 : +1;
      int _offset = _chunk * _step;
      // record original work function
      SMPDD &dd = ( SMPDD & ) work.getActiveDevice();
      // setting new arguments
      nli = (nanos_loop_info_t *) work.getData();
      nli->lower = _lower;
      nli->upper = _upper; 
      nli->step = _step;
      nli->chunk = _offset; 
      nli->stride = _offset * num_threads; 
      nli->args = ( void * ) dd.getWorkFct();
      // change to our wrapper
      dd = SMPDD(interleavedLoop);
      // Init and Submit WorkDescriptors: 1..N
      for ( i = 1; i < num_threads; i++ ) {
         // Avoiding to create 'empty' WorkDescriptors
         if ( ((_lower + (i * _offset)) * _sign) > ( _upper * _sign ) ) break;
         // Duplicating slice into wd
         slice = NULL;
         sys.duplicateWD( &slice, &work );

         debug ( "Creating task " << slice << ":" << slice->getId() << " from sliced one " << &work << ":" << work.getId() );

         // Computing specific loop boundaries for current slice
         nli = ( nanos_loop_info_t * ) slice->getData();
         nli->lower = _lower + ( i * _offset);
         nli->upper = _upper;
         nli->step = _step;
         nli->chunk = _offset;
         nli->stride = _offset * num_threads; 

         sys.setupWD(*slice, work.getParent());
         // Submit: slice (WorkDescriptor i, running on Thread i)
         slice->tieTo( (*team)[i] );
         (*team)[i].addNextWD(slice);
      }
   }
   // Submit: work
   work.convertToRegularWD();
   work.tieTo( (*team)[0] );
   (*team)[0].addNextWD( (WorkDescriptor *) &work);
}




#else
{
   debug ( "Submitting sliced task " << &work << ":" << work.getId() );
   
   BaseThread *mythread = myThread;
   ThreadTeam *team = mythread->getTeam();
   int i, num_threads = team->getFinalSize();
   WorkDescriptor *slice = NULL;
   nanos_loop_info_t *nli;

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

   // copying rest of slicer data values and computing sign value
   // getting user defined chunk, lower, upper and step
   nli = ( nanos_loop_info_t * ) work.getData();
   int _chunk = nli->chunk;
   int _lower = nli->lower;
   int _upper = nli->upper;
   int _step  = nli->step;

   if ( _chunk == 0 ) {
      // Compute chunk and adjustment
      int _niters = (((_upper - _lower) / _step ) + 1 );
      int _adjust = _niters % valid_threads;
      _chunk = ((_niters / valid_threads) -1 ) * _step;
      // Computing upper bound
      _upper = _lower + _chunk + (( _adjust > 0) ? _step : 0);
      // Computing specific loop boundaries for WorkDescriptor 0
      nli->upper = _upper; 
      nli->last = (valid_threads == 1 );
      // Creating additional WorkDescriptors: 1..N
      int j = 0; /* initializing thread id */
      for ( i = 1; i < valid_threads; i++ ) {

         // Finding 'j', as the next valid thread 
         while ( (j < num_threads) && (thread_map[j] != i) ) j++;

         // Computing lower and upper bound
         _lower = _upper + _step;
         _upper = _lower + _chunk + (( _adjust > j ) ? _step : 0);
         // Duplicating slice
         slice = NULL;
         sys.duplicateWD( &slice, &work );
   
         debug ( "Creating task " << slice << ":" << slice->getId() << " from sliced one " << &work << ":" << work.getId() );

         // Computing specific loop boundaries for current slice
         nli = ( nanos_loop_info_t * ) slice->getData();
         nli->lower = _lower;
         nli->upper = _upper;
         nli->last = ( j == (valid_threads - 1) );
         // Submit: slice (WorkDescriptor i, running on Thread j)
         sys.setupWD ( *slice, work.getParent() );
         slice->tieTo( (*team)[j] );
         (*team)[j].addNextWD(slice);
      }
   } else {
      // Computing offset between threads
      int _sign = ( _step < 0 ) ? -1 : +1;
      int _offset = _chunk * _step;
      // record original work function
      SMPDD &dd = ( SMPDD & ) work.getActiveDevice();
      // setting new arguments
      nli = (nanos_loop_info_t *) work.getData();
      nli->lower = _lower;
      nli->upper = _upper; 
      nli->step = _step;
      nli->chunk = _offset; 
      nli->stride = _offset * valid_threads; 
      nli->args = ( void * ) dd.getWorkFct();
      // change to our wrapper
      dd = SMPDD(interleavedLoop);
      // Init and Submit WorkDescriptors: 1..N
      int j = 0; /* initializing thread id */
      for ( i = 1; i < valid_threads; i++ ) {
         // Finding 'j', as the next valid thread 
         while ( (j < num_threads) && (thread_map[j] != i) ) j++;
         // Avoiding to create 'empty' WorkDescriptors
         if ( ((_lower + (j * _offset)) * _sign) > ( _upper * _sign ) ) break;
         // Duplicating slice into wd
         slice = NULL;
         sys.duplicateWD( &slice, &work );

         debug ( "Creating task " << slice << ":" << slice->getId() << " from sliced one " << &work << ":" << work.getId() );

         // Computing specific loop boundaries for current slice
         nli = ( nanos_loop_info_t * ) slice->getData();
         nli->lower = _lower + ( j * _offset);
         nli->upper = _upper;
         nli->step = _step;
         nli->chunk = _offset;
         nli->stride = _offset * valid_threads; 
         sys.setupWD ( *slice, work.getParent() );
         // Submit: slice (WorkDescriptor i, running on Thread j)
         slice->tieTo( (*team)[j] );
         (*team)[j].addNextWD(slice);
      }
   }
   // Submit: work (WorkDescriptor 0, running on thread 'first')
   work.convertToRegularWD();
   work.tieTo( (*team)[first_valid_thread] );
   if ( mythread == &((*team)[first_valid_thread]) ) {
      if ( Scheduler::inlineWork( &work, false ) ) {
         work.~WorkDescriptor();
         delete[] (char *) &work;
      }
   }
   else (*team)[first_valid_thread].addNextWD( (WorkDescriptor *) &work);
}
#endif

class SlicerStaticForPlugin : public Plugin {
   public:
      SlicerStaticForPlugin () : Plugin("Slicer for Loops using a static policy",1) {}
      ~SlicerStaticForPlugin () {}

      virtual void config( Config& cfg ) {}

      void init ()
      {
         sys.registerSlicer("static_for", NEW SlicerStaticFor() );	
      }
};

} // namespace ext
} // namespace nanos

DECLARE_PLUGIN("slicer-static_for",nanos::ext::SlicerStaticForPlugin);
