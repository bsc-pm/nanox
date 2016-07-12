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

static void staticLoop ( void *arg )
{
   debug ( "Executing static loop wrapper");
   int _upper, _stride, _chunk;

   nanos_loop_info_t * loop_info = (nanos_loop_info_t *) arg;

   //! Checking empty iteration spaces 
   if ( (loop_info->upper > loop_info->lower)  && ( loop_info->step <= 0 ) ) return;
   if ( (loop_info->upper < loop_info->lower)  && ( loop_info->step >= 0 ) ) return;

   //! Forcing 'last' to be false
   loop_info->last = false;

   //! Getting initial parameters: 'upper' and 'stride'
   _upper = loop_info->upper;
   _stride = loop_info->stride;

   //! Loop replication (according to step) related with performance issues
   if ( loop_info->step < 0 ) {
      _chunk = loop_info->chunk + 1;
      for ( ; loop_info->lower >= _upper; loop_info->lower += _stride )
      {
         //! Computing current parameters
         loop_info->upper = loop_info->lower + _chunk;
         if ( loop_info->upper  <= _upper ) {
            loop_info->upper = _upper;
            loop_info->last = true;
         }
         //! Calling realwork
         ((DeviceData::work_fct)(loop_info->args))(arg);
      }
   } else {
      _chunk = loop_info->chunk - 1;
      for ( ; loop_info->lower <= _upper; loop_info->lower += _stride )
      {
         // Computing current parameters
         loop_info->upper = loop_info->lower + _chunk;
         if ( loop_info->upper  >= _upper ) {
            loop_info->upper = _upper;
            loop_info->last = true;
         }
         // Calling realwork
         ((DeviceData::work_fct)(loop_info->args))(arg);
      }
   }
}

void SlicerStaticFor::submit ( WorkDescriptor &work )
{
   debug ( "Submitting sliced task " << &work << ":" << work.getId() );
   
   BaseThread *mythread = myThread;
   ThreadTeam *team = mythread->getTeam();
   WorkDescriptor *slice = NULL;
   nanos_loop_info_t *loop_info;
   int i;

   // Ensure team stability during the job distribution
   while ( !team->isStable() ) memoryFence();
   team->lock();

   /* Determine which is the number of threads are compatible with the work descriptor:
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
   int num_threads = team->getFinalSize();
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
   loop_info = ( nanos_loop_info_t * ) work.getData();

   SMPDD &dd = ( SMPDD & ) work.getActiveDevice();
   loop_info->args = ( void * ) dd.getWorkFct();
   dd = SMPDD(staticLoop);

   int _chunk = loop_info->chunk;
   int _lower = loop_info->lower;
   int _upper = loop_info->upper;
   int _step  = loop_info->step;

   if ( _chunk == 0 ) {

      // Compute chunk and adjustment
      int _niters = (((_upper - _lower) / _step ) + 1 );
      int _adjust = _niters % valid_threads;
      _chunk = ((_niters / valid_threads) ) * _step;
      // Computing specific loop boundaries for WorkDescriptor 0
      loop_info->chunk = _chunk + (( _adjust > 0 ) ? _step : 0);
      loop_info->stride = _niters * _step; 
      // Creating additional WorkDescriptors: 1..N
      int j = 0; /* initializing thread id */
      for ( i = 1; i < valid_threads; i++ ) {

         // Finding 'j', as the next valid thread 
         while ( (j < num_threads) && (thread_map[j] != i) ) j++;

         // Computing lower and upper bound
         _lower += _chunk + (( _adjust > (j-1) ) ? _step : 0);
         // Duplicating slice
         slice = NULL;
         sys.duplicateWD( &slice, &work );
   
         debug ( "Creating task " << slice << ":" << slice->getId() << " from sliced one " << &work << ":" << work.getId() );

         // Computing specific loop boundaries for current slice
         loop_info = ( nanos_loop_info_t * ) slice->getData();
         loop_info->lower = _lower;
         loop_info->chunk = _chunk + (( _adjust > j ) ? _step : 0);
         loop_info->stride = _niters * _step; 

         // Submit: slice (WorkDescriptor i, running on Thread j)
         sys.setupWD ( *slice, work.getParent() );
         BaseThread &target_thread = (*team)[j];
         slice->tieTo( target_thread );
         target_thread.addNextWD(slice);
      }
   } else {
      // Computing offset between threads
      int _sign = ( _step < 0 ) ? -1 : +1;
      int _offset = _chunk * _step;
      // setting new arguments
      loop_info = (nanos_loop_info_t *) work.getData();
      loop_info->lower = _lower;
      loop_info->upper = _upper; 
      loop_info->step = _step;
      loop_info->chunk = _offset; 
      loop_info->stride = _offset * valid_threads; 
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
         loop_info = ( nanos_loop_info_t * ) slice->getData();
         loop_info->lower = _lower + ( j * _offset);
         loop_info->upper = _upper;
         loop_info->step = _step;
         loop_info->chunk = _offset;
         loop_info->stride = _offset * valid_threads; 

         // Submit: slice (WorkDescriptor i, running on Thread j)
         sys.setupWD ( *slice, work.getParent() );
         BaseThread &target_thread = (*team)[j];
         slice->tieTo( target_thread );
         target_thread.addNextWD(slice);
      }
   }

   // Only 1 WD left, we can unlock after obtaining the last target
   BaseThread &first_thread = (*team)[first_valid_thread];
   team->unlock();

   // Submit: work (WorkDescriptor 0, running on thread 'first')
   work.convertToRegularWD();
   work.tieTo( first_thread );
   if ( mythread == &first_thread ) {
      if ( Scheduler::inlineWork( &work, false ) ) {
         work.~WorkDescriptor();
         delete[] (char *) &work;
      }
   }
   else
   {
      first_thread.addNextWD( (WorkDescriptor *) &work);
   }
}

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
