/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

/*
<testinfo>
test_generator=gens/core-generator
test_generator_ENV=( "NX_TEST_SCHEDULE=bf --schedule-priority" )
test_LDFLAGS="-Wl,--export-dynamic"
</testinfo>
*/

extern "C"{
   __attribute__((weak)) void nanos_needs_priorities_fun(void) {}
}

#include "config.hpp"
#include "nanos.h"
#include "atomic.hpp"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"

using namespace std;

using namespace nanos;
using namespace nanos::ext;

#define USE_NANOS     true
#define NUM_ITERS     100
#define VECTOR_SIZE   100

int A[VECTOR_SIZE];
// The high priority task should only act once on A
int done = 0;
Lock l, doneLock;;
typedef struct {
   nanos_loop_info_t loop_info;
} main__loop_1_data_t;

void main__loop_1 ( void *args );

/**
 * This task increments by 1 the elements in the array.
 */
void main__loop_1 ( void *args )
{
   int i;
   main__loop_1_data_t *hargs = (main__loop_1_data_t * ) args;

   for ( i = hargs->loop_info.lower; i < hargs->loop_info.upper; i += hargs->loop_info.step) {
      LockBlock lock( l );
      ++A[i];
      memoryFence();
   }
}

/**
 * This loop will set all elements to zero.
 * If the priority scheduler is working properly, after both
 * loops have run, the resulting array will contains elements
 * with value NUM_ITERS (or close to).
 */
void main__loop_2 ( void *args );

void main__loop_2 ( void *args )
{
   int i;
   main__loop_1_data_t *hargs = (main__loop_1_data_t * ) args;
   LockBlock lock( doneLock );
   if ( done != 0 ) {
      return;
   }
   done = 1;

   for ( i = hargs->loop_info.lower; i < hargs->loop_info.upper; i += hargs->loop_info.step) {
      A[i]=0;
   }
}

int main ( int argc, char **argv )
{
   int i;
   bool check = true;

   main__loop_1_data_t _loop_data;

   // initialize vector
   for ( i = 0; i < VECTOR_SIZE; i++ ) A[i] = 0;

   // Stop scheduler
   sys.stopScheduler();
   sys.waitUntilThreadsPaused();
   WD *wg = getMyThreadSafe()->getCurrentWD();
   // increment vector
   for ( i = 0; i < NUM_ITERS; i++ ) {
#if USE_NANOS
      // loop info initialization
      _loop_data.loop_info.lower = 0;
      _loop_data.loop_info.upper = VECTOR_SIZE;
      _loop_data.loop_info.step = + 1;

      // Work descriptor creation
      WD * wd = new WD( new SMPDD( main__loop_1 ), sizeof( _loop_data ), __alignof__(nanos_loop_info_t), ( void * ) &_loop_data );
      wd->setPriority( 100 );

      // Work Group affiliation
      wg->addWork( *wd );

      // Work submission
      sys.submit( *wd );

#else
      for ( int j = 0; j < VECTOR_SIZE; j++ ) A[j] += 100;
#endif
   }
   for ( i = 0; i < sys.getNumWorkers(); ++i )
   {
#if USE_NANOS
      // Second task: set to 0
      WD* wd = new WD( new SMPDD( main__loop_2 ), sizeof( _loop_data ), __alignof__(nanos_loop_info_t), ( void * ) &_loop_data );
      // Use a higher priority
      wd->setPriority( 150 );
      wg->addWork( *wd );
      // Work submission
      sys.submit( *wd );

#else
      for ( int j = 0; j < VECTOR_SIZE; j++ ) A[j] = 0;
#endif
   }
   // Re-enable the scheduler
   sys.startScheduler();
   sys.waitUntilThreadsUnpaused();

   
   // barrier (kind of)
   wg->waitCompletion();
   

   /*
    * Verification criteria: The priority scheduler must ensure that the
    * highest priority task that was submitted the latest is executed before
    * at least one lower priority task.
    * In this case, as the highest priority task sets the elements in the A
    * array to 0, it is as simple as checking if that's the value at the end of
    * the execution. If it is, the test failed, otherwise, succeeded.
    */
   for ( i = 0; i < VECTOR_SIZE; i++ ) if ( A[i] == 0 ) check = false;

   if ( check ) {
      fprintf(stderr, "%s : %s\n", argv[0], "successful");
      return 0;
   }
   else {
      fprintf(stderr, "%s: %s\n", argv[0], "unsuccessful");
      return -1;
   }
}

