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
test_generator_ENV=( "NX_TEST_SCHEDULE=bf --schedule-smart-priority" )
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

#define NUM_ITERS     100
#define VECTOR_SIZE   100

int A[VECTOR_SIZE];
Lock l;
typedef struct {
   nanos_loop_info_t loop_info;
} task_data_t;

void task_a ( void *args );

/**
 * This task increments by 1 the elements in the array.
 */
void task_a ( void *args )
{
   debug( "Task A" );
   int i;
   task_data_t *hargs = (task_data_t * ) args;

   for ( i = hargs->loop_info.lower; i < hargs->loop_info.upper; i += hargs->loop_info.step) {
      LockBlock lock( l );
      ++A[i];
      memoryFence();
   }
   //usleep( 1000 );
}

/**
 * This loop will set all elements to zero.
 * If the priority scheduler is working properly, after both
 * loops have run, the resulting array will contains elements
 * with value NUM_ITERS (or close to).
 */
void task_b ( void *args );

void task_b ( void *args )
{
   debug( "Task B" );
   int i;
   task_data_t *hargs = (task_data_t * ) args;

   for ( i = hargs->loop_info.lower; i < hargs->loop_info.upper; i += hargs->loop_info.step) {
      LockBlock lock( l );
      A[i]=0;
      memoryFence();
   }
}

void task_c ( void *args );

void task_c ( void *args )
{
   debug( "Task C" );
   int i;
   task_data_t *hargs = (task_data_t * ) args;

   for ( i = hargs->loop_info.lower; i < hargs->loop_info.upper; i += hargs->loop_info.step) {
      LockBlock lock( l );
      A[i]*=10;
      memoryFence();
   }
}

void task_d ( void *args );

void task_d ( void *args )
{
   debug( "Task D" );
   int i;
   task_data_t *hargs = (task_data_t * ) args;

   for ( i = hargs->loop_info.lower; i < hargs->loop_info.upper; i += hargs->loop_info.step) {
      LockBlock lock( l );
      A[i]/=2;
      memoryFence();
   }
}

int main ( int argc, char **argv )
{
   int i;
   bool check = true;

   task_data_t task_data;
   
   int depA;
   nanos_region_dimension_t dimA[1] = {{ sizeof( int ), 0, sizeof( int ) }};
   nanos_data_access_t depsA[] = {{(void *)&depA, {0,1,0,0,0}, 1, dimA, 0} };
   
   int depB;
   nanos_region_dimension_t dimB[1] = {{ sizeof( int ), 0, sizeof( int ) }};
   nanos_data_access_t depsB[] = {{(void *)&depB, {0,1,0,0,0}, 1, dimB, 0} };
   
   int depC;
   nanos_region_dimension_t dimC[1] = {{ sizeof( int ), 0, sizeof( int ) }};
   nanos_data_access_t depsC[] = {{(void *)&depA, {1,0,0,0,0}, 1, dimA, 0},
      {(void *)&depC, {0,1,0,0,0}, 1, dimC, 0} };
   
   nanos_data_access_t depsD[] = {{(void *)&depB, {1,0,0,0,0}, 1, dimB, 0},
      {(void *)&depC, {1,0,0,0,0}, 1, dimC, 0} };

   // initialize vector
   for ( i = 0; i < VECTOR_SIZE; i++ ) A[i] = 0;

   // Stop scheduler
   sys.stopScheduler();
   sys.waitUntilThreadsPaused();
   WD *wg = getMyThreadSafe()->getCurrentWD();
   WD * wd;
   // loop info initialization
   task_data.loop_info.lower = 0;
   task_data.loop_info.upper = VECTOR_SIZE;
   task_data.loop_info.step = + 1;
   WD* wdLastA = NULL;
   // increment vector
   for ( i = 0; i < NUM_ITERS; i++ ) {

      // Work descriptor creation
      wd = new WD( new SMPDD( task_a ), sizeof( task_data ), __alignof__(nanos_loop_info_t), ( void * ) &task_data );
      //wd->setPriority( 0 );

      // Work Group affiliation
      wg->addWork( *wd );

      // Work submission
      sys.submitWithDependencies( *wd, 1, (nanos::DataAccess*)&depsA );
      //sys.submit( *wd );

   }
   wdLastA = wd;
   
   // Second task: set to 0
   wd = new WD( new SMPDD( task_b ), sizeof( task_data ), __alignof__(nanos_loop_info_t), ( void * ) &task_data );
   // Use a higher priority
   //wd->setPriority( 300 );
   wg->addWork( *wd );
   // Work submission
   debug( "Submitting B" );
   sys.submitWithDependencies( *wd, 1, (nanos::DataAccess*)&depsB );
   // Keep a pointer to this WD
   WD* wdB = wd;
   
   // Third task: multiply by 10
   wd = new WD( new SMPDD( task_c ), sizeof( task_data ), __alignof__(nanos_loop_info_t), ( void * ) &task_data );
   //wd->setPriority( 251 );
   wg->addWork( *wd );
   // Work submission
   debug( "Submitting C" );
   sys.submitWithDependencies( *wd, 2, (nanos::DataAccess*)&depsC );
   WD* wdC = wd;
   
   // Fourth task: divide by 2
   wd = new WD( new SMPDD( task_d ), sizeof( task_data ), __alignof__(nanos_loop_info_t), ( void * ) &task_data );
   wd->setPriority( 250 );
   wg->addWork( *wd );
   // Work submission
   debug( "Submitting D" );
   sys.submitWithDependencies( *wd, 2, (nanos::DataAccess*)&depsD );
   
   // D's priority should've been propagated to B and C
   if ( ( wdB->getPriority() != 250 ) || ( wdC->getPriority() != 250 ) ) {
      check = false;
      fprintf(stderr, "Priority of task D not propagated to task B and task C (%d)\n", wdB->getPriority() );
   }
   if ( wdLastA->getPriority() != 0 ) {
      check = false;
      fprintf(stderr, "Priority of task A was altered by someone (is %d)\n", wdLastA->getPriority() );
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

