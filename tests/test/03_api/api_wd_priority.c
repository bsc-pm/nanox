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

// FIXME: add priority queue option to this benchmark see #807
/*
<testinfo>
test_generator=gens/api-omp-generator
test_generator_ENV=( "NX_TEST_SCHEDULE=bf --schedule-priority" )
test_LDFLAGS="-Wl,--export-dynamic"
</testinfo>
*/

__attribute__((weak)) void nanos_needs_priorities_fun(void) {}

/*
 * Test description:
 * This tests setting a WD's priority via the C API. It will create several
 * tasks with zero priority, and a high-priority task. If the higher priority
 * task is not executed before (at least) one of the others, the shared variable
 * will be 0, so the test will fail.
 *
 * To ensure no low priority task is executed until the high priority task is
 * submitted (the last one to be submitted), this test stops the scheduler.
 */

#include <stdio.h>
#include <nanos.h>
#include "omp.h"
#include <alloca.h>

#define NUM_ITERS   100

// This is to ensure the high priority task is not executed twice or more
int done = 0;
omp_lock_t mylock = 0;

/* ******************************* SECTION 1 ***************************** */
// compiler: outlined function arguments
typedef struct { int *M; } task_arguments_t;
// compiler: outlined function
void task_1 ( void *p_args );
void task_1 ( void *p_args )
{
   int i;
   task_arguments_t *args = (task_arguments_t *) p_args;
   ++(*args->M);
}

// compiler: smp device for task_1 function
nanos_smp_args_t task_1_device_args = { task_1 };

void task_2 ( void *p_args );
void task_2 ( void *p_args )
{
   int i;
   task_arguments_t *args = (task_arguments_t *) p_args;
   omp_set_lock( &mylock );
   if( done == 0 )
      *args->M=0;
   done = 1;
   omp_unset_lock( &mylock );

}

// compiler: smp device for task_2 function
nanos_smp_args_t task_2_device_args = { task_2 };

/* ******************************* SECTIONS ***************************** */

/* ************** CONSTANT PARAMETERS IN WD CREATION ******************** */

struct nanos_const_wd_definition_1
{
     nanos_const_wd_definition_t base;
     nanos_device_t devices[1];
};

struct nanos_const_wd_definition_1 const_data1 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   0,//__alignof__(section_data_1),
   0,
   1,0,NULL},
   {
      {
         nanos_smp_factory,
         &task_1_device_args
      }
   }
};
struct nanos_const_wd_definition_1 const_data2 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   0,//__alignof__(section_data_2),
   0,
   1,0,NULL},
   {
      {
         nanos_smp_factory,
         &task_2_device_args
      }
   }
};
int main ( int argc, char **argv )
{
   int A = 0;
   bool check = true;
   int i;

   // Stop scheduler, no task should be run until told so
   nanos_stop_scheduler();
   nanos_wait_until_threads_paused();
   nanos_wd_dyn_props_t dyn_props = {0,0};

   for ( i = 0; i < NUM_ITERS; ++i ) {
      nanos_wd_t wd = NULL;
      
      task_arguments_t *section_data_1 = NULL;
      const_data1.base.data_alignment = __alignof__(section_data_1);
      NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_data1.base, &dyn_props, sizeof(section_data_1), (void **) &section_data_1,
                                nanos_current_wd(), NULL, NULL ) );
      section_data_1->M = &A;
      
      NANOS_SAFE( nanos_submit( wd,0,0,0 ) );
   }
   // Second task
   for ( i = 0; i < omp_get_num_threads(); ++i )
   {
      task_arguments_t *section_data_2 = NULL;
      const_data2.base.data_alignment = __alignof__(section_data_2);
      dyn_props.priority = 100;
      nanos_wd_t wd = NULL;
      NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_data2.base, &dyn_props, sizeof(section_data_2), (void **) &section_data_2,
                                nanos_current_wd(), NULL, NULL ) );
      
      section_data_2->M = &A;
   
      NANOS_SAFE( nanos_submit( wd,0,0,0 ) );
   }
   
   
   // Now the scheduler can make put the threads to work
   nanos_start_scheduler();
   nanos_wait_until_threads_unpaused();
   
   // Wait until all tasks have been executed
   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   
   /*
    * Condition: A is > 0, that means that
    * the second task was executed before at least one of the first ones.
    */
   check = A != 0;
   fprintf( stderr, "A == %d?: A = %d\n", NUM_ITERS, A );
   
   fprintf(stderr, "%s : %s\n", argv[0], check ? "  successful" : "unsuccessful");
   if (check) { return 0; } else { return -1; }
}

