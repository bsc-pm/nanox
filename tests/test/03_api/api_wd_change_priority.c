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
test_generator_ENV=( "NX_TEST_MAX_CPUS=1"
                     "NX_TEST_SCHEDULE=bf --schedule-priority" )
</testinfo>
*/

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

/* ******************************* TASK INC ***************************** */

// compiler: outlined function arguments
typedef struct { int *M; } task_inc_args_t;

// compiler: outlined function
void task_inc ( void *p_args )
{
   task_inc_args_t *args = (task_inc_args_t *) p_args;
   (*args->M)++;
}
// compiler: smp device for task_inc function
nanos_smp_args_t task_inc_device_args = { task_inc };

// compiler: const data for task_inc function
struct nanos_const_wd_definition_task_inc
{
     nanos_const_wd_definition_t base;
     nanos_device_t devices[1];
};
struct nanos_const_wd_definition_task_inc const_data_task_inc = 
{
   {
      {.mandatory_creation = true, .tied = false },
      0, //__alignof__(section_data_1),
      0,
      1,
      0,
      NULL
   } /* base */
   ,
   {
      { nanos_smp_factory, &task_inc_device_args }
   } /* devices[1]*/
};


// compiler: outlined function arguments
typedef struct { int *M; } task_init_args_t;

// compiler: outlined function
void task_init ( void *p_args )
{
   task_init_args_t *args = (task_init_args_t *) p_args;
   (*args->M) = 0;
}
// compiler: smp device for task_init function
nanos_smp_args_t task_init_device_args = { task_init };

// compiler: const data for task_init function
struct nanos_const_wd_definition_task_init
{
     nanos_const_wd_definition_t base;
     nanos_device_t devices[1];
};
struct nanos_const_wd_definition_task_init const_data_task_init = 
{
   {
      {.mandatory_creation = true, .tied = false },
      0, // default data alignment
      0,
      1,
      0,
      NULL
   } /* base */
   ,
   {
      { nanos_smp_factory, &task_init_device_args }
   } /* devices[1]*/
};




int main ( int argc, char **argv )
{
   int i, A = 0;
   bool check = true;

   // Stop scheduler, no task should be run until told so
   nanos_stop_scheduler();
   nanos_wait_until_threads_paused();

   nanos_wd_t twd = NULL;

   {
      nanos_wd_dyn_props_t dyn_props = {0,0};
      nanos_wd_t wd = NULL;

      dyn_props.priority = 50;
      
      task_inc_args_t *data_task_inc = NULL;
      const_data_task_inc.base.data_alignment = __alignof__(data_task_inc);
      NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_data_task_inc.base, &dyn_props, sizeof(data_task_inc), (void **) &data_task_inc,
                                            nanos_current_wd(), NULL, NULL ) );
      data_task_inc->M = &A;
      
      NANOS_SAFE( nanos_submit( wd,0,0,0 ) );

      twd = wd; // keeping target wd
   }

   {
      nanos_wd_dyn_props_t dyn_props = {0,0};
      nanos_wd_t wd = NULL;

      dyn_props.priority = 100;
      
      task_init_args_t *data_task_init = NULL;
      const_data_task_init.base.data_alignment = __alignof__(data_task_init);
      NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_data_task_init.base, &dyn_props, sizeof(data_task_init), (void **) &data_task_init,
                                            nanos_current_wd(), NULL, NULL ) );
      data_task_init->M = &A;
      
      NANOS_SAFE( nanos_submit( wd,0,0,0 ) );

   }

   nanos_set_wd_priority ( twd, 150 );

   // Now the scheduler can make put the threads to work
   nanos_start_scheduler();
   nanos_wait_until_threads_unpaused();
   
   // Wait until all tasks have been executed
   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   
   check = A == 0;

   fprintf(stderr, "%s : final value %d : %s\n", argv[0], A,  check ? "  successful" : "unsuccessful");

   if (check) { return 0; } else { return -1; }
}

