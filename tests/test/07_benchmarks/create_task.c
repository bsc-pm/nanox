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

#include "common.h"

/*
<testinfo>
test_generator=gens/mcc-openmp-generator
test_generator_ENV=( "NX_TEST_MODE=performance" )
</testinfo>
*/

// TEST: Task Creation Overhead ********************************************************************
typedef struct _nx_data_env_1_t_tag { } _nx_data_env_1_t;
static void _smp__ol_test_task_creation_overhead_1(_nx_data_env_1_t *const __restrict__ _args) { task(TEST_TUSECS); }
void test_task_creation_overhead ( stats_t *s )
{
   int i;
   double times[TEST_NSAMPLES];
   for ( i = 0; i < TEST_NSAMPLES; i++ ) {
      {
         times[i] = GET_TIME;
         /* SMP device descriptor, code generated on June 3th, 2012 */
         static nanos_smp_args_t _ol_test_task_creation_overhead_1_smp_args = {(void (*)(void *)) _smp__ol_test_task_creation_overhead_1};
         _nx_data_env_1_t *ol_args = (_nx_data_env_1_t *) 0;
         nanos_wd_t wd = (nanos_wd_t) 0;
         struct nanos_const_wd_definition_local_t { nanos_const_wd_definition_t base; nanos_device_t devices[1];
         };
         static struct nanos_const_wd_definition_local_t _const_def = { 
            { { 0, 1, 0, 0, 0, 0, 0, 0 }, __alignof__(_nx_data_env_1_t), 0, 1, 0, NULL }, {{ nanos_smp_factory, &_ol_test_task_creation_overhead_1_smp_args }}
         };
         nanos_wd_dyn_props_t dyn_props = {0};
         nanos_err_t err;
         dyn_props.priority = 0;

         err = nanos_create_wd_compact(&wd, &_const_def.base, &dyn_props, sizeof(_nx_data_env_1_t),
                                       (void **) &ol_args, nanos_current_wd(), (nanos_copy_data_t **) 0, NULL
               );

         if (err != NANOS_OK) nanos_handle_error(err);
         times[i] = GET_TIME - times[i];


         if (wd != (nanos_wd_t) 0) {
            err = nanos_submit(wd, 0, (nanos_data_access_t *) 0, (nanos_team_t) 0);
            if (err != NANOS_OK) nanos_handle_error(err);
         } else {
            _nx_data_env_1_t imm_args;
            dyn_props.priority = 0;
            err = nanos_create_wd_and_run_compact(&_const_def.base, &dyn_props, sizeof(_nx_data_env_1_t), &imm_args, 0, (nanos_data_access_t *) 0, (nanos_copy_data_t *) 0, (void *) 0, NULL);
            if (err != NANOS_OK) nanos_handle_error(err);
         }
      }
   }
#pragma omp taskwait
   stats( s, times, TEST_NSAMPLES);
}

int main ( int argc, char *argv[] )
{
   stats_t s;

   test_task_creation_overhead( &s ); 
   print_stats ( "Create task overhead","warm-up", &s );
   test_task_creation_overhead( &s );
   print_stats ( "Create task overhead","test", &s );

   return 0;
}
