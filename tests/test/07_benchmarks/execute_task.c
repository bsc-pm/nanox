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

#include "nanos.h"
#include "common.h"

/*
<testinfo>
test_generator=gens/mcc-openmp-generator
test_generator_ENV=( "NX_TEST_MODE=performance" )
</testinfo>
*/

// TEST: Task Execution Overhead *******************************************************************
void test_task_execution_overhead ( stats_t *s )
{
   int i,j, nthreads = omp_get_max_threads();
   double times_seq[TEST_NSAMPLES];
   double times[TEST_NSAMPLES];

   for ( i = 0; i < TEST_NSAMPLES; i++ ) {
      times_seq[i] = GET_TIME;
      for ( j = 0; j < TEST_NTASKS; j++ ) {
         task(TEST_TUSECS);
      }
      times_seq[i] = GET_TIME - times_seq[i];
   }

   for ( i = 0; i < TEST_NSAMPLES; i++ ) {
      times[i] = GET_TIME;
      for ( j = 0; j < TEST_NTASKS; j++ ) {
#pragma omp task
         task(TEST_TUSECS);
      }
#pragma omp taskwait
      times[i] = (((GET_TIME - times[i]) - times_seq[i]) * nthreads) / TEST_NTASKS;
   }
   stats( s, times, TEST_NSAMPLES);
}

int main ( int argc, char *argv[] )
{
   stats_t s;

   test_task_execution_overhead( &s );
   print_stats ( "Execute task overhead","warm-up", &s );
   test_task_execution_overhead( &s );
   print_stats ( "Execute task overhead","test", &s );

   return 0;
}
