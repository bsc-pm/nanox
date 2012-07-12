/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
#include "nanos.h"
#include "common.h"

/*
<testinfo>
test_generator=gens/mcc-openmp-generator
</testinfo>
*/

// TEST: Task Wait Overhead ***********************************************************************
void test_taskwait_execution_overhead ( stats_t *s )
{
   int i;
   double times[TEST_NSAMPLES];
   for ( i = 0; i < TEST_NSAMPLES; i++ ) {
      times[i] = GET_TIME;
#pragma omp taskwait
      times[i] = (GET_TIME - times[i]);
   }
   stats( s, times, TEST_NSAMPLES);
}

int main ( int argc, char *argv[] )
{
   stats_t s;

   test_taskwait_execution_overhead( &s );
   print_stats ( "Execute taskwait overhead","warm-up", &s );
   test_taskwait_execution_overhead( &s );
   print_stats ( "Execute taskwait overhead","test", &s );

   return 0;
}
