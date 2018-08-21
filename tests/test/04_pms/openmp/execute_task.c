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

#include "stdio.h"
#include "omp.h"

/*
<testinfo>
test_generator=gens/mcc-openmp-generator
</testinfo>
*/

#define TEST_NSAMPLES   50 // Number of samples for each test
#define TEST_NTASKS     50 // Number of tasks (when multiple tasks created)
#define TEST_TUSECS     50 // Task granularity (usecs) in warming up phase

int A[TEST_NTASKS];

void task ( int i ) { A[i]++; }

// TEST: Task Execution ***************************************************************************
int main ( int argc, char *argv[] )
{
   int i,j; // indexes
   int result = 0;

   for ( i = 0; i < TEST_NSAMPLES; i++ ) {
      for ( j = 0; j < TEST_NTASKS; j++ ) {
#pragma omp task firstprivate(j)
         task(j);
      }
#pragma omp taskwait
   }

      for ( j = 0; j < TEST_NTASKS; j++ ) 
         if ( A[j] != TEST_NSAMPLES ) { result = 1; break; }

   fprintf(stderr, "Result is %s\n", result? "successful": "UNSUCCESSFUL"); return result;
}
