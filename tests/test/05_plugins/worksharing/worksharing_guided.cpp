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
test_generator=gens/api-omp-generator
</testinfo>
*/

#include "nanos_omp.h"
#include <iostream>
#include <cstdlib>

// The program will create all possible permutation using NUM_{A,B,C}
// for step and chunk. For a complete testing purpose they have to be:
// -  single step/chunk: 1 ('one')
// -  a divisor of VECTOR_SIZE  (e.g. 5, using a VECTOR_SIZE of 1000)
// -  a non-divisor of VECTOR_SIZE (e.g. 13 using a VECTOR_SIZE 1000)
#define NUM_A          1
#define NUM_B          5
#define NUM_C          13

// Mandatory definitions before including "worksharing.hpp"
#define NUM_ITERS      20
#define VECTOR_SIZE    1000
#define VECTOR_MARGIN  20

// Optional definitions before including "worksharing.hpp"
//#define VERBOSE
//#define EXTRA_VERBOSE

#include "worksharing.hpp"

int main(int argc, char **argv)
{
   int error = 0;

   error += ws_test(nanos_omp_sched_guided, "sched_guided (A,A)", NUM_A, NUM_A);
   error += ws_test(nanos_omp_sched_guided, "sched_guided (A,B)", NUM_A, NUM_B);
   error += ws_test(nanos_omp_sched_guided, "sched_guided (A,C)", NUM_A, NUM_C);
   error += ws_test(nanos_omp_sched_guided, "sched_guided (B,A)", NUM_B, NUM_A);
   error += ws_test(nanos_omp_sched_guided, "sched_guided (B,B)", NUM_B, NUM_B);
   error += ws_test(nanos_omp_sched_guided, "sched_guided (B,C)", NUM_B, NUM_C);
   error += ws_test(nanos_omp_sched_guided, "sched_guided (C,A)", NUM_C, NUM_A);
   error += ws_test(nanos_omp_sched_guided, "sched_guided (C,B)", NUM_C, NUM_B);
   error += ws_test(nanos_omp_sched_guided, "sched_guided (C,C)", NUM_C, NUM_C);

   std::cout << argv[0] << (!error ? ": successful" : ": unsuccessful") << std::endl;
   return error ? EXIT_FAILURE : EXIT_SUCCESS;
}
