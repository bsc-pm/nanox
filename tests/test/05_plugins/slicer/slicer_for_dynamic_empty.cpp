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
</testinfo>
*/

// The program will create all possible permutation using NUM_{A,B,C}
// for step and chunk. For a complete testing purpose they have to be:
// -  single step/chunk: 1 ('one')
// -  a divisor of VECTOR_SIZE  (e.g. 5, using a VECTOR_SIZE of 1000)
// -  a non-divisor of VECTOR_SIZE (e.g. 13 using a VECTOR_SIZE 1000)
#define NUM_A          1
#define NUM_B          5
#define NUM_C          13

// Mandatory definitions before including "slicer_for.hpp"
#define NUM_ITERS      20
#define VECTOR_SIZE    1000
#define VECTOR_MARGIN  20

// Optional definitions before including "slicer_for.hpp"
//#define VERBOSE
//#define EXTRA_VERBOSE
#define INVERT_LOOP_BOUNDARIES

#include "slicer_for.hpp"
#include <iostream>

int main ( int argc, char **argv )
{
   int error = 0;

#ifdef VERBOSE
   fprintf(stderr,"SLICER_FOR: dynamic_for begins.\n");
#endif
   error += slicer_test("dynamic_for", "sched_dynamic (A,A)", NUM_A, NUM_A);
   error += slicer_test("dynamic_for", "sched_dynamic (B,A)", NUM_B, NUM_A);
   error += slicer_test("dynamic_for", "sched_dynamic (C,A)", NUM_C, NUM_A);
   error += slicer_test("dynamic_for", "sched_dynamic (A,B)", NUM_A, NUM_B);
   error += slicer_test("dynamic_for", "sched_dynamic (B,B)", NUM_B, NUM_B);
   error += slicer_test("dynamic_for", "sched_dynamic (C,B)", NUM_C, NUM_B);
   error += slicer_test("dynamic_for", "sched_dynamic (A,C)", NUM_A, NUM_C);
   error += slicer_test("dynamic_for", "sched_dynamic (B,C)", NUM_B, NUM_C);
   error += slicer_test("dynamic_for", "sched_dynamic (C,C)", NUM_C, NUM_C);
#ifdef VERBOSE
   fprintf(stderr,"SLICER_FOR: dynamic_for ends.\n");
#endif

   std::cout << argv[0] << (!error ? ": successful" : ": unsuccessful") << std::endl;
   return error ? EXIT_FAILURE : EXIT_SUCCESS;
}
