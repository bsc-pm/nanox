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
   fprintf(stderr,"SLICER_FOR: static_for (static) begins.\n");
#endif
   error += slicer_test("static_for", "sched_static (A,0)", NUM_A, 0);
   error += slicer_test("static_for", "sched_static (B,0)", NUM_B, 0);
   error += slicer_test("static_for", "sched_static (C,0)", NUM_C, 0);
#ifdef VERBOSE
   fprintf(stderr,"SLICER_FOR: static_for (static) ends.\n");
#endif

   std::cout << argv[0] << (!error ? ": successful" : ": unsuccessful") << std::endl;
   return error ? EXIT_FAILURE : EXIT_SUCCESS;
}
