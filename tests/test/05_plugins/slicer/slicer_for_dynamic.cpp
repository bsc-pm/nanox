/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

/*
<testinfo>
test_generator=gens/core-generator
</testinfo>
*/

#include "config.hpp"
#include <nanos.h>
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include "slicer.hpp"
#include "plugin.hpp"
#include "slicer_for.h"

using namespace std;

using namespace nanos;
using namespace nanos::ext;

#define NUM_ITERS      1
#define VECTOR_SIZE    1000
#define VECTOR_MARGIN  20

// The program will create all possible permutation using NUM_{A,B,C}
// for step and chunk. For a complete testing purpose they have to be:
// -  single step/chunk: 1 ('one')
// -  a divisor of VECTOR_SIZE  (e.g. 5, using a VECTOR_SIZE of 1000)
// -  a non-divisor of VECTOR_SIZE (e.g. 13 using a VECTOR_SIZE 1000)
#define NUM_A          1
#define NUM_B          5
#define NUM_C          13

#define STEP_ERROR     17

// Output information level:
//#define VERBOSE
//#define EXTRA_VERBOSE

int *A;

void print_vector();

typedef struct {
   nanos_loop_info_t loop_info;
   int offset;
} main__loop_1_data_t;

void main__loop_1 ( void *args );

void main__loop_1 ( void *args )
{
   int i;
   main__loop_1_data_t *hargs = (main__loop_1_data_t * ) args;
#ifdef VERBOSE
   fprintf(stderr,"[%d..%d:%d/%d]",
      hargs->loop_info.lower, hargs->loop_info.upper, hargs->loop_info.step, hargs->offset);
#endif
   if ( hargs->loop_info.step > 0 )
   {
      for ( i = hargs->loop_info.lower; i <= hargs->loop_info.upper; i += hargs->loop_info.step) {
         A[i+hargs->offset]++;
      }
   }
   else if ( hargs->loop_info.step < 0 )
   {
      for ( i = hargs->loop_info.lower; i >= hargs->loop_info.upper; i += hargs->loop_info.step) {
         A[i+hargs->offset]++;
      }
   }
   else {A[-VECTOR_MARGIN] = STEP_ERROR; }

}

void print_vector ()
{
#ifdef EXTRA_VERBOSE
   for ( int j = -5; j < 0; j++ ) fprintf(stderr,"%d:",A[j]);
   fprintf(stderr,"[");
   for ( int j = 0; j <= VECTOR_SIZE; j++ ) fprintf(stderr,"%d:",A[j]);
   fprintf(stderr,"]");
   for ( int j = VECTOR_SIZE+1; j < VECTOR_SIZE+6; j++ ) fprintf(stderr,"%d:",A[j]);
   fprintf(stderr,"\n");
#endif
}

int main ( int argc, char **argv )
{
   int i;
   bool check = true; 
   bool p_check = true, out_of_range = false, race_condition = false, step_error= false;
   int I[VECTOR_SIZE+2*VECTOR_MARGIN];
   main__loop_1_data_t _loop_data;
   
   A = &I[VECTOR_MARGIN];

#ifdef VERBOSE
   fprintf(stderr,"SLICER_FOR: Initializing vector.\n");
#endif
   // initialize vector
   for ( i = 0; i < VECTOR_SIZE+2*VECTOR_MARGIN; i++ ) I[i] = 0;

   // omp for: dynamic policy (chunk != 0)
#ifdef VERBOSE
   fprintf(stderr,"SLICER_FOR: dynamic_for begins.\n");
#endif
   TEST_SLICER("dynamic_for", SlicerDataFor)
#ifdef VERBOSE
   fprintf(stderr,"SLICER_FOR: dynamic_for ends.\n");
#endif

   // final result
   //fprintf(stderr, "%s : %s\n", argv[0], check ? "  successful" : "unsuccessful");
   if (check) { return 0; } else { return -1; }
}

