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

#include "config.hpp"
#include "nanos.h"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include "slicer.hpp"

using namespace std;

using namespace nanos;
using namespace nanos::ext;

#define USE_NANOS      true
#define NUM_ITERS       100
#define VECTOR_SIZE     100 
#define VECTOR_MARGIN    10

int *A;

typedef struct {
   nanos_loop_info_t loop_info;
} main__loop_1_data_t;


void main__loop_1 ( void *args )
{
   int i;
   main__loop_1_data_t *hargs = (main__loop_1_data_t * ) args;

   for ( i = hargs->loop_info.lower; i < hargs->loop_info.upper; i += hargs->loop_info.step) {
      A[i]++;
   }
}

int main ( int argc, char **argv )
{
   int i;
   bool check = true; 
   bool p_check = true, out_of_range = false, race_condition = false;
   int I[VECTOR_SIZE+2*VECTOR_MARGIN];
   
   A = &I[VECTOR_MARGIN];

   main__loop_1_data_t _loop_data;

   // initialize vector
   for ( i = 0; i < VECTOR_SIZE+2*VECTOR_MARGIN; i++ ) I[i] = 0;

   // LOOP: (Dynamic, lower(+), upper(+), step(+1), chunk(5), data(local)
   for ( i = 0; i < NUM_ITERS; i++ ) {

      // Work descriptor creation, loop info included in SlicerDataDynamicFor
      WD * wd = new SlicedWD( sys.getSlicerDynamicFor(), *new SlicerDataDynamicFor(0,VECTOR_SIZE,+1,5),
                        new SMPDD( main__loop_1 ), sizeof( _loop_data ),( void * ) &_loop_data );

      // Work Group affiliation
      WG *wg = myThread->getCurrentWD();
      wg->addWork( *wd );

      // Work submission
      sys.submit( *wd );
 
      // barrier (kind of)
      wg->waitCompletation();

      // UNDO
      for ( int j = 0; j < VECTOR_SIZE; j++ ) A[j]--;
   }
   // check and initialize vector for next test
   for ( i = 0; i < VECTOR_SIZE+2*VECTOR_MARGIN; i++ )
      if ( I[i] != 0 ) {
         if ( (i < VECTOR_MARGIN) || (i > (VECTOR_SIZE + VECTOR_MARGIN))) out_of_range = true;
         if ( I[i] != NUM_ITERS ) race_condition = true;
         I[i] = 0; check = false; p_check = false;
      }
   // print partial result
   fprintf(stderr, "dynamic \n" );
   // initialize partial checks
   p_check = true; out_of_range = false; race_condition = false;



   // END:
   fprintf(stderr, "%s : %s\n", argv[0], check ? "successful" : "unsuccessful");
   if (check) { return 0; } else { return -1; }
}

