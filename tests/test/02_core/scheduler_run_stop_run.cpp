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
test_generator_ENV=( "NX_TEST_SCHEDULE=wf" )
</testinfo>
*/

#include "config.hpp"
#include "nanos.h"
#include "atomic.hpp"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"

using namespace std;

using namespace nanos;
using namespace nanos::ext;

#define NUM_ITERS     2000
#define VECTOR_SIZE   100

Atomic<int> A;

typedef struct {
   nanos_loop_info_t loop_info;
} main__loop_1_data_t;

void main__loop_1 ( void *args );

void main__loop_1 ( void *args )
{
   A++;
}


int main ( int argc, char **argv )
{
   int i;
   bool check = true;

   main__loop_1_data_t _loop_data;

   A = 0;

   WD *wg = getMyThreadSafe()->getCurrentWD(); 
   for ( i = 0; i < NUM_ITERS; i++ ) {
      
      // If we're done processing half of the dataset
      if ( i == NUM_ITERS/2 ) {
         // Stop scheduler
         sys.stopScheduler();
      }
      
      // Work descriptor creation
      WD * wd = new WD( new SMPDD( main__loop_1 ), sizeof( _loop_data ), __alignof__(nanos_loop_info_t), ( void * ) &_loop_data );
      wd->setPriority( 100 );

      // Work Group affiliation
      wg->addWork( *wd );

      // Work submission
      sys.submit( *wd );
      
      if ( i == ( NUM_ITERS/2 + 5 ) ){
         // Keep going
         sys.startScheduler();
      }
   }
   // barrier (kind of)
   wg->waitCompletion();
   

   /*
    * How can we be sure the test passed? Each task increments A. If we run N
    * tasks, A should be equal to N.
    * If it's less than N, that'd mean the scheduler lost something.
    */
   if ( A.value() != NUM_ITERS ) check = false;

   if ( check ) {
      fprintf(stderr, "%s : %s\n", argv[0], "successful");
      return 0;
   }
   else {
      fprintf(stderr, "%s: %s\n", argv[0], "unsuccessful");
      return -1;
   }
}

