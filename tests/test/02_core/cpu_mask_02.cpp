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
   test_generator="gens/core-generator -a --no-warmup-threads|--warmup-threads"
   test_generator_ENV=( "NX_TEST_MAX_CPUS=1"
                        "NX_TEST_SCHEDULE=bf"
                        "NX_TEST_ARCH=smp")
   test_exec_command="timeout 1m"
</testinfo>
*/

#include <sched.h>
#include <cstdlib>
#include <iostream>
#include "config.hpp"
#include "nanos.h"
#include "os.hpp"
#include "atomic.hpp"
#include "smpprocessor.hpp"
#include "system.hpp"

using namespace nanos;
using namespace nanos::ext;

#define NUM_ITERS  1000
#define NUM_RUNS   10
#define MIN_PROCS  4

#ifndef min
#define min(x,y) ((x<y)?x:y)
#endif

Atomic<int> A;

typedef struct {
   nanos_loop_info_t loop_info;
} main__loop_1_data_t;


void print_mask( const char *pre, cpu_set_t *mask );
void set_random_mask( void );
void main__loop_1 ( void *args );

void print_mask( const char *pre, cpu_set_t *mask )
{
   char str[CPU_SETSIZE*2 + 8];
   int i, max_cpu = 0;

   strcpy( str, "[ " );
   for (i=0; i<CPU_SETSIZE; i++) {
      if ( CPU_ISSET(i, mask ) ) {
         strcat( str, "1 " );
         max_cpu = i;
      } else {
         strcat( str, "0 " );
      }
   }
   str[ (max_cpu+2)*2 ] = ']';
   str[ (max_cpu+2)*2+1] = '\0';
   std::cout << pre << str << std::endl;
}

void set_random_mask( void )
{
   cpu_set_t new_mask;
   CPU_ZERO( &new_mask );

   int i;
   for (i=0; i<MIN_PROCS; i++) {
      if (rand()%2) {
         CPU_SET( i, &new_mask );
      }
   }
   sys.setCpuActiveMask( &new_mask );
}

void main__loop_1 ( void *args )
{
   A++;
}


int main ( int argc, char **argv )
{
   if ( OS::getMaxProcessors() < MIN_PROCS ) {
      fprintf(stdout, "Skiping %s test\n", argv[0]);
      return EXIT_SUCCESS;
   }
   if ( !sys.getSMPPlugin()->getBinding() )
      return EXIT_SUCCESS;

   int i;
   bool check = true;

   main__loop_1_data_t _loop_data;

   // Repeat the test NUM_RUNS times
   for ( int testNumber = 0; testNumber < NUM_RUNS; ++testNumber ) {
      A = 0;

      WD *wg = getMyThreadSafe()->getCurrentWD();
      // increment variable
      for ( i = 0; i < NUM_ITERS; i++ ) {
         // Work descriptor creation
         WD * wd = new WD( new SMPDD( main__loop_1 ), sizeof( _loop_data ),
               __alignof__(nanos_loop_info_t), ( void * ) &_loop_data );
         wd->setPriority( 100 );

         // Work Group affiliation
         wg->addWork( *wd );

         // Work submission
         sys.submit( *wd );

      }
      // barrier (kind of)
      wg->waitCompletion();
      set_random_mask();

      /*
       * The verification criteria is that A is equal to the number of tasks
       * run. Should A be lower, that would indicate that not all tasks
       * successfuly finished.
       */
      if ( A.value() != NUM_ITERS ) check = false;
   }

   if ( check ) {
      fprintf(stderr, "%s : %s\n", argv[0], "successful");
      return EXIT_SUCCESS;
   }
   else {
      fprintf(stderr, "%s: %s\n", argv[0], "unsuccessful");
      return EXIT_FAILURE;
   }
}

