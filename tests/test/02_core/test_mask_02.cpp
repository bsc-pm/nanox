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

/*
<testinfo>
test_generator=gens/mixed-generator
test_max_cpus=1
test_ignore_fail=1
</testinfo>
*/

#include <sched.h>
#include "config.hpp"
#include "nanos.h"
#include "atomic.hpp"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"

using namespace nanos;
using namespace nanos::ext;

#define NUM_ITERS     1000
#define NUM_RUNS   10

Atomic<int> A;

typedef struct {
   nanos_loop_info_t loop_info;
} main__loop_1_data_t;

cpu_set_t master_cpu;
cpu_set_t default_mask;

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
   for (i=0; i<CPU_SETSIZE; i++) {
      if ( CPU_ISSET( i, &default_mask ) ) {
         if (rand()%2) {
            CPU_SET( i, &new_mask );
         }
      }
   }
   CPU_OR( &new_mask, &new_mask, &master_cpu );
   //print_mask( "New mask: ", &new_mask );
   sys.setCpuMask( &new_mask, true );
}

void main__loop_1 ( void *args )
{
   A++;
}


int main ( int argc, char **argv )
{
   sys.setUntieMaster( false );

   int i;
   bool check = true;

   main__loop_1_data_t _loop_data;

   CPU_ZERO( &default_mask );
   CPU_SET( 0, &default_mask );
   CPU_SET( 1, &default_mask );
   CPU_SET( 2, &default_mask );
   CPU_SET( 3, &default_mask );
   CPU_ZERO( &master_cpu );
   CPU_SET( 0, &master_cpu );

   // Repeat the test NUM_RUNS times
   for ( int testNumber = 0; testNumber < NUM_RUNS; ++testNumber ) {
      A = 0;
   
      WG *wg = getMyThreadSafe()->getCurrentWD();   
      // increment variable
      for ( i = 0; i < NUM_ITERS; i++ ) {
         // Work descriptor creation
         WD * wd = new WD( new SMPDD( main__loop_1 ), sizeof( _loop_data ), __alignof__(nanos_loop_info_t), ( void * ) &_loop_data );
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
   sys.setCpuMask( &default_mask, true );

   if ( check ) {
      fprintf(stderr, "%s : %s\n", argv[0], "successful");
      return 0;
   }
   else {
      fprintf(stderr, "%s: %s\n", argv[0], "unsuccessful");
      return -1;
   }
}

