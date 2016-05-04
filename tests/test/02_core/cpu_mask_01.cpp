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

//#define _GNU_SOURCE
#include <sched.h>
#include <string.h>
#include <iostream>
#include "os.hpp"
#include "cpuset.hpp"
#include "nanos.h"
#include "system.hpp"

/*
<testinfo>
   test_generator="gens/core-generator -a --no-warmup-threads|--warmup-threads"
   test_generator_ENV=( "NX_TEST_MAX_CPUS=1"
                        "NX_TEST_SCHEDULE=bf"
                        "NX_TEST_ARCH=smp")
   test_exec_command="timeout 1m"
</testinfo>
*/

#define SIZE 100

#ifndef min
#define min(x,y) ((x<y)?x:y)
#endif

using namespace nanos;
using namespace nanos::ext;

void print_mask( const char *pre, cpu_set_t *mask );

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

int main ( int argc, char *argv[])
{
   if ( OS::getMaxProcessors() < 2 ) {
      fprintf(stdout, "Skiping %s test\n", argv[0]);
      return 0;
   }

   int error = 0;
   int max_procs = OS::getMaxProcessors();

   cpu_set_t nanos_mask1, nanos_mask2;
   cpu_set_t sched_mask1, sched_mask2;
   CPU_ZERO( &nanos_mask1 );
   CPU_ZERO( &nanos_mask2 );
   CPU_ZERO( &sched_mask1 );
   CPU_ZERO( &sched_mask2 );

   // init
   const CpuSet &active_mask = sys.getCpuActiveMask();
   active_mask.copyTo( &nanos_mask1 );
   sched_getaffinity( 0, sizeof(cpu_set_t), &sched_mask1 );

   // test
   CPU_SET( 0, &nanos_mask2 );
   CPU_SET( 1, &nanos_mask2 );
   sys.setCpuActiveMask( &nanos_mask2 );
   sched_getaffinity( 0, sizeof(cpu_set_t), &sched_mask2 );

   fprintf(stdout,"Thread team final size will be %d and %d is expected\n",
      (int) myThread->getTeam()->getFinalSize(),
            min(CPU_COUNT(&nanos_mask2),max_procs)
   );
   if ( sys.getPMInterface().isMalleable() && myThread->getTeam()->getFinalSize() != (size_t) CPU_COUNT(&nanos_mask2) ) error++;


   /* If binding is disabled further tests make no sense */
   if ( !sys.getSMPPlugin()->getBinding() ) {
      fprintf(stdout,"Result is %s\n", error? "UNSUCCESSFUL":"successful");
      return error;
   }

   // check intersections
   cpu_set_t intxn;

   CPU_AND( &intxn, &nanos_mask1, &sched_mask1);
   if ( !CPU_EQUAL( &intxn, &sched_mask1 ) ) error++;
   //print_mask( "nanos_mask1: ", &nanos_mask1 );
   //print_mask( "sched_mask1: ", &sched_mask1 );
   //print_mask( "intxn: ", &intxn );

   CPU_AND( &intxn, &nanos_mask2, &sched_mask2);
   if ( !CPU_EQUAL( &intxn, &sched_mask2 ) ) error++;
   //print_mask( "nanos_mask2: ", &nanos_mask2 );
   //print_mask( "sched_mask2: ", &sched_mask2 );
   //print_mask( "intxn: ", &intxn );

   fprintf(stdout,"Result is %s\n", error? "UNSUCCESSFUL":"successful");

   return error;
}

