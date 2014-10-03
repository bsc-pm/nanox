
//#define _GNU_SOURCE
#include <sched.h>
#include <string.h>
#include <iostream>
#include "os.hpp"
#include "nanos.h"
#include "system.hpp"

/*
<testinfo>
test_generator=gens/mixed-generator
test_max_cpus=1
test_schedule="bf"
</testinfo>
*/

#define SIZE 100

#ifndef min
#define min(x,y) ((x<y)?x:y)
#endif

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
   sys.getCpuMask( &nanos_mask1 );
   sched_getaffinity( 0, sizeof(cpu_set_t), &sched_mask1 );

   // test
   CPU_SET( 0, &nanos_mask2 );
   CPU_SET( 1, &nanos_mask2 );
   sys.setCpuMask( &nanos_mask2 );
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

