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
test_generator=gens/api-omp-generator
</testinfo>
*/

#include <stdio.h>
#include <nanos.h>
#include "omp.h"

#include <alloca.h>

#define NUM_RUNS   100

omp_lock_t mylock = 0;
int A;

/* ******************************* SECTION 1 ***************************** */
// compiler: outlined function arguments
typedef struct { int *M; } main__section_1_data_t;
// compiler: outlined function
void main__section_1 ( void *p_args )
{
   omp_set_lock( &mylock );
   bool schedEnabled;
   NANOS_SAFE( nanos_scheduler_enabled( &schedEnabled ) );
   fprintf( stderr, "Increasing A (%d) WD(%d)\n", schedEnabled, nanos_get_wd_id(nanos_current_wd() ) );
   A++;
   omp_unset_lock( &mylock );
}
// compiler: smp device for main__section_1 function
nanos_smp_args_t main__section_1_device_args = { main__section_1 };

/* ******************************* SECTIONS ***************************** */

int main ( int argc, char **argv )
{
   int i;
   bool check = true;

   /* COMMON INFO */
   nanos_wd_props_t props = {
      .mandatory_creation = true,
      .tied = false,
      .tie_to = false,
      .priority = 0
   };
   
   omp_init_lock( &mylock );

   // Repeat the test NUM_RUNS times
   int testNumber;
   for ( testNumber = 0; testNumber < /*NUM_RUNS*/1000; ++testNumber ) {
      fprintf( stderr, "[TEST] Starting\n" );
      NANOS_SAFE( nanos_stop_scheduler() );
      A = 0;
      nanos_wd_t wd[4] = { NULL, NULL, NULL, NULL };
   
      /* Creating section 1 wd */
      nanos_device_t main__section_1_device[1] = { NANOS_SMP_DESC( main__section_1_device_args ) };
      main__section_1_data_t *section_data_1 = NULL;
      fprintf(stderr, "[TEST] Creating WD\n" );
      NANOS_SAFE( nanos_create_wd ( &wd[0], 1, main__section_1_device, sizeof(section_data_1), __alignof__(section_data_1), (void **) &section_data_1,
                                nanos_current_wd(), &props , 0, NULL ) );
      fprintf(stderr, "[TEST] Created WD %d\n", nanos_get_wd_id( wd[0]) );
   
      NANOS_SAFE( nanos_submit( wd[0],0,0,0 ) );
      
      fprintf( stderr, "[TEST] Esperando por la tarea\n" );
      NANOS_SAFE( nanos_wait_until_threads_paused() );
      fprintf( stderr, "[TEST] A partir de ahora no debería pasar nada\n" );
      // We should be guaranteed that A will not be modified since this point
      omp_set_lock( &mylock );
      int previousA = A;
      omp_unset_lock( &mylock );
      
      usleep( 100 );
      omp_set_lock( &mylock );
      if ( A != previousA ) {
         check = false;
         NANOS_SAFE( nanos_start_scheduler() );
         break;
      }
      omp_unset_lock( &mylock );
      fprintf( stderr, "[TEST] Poniendo en marcha el scheduler\n" );
      NANOS_SAFE( nanos_start_scheduler() );
      fprintf( stderr, "[TEST] Esperando ejecución de la tarea\n" );
      NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
      fprintf( stderr, "[TEST] Tarea finalizada. A = %d\n", A );
      if ( A != 1 ) {
         fprintf( stderr, "[TEST] A is not 1\n" );
         check = false;
         break;
      }
      fprintf( stderr, "[TEST] Finished\n" );
      // Make sure all threads are unpaused before running the next test
      //usleep( 1000 );
      //NANOS_SAFE( nanos_wait_until_threads_unpaused() );
      fprintf( stderr, "[TEST] Esperando a threads en pausa\n" );
      NANOS_SAFE( nanos_wait_until_threads_unpaused() );
   }

   omp_destroy_lock( &mylock );

   fprintf(stderr, "%s : %s\n", argv[0], check ? "  successful" : "unsuccessful");
   if (check) { return 0; } else { return -1; }
}

