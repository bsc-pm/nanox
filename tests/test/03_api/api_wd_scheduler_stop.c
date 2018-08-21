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

/* ************** CONSTANT PARAMETERS IN WD CREATION ******************** */

struct nanos_const_wd_definition_1
{
     nanos_const_wd_definition_t base;
     nanos_device_t devices[1];
};

struct nanos_const_wd_definition_1 const_data1 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   0,//__alignof__(section_data_1),
   0,
   1,0,NULL},
   {
      {
         nanos_smp_factory,
         &main__section_1_device_args
      }
   }
};

int main ( int argc, char **argv )
{
   int i;
   bool check = true;

   /* COMMON INFO */
   nanos_wd_props_t props = {
      .mandatory_creation = true,
      .tied = false};
   
   omp_init_lock( &mylock );

   // Repeat the test NUM_RUNS times
   nanos_wd_dyn_props_t dyn_props = {0};
   int testNumber;
   for ( testNumber = 0; testNumber < NUM_RUNS; ++testNumber ) {
      NANOS_SAFE( nanos_stop_scheduler() );
      A = 0;
      nanos_wd_t wd[4] = { NULL, NULL, NULL, NULL };

      /* Creating section 1 wd */
      main__section_1_data_t *section_data_1 = NULL;
      const_data1.base.data_alignment = __alignof__(section_data_1);
      NANOS_SAFE( nanos_create_wd_compact ( &wd[0], &const_data1.base, &dyn_props, sizeof(section_data_1), (void **) &section_data_1,
                                nanos_current_wd(), NULL, NULL ) );
   
      NANOS_SAFE( nanos_submit( wd[0],0,0,0 ) );
      
      NANOS_SAFE( nanos_wait_until_threads_paused() );
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
      NANOS_SAFE( nanos_start_scheduler() );
      NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
      if ( A != 1 ) {
         fprintf( stderr, "[TEST] A is not 1\n" );
         check = false;
         break;
      }
      // Make sure all threads are unpaused before running the next test
      NANOS_SAFE( nanos_wait_until_threads_unpaused() );
   }

   omp_destroy_lock( &mylock );

   fprintf(stderr, "%s : %s\n", argv[0], check ? "  successful" : "unsuccessful");
   if (check) { return 0; } else { return -1; }
}

