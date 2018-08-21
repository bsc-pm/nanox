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
test_generator=gens/api-generator
test_exec_fail=yes
</testinfo>
*/

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <nanos.h>

#define FIRST 10
#define SECOND 10


typedef struct {
   int a;
   int *b;
} my_args;

int base = 0;

void first( void *ptr );
void second( void *ptr );

nanos_smp_args_t TASK_1 = { first };
nanos_smp_args_t TASK_2 = { second };
void submit_task( nanos_smp_args_t task, int intarg, int* text );

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
      .tied = true},
   __alignof__(my_args),
   2,
   1,
   2,NULL},
   {
      {
         nanos_smp_factory,
         0
      }
   }
};

void first( void *ptr )
{
   int i,j;
   my_args local;
   nanos_copy_value( (void *)&local.a, 0, nanos_current_wd() );
   nanos_get_addr( 1, (void **)&local.b, nanos_current_wd() );

   for (i=0; i < local.a; i++) {
      submit_task( TASK_2, i, local.b );
      NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
      printf( "Checking in level 1 task...  " );
      for (j=0; j < 10; j++) {
         if ( local.b[j] != ( base + 1 + i ) ) {
            printf("FAIL local.b[%d] was '%i' instead of '%i'\n",i,local.b[i],( base + 1 + i ));
            abort();
         }
      }
      printf("PASS\n");
   }

}

void second( void *ptr )
{
   int i;

   my_args local;
   nanos_copy_value( &local.a, 0, nanos_current_wd() );
   nanos_get_addr( 1, (void **)&local.b, nanos_current_wd() );

   printf( "Checking for copy-in correctness...  " );
   for (i=0; i < 10; i++) {
      if ( local.b[i] != ( base + local.a ) ) {
         printf("FAIL local.b[%d] was '%i' instead of '%i'\n",i,local.b[i],( base + local.a ));
         abort();
      }
      local.b[i]++;
   }
   printf("PASS\n");
}

void submit_task( nanos_smp_args_t task, int intarg, int* text )
{
   my_args* args = 0;
   nanos_copy_data_t *cd = 0;

   nanos_wd_t wd1=0;
   nanos_wd_dyn_props_t dyn_props = {0};
   const_data1.devices[0].arg = &task;
   nanos_region_dimension_internal_t *dims = 0;
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof(my_args), (void**)&args, nanos_current_wd(), &cd, &dims ) );

   args->a = intarg;
   args->b = text;

   dims[0] = ( nanos_region_dimension_internal_t ){ sizeof(args->a), 0, sizeof(args->a)};
   dims[1] = ( nanos_region_dimension_internal_t ){ sizeof(int)*10, 0, sizeof(int)*10};

   cd[0] = (nanos_copy_data_t) {(void*)&(args->a), NANOS_PRIVATE, {true, false}, 1, &dims[0], 0};
   cd[1] = (nanos_copy_data_t) {(void*)args->b, NANOS_SHARED, {true, true}, 1, &dims[1], 0}; 

   NANOS_SAFE( nanos_submit( wd1,0,0,0 ) );
}


int main ( int argc, char **argv )
{
   int i;
   int text[10] = {0,0,0,0,0,0,0,0,0,0};
   int* dummy1 = text;

   submit_task( TASK_1, FIRST, dummy1 );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   base = FIRST;

   printf( "Checking for copy-back correctness..." );
   for ( i = 0; i < 10; i++ ) {
      if ( dummy1[i] != FIRST ) {
         printf("  FAIL text[%i] should be %i and is %i\n",i,FIRST,dummy1[i]);
         exit(1);
      }
   }
   printf("  PASS\n");

   submit_task( TASK_1, SECOND, dummy1 );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   printf( "Checking for copy-back correctness..." );
   for ( i = 0; i < 10; i++ ) {
      if ( dummy1[i] != FIRST+SECOND ) {
         printf("  FAIL text[%i] should be %i and is %i\n",i,FIRST+SECOND,dummy1[i]);
         exit(1);
      }
   }
   printf("  PASS\n");

   return 0;
}

