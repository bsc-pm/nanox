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

typedef struct {
   int a;
   char *b;
} my_args;

void first( void *ptr );
void first( void *ptr )
{
   int i;

   my_args *args = (my_args *)ptr;
   my_args local;
   nanos_copy_value( &local.a, 0, nanos_current_wd() );
   nanos_get_addr( 1, (void **)&local.b, nanos_current_wd() );

   if ( args->a != local.a ) {
      printf( "Error private argument is incorrect, %d in args and %d through the copies  FAIL\n", args->a, local.a );
      abort();
   } else {
      printf( "Checking private argument ...          PASS\n" );
   }

   printf(    "Checking for shared argument ...");

   if ( strcmp(args->b, local.b) == 0) {
      printf(                                 "       PASS\n");
   } else {
      printf(                                 "       FAIL\n");
      printf( "Argument is '%s' while the copy is '%s'\n", args->b, local.b );
      abort();
   }
   for ( i = 0; i < 9; i++ )
      local.b[i] = '9'-i;
}

void second( void *ptr );
void second( void *ptr )
{
   my_args *args = (my_args *)ptr;
   my_args local;
   nanos_copy_value( &local.a, 0, nanos_current_wd() );
   nanos_get_addr( 1, (void **)&local.b, nanos_current_wd() );

   if ( args->a != local.a ) {
      printf( "Error private argument is incorrect, %d in args and %d through the copies  FAIL\n", args->a, local.a );
      abort();
   } else {
      printf( "Checking private argument ...          PASS\n" );
   }

}




nanos_smp_args_t test_device_arg_1 = { first };
nanos_smp_args_t test_device_arg_2 = { second };

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
   __alignof__(my_args),
   2,
   1,
   2,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_1
      }
   }
};
struct nanos_const_wd_definition_1 const_data2 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   __alignof__(my_args),
   2,
   1,
   2,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_2
      }
   }
};

int main ( int argc, char **argv )
{
   char text[10] = "123456789";
   char text2[10] = "987654321";
   char* dummy1 = text;
   
   my_args* args = 0;
   nanos_copy_data_t *cd = 0;

   nanos_wd_t wd1=0;
   nanos_wd_dyn_props_t dyn_props = {0};
   nanos_region_dimension_internal_t *dims = 0;
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof(my_args), (void**)&args, nanos_current_wd(), &cd, &dims) );

   args->a = 1;
   args->b = dummy1;

   dims[0] = (nanos_region_dimension_internal_t) {sizeof(args->a), 0, sizeof(args->a)};
   dims[1] = (nanos_region_dimension_internal_t) {sizeof(char)*10, 0, sizeof(char)*10};

   cd[0] = (nanos_copy_data_t) {(void*)&(args->a), NANOS_PRIVATE, {true, false}, 1, &dims[0], 0};
   cd[1] = (nanos_copy_data_t) {(void*)args->b, NANOS_SHARED, {true, true}, 1, &dims[1], 0}; 

   NANOS_SAFE( nanos_submit( wd1,0,0,0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   int i;

   for (i=0; i < 100; i++) {
      args = 0;
      cd = 0;
      wd1=0;
      dims = 0;
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data2.base, &dyn_props, sizeof(my_args), (void**)&args, nanos_current_wd(), &cd, &dims) );
      args->a = 1;
      args->b = dummy1;
      dims[0] = (nanos_region_dimension_internal_t) {sizeof(args->a), 0, sizeof(args->a)};
      dims[1] = (nanos_region_dimension_internal_t) {sizeof(char)*1, 0, sizeof(char)*1};
      cd[0] = (nanos_copy_data_t) {(void*)&(args->a), NANOS_PRIVATE, {true, false}, 1, &dims[0], 0};
      cd[1] = (nanos_copy_data_t) {(void*)args->b, NANOS_SHARED, {true, false}, 1, &dims[1], 0}; 

      NANOS_SAFE( nanos_submit( wd1,0,0,0 ) );

      args = 0;
      cd = 0;
      wd1=0;
      dims = 0;
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data2.base, &dyn_props, sizeof(my_args), (void**)&args, nanos_current_wd(), &cd, &dims) );
      args->a = 1;
      args->b = dummy1;
      dims[0] = (nanos_region_dimension_internal_t) {sizeof(args->a), 0, sizeof(args->a)};
      dims[1] = (nanos_region_dimension_internal_t) {sizeof(char)*10, 0, sizeof(char)*10};
      cd[0] = (nanos_copy_data_t) {(void*)&(args->a), NANOS_PRIVATE, {true, false}, 1, &dims[0], 0};
      cd[1] = (nanos_copy_data_t) {(void*)args->b, NANOS_SHARED, {true, false}, 1, &dims[1], 0}; 

      NANOS_SAFE( nanos_submit( wd1,0,0,0 ) );
   }


   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );



   if ( strcmp( text2, dummy1 ) == 0 ) {
      printf( "Checking for copy-back correctness...  PASS\n" );
   } else {
      printf( "Checking for copy-back correctness...  FAIL\n" );
      printf( "expecting '%s', copied back: '%s'\n", text2, dummy1 );
      return 1;
   }

   return 0;
}

