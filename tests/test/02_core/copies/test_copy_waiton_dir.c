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
exec_versions="smp_shared_mem smp_private_mem"

declare test_ENV_smp_private_mem="NX_SMP_PRIVATE_MEMORY=true"

</testinfo>
*/

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <nanos.h>

typedef struct {
   //int a;
   int *a;
   char *b;
} my_args;

void first( void *ptr );
void first( void *ptr )
{
   int i;

   my_args *args = (my_args *)ptr;
   my_args local;
   nanos_get_addr( 0, (void **)&local.a, nanos_current_wd() );
   nanos_get_addr( 1, (void **)&local.b, nanos_current_wd() );

   if ( *(local.a) != 42 ) {
      printf( "Error in argument is incorrect, received %d and it should be 42  FAIL\n", *(local.a) );
      abort();
   } else {
      printf( "Checking in argument ...               PASS\n" );
   }

   printf(    "Checking for inout argument ...");

   if ( strcmp(args->b, local.b) == 0) {
      printf(                                "        PASS\n");
   } else {
      printf(                                "        FAIL\n");
      printf( "Argument is '%s' while the copy is '%s'\n", args->b, local.b );
      abort();
   }
   for ( i = 0; i < 9; i++ )
      local.b[i] = '9'-i;
}

void second( void *ptr );
void second( void *ptr )
{
   int i;

   my_args *args = (my_args *)ptr;
   my_args local;
   nanos_get_addr( 0, (void **)&local.a, nanos_current_wd() );
   nanos_get_addr( 1, (void **)&local.b, nanos_current_wd() );

   if ( *(local.a) != 24 ) {
      printf( "Error in argument is incorrect, received %d and it should be 24  FAIL\n", *(local.a) );
      abort();
   } else {
      printf( "Checking in argument ...               PASS\n" );
   }

   printf(    "Checking for inout argument ...");

   if ( strcmp(args->b, local.b) == 0) {
      printf(                                "        PASS\n");
   } else {
      printf(                                "        FAIL\n");
      printf( "Argument is '%s' while the copy is '%s'\n", args->b, local.b );
      abort();
   }
   for ( i = 0; i < 9; i++ )
      local.b[i] = '9'-i;
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

int int_arg_1 = 42;
int int_arg_2 = 24;

int main ( int argc, char **argv )
{
   char text[10] = "123456789";
   char text2[10] = "987654321";
   char* dummy1 = text;
   int i;

   
   my_args* args = 0;
   nanos_copy_data_t *cd = 0;

   nanos_wd_t wd1=0;
   nanos_wd_dyn_props_t dyn_props = {0};
   nanos_region_dimension_internal_t *dims = 0;
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof(my_args), (void**)&args, nanos_current_wd(), &cd, &dims) );

   args->a = &int_arg_1;
   args->b = dummy1;
   nanos_region_dimension_t dimensions[1] = {{strlen(args->b)+1, 0, strlen(args->b)+1}};
   nanos_data_access_t data_accesses[1] = {{args->b, {1,1,0,0,0}, 1, dimensions}};

   dims[0] = (nanos_region_dimension_internal_t) {sizeof(int), 0, sizeof(int)};
   dims[1] = (nanos_region_dimension_internal_t) {sizeof(char)*10, 0, sizeof(char)*10};

   cd[0] = (nanos_copy_data_t) {(void*)args->a, NANOS_SHARED, {true, false}, 1, &dims[0], 0};
   cd[1] = (nanos_copy_data_t) {(void*)args->b, NANOS_SHARED, {true, true}, 1, &dims[1], 0}; 

   NANOS_SAFE( nanos_submit( wd1,1,data_accesses,0 ) );

   nanos_region_dimension_t dimensions1[1] = {{strlen(dummy1)+1, 0, strlen(dummy1)+1}};
   nanos_data_access_t data_accesses1[1] = {{dummy1, {1,1,0,0,0}, 1, dimensions1}};
   
   //FIXME: wait_on is not working... will enable this after fixing it.
   //NANOS_SAFE( nanos_wait_on( 1, data_accesses1 ) );
   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   for ( i = 0; i < 9; i++ )
      text[i] = '1'+i;

   args = 0;
   cd = 0;
   wd1=0;
   dims=0;
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data2.base, &dyn_props, sizeof(my_args), (void**)&args, nanos_current_wd(), &cd, &dims) );

   args->a = &int_arg_2;
   args->b = dummy1;
   nanos_region_dimension_t dimensions2[1] = {{strlen(args->b)+1, 0, strlen(args->b)+1}};
   nanos_data_access_t data_accesses2[1] = {{args->b, {1,1,0,0,0}, 1, dimensions2}};
   dims[0] = (nanos_region_dimension_internal_t) {sizeof(int), 0, sizeof(int)};
   dims[1] = (nanos_region_dimension_internal_t) {sizeof(char)*10, 0, sizeof(char)*10};

   cd[0] = (nanos_copy_data_t) {(void*)args->a, NANOS_SHARED, {true, false}, 1, &dims[0], 0};
   cd[1] = (nanos_copy_data_t) {(void*)args->b, NANOS_SHARED, {true, true}, 1, &dims[1], 0}; 


   NANOS_SAFE( nanos_submit( wd1,1,data_accesses2,0 ) );

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

