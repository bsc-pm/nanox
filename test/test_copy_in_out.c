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

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <nanos.h>
#include <nanos_pe.h>

typedef struct {
   int a;
   char *b;
} my_args;

void check_hardcoded_copy_data();
void* aux_get_copies_addr( unsigned int );
nanos_sharing_t aux_get_sharing( unsigned int );

void first( void *ptr )
{
   int i;
//   check_hardcoded_copy_data ();

   my_args *args = (my_args *)ptr;
   my_args local;
   nanos_copy_value( &local.a, aux_get_copies_addr(0), aux_get_sharing(0), sizeof(local.a) );
   nanos_get_addr( aux_get_copies_addr(1), aux_get_sharing(1), (void **)&local.b );

   if ( args->a != local.a ) {
      printf( "Error private argument is incorrect, %d in args and %d through the copies  FAIL\n", args->a, local.a );
   } else {
      printf( "Checking private argument ...          PASS\n" );
   }

   printf(    "Checking for shared argument ...");

   if ( strcmp(args->b, local.b) == 0) {
   printf(                                    "       PASS\n");
   } else {
   printf(                                    "       FAIL\n");
   printf( "Argument is '%s' while the copy is '%s'\n", args->b, local.b );
   }

   for ( i = 0; i < 9; i++ )
      local.b[i] = '9'-i;
}

nanos_smp_args_t test_device_arg_1 = { first };

int main ( int argc, char **argv )
{
   char text[10] = "123456789";
   char text2[10] = "987654321";
   char* dummy1 = text;
   
   my_args* args = 0;
   nanos_wd_props_t props = {
     .mandatory_creation = true,
     .tied = false,
     .tie_to = false,
   };

   nanos_copy_data_t *cd = 0;

   nanos_wd_t wd1=0;
   nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_1) };
   NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_1, sizeof(my_args), (void**)&args, nanos_current_wd(), &props, 2, &cd) );

   args->a = 1;
   args->b = dummy1;

   cd[0] = (nanos_copy_data_t) {&(args->a), NX_PRIVATE, {true, false}, sizeof(args->a)};
   cd[1] = (nanos_copy_data_t) {args->b, NX_SHARED, {true, true}, sizeof(char)*10}; 

   NANOS_SAFE( nanos_submit( wd1,0,0,0 ) );

   NANOS_SAFE( nanos_wg_wait_completation( nanos_current_wd() ) );

   if ( strcmp( text2, dummy1 ) == 0 ) {
      printf( "Checking for copy-back correctness...  PASS\n" );
   } else {
      printf( "Checking for copy-back correctness...  FAIL\n" );
      printf( "expecting '%s', copied back: '%s'\n", text2, dummy1 );
   }

   return 0;
}

