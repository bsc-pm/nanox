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
test_generator='gens/api-generator -a '--numa-cache-size=512''
</testinfo>
*/

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <nanos.h>

typedef struct {
   int a;
   char *b;
} my_args;

void first( void *ptr )
{
}

nanos_smp_args_t test_device_arg_1 = { first };

/* ************** CONSTANT PARAMETERS IN WD CREATION ******************** */
nanos_const_wd_definition_t const_data1 = 
{
   {
      .mandatory_creation = true,
      .tied = false,
      .tie_to = false,
      .priority = 0
   },
   __alignof__(my_args),
   2,
   1,
   {
      {
         nanos_smp_factory,
         0,//nanos_smp_dd_size,
         &test_device_arg_1
      }
   }
};

int main ( int argc, char **argv )
{
   char text[10] = "123456789";
   char text2[10] = "987654321";
   char* array = malloc(1024*sizeof(char));
   char* dummy1 = text;
   int i;
   
   for (i=0; i < 8; i++) {
      my_args* args = 0;
      nanos_copy_data_t *cd = 0;
   
      nanos_wd_t wd1=0;
      const_data1.devices[0].dd_size = nanos_smp_dd_size;
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1, sizeof(my_args), (void**)&args, nanos_current_wd(), &cd) );
   
      args->a = 1;
      args->b = &array[i*128];
   
      cd[0] = (nanos_copy_data_t) {(uint64_t)&(args->a), NANOS_PRIVATE, {true, false}, sizeof(args->a)};
      cd[1] = (nanos_copy_data_t) {(uint64_t)args->b, NANOS_SHARED, {true, false}, sizeof(char)*128}; 
   
      NANOS_SAFE( nanos_submit( wd1,0,0,0 ) );
   
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   return 0;
}

