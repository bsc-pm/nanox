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
test_generator="gens/api-generator -d plain,regions,perfect-regions"
</testinfo>
*/

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <nanos.h>

int orderer = 0;

void first();
void first()
{
   orderer++;
   printf("first task!\n");
   fflush(stdout);
}

void second();
void second()
{
   if ( orderer != 1 ) {
      printf("Error, order of tasks not respected!\n");
      abort();
   }
   orderer++;
   printf("second task!\n");
   fflush(stdout);
}


nanos_smp_args_t test_device_arg_1 = { (void(*)())first };
nanos_smp_args_t test_device_arg_2 = { (void(*)())second };

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
   1,
   0,NULL},
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
   0,//__alignof__(section_data_1),
   0,
   1,
   0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_2
      }
   }
};

int main ( int argc, char **argv )
{
   int dep;
   nanos_region_dimension_t dimensions[1] = {{sizeof(int), 0, sizeof(int)}};
   nanos_data_access_t data_accesses[1] = {{&dep, {1,1,0,0,0}, 1, dimensions}};
   nanos_wd_t wd1=0;
   nanos_wd_dyn_props_t dyn_props = {0};
   const_data1.base.data_alignment = 1;
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, 0, NULL, nanos_current_wd(), NULL, NULL ) );
   NANOS_SAFE( nanos_submit( wd1, 1, data_accesses, 0 ) );


   nanos_wd_t wd2=0;
   const_data2.base.data_alignment = 1;
   NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data2.base, &dyn_props, 0, NULL, nanos_current_wd(), NULL, NULL ) );
   NANOS_SAFE( nanos_submit( wd2, 1, data_accesses,0 ) );


   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   if ( orderer != 2 ) {
      printf("Error: Dependencies have not been respected or a task(s) has not been executed.\n");
      return 1;
   }

   return 0;
}

