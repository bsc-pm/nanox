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
</testinfo>
*/

#include <stdio.h>
#include <nanos.h>

// compiler: outlined function arguments
typedef struct {
   int value;
} main__task_1_data_t;

// compiler: outlined function
void main__task_1 ( void *args );
void main__task_1 ( void *args )
{
   main__task_1_data_t *hargs = (main__task_1_data_t * ) args;

   usleep ( hargs->value );
}

// compiler: smp device for main__loop_1 function
nanos_smp_args_t main__task_1_device_args = { main__task_1 };

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
   __alignof__( main__task_1_data_t),
   0,
   1,0,NULL},
   {
      {
         nanos_smp_factory,
         &main__task_1_device_args
      }
   }
};

nanos_wd_dyn_props_t dyn_props = {0};

int main ( int argc, char **argv )
{

      nanos_wd_t wd = NULL;
      main__task_1_data_t *task_data = NULL;

      NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_data1.base, &dyn_props, sizeof( main__task_1_data_t ),
                                    (void **) &task_data, nanos_current_wd(), NULL, NULL ));

      task_data->value = 100;

      NANOS_SAFE( nanos_submit( wd,0,0,0 ) );
      NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

      return 0; 
}

