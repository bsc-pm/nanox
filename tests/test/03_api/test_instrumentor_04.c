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
#include <alloca.h>


nanos_wd_dyn_props_t dyn_props = {0};

void sleep_100 ( void );
void sleep_100 ( void )
{
   nanos_event_t event;

   nanos_instrument_get_key ("user-funct-name", &(event.key));
   nanos_instrument_register_value ( &(event.value), "user-funct-name", "sleep_100", "Function: sleep_100", false );
   
   event.type = NANOS_BURST_START;
   nanos_instrument_events(1, &event);

   usleep ( 100 );
   nanos_yield();
   usleep ( 100 );
   nanos_yield();
   usleep ( 100 );

   event.type = NANOS_BURST_END;
   nanos_instrument_events(1, &event);
}
// compiler: outlined function arguments
typedef struct {
   int value;
} main__task_1_data_t;

typedef struct {
   int value;
} main__task_2_data_t;

void main__task_2 ( void *args );
void main__task_2 ( void *args )
{
   nanos_event_t event;

   nanos_instrument_get_key ("user-funct-name", &(event.key));
   nanos_instrument_register_value ( &(event.value), "user-funct-name", "task-2", "Function: main__task_2", false );
   
   event.type = NANOS_BURST_START;
   nanos_instrument_events(1, &event);

   main__task_2_data_t *hargs = (main__task_2_data_t * ) args;

   nanos_yield();
   usleep ( hargs->value );
   nanos_yield();

   event.type = NANOS_BURST_END;
   nanos_instrument_events(1, &event);
}
// compiler: smp device for main__task_2 function
nanos_smp_args_t main__task_2_device_args = { main__task_2 };

/* ************** CONSTANT PARAMETERS IN WD CREATION ******************** */
struct nanos_const_wd_definition_1
{
     nanos_const_wd_definition_t base;
     nanos_device_t devices[1];
};

struct nanos_const_wd_definition_1 const_data2 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   __alignof__(main__task_2_data_t),
   0,
   1,0,NULL},
   {
      {
         nanos_smp_factory,
         &main__task_2_device_args
      }
   }
};

// compiler: outlined function
void main__task_1 ( void *args );
void main__task_1 ( void *args )
{
   nanos_event_t event;

   nanos_instrument_get_key ("user-funct-name", &(event.key));
   nanos_instrument_register_value ( &(event.value), "user-funct-name", "task-1", "Function: main__task_1", false );
   
   event.type = NANOS_BURST_START;
   nanos_instrument_events(1, &event);

   int i;
   main__task_1_data_t *hargs = (main__task_1_data_t * ) args;

   usleep ( hargs->value );

   for (i = 0; i < 10; i++ ) {
      nanos_wd_t wd = NULL;
      main__task_2_data_t *task_data = NULL;

      NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_data2.base, &dyn_props, sizeof( main__task_2_data_t ),
                                    (void **) &task_data, nanos_current_wd(), NULL, NULL ));

      task_data->value = 1000;

      NANOS_SAFE( nanos_submit( wd,0,0,0 ) );
   }

   usleep ( hargs->value );
   sleep_100 ();
   usleep ( hargs->value );
   nanos_yield();
   usleep ( hargs->value );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   event.type = NANOS_BURST_END;
   nanos_instrument_events(1, &event);
}
// compiler: smp device for main__task_1 function
nanos_smp_args_t main__task_1_device_args = { main__task_1 };

/* ************** CONSTANT PARAMETERS IN WD CREATION ******************** */
struct nanos_const_wd_definition_1 const_data1 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   __alignof__(main__task_1_data_t),
   0,
   1,0,NULL},
   {
      {
         nanos_smp_factory,
         &main__task_1_device_args
      }
   }
};

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

