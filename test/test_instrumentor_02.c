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
#include <nanos.h>
#include <alloca.h>
#define GENERIC_INSTRUMENTOR_API

// compiler: outlined function arguments
typedef struct {
   int value;
} main__task_1_data_t;

typedef struct {
   int value;
} main__task_2_data_t;

// compiler: outlined function
void main__task_1 ( void *args )
{
#ifdef GENERIC_INSTRUMENTOR_API
   nanos_event_t event_ini;
   event_ini.type = BURST_START;
   event_ini.info.burst.key = 133;
   event_ini.info.burst.value = 12345;

   nanos_instrument_events ( 1, &event_ini );
#else
   nanos_instrument_enter_burst( 133, 12345 );
#endif

#ifdef GENERIC_INSTRUMENTOR_API
   nanos_event_t event_create;
   event_create.type = PTP_END;
   event_create.info.ptp.domain = (nanos_event_domain_t) 0;
   event_create.info.ptp.id = (nanos_event_id_t) 1;

   event_create.info.ptp.nkvs = 2;
   event_create.info.ptp.keys = (nanos_event_key_t *) alloca ( sizeof(nanos_event_key_t) * event_create.info.ptp.nkvs );
   event_create.info.ptp.values = (nanos_event_value_t *) alloca ( sizeof(nanos_event_value_t) * event_create.info.ptp.nkvs );

   event_create.info.ptp.keys[0] = 110; // PTPBase + 110
   event_create.info.ptp.keys[1] = 111; // PTPBase + 111
   event_create.info.ptp.values[0] = 1133;
   event_create.info.ptp.values[1] = 3311;

   nanos_instrument_events ( 1, &event_create );
#else
#endif

   main__task_1_data_t *hargs = (main__task_1_data_t * ) args;

   usleep ( hargs->value );
   nanos_yield();
   usleep ( hargs->value );

#ifdef GENERIC_INSTRUMENTOR_API
   /* structure */
   nanos_event_t event_point;

   /* initialization */
   event_point.type = POINT;
   event_point.info.point.nkvs = 3;
   event_point.info.point.keys = (nanos_event_key_t * ) alloca ( sizeof(nanos_event_key_t) * event_point.info.point.nkvs );
   event_point.info.point.values = (nanos_event_value_t * ) alloca ( sizeof(nanos_event_value_t) * event_point.info.point.nkvs );
   event_point.info.point.keys[0] = 100;  // PointBase + 100
   event_point.info.point.keys[1] = 101;  // PointBase + 101
   event_point.info.point.keys[2] = 102;  // PointBase + 102
   event_point.info.point.values[0] = 2200; 
   event_point.info.point.values[1] = 2201; 
   event_point.info.point.values[2] = 2202; 

   /* create events */
   nanos_instrument_events ( 1, &event_point );
#else
   unsigned int nkvs = 3;
   nanos_event_key_t *keys = (nanos_event_key_t * ) alloca ( sizeof(nanos_event_key_t) * nkvs );
   nanos_event_value_t *values = (nanos_event_value_t * ) alloca ( sizeof(nanos_event_value_t) * nkvs );

   /* insert values in vectors keys and values */
   keys[0] = 200;  // PointBase + 200
   keys[1] = 201;  // PointBase + 201 
   keys[2] = 202;  // PointBase + 202 = 9802
   values[0] = 3200; 
   values[1] = 3201; 
   values[2] = 3202; 

   /* create event */
   nanos_instrument_point_event ( nkvs, keys, values );
#endif

   usleep ( hargs->value );
   nanos_yield();
   usleep ( hargs->value );

#ifdef GENERIC_INSTRUMENTOR_API
   nanos_event_t event_fini;
   event_fini.type = BURST_END;
   event_fini.info.burst.key = 133;
   event_fini.info.burst.value = 12345;

   nanos_instrument_events ( 1, &event_fini );
#else
   nanos_instrument_leave_burst( 133, 12345 );
#endif

}
void main__task_2 ( void *args )
{
   main__task_2_data_t *hargs = (main__task_2_data_t * ) args;

   nanos_yield();
   usleep ( hargs->value );
   nanos_yield();

}
// compiler: smp device for main__loop_1 function
nanos_smp_args_t main__task_1_device_args = { main__task_1 };
nanos_smp_args_t main__task_2_device_args = { main__task_2 };

int main ( int argc, char **argv )
{
   int i;

   for ( i = 0; i < 10; i++ ) {
      nanos_wd_t wd = NULL;
      nanos_device_t main__task_2_device[1] = { NANOS_SMP_DESC( main__task_2_device_args ) };
      main__task_2_data_t *task_data = NULL;
      nanos_wd_props_t props = {
         .mandatory_creation = true,
         .tied = false,
         .tie_to = false
      };

      NANOS_SAFE( nanos_create_wd ( &wd, 1, main__task_2_device , sizeof( main__task_2_data_t ),
                                    (void **) &task_data, nanos_current_wd(), &props , 0, NULL ));

      task_data->value = 100;

      NANOS_SAFE( nanos_submit( wd,0,0,0 ) );

   }

   nanos_wd_t wd = NULL;
   nanos_device_t main__task_1_device[1] = { NANOS_SMP_DESC( main__task_1_device_args ) };
   main__task_1_data_t *task_data = NULL;
   nanos_wd_props_t props = {
      .mandatory_creation = true,
      .tied = false,
      .tie_to = false
   };

   NANOS_SAFE( nanos_create_wd ( &wd, 1, main__task_1_device , sizeof( main__task_1_data_t ),
                                    (void **) &task_data, nanos_current_wd(), &props , 0, NULL ));

   task_data->value = 100;

   NANOS_SAFE( nanos_submit( wd,0,0,0 ) );
#ifdef GENERIC_INSTRUMENTOR_API
   nanos_event_t event_create;
   event_create.type = PTP_START;
   event_create.info.ptp.domain = (nanos_event_domain_t) 0;
   event_create.info.ptp.id = (nanos_event_id_t) 1;

   event_create.info.ptp.nkvs = 2;
   event_create.info.ptp.keys = (nanos_event_key_t *) alloca ( sizeof(nanos_event_key_t) * event_create.info.ptp.nkvs );
   event_create.info.ptp.values = (nanos_event_value_t *) alloca ( sizeof(nanos_event_value_t) * event_create.info.ptp.nkvs );

   event_create.info.ptp.keys[0] = 110; // PTPBase + 110
   event_create.info.ptp.keys[1] = 111; // PTPBase + 111
   event_create.info.ptp.values[0] = 1133;
   event_create.info.ptp.values[1] = 3311;

   nanos_instrument_events ( 1, &event_create );
#else
#endif

   NANOS_SAFE( nanos_wg_wait_completation( nanos_current_wd() ) );

   return 0; 
}

