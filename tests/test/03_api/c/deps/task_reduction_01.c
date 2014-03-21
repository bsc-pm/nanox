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
test_generator=gens/api-generator
</testinfo>
*/

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <nanos.h>

#define NUM_TASKS 500

typedef struct { int *sh_val; } my_args;

void initialize(void *ptr)
{
   int *value = ((my_args *) ptr)->sh_val;
   *value = 0;

#ifdef VERBOSE
   printf("initialize task: %p set to %d\n",value,*value);
   fflush(stdout);
#endif
}

void increment(void *ptr)
{
   int *value = ((my_args *) ptr)->sh_val;
#if 0
   (*value)++;
#ifdef VERBOSE
   printf("increment task: %p set to %d\n",value,*value);
   fflush(stdout);
#endif

#else
   // instead of working with the original symbol, get a thread private one
   int *copy = NULL;
   nanos_task_reduction_get_thread_storage( value, (void **) &copy );

   // increment thread private value
   (*copy)++;
#ifdef VERBOSE
   printf("increment task (copy): %p set to %d\n",copy,*copy);
   fflush(stdout);
#endif
#endif
}

void print(void *ptr)
{
   int *value = ((my_args *) ptr)->sh_val;
   printf("print task: %p is %d\n",value,*value);
   fflush(stdout);
}

nanos_smp_args_t smp_initialize = { initialize };
nanos_smp_args_t smp_increment = { increment };
nanos_smp_args_t smp_print = { print };

/* ************** CONSTANT PARAMETERS IN WD CREATION ******************** */

struct nanos_const_wd_definition
{
     nanos_const_wd_definition_t base;
     nanos_device_t devices[1];
};

struct nanos_const_wd_definition const_init = 
{
   { 
      { .mandatory_creation = true, .tied = false},
      __alignof__(my_args), 0, 1, 0,NULL
   },
   { { nanos_smp_factory, &smp_initialize } }
};
struct nanos_const_wd_definition const_increment = 
{
   { 
      { .mandatory_creation = true, .tied = false},
      __alignof__(my_args), 0, 1, 0,NULL
   },
   { { nanos_smp_factory, &smp_increment } }
};
struct nanos_const_wd_definition const_print = 
{
   { 
      { .mandatory_creation = true, .tied = false},
      __alignof__(my_args), 0, 1, 0,NULL
   },
   { { nanos_smp_factory, &smp_print } }
};

nanos_wd_dyn_props_t dyn_props = {0};

void my_init ( void *p ) { *((int *)p) = 0 ; }
void my_red ( void *p, void *q ) { *((int *)p) = *((int *)p) + *((int *)q);  }

int main ( int argc, char **argv )
{
   int my_value;

   {
      // Initialize
      my_args *args = 0;
      nanos_wd_t wd = 0;

      // Create
      NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_init.base, &dyn_props, sizeof( my_args ), ( void ** )&args, nanos_current_wd(), NULL, NULL ) );

      args->sh_val = &my_value;

      // Submit
      nanos_region_dimension_t dimensions[1] = { {sizeof(my_value), 0, sizeof(my_value)} };
      nanos_data_access_t data_accesses[1] = {{&my_value, {0,1,0,0,0}, 1, dimensions, 0}};
      NANOS_SAFE( nanos_submit( wd,1,data_accesses,0 ) );
   }
   int j;
   for ( j = 0; j < NUM_TASKS; j++ ) {
      // Initialize
      my_args *args = 0;
      nanos_wd_t wd = 0;

      // Create
      nanos_task_reduction_register ( &my_value, sizeof(my_value), sizeof(my_value), &my_init, &my_red ) ;
      NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_increment.base, &dyn_props, sizeof( my_args ), ( void ** )&args, nanos_current_wd(), NULL, NULL ) );

      args->sh_val = &my_value;

      // Submit
      nanos_region_dimension_t dimensions[1] = { {sizeof(my_value), 0, sizeof(my_value)} };
      nanos_data_access_t data_accesses[1] = {{&my_value, {1,1,0,1,0}, 1, dimensions, 0}};
      NANOS_SAFE( nanos_submit( wd,1,data_accesses,0 ) );
   }

   {
      // Initialize
      my_args *args = 0;
      nanos_wd_t wd = 0;

      // Create
      NANOS_SAFE( nanos_create_wd_compact ( &wd, &const_print.base, &dyn_props, sizeof( my_args ), ( void ** )&args, nanos_current_wd(), NULL, NULL ) );

      args->sh_val = &my_value;

      // Submit
      nanos_region_dimension_t dimensions[1] = { {sizeof(my_value), 0, sizeof(my_value)} };
      nanos_data_access_t data_accesses[1] = {{&my_value, {1,1,0,0,0}, 1, dimensions, 0}};
      NANOS_SAFE( nanos_submit( wd,1,data_accesses,0 ) );
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   fprintf(stderr,"Final result", my_value );

   if( my_value == NUM_TASKS ) return 0;
   else return 1;
}


