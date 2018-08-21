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
test_generator=gens/api-omp-generator
</testinfo>
*/

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#include "nanos.h"
#include "omp.h"

omp_lock_t mylock = 0;

void first()
{
   omp_set_lock( &mylock );
   printf("first task!\n");
   fflush(stdout);
   omp_unset_lock( &mylock );
}

nanos_smp_args_t test_device_arg_1 = { first };

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
   1,
   0,
   1,0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_1
      }
   }
};

int test_single_lock()
{
   int i;

   omp_init_lock( &mylock );

   nanos_wd_dyn_props_t dyn_props = {0};

   for ( i=0; i < 10; i++ ) {
      nanos_wd_t wd1=0;

      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1, &dyn_props, 0, NULL, nanos_current_wd(), NULL, NULL ) );

      NANOS_SAFE( nanos_submit( wd1,1,0,0 ) );
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   omp_destroy_lock( &mylock );

   return 0;
}

int count = 0;
omp_nest_lock_t mynlock;

void second()
{
   omp_set_nest_lock( &mynlock );
   omp_set_nest_lock( &mynlock );
   omp_set_nest_lock( &mynlock );
   omp_set_nest_lock( &mynlock );
   omp_set_nest_lock( &mynlock );
   omp_set_nest_lock( &mynlock );
   count++;
   omp_unset_nest_lock( &mynlock );
   omp_unset_nest_lock( &mynlock );
   omp_unset_nest_lock( &mynlock );
   omp_unset_nest_lock( &mynlock );
   omp_unset_nest_lock( &mynlock );
   omp_unset_nest_lock( &mynlock );
}

void third()
{
   if ( omp_test_nest_lock( &mynlock) ) {
      count++;
      omp_unset_nest_lock( &mynlock );
   } else {
      omp_set_nest_lock( &mynlock );
      count++;
      omp_unset_nest_lock( &mynlock );
   }
}

nanos_smp_args_t test_device_arg_2 = { second };
nanos_smp_args_t test_device_arg_3 = { third };

/* ************** CONSTANT PARAMETERS IN WD CREATION ******************** */

struct nanos_const_wd_definition_1 const_data2 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   1,
   0,
   1,0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_2
      }
   }
};
struct nanos_const_wd_definition_1 const_data3 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   1,
   0,
   1,0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_3
      }
   }
};

int test_nest_lock()
{
   int i;

   omp_init_nest_lock( &mynlock );

   nanos_wd_dyn_props_t dyn_props = {0};

   for ( i = 0; i < 100; i++ ) {
      nanos_wd_t wd1=0;
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data2.base, &dyn_props,  0, NULL, nanos_current_wd(), NULL, NULL ) );

      NANOS_SAFE( nanos_submit( wd1,1,0,0 ) );

      wd1=0;
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data3.base, &dyn_props, 0, NULL, nanos_current_wd(), NULL, NULL ) );

      NANOS_SAFE( nanos_submit( wd1,1,0,0 ) );
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   omp_destroy_nest_lock( &mynlock );
   if ( count == 200 ) return 0;
   return count;
}

int main ( int argc, char **argv )
{
   int result = 0;
   result += test_single_lock();
   if ( result > 0 ) printf("Error in test_single_lock. value: %d\n",result);
   result += test_nest_lock();
   if ( result > 0 ) printf("Error in test_nest_lock. value: %d\n",result);
   return result;
}

