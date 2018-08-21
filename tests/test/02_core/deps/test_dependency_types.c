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

typedef struct {
   int* p_i;
} my_args;

typedef struct {
   int* p_i;
   int  index;
} my_args2;

typedef struct {
   int* p_i;
   int* p_result;
   int  index;
} my_args3;


void first(void *ptr);
void first(void *ptr)
{
   int *i = ((my_args *) ptr)->p_i;
   *i = 0;
#ifdef VERBOSE
   printf("first task: %p set to %d\n",i,*i);
   fflush(stdout);
#endif
}

void second(void *ptr);
void second(void *ptr)
{
   int *i = ((my_args *) ptr)->p_i;
   (*i)++;
#ifdef VERBOSE
   printf("successor: %p set to %d\n",i,*i);
   fflush(stdout);
#endif
}

void third(void *ptr);
void third(void *ptr)
{
   int j;
   for ( j = 0; j < 100; j++ ) {
      int *i = ((my_args *) ptr)[j].p_i;
      (*i)++;
   }
}

void fourth(void *ptr);
void fourth(void *ptr)
{
   int *i = ((my_args *) ptr)[0].p_i;
   int *j = ((my_args *) ptr)[1].p_i;
   (*j) = *i;
}

void fifth(void *ptr);
void fifth(void *ptr)
{
#ifdef VERBOSE
   printf("fifth\n");
   fflush(stdout);
#endif
   int *red_array = ((my_args2 *) ptr)->p_i;
   int index = ((my_args2 *) ptr)->index;
   red_array[index]++;
}

void sixth(void *ptr);
void sixth(void *ptr)
{
#ifdef VERBOSE
   printf("sixth\n");
   fflush(stdout);
#endif
   int i;
   int *red_array = ((my_args2 *) ptr)->p_i;
   int size = ((my_args2 *) ptr)->index;
   for ( i = 0; i < size; i++ )
   {
      red_array[i]++;
   }
}

void seventh(void *ptr);
void seventh(void *ptr)
{
#ifdef VERBOSE
   printf("seventh\n");
   fflush(stdout);
#endif
   int *array = ((my_args3 *) ptr)->p_i;
   int *result = ((my_args3 *) ptr)->p_result;
   int index = ((my_args3 *) ptr)->index;
   if ( array[index] != 2 ) {
      *result = -1;
   }
}

void eighth(void *ptr);
void eighth(void *ptr)
{
#ifdef VERBOSE
   printf("eighth\n");
   fflush(stdout);
#endif
}




nanos_smp_args_t test_device_arg_1 = { first };
nanos_smp_args_t test_device_arg_2 = { second };
nanos_smp_args_t test_device_arg_3 = { third };
nanos_smp_args_t test_device_arg_4 = { fourth };
nanos_smp_args_t test_device_arg_5 = { fifth };
nanos_smp_args_t test_device_arg_6 = { sixth };
nanos_smp_args_t test_device_arg_7 = { seventh };
nanos_smp_args_t test_device_arg_8 = { eighth };

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
   __alignof__(my_args),
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

struct nanos_const_wd_definition_1 const_data3 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   __alignof__(my_args),
   0,
   1,
   0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_3
      }
   }
};

struct nanos_const_wd_definition_1 const_data4 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   __alignof__(my_args),
   0,
   1,
   0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_4
      }
   }
};

struct nanos_const_wd_definition_1 const_data5 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   __alignof__(my_args2),
   0,
   1,
   0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_5
      }
   }
};

struct nanos_const_wd_definition_1 const_data6 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   __alignof__(my_args2),
   0,
   1,
   0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_6
      }
   }
};

struct nanos_const_wd_definition_1 const_data7 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   __alignof__(my_args3),
   0,
   1,
   0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_7
      }
   }
};

struct nanos_const_wd_definition_1 const_data8 = 
{
   {{
      .mandatory_creation = true,
      .tied = false},
   __alignof__(my_args2),
   0,
   1,
   0,NULL},
   {
      {
         nanos_smp_factory,
         &test_device_arg_8
      }
   }
};


nanos_wd_dyn_props_t dyn_props = {0};

bool single_dependency();
bool single_dependency()
{
   int my_value;
   int * dep_addr = &my_value;
   my_args *args1=0;
   nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses1[1] = {{&my_value, {0,1,0,0,0}, 1, dimensions1}};
   nanos_wd_t wd1=0;
   const_data1.base.data_alignment = __alignof__(my_args);
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof( my_args ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
   args1->p_i = dep_addr;
   NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );

   my_args *args2=0;
   nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses2[1] = {{&my_value, {1,1,0,0,0}, 1, dimensions2}};
   nanos_wd_t wd2 = 0;
   const_data2.base.data_alignment = __alignof__(my_args);
   NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data2.base, &dyn_props, sizeof( my_args ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
   args2->p_i = dep_addr;
   NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   
   return (my_value == 1);
}

bool single_inout_chain();
bool single_inout_chain()
{
   int i;
   int my_value;
   int * dep_addr = &my_value;
   my_args *args1=0;
   nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses1[1] = {{&my_value, {0,1,0,0,0}, 1, dimensions1, 0}};
   nanos_wd_t wd1=0;
   const_data1.base.data_alignment = __alignof__(args1);
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof( args1 ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
   args1->p_i = dep_addr;
   NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );

   for ( i = 0; i < 100; i++ ) {
      my_args *args2=0;
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{&my_value, {1,1,0,0,0}, 1, dimensions2, 0}};
      nanos_wd_t wd2 = 0;
      
      const_data2.base.data_alignment = __alignof__(my_args);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data2.base, &dyn_props, sizeof( my_args ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = dep_addr;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   
   return (my_value == 100);
}

bool multiple_inout_chains();
bool multiple_inout_chains()
{
   int i, j;
   int size = 10;
   int __attribute__( ( aligned( 16 ) ) ) my_value[size];

   for ( i = 0; i < size; i++ ) {
      int * dep_addr = &my_value[i];
      my_args *args1=0;
      nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value[i]), 0, sizeof(my_value[i])}};
      nanos_data_access_t data_accesses1[1] = {{&my_value[i], {0,1,0,0,0}, 1, dimensions1, 0}};
      nanos_wd_t wd1=0;
      const_data1.base.data_alignment = __alignof__(my_args);
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof( my_args ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
      args1->p_i = dep_addr;
      NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );

      for ( j = 0; j < size; j++ ) {
         my_args *args2=0;
         nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value[i]), 0, sizeof(my_value[i])}};
         nanos_data_access_t data_accesses2[1] = {{&my_value[i], {1,1,0,0,0}, 1, dimensions2, 0}};
         nanos_wd_t wd2 = 0;
         
         const_data2.base.data_alignment = __alignof__(my_args);
         NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data2.base, &dyn_props, sizeof( my_args ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
         args2->p_i = dep_addr;
         NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
      }
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   for ( i = 0; i < size; i++ ) {
      if ( my_value[i] != size ) return false;
   }
   return true;
}

bool multiple_predecessors();
bool multiple_predecessors()
{
   int j;
   int size=100;
   int __attribute__( ( aligned( 128 ) ) ) my_value[size];

   for ( j = 0; j < size; j++ ) {
      int * dep_addr1 = &my_value[j];
      my_args *args1=0;
      nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value[j]), 0, sizeof(my_value[j])}};
      nanos_data_access_t data_accesses1[1] = {{&my_value[j], {0,1,0,0,0}, 1, dimensions1, 0}};
      nanos_wd_t wd1 = 0;

      const_data1.base.data_alignment = __alignof__(my_args);
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof( my_args ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
      args1->p_i = dep_addr1;
      NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );
   }

   nanos_region_dimension_t dimensions2[size][1];
   nanos_data_access_t data_accesses2[size];
   int *dep_addr2[size];
   my_args *args2=0;
   for ( j = 0; j < size; j++ ) {
      dep_addr2[j] = &my_value[j];
      nanos_region_dimension_t dim = {sizeof(my_value[j]), 0, sizeof(my_value[j])};
      dimensions2[j][0] = dim;
      nanos_data_access_t da = {&my_value[j], {1,1,0,0,0}, 1, dimensions2[j],0};
      data_accesses2[j] = da;
   }

   nanos_wd_t wd2=0;
   const_data3.base.data_alignment = __alignof__(my_args);
   NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data3.base, &dyn_props, sizeof( my_args )*size, ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
   for ( j = 0; j < size; j++);
   for ( j = 0; j < size; j++)
      args2[j].p_i = dep_addr2[j];
   NANOS_SAFE( nanos_submit( wd2,size,data_accesses2,0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   for ( j = 0; j < size; j++ ) {
      if ( my_value[j] != 1 ) return false;
   }
   return true;
}

bool multiple_antidependencies();
bool multiple_antidependencies()
{
   int j;
   int my_value=1500;
   int __attribute__( ( aligned( 128 ) ) ) my_reslt[100];

   for ( j = 0; j < 100; j++ ) {
      int * dep_addr1 = &my_value;
      int * reslt_addr =&my_reslt[j];
      my_args *args1=0;
      nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses1[1] = {{&my_value, {1,0,0,0,0}, 1, dimensions1, 0}};

      nanos_wd_t wd1 = 0;
      const_data4.base.data_alignment = __alignof__(my_args);
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data4.base, &dyn_props, sizeof( my_args )*2, ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
      args1[0].p_i = dep_addr1;
      args1[1].p_i = reslt_addr;
      NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );
   }

   int *dep_addr2 = &my_value;
   nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses2[1] = {{&my_value, {1,1,0,0,0}, 1, dimensions2, 0}};
   my_args *args2=0;

   nanos_wd_t wd2=0;
   const_data2.base.data_alignment = __alignof__(my_args);
   NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data2.base, &dyn_props, sizeof( my_args ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
   args2->p_i = dep_addr2;
   NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   for ( j = 0; j < 100; j++ ) {
      if ( my_reslt[j] != 1500 ) return false;
   }
   if (my_value != 1501) return false;
   return true;
}

bool out_dep_chain();
bool out_dep_chain()
{
   int i;
   int my_value;
   int * dep_addr = &my_value;

   for ( i = 0; i < 100; i++ ) {
      my_args *args2=0;
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{&my_value, {0,1,0,0,0}, 1, dimensions2, 0}};
      nanos_wd_t wd2 = 0;
      const_data1.base.data_alignment = __alignof__(my_args);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data1.base, &dyn_props, sizeof( my_args ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = dep_addr;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }

   int input=500;
   int * input_addr = &input;
   nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses1[1] = {{&my_value, {0,1,0,0,0}, 1, dimensions1, 0}};
   my_args *args1=0;
   nanos_wd_t wd1=0;
   const_data4.base.data_alignment = __alignof__(my_args);
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data4.base, &dyn_props, sizeof( my_args )*2, ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
   args1[0].p_i = input_addr;
   args1[1].p_i = dep_addr;
   NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   
   return (my_value == 500);
}

bool wait_on_test();
bool wait_on_test()
{
   int j;
   int size=10;
   int __attribute__( ( aligned( 16 ) ) ) my_value[size];

   for ( j = 0; j < size; j++ ) {
      my_value[j] = 500;
      int * dep_addr1 = &my_value[j];
      my_args *args1=0;
      nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value[j]), 0, sizeof(my_value[j])}};
      nanos_data_access_t data_accesses1[1] = {{&my_value[j], {0,1,0,0,0}, 1, dimensions1, 0}};
      nanos_wd_t wd1 = 0;

      const_data1.base.data_alignment = __alignof__(my_args);
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof( my_args ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
      args1->p_i = dep_addr1;
      NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );
   }

   nanos_region_dimension_t dimensions2[size][1];
   nanos_data_access_t data_accesses2[size];
   for ( j = 0; j < size; j++ ) {
      nanos_region_dimension_t dim = {sizeof(my_value[j]), 0, sizeof(my_value[j])};
      dimensions2[j][0] = dim;
      nanos_data_access_t da = {&my_value[j], {1,0,0,0,0}, 1, dimensions2[j], 0};
      data_accesses2[j] = da;
   }
   
   NANOS_SAFE( nanos_wait_on( size, data_accesses2 ));

   for ( j = 0; j < size; j++ ) {
    if ( my_value[j] != 0 ) return false;
   }
   return true;
}

bool create_and_run_test();
bool create_and_run_test()
{
   int j;
   int __attribute__( ( aligned( 128 ) ) ) my_value[100];
   int other_value=0;

   for ( j = 0; j < 100; j++ ) {
      my_value[j] = 500;
      int * dep_addr1 = &my_value[j];
      my_args *args1=0;
      nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value[j]), 0, sizeof(my_value[j])}};
      nanos_data_access_t data_accesses1[1] = {{&my_value[j], {0,1,0,0,0}, 1, dimensions1, 0}};
      nanos_wd_t wd1 = 0;

      const_data1.base.data_alignment = __alignof__(my_args);
      NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof( my_args ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
      args1->p_i = dep_addr1;
      NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );
   }

   nanos_region_dimension_t dimensions2[100][1];
   nanos_data_access_t data_accesses2[100];
   for ( j = 0; j < 100; j++ ) {
      nanos_region_dimension_t dim = {sizeof(my_value[j]), 0, sizeof(my_value[j])};
      dimensions2[j][0] = dim;
      nanos_data_access_t da = {&my_value[j], {1,0,0,0,0}, 1, dimensions2[j], 0};
      data_accesses2[j] = da;
   }

   my_args arg;
   arg.p_i = &other_value;
   nanos_device_t test_devices_2[1] = { NANOS_SMP_DESC( test_device_arg_1 ) };

   const_data1.base.data_alignment = __alignof__(my_args);
   NANOS_SAFE( nanos_create_wd_and_run_compact ( &const_data1.base, &dyn_props, sizeof( my_args ), ( void * )&arg, 100, data_accesses2, NULL, NULL, NULL ) );

   for ( j = 0; j < 100; j++ ) {
    if ( my_value[j] != 0 ) return false;
   }
   return true;
}

// Test concurrent tasks, this test creates a task with an inout dependency on an array an then
// a bunch of concurrent (reduction) tasks that update it. Finally it waits for them all to finish and
// checks the result
bool concurrent_task_1();
bool concurrent_task_1()
{
   int i, j;
   int size = 100;
   int __attribute__( ( aligned( 128 ) ) ) my_value[size];
   int *value_ref = (int *)&my_value;

   for ( i = 0; i < size; i++ ) {
      my_value[i] = 0;
   }

   my_args2 *args1=0;
   nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses1[1] = {{my_value, {0,1,0,0,0}, 1, dimensions1, 0}};
   nanos_wd_t wd1 = 0;

   const_data6.base.data_alignment = __alignof__(my_args2);
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data6.base, &dyn_props, sizeof( my_args2 ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
   args1->p_i = my_value;
   args1->index = size;
   NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );

   for ( j = 0; j < size; j++ ) {
      my_args2 *args2=0;
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{my_value, {1,1,0,1,0}, 1, dimensions2, 0}};
      nanos_wd_t wd2 = 0;

      const_data5.base.data_alignment = __alignof__(my_args2);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data5.base, &dyn_props, sizeof( my_args2 ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = my_value;
      args2->index = j;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   for ( j = 0; j < 100; j++ ) {
      if ( my_value[j] != 2 ) return false;
   }
   return true;
}

// Test concurrent tasks, this test creates a task with an inout dependency on an array an then
// a bunch of concurrent (reduction) tasks that update it. Then, another set of tasks are successors
// of the concurrent ones. This checks that the concurrent task behaves correctly
bool concurrent_task_2();
bool concurrent_task_2()
{
   int i, j;
   int size = 100;
   int __attribute__( ( aligned( 128 ) ) ) my_value[size];
   int *value_ref = (int *)&my_value;
   int my_results[size];

   for ( i = 0; i < size; i++ ) {
      my_value[i] = 0;
      my_results[i] = 0;
   }

   my_args2 *args1=0;
   nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses1[1] = {{my_value, {0,1,0,0,0}, 1, dimensions1, 0}};
   nanos_wd_t wd1 = 0;

   const_data6.base.data_alignment = __alignof__(my_args2);
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data6.base, &dyn_props, sizeof( my_args2 ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
   args1->p_i = my_value;
   args1->index = size;
   NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );

   for ( j = 0; j < size; j++ ) {
      my_args2 *args2=0;
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{my_value, {1,1,0,1,0}, 1, dimensions2, 0}};
      nanos_wd_t wd2 = 0;

      const_data5.base.data_alignment = __alignof__(my_args2);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data5.base, &dyn_props, sizeof( my_args2 ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = my_value;
      args2->index = j;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }

   for ( j = 0; j < size; j++ ) {
      my_args3 *args2=0;
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{my_value, {1,0,0,0,0}, 1, dimensions2, 0}};
      nanos_wd_t wd2 = 0;

      const_data7.base.data_alignment = __alignof__(my_args3);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data7.base, &dyn_props, sizeof( my_args3 ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = my_value;
      args2->p_result = &my_results[j];
      args2->index = j;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   for ( j = 0; j < size; j++ ) {
      if ( my_results[j] < 0 ) return false;
   }
   return true;
}

// Test concurrent tasks, this test creates a task with an inout dependency on an array an then
// a bunch of tasks that read the dependency, then, again, a bunch of concurrent (reduction) tasks 
// that update it. Then, another set of tasks are successors
// of the concurrent ones. This checks that the concurrent task behaves correctly
bool concurrent_task_3();
bool concurrent_task_3()
{
   int i, j;
   int size = 100;
   int __attribute__( ( aligned( 128 ) ) ) my_value[size];
   int *value_ref = (int *)&my_value;
   int __attribute__( ( aligned( 128 ) ) ) my_results[size];
   

   for ( i = 0; i < size; i++ ) {
      my_value[i] = 0;
      my_results[i] = 0;
   }

   my_args2 *args1=0;
   nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses1[1] = {{&my_value, {0,1,0,0,0}, 1, dimensions1, 0}};
   nanos_wd_t wd1 = 0;

   const_data6.base.data_alignment = __alignof__(my_args2);
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data6.base, &dyn_props, sizeof( my_args2 ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
   args1->p_i = my_value;
   args1->index = size;
   NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );

   for ( j = 0; j < size; j++ ) {
      my_args2 *args2=0;
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{&my_value, {1,0,0,0,0}, 1, dimensions2, 0}};
      nanos_wd_t wd2 = 0;

      const_data8.base.data_alignment = __alignof__(my_args2);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data8.base, &dyn_props, sizeof( my_args2 ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = my_value;
      args2->index = j;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }

   for ( j = 0; j < size; j++ ) {
      my_args2 *args2=0;
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{&my_value, {1,1,0,1,0}, 1, dimensions2, 0}};
      nanos_wd_t wd2 = 0;

      const_data5.base.data_alignment = __alignof__(my_args2);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data5.base, &dyn_props, sizeof( my_args2 ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = my_value;
      args2->index = j;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }

   for ( j = 0; j < size; j++ ) {
      my_args3 *args2=0;
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{&my_value, {1,0,0,0,0}, 1, dimensions2, 0}};
      nanos_wd_t wd2 = 0;

      const_data7.base.data_alignment = __alignof__(my_args3);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data7.base, &dyn_props, sizeof( my_args3 ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = my_value;
      args2->p_result = &my_results[j];
      args2->index = j;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   for ( j = 0; j < size; j++ ) {
      if ( my_results[j] < 0 ) return false;
   }
   return true;
}

// This test creates a task that increases the the elements of an array by once,
// and then 100 tasks that do the same but with the commutative property set.
// Concurrent should fail while commutative should work.
bool commutative_task_1();
bool commutative_task_1()
{
   int i, j;
   int size = 100;
   int __attribute__( ( aligned( 128 ) ) ) my_value[size];
   int *value_ref = (int *)&my_value;

   for ( i = 0; i < size; i++ ) {
      my_value[i] = 0;
   }
   // This will help triggering issues :)
   nanos_stop_scheduler();
   
   my_args2 *args1=0;
   nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses1[1] = {{my_value, {0,1,0,0,0}, 1, dimensions1, 0}};
   nanos_wd_t wd1 = 0;

   const_data6.base.data_alignment = __alignof__(my_args2);
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data6.base, &dyn_props, sizeof( my_args2 ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
   args1->p_i = my_value;
   args1->index = size;
   NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );

   for ( j = 0; j < size; j++ ) {
      my_args2 *args2=0;
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{my_value, {1,1,0,0,1}, 1, dimensions2, 0}};
      nanos_wd_t wd2 = 0;

      const_data6.base.data_alignment = __alignof__(my_args2);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data6.base, &dyn_props, sizeof( my_args2 ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = my_value;
      args2->index = size;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }
   
   nanos_start_scheduler();
   
   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   for ( j = 0; j < size; j++ ) {
      if ( my_value[j] != size+1 ) return false;
   }
   return true;
}

bool dependency_offset();
bool dependency_offset()
{
   int i;
   int __attribute__( ( aligned( 128 ) ) ) my_value;
   int * dep_addr = &my_value;
   my_args *args1=0;
   nanos_region_dimension_t dimensions1[1] = {{sizeof(my_value), 0, sizeof(my_value)}};
   nanos_data_access_t data_accesses1[1] = {{&my_value, {0,1,0,0,0}, 1, dimensions1, 0}};
   nanos_wd_t wd1=0;
   const_data1.base.data_alignment = __alignof__(my_args);
   NANOS_SAFE( nanos_create_wd_compact ( &wd1, &const_data1.base, &dyn_props, sizeof( my_args ), ( void ** )&args1, nanos_current_wd(), NULL, NULL ) );
   args1->p_i = dep_addr;
   NANOS_SAFE( nanos_submit( wd1,1,data_accesses1,0 ) );

   for ( i = 0; i < 100; i++ ) {
      my_args *args2=0;
      // It is easier to check that using offsets that lead to the same dependency they are not broken than the oposite
      int * local_dep_addr = &my_value + i; // NOTE: if renaming is implemented this is not safe
      nanos_region_dimension_t dimensions2[1] = {{sizeof(my_value), sizeof(my_value), sizeof(my_value)}};
      nanos_data_access_t data_accesses2[1] = {{local_dep_addr, {1,1,0,0,0}, 1, dimensions2, -1L*i*sizeof(my_value)}};
      nanos_wd_t wd2 = 0;
      const_data2.base.data_alignment = __alignof__(my_args);
      NANOS_SAFE( nanos_create_wd_compact ( &wd2, &const_data2.base, &dyn_props, sizeof( my_args ), ( void ** )&args2, nanos_current_wd(), NULL, NULL ) );
      args2->p_i = dep_addr;
      NANOS_SAFE( nanos_submit( wd2,1,data_accesses2,0 ) );
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
   fprintf( stderr, "Value: %d\n", my_value );
   
   return (my_value == 100);
}



int main ( int argc, char **argv )
{
   printf("Single dependency test... \n");
   fflush(stdout);
   if ( single_dependency() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
   }
   
   printf("Single inout chain test... \n");
   fflush(stdout);
   if ( single_inout_chain() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
   }

   printf("Multiple inout chains test... \n");
   fflush(stdout);
   if ( multiple_inout_chains() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   printf("task with multiple predecessors... \n");
   fflush(stdout);
   if ( multiple_predecessors() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }
   printf("task with multiple anti-dependencies... \n");
   fflush(stdout);
   if ( multiple_antidependencies() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   printf("Out dependencies chain... \n");
   fflush(stdout);
   if ( out_dep_chain() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   printf("Wait on test...\n");
   fflush(stdout);
   if ( wait_on_test() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   printf("create and run test...\n");
   fflush(stdout);
   if ( create_and_run_test() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   printf("concurrent tasks test...\n");
   fflush(stdout);
   if ( concurrent_task_1() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   printf("concurrent tasks 2 test...\n");
   fflush(stdout);
   if ( concurrent_task_2() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   printf("concurrent tasks 3 test...\n");
   fflush(stdout);
   if ( concurrent_task_3() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   printf("commutative tasks 1 test...\n");
   fflush(stdout);
   if ( commutative_task_1() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   printf("offset in dependencies test...\n");
   fflush(stdout);
   if ( dependency_offset() ) {
      printf("PASS\n");
      fflush(stdout);
   } else {
      printf("FAIL\n");
      fflush(stdout);
      return 1;
   }

   return 0;
}

