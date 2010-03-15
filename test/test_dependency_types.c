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
#include <nanos.h>


typedef struct {
   int* p_i;
} my_args;

void first(void *ptr)
{
   int *i = ((my_args *) ptr)->p_i;
   *i = 0;
#ifdef VERBOSE
   printf("first task: %x set to %d\n",(unsigned int)i,*i);
   fflush(stdout);
#endif
}

void second(void *ptr)
{
   int *i = ((my_args *) ptr)->p_i;
   (*i)++;
#ifdef VERBOSE
   printf("successor: %x set to %d\n",(unsigned int)i,*i);
   fflush(stdout);
#endif
}

void third(void *ptr)
{
   int j;
   for ( j = 0; j < 100; j++ ) {
      int *i = ((my_args *) ptr)[j].p_i;
      (*i)++;
   }
}

void fourth(void *ptr)
{
   int *i = ((my_args *) ptr)[0].p_i;
   int *j = ((my_args *) ptr)[1].p_i;
   (*j) = *i;
}


nanos_smp_args_t test_device_arg_1 = { first };
nanos_smp_args_t test_device_arg_2 = { second };
nanos_smp_args_t test_device_arg_3 = { third };
nanos_smp_args_t test_device_arg_4 = { fourth };

bool single_dependency()
{
   int my_value;
   int * dep_addr = &my_value;
   my_args *args1=0;
   nanos_dependence_t deps1 = {(void **)&dep_addr,0, {0,1,0}, 0};
   nanos_wd_props_t props = {
     .mandatory_creation = true,
     .tied = false,
     .tie_to = false,
   };
   nanos_wd_t wd1=0;
   nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_1) };
   NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_1, sizeof(my_args), (void**)&args1, nanos_current_wd(), &props, 0, NULL) );
   args1->p_i = dep_addr;
   NANOS_SAFE( nanos_submit( wd1,1,&deps1,0 ) );

   my_args *args2=0;
   nanos_dependence_t deps2 = {(void **)&dep_addr,0, {1,1,0}, 0};
   nanos_wd_t wd2 = 0;
   nanos_device_t test_devices_2[1] = { NANOS_SMP_DESC( test_device_arg_2 ) };
   NANOS_SAFE( nanos_create_wd ( &wd2, 1,test_devices_2, sizeof(my_args), (void**)&args2, nanos_current_wd(), &props, 0, NULL) );
   args2->p_i = dep_addr;
   NANOS_SAFE( nanos_submit( wd2,1,&deps2,0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd() ) );
   
   return (my_value == 1);
}

bool single_inout_chain()
{
   int i;
   int my_value;
   int * dep_addr = &my_value;
   my_args *args1=0;
   nanos_dependence_t deps1 = {(void **)&dep_addr,0, {0,1,0}, 0};
   nanos_wd_props_t props = {
     .mandatory_creation = true,
     .tied = false,
     .tie_to = false,
   };
   nanos_wd_t wd1=0;
   nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_1) };
   NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_1, sizeof(my_args), (void**)&args1, nanos_current_wd(), &props, 0, NULL) );
   args1->p_i = dep_addr;
   NANOS_SAFE( nanos_submit( wd1,1,&deps1,0 ) );

   for ( i = 0; i < 100; i++ ) {
      my_args *args2=0;
      nanos_dependence_t deps2 = {(void **)&dep_addr,0, {1,1,0}, 0};
      nanos_wd_t wd2 = 0;
      nanos_device_t test_devices_2[1] = { NANOS_SMP_DESC( test_device_arg_2 ) };
      NANOS_SAFE( nanos_create_wd ( &wd2, 1,test_devices_2, sizeof(my_args), (void**)&args2, nanos_current_wd(), &props, 0, NULL) );
      args2->p_i = dep_addr;
      NANOS_SAFE( nanos_submit( wd2,1,&deps2,0 ) );
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd() ) );
   
   return (my_value == 100);
}

bool multiple_inout_chains()
{
   int i, j;
   int my_value[100];

   for ( i = 0; i < 100; i++ ) {
      int * dep_addr = &my_value[i];
      my_args *args1=0;
      nanos_dependence_t deps1 = {(void **)&dep_addr,0, {0,1,0}, 0};
      nanos_wd_props_t props = {
        .mandatory_creation = true,
        .tied = false,
        .tie_to = false,
      };
      nanos_wd_t wd1=0;
      nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_1) };
      NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_1, sizeof(my_args), (void**)&args1, nanos_current_wd(), &props, 0, NULL) );
      args1->p_i = dep_addr;
      NANOS_SAFE( nanos_submit( wd1,1,&deps1,0 ) );

      for ( j = 0; j < 100; j++ ) {
         my_args *args2=0;
         nanos_dependence_t deps2 = {(void **)&dep_addr,0, {1,1,0}, 0};
         nanos_wd_t wd2 = 0;
         nanos_device_t test_devices_2[1] = { NANOS_SMP_DESC( test_device_arg_2 ) };
         NANOS_SAFE( nanos_create_wd ( &wd2, 1,test_devices_2, sizeof(my_args), (void**)&args2, nanos_current_wd(), &props, 0, NULL) );
         args2->p_i = dep_addr;
         NANOS_SAFE( nanos_submit( wd2,1,&deps2,0 ) );
      }
   }

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd() ) );

   for ( i = 0; i < 100; i++ ) {
      if ( my_value[i] != 100 ) return false;
   }
   return true;
}

bool multiple_predecessors()
{
   int j;
   int my_value[100];
   nanos_wd_props_t props = {
     .mandatory_creation = true,
     .tied = false,
     .tie_to = false,
   };

   for ( j = 0; j < 100; j++ ) {
      int * dep_addr1 = &my_value[j];
      my_args *args1=0;
      nanos_dependence_t deps1 = {(void **)&dep_addr1,0, {0,1,0}, 0};
      nanos_wd_t wd1 = 0;

      nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_1 ) };
      NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_1, sizeof(my_args), (void**)&args1, nanos_current_wd(), &props, 0, NULL) );
      args1->p_i = dep_addr1;
      NANOS_SAFE( nanos_submit( wd1,1,&deps1,0 ) );
   }

   nanos_dependence_t deps2[100];
   int *dep_addr2[100];
   my_args *args2=0;
   for ( j = 0; j < 100; j++ ) {
      dep_addr2[j] = &my_value[j];
      deps2[j] = (nanos_dependence_t){(void **) &dep_addr2[j],0, {1,1,0},0};
   }

   nanos_wd_t wd2=0;
   nanos_device_t test_devices_3[1] = { NANOS_SMP_DESC( test_device_arg_3) };
   NANOS_SAFE( nanos_create_wd ( &wd2, 1,test_devices_3, sizeof(my_args)*100, (void**)&args2, nanos_current_wd(), &props, 0, NULL) );
   for ( j = 0; j < 100; j++)
      args2[j].p_i = dep_addr2[j];
   NANOS_SAFE( nanos_submit( wd2,100,&deps2[0],0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd() ) );
   for ( j = 0; j < 100; j++ ) {
      if ( my_value[j] != 1 ) return false;
   }
   return true;
}

bool multiple_antidependencies()
{
   int j;
   int my_value=1500;
   int my_reslt[100];
   nanos_wd_props_t props = {
     .mandatory_creation = true,
     .tied = false,
     .tie_to = false,
   };

   for ( j = 0; j < 100; j++ ) {
      int * dep_addr1 = &my_value;
      int * reslt_addr =&my_reslt[j];
      my_args *args1=0;
      nanos_dependence_t deps1 = {(void **)&dep_addr1,0, {1,0,0}, 0};

      nanos_wd_t wd1 = 0;
      nanos_device_t test_devices_4[1] = { NANOS_SMP_DESC( test_device_arg_4 ) };
      NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_4, sizeof(my_args)*2, (void**)&args1, nanos_current_wd(), &props, 0, NULL) );
      args1[0].p_i = dep_addr1;
      args1[1].p_i = reslt_addr;
      NANOS_SAFE( nanos_submit( wd1,1,&deps1,0 ) );
   }

   int *dep_addr2 = &my_value;
   nanos_dependence_t deps2 = (nanos_dependence_t){(void **) &dep_addr2,0, {1,1,0},0};
   my_args *args2=0;

   nanos_wd_t wd2=0;
   nanos_device_t test_devices_2[1] = { NANOS_SMP_DESC( test_device_arg_2) };
   NANOS_SAFE( nanos_create_wd ( &wd2, 1,test_devices_2, sizeof(my_args), (void**)&args2, nanos_current_wd(), &props, 0, NULL) );
   args2->p_i = dep_addr2;
   NANOS_SAFE( nanos_submit( wd2,1,&deps2,0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd() ) );
   for ( j = 0; j < 100; j++ ) {
      if ( my_reslt[j] != 1500 ) return false;
   }
   if (my_value != 1501) return false;
   return true;
}

bool out_dep_chain()
{
   int i;
   int my_value;
   int * dep_addr = &my_value;
   nanos_wd_props_t props = {
     .mandatory_creation = true,
     .tied = false,
     .tie_to = false,
   };

   for ( i = 0; i < 100; i++ ) {
      my_args *args2=0;
      nanos_dependence_t deps2 = {(void **)&dep_addr,0, {0,1,0}, 0};
      nanos_wd_t wd2 = 0;
      nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_1 ) };
      NANOS_SAFE( nanos_create_wd ( &wd2, 1,test_devices_1, sizeof(my_args), (void**)&args2, nanos_current_wd(), &props, 0, NULL) );
      args2->p_i = dep_addr;
      NANOS_SAFE( nanos_submit( wd2,1,&deps2,0 ) );
   }

   int input=500;
   int * input_addr = &input;
   nanos_dependence_t deps1 = {(void **)&dep_addr,0, {0,1,0}, 0};
   my_args *args1=0;
   nanos_wd_t wd1=0;
   nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_4) };
   NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_1, sizeof(my_args)*2, (void**)&args1, nanos_current_wd(), &props, 0, NULL) );
   args1[0].p_i = input_addr;
   args1[1].p_i = dep_addr;
   NANOS_SAFE( nanos_submit( wd1,1,&deps1,0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd() ) );
   
   return (my_value == 500);
}

bool wait_on_test()
{
   int j;
   int my_value[100];
   nanos_wd_props_t props = {
     .mandatory_creation = true,
     .tied = false,
     .tie_to = false,
   };

   for ( j = 0; j < 100; j++ ) {
      my_value[j] = 500;
      int * dep_addr1 = &my_value[j];
      my_args *args1=0;
      nanos_dependence_t deps1 = {(void **)&dep_addr1,0, {0,1,0}, 0};
      nanos_wd_t wd1 = 0;

      nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_1 ) };
      NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_1, sizeof(my_args), (void**)&args1, nanos_current_wd(), &props, 0, NULL) );
      args1->p_i = dep_addr1;
      NANOS_SAFE( nanos_submit( wd1,1,&deps1,0 ) );
   }

   nanos_dependence_t deps2[100];
   int *dep_addr2[100];
   for ( j = 0; j < 100; j++ ) {
      dep_addr2[j] = &my_value[j];
      deps2[j] = (nanos_dependence_t){(void **) &dep_addr2[j],0, {1,0,0},0};
   }
   
   NANOS_SAFE( nanos_wait_on( 100, &deps2[0] ));

   for ( j = 0; j < 100; j++ ) {
    if ( my_value[j] != 0 ) return false;
   }
   return true;
}

bool create_and_run_test()
{
   int j;
   int my_value[100];
   int other_value=0;
   nanos_wd_props_t props = {
     .mandatory_creation = true,
     .tied = false,
     .tie_to = false,
   };

   for ( j = 0; j < 100; j++ ) {
      my_value[j] = 500;
      int * dep_addr1 = &my_value[j];
      my_args *args1=0;
      nanos_dependence_t deps1 = {(void **)&dep_addr1,0, {0,1,0}, 0};
      nanos_wd_t wd1 = 0;

      nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_1 ) };
      NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_1, sizeof(my_args), (void**)&args1, nanos_current_wd(), &props, 0, NULL) );
      args1->p_i = dep_addr1;
      NANOS_SAFE( nanos_submit( wd1,1,&deps1,0 ) );
   }

   nanos_dependence_t deps2[100];
   int *dep_addr2[100];
   for ( j = 0; j < 100; j++ ) {
      dep_addr2[j] = &my_value[j];
      deps2[j] = (nanos_dependence_t){(void **) &dep_addr2[j],0, {1,0,0},0};
   }

   my_args arg;
   arg.p_i = &other_value;
   nanos_device_t test_devices_2[1] = { NANOS_SMP_DESC( test_device_arg_1 ) };

   NANOS_SAFE( nanos_create_wd_and_run( 1, test_devices_2, sizeof(my_args), (void *)&arg, 100, &deps2[0], &props , 0, NULL) );

   for ( j = 0; j < 100; j++ ) {
    if ( my_value[j] != 0 ) return false;
   }
   return true;
}

int main ( int argc, char **argv )
{

   printf("Single dependency test... \n");
   if ( single_dependency() ) {
      printf("PASS\n");
   } else {
      printf("FAIL\n");
   }
   
   printf("Single inout chain test... \n");
   if ( single_inout_chain() ) {
      printf("PASS\n");
   } else {
      printf("FAIL\n");
   }

   printf("Multiple inout chains test... \n");
   if ( multiple_inout_chains() ) {
      printf("PASS\n");
   } else {
      printf("FAIL\n");
   }
 
   printf("task with multiple predecessors... \n");
   if ( multiple_predecessors() ) {
      printf("PASS\n");
   } else {
      printf("FAIL\n");
   }
 
   printf("task with multiple anti-dependencies... \n");
   if ( multiple_antidependencies() ) {
      printf("PASS\n");
   } else {
      printf("FAIL\n");
   }

   printf("Out dependencies chain... \n");
   if ( out_dep_chain() ) {
      printf("PASS\n");
   } else {
      printf("FAIL\n");
   }

   printf("Wait on test...\n");
   if ( wait_on_test() ) {
      printf("PASS\n");
   } else {
      printf("FAIL\n");
   }

   printf("create and run test...\n");
   if ( create_and_run_test() ) {
      printf("PASS\n");
   } else {
      printf("FAIL\n");
   }

   return 0;
}

