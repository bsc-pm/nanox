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
#include <nanos.h>
#include <alloca.h>


/* ******************************* SECTION 1 ***************************** */
// compiler: outlined function arguments
typedef struct { int *M; } main__section_1_data_t;
// compiler: outlined function
void main__section_1 ( void *p_args )
{
   int i;
   main__section_1_data_t *args = (main__section_1_data_t *) p_args;
   fprintf( stderr,"Section 1\n" );
//fprintf(stderr,"section 1: vector @=%p\n",args->M );
//   for ( i = 0; i < VECTOR_SIZE; i++) args->M[i]++;
//fprintf(stderr,"section 1: vector @=%p has finished\n",args->M );
}
// compiler: smp device for main__section_1 function
nanos_smp_args_t main__section_1_device_args = { main__section_1 };

/* ******************************* SECTIONS ***************************** */
// compiler: outlined function
void main__sections ( void *p_args ) { fprintf(stderr,"es\n"); }

int main ( int argc, char **argv )
{
   int i;
   bool check = true;

   /* COMMON INFO */
   nanos_wd_props_t props = {
      .mandatory_creation = true,
      .tied = false,
      .tie_to = false,
      .priority = 0
   };


   nanos_wd_t wd[4] = { NULL, NULL, NULL, NULL };

   /* Creating section 1 wd */
   nanos_device_t main__section_1_device[1] = { NANOS_SMP_DESC( main__section_1_device_args ) };
   main__section_1_data_t *section_data_1 = NULL;
   fprintf(stderr, "Creating WD\n" );
   NANOS_SAFE( nanos_create_wd ( &wd[0], 1, main__section_1_device, sizeof(section_data_1), __alignof__(section_data_1), (void **) &section_data_1,
                             nanos_current_wd(), &props , 0, NULL ) );
   fprintf(stderr, "Created WD\n" );

   NANOS_SAFE( nanos_submit( wd[0],0,0,0 ) );

   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   // WD creation (and run
   fprintf(stderr, "Creating and running WD\n" );
   props.priority = 3;
   NANOS_SAFE( nanos_create_wd_and_run( 1, main__section_1_device, sizeof(section_data_1), __alignof__(section_data_1), (void *) section_data_1,
             0, (nanos_dependence_t *) 0, &props, 0, NULL, NULL ) );
   fprintf(stderr, "Created WD\n" );
   NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );

   fprintf(stderr, "%s : %s\n", argv[0], check ? "  successful" : "unsuccessful");
   if (check) { return 0; } else { return -1; }
}

