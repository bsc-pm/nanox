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


void check_hardcoded_copy_data();

void first()
{
   check_hardcoded_copy_data ();
}

nanos_smp_args_t test_device_arg_1 = { first };

int main ( int argc, char **argv )
{
   int dummy=0;
   void* dummy1 = (void *) 1280;
   void* dummy2 = (void *) 1024;
   nanos_wd_props_t props = {
     .mandatory_creation = true,
     .tied = false,
     .tie_to = false,
   };

   nanos_copy_data_t *cd = 0;

   nanos_wd_t wd1=0;
   nanos_device_t test_devices_1[1] = { NANOS_SMP_DESC( test_device_arg_1) };
   NANOS_SAFE( nanos_create_wd ( &wd1, 1,test_devices_1, 0, (void*)&dummy, nanos_current_wd(), &props, 2, &cd) );

printf("Recieved: %lx\n",(unsigned long)cd);
fflush(stdout);

   cd[0] = (nanos_copy_data_t) {dummy1, NX_SHARED, {true, false}, 255};
   cd[1] = (nanos_copy_data_t) {dummy2, NX_PRIVATE, {false, true}, 127}; 

   NANOS_SAFE( nanos_submit( wd1,0,0,0 ) );

   NANOS_SAFE( nanos_wg_wait_completation( nanos_current_wd() ) );
   return 0;
}

