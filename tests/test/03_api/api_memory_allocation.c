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

#define NUM_ITERS      20
#define VECTOR_SIZE    1000
// -  a non-divisor of VECTOR_SIZE (e.g. 13 using a VECTOR_SIZE 1000)
#define NUM_A          1
#define NUM_B          5
#define NUM_C          13

#define STEP_ERROR     17

#include <stdlib.h>


int main (int argc, char *argv[])
{
   bool check = true;
   int i,it, *m = NULL;

   for ( it = 0; it < NUM_ITERS; it++ ) 
   {
      nanos_malloc ( (void *) &m, sizeof(int)*VECTOR_SIZE, NULL, 0 );
      for (i = 0; i < VECTOR_SIZE; i++) m[i] = 0;
      for (i = 0; i < VECTOR_SIZE; i++) m[i]++;
      for (i = 0; i < VECTOR_SIZE; i++) if ( m[i] != 1 ) { check = false; break; }
      nanos_free (m);

   }

   if (check) { return 0; } else { return -1; }
}
