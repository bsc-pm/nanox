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

#include<stdio.h>
#include"nanos.h"

/*
<testinfo>
test_generator=gens/mcc-openmp-generator
</testinfo>
*/

int main ( int argc, char *argv[] )
{
   int error = 0, a = 0;

   #pragma omp task shared(a) inout(a)
   {
      a++;
      fprintf(stderr,"1");
      nanos_dependence_release_all();
      nanos_yield();
      usleep(10000);
      fprintf(stderr,"3");
      a--;
   }

   #pragma omp task shared(a,error) in(a)
   {
      fprintf(stderr,"2");
      if (!a) error++;
   }

   #pragma omp taskwait

   fprintf(stderr,"4:verification=%s\n",error?"UNSUCCESSFUL":"successful");

   return error;
}
