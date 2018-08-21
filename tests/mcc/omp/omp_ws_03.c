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

#include <stdio.h>
#include "omp.h"

#define SIZE 16
#define ITERS 4

int main ( int argc, char *argv[] )
{
   int rv = 0, i, a[SIZE], it;

   for (i=0;i<SIZE;i++) a[i] = 0;

   for (it=0;it<ITERS;it++) {
#pragma omp parallel
#pragma omp for schedule(omp_dynamic)
      for (i=0;i<SIZE;i++) a[i]++;
   }

   for (i=0;i<SIZE;i++) if (a[i] != ITERS) rv++;

   if ( rv ) fprintf(stderr,"%s: Error, final result is not valid \n",argv[0]);

   return rv;
}
