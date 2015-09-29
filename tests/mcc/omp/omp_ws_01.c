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
#pragma omp for schedule(omp_static)
      for (i=0;i<SIZE;i++) a[i]++;
   }

   for (i=0;i<SIZE;i++) if (a[i] != ITERS) rv++;

   if ( rv ) fprintf(stderr,"%s: Error, final result is not valid \n",argv[0]);

   return rv;
}
