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
#include <math.h>
#include <time.h>
#include "nanos.h"

double get_usecs () {
   struct timespec tp;
   int error = clock_gettime( CLOCK_REALTIME, &tp);
   if (error != 0) return 0.0;
   return (tp.tv_sec * 1.0e6) + (tp.tv_nsec * 1.0e-3);
}
double get_msecs () {
   struct timespec tp;
   int error = clock_gettime( CLOCK_REALTIME, &tp);
   if (error != 0) return 0.0;
   return (tp.tv_sec * 1.0e3) + (tp.tv_nsec * 1.0e-6);
}
double get_secs () {
   struct timespec tp;
   int error = clock_gettime( CLOCK_REALTIME, &tp);
   if (error != 0) return 0.0;
   return (tp.tv_sec) + (tp.tv_nsec * 1.0e-9);
}
long get_epoch () {
   struct timespec tp;
   int error = clock_gettime( CLOCK_REALTIME, &tp);
   if (error != 0) return 0;
   return (((tp.tv_sec / 60 ) / 60) / 24);
}

typedef struct stats_t {
   double mean;
   double sd;
   double min;
   double max;
   double outliers;
} stats_t;

#define GET_TIME get_usecs()
#define TEST_NSAMPLES   50 // Number of samples for each test
#define TEST_NTASKS     50 // Number of tasks (when multiple tasks created)
#define TEST_TUSECS     50 // Task granularity (usecs) in warming up phase

void stats ( stats_t *s, double *values, unsigned size ) 
{
  int i;
  unsigned nr = 0;
  double total = 0.0, sumsq = 0.0;

  for (i=0; i<size; i++){
    if ( s->min > values[i] ) s->min = values[i];
    if ( s->max < values[i] ) s->max = values[i];
    total +=values[i]; 
//fprintf(stderr,"%f ",values[i]); /*FIXME*/
  } 
  s->mean  = total / size; // Computing initial mean

  for (i=0; i<size; i++){
    sumsq += (values[i] - (s->mean)) * (values[i] - (s->mean)); 
  } 
  s->sd = sqrt(sumsq / (size-1)); // Computing initial sd

  // Ignoring outliers, computing statistics again
  total = 0.0, sumsq = 0.0;
  s->min = 1.0e10; s->max = 0.0;

  for (i=0; i<size; i++){
    if ( values[i] < 0 ) continue;
    if ( (values[i] > (s->mean + (3.0 * s->sd))) || (values[i] < (s->mean - (3.0 * s->sd ))) ) {
       values[i] = -1.0;
       continue;
    }
    if ( s->min > values[i] ) s->min = values[i];
    if ( s->max < values[i] ) s->max = values[i];
    total +=values[i]; nr++;
  } 
  s->mean  = total / nr; // Computing final mean

  for (i=0; i<size; i++){
    if ( values[i] < 0 ) continue;
    sumsq += (values[i] - (s->mean)) * (values[i] - (s->mean)); 
  } 
  s->sd = sqrt(sumsq / (nr-1)); // Computing final sd

  s->outliers = ((double)(size - nr)) / size;
}

void print_stats ( const char *name, const char *desc, stats_t *s )
{
   bool binding;
   nanos_get_default_binding( &binding );


   fprintf(stderr, "*:Nanos++:%d:%s:%s:%3.3f:%3.3f:%3.3f:%3.3f:%0.3f:%d:%s:%s:%s:%s:%s\n",
                   get_epoch(),
                   name, desc,
                   s->mean, s->sd, s->min, s->max, s->outliers,
                   omp_get_max_threads(),
                   nanos_get_mode(),
                   nanos_get_pm(),
                   binding ? "binding" : "no-biding",
                   nanos_get_default_architecture(),
                   nanos_get_default_scheduler()
          );
}

void task ( int usecs ) { usleep(usecs); }

