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

#ifndef _SLICER_FOR_H
#define _SLICER_FOR_H

#if !defined(NUM_ITERS) || \
   !defined(VECTOR_SIZE) || \
   !defined(VECTOR_MARGIN)
#error "Define NUM_ITERS, VECTOR_SIZE and VECTOR_MARGIN before including worksharing.hpp"
#endif
#ifndef NUM_ITERS
#error Define NUM_ITERS before including __FILE__
#endif

#include "nanos.h"
#include "system_decl.hpp"
#include "slicer_decl.hpp"
#include "smpprocessor.hpp"
#include "plugin.hpp"
#include <stdio.h>
#include <string>

using namespace nanos;
using namespace nanos::ext;

enum { STEP_ERR = -17 };

typedef struct {
   nanos_loop_info_t loop_info;
   int offset;
} main__loop_1_data_t;

void main__loop_1(void *args);
void execute(std::string slicer_name, int lower, int upper, int offset, int step, int chunk);
int check(const char *desc, const char *lower, const char *upper,
      const char *offset, int step, int chunk);
int slicer_test(std::string slicer, const char *desc, int step, int chunk);
void print_vector();

static int I[VECTOR_SIZE+2*VECTOR_MARGIN] = {0};
static int *A = &I[VECTOR_MARGIN];


void main__loop_1 ( void *args )
{
   main__loop_1_data_t *hargs = (main__loop_1_data_t*) args;
   int lower = hargs->loop_info.lower;
   int upper = hargs->loop_info.upper;
   int step = hargs->loop_info.step;
   int offset = hargs->offset;

#ifdef VERBOSE
   fprintf(stderr,"[%d..%d:%d/%d]", lower, upper, step, offset);
#endif

   if ( step > 0 ) {
      for ( int i = lower; i <= upper; i += step) {
         A[i+offset]++;
      }
   }
   else if ( step < 0 ) {
      for ( int i = lower; i >= upper; i += step) {
         A[i+offset]++;
      }
   }
   else {
      A[-VECTOR_MARGIN] = STEP_ERR;
   }
}

void execute(std::string slicer_name, int lower, int upper, int offset, int step, int chunk)
{
#ifdef INVERT_LOOP_BOUNDARIES
   int _lower = lower;
   lower = upper;
   upper = _lower;
#endif

   for (int i = 0; i < NUM_ITERS; i++) {
      /* Load plugin */
      sys.loadPlugin( "slicer-" + slicer_name );
      Slicer *slicer = sys.getSlicer ( slicer_name );

      /* set up Work Descriptor */
      main__loop_1_data_t _loop_data;
      _loop_data.offset = -offset;
      WD * wd = new WorkDescriptor( new SMPDD( main__loop_1 ), sizeof( _loop_data ),
            __alignof__(nanos_loop_info_t),( void * ) &_loop_data, 0, NULL, NULL );
      wd->setSlicer( slicer );
      _loop_data.loop_info.lower = lower + offset;
      _loop_data.loop_info.upper = upper + offset;
      _loop_data.loop_info.step = step;
      _loop_data.loop_info.chunk = chunk;

      /* Submit and wait completion */
      WD *wg = getMyThreadSafe()->getCurrentWD();
      wg->addWork( *wd );
      if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
         char *idata = NEW char[sys.getPMInterface().getInternalDataSize()];
         sys.getPMInterface().initInternalData( idata );
         wd->setInternalData( idata );
      }
      sys.setupWD( *wd, (nanos::WD *) wg );
      sys.submit( *wd );
      wg->waitCompletion();

      /* Undo increment done by the worksharing */
      if (step > 0)
         for (int j = lower+offset; j <= upper+offset; j+= step)
            A[j-offset]--;
      else if (step < 0)
         for (int j = lower+offset; j >= upper+offset; j+= step)
            A[j-offset]--;
   }
}

int check(const char *desc, const char *lower, const char *upper,
      const char *offset, int step, int chunk)
{
   int error = 0;
   bool out_of_range = false, race_condition = false, step_error = false;

   if (I[0] == STEP_ERR) {
      step_error = true;
      I[0] = 0;
   }

   for (int i = 0; i < VECTOR_SIZE+2*VECTOR_MARGIN; i++) {
      if (I[i] != 0) {
         if ((i < VECTOR_MARGIN) || (i > (VECTOR_SIZE + VECTOR_MARGIN))) out_of_range = true;
         if ((I[i] != NUM_ITERS) && (I[i] != -NUM_ITERS)) race_condition = true;
         I[i] = 0;
         error++;
      }
   }

   fprintf(stderr,
         "%s, lower (%s), upper (%s), offset (%s), step(%+03d), chunk(%02d): %s%s%s%s\n",
         desc, lower, upper, offset, step, chunk,
         error == 0 ?     " successful" : " unsuccessful",
         out_of_range ?   " - Out of Range" : "",
         race_condition ? " - Race Condition" : "",
         step_error ?     " - Step Error" : ""
         );

   return error;
}

int slicer_test(std::string slicer, const char *desc, int step, int chunk)
{
   int error = 0;

   execute(slicer, 0, VECTOR_SIZE, 0, +step, chunk);
   error += check(desc, "0", "+", "+0000", +step, chunk);
   execute(slicer, VECTOR_SIZE-1, -1, 0, -step, chunk);
   error += check(desc, "+", "0", "+0000", -step, chunk);
   execute(slicer, 0, VECTOR_SIZE, -VECTOR_SIZE, +step, chunk);
   error += check(desc, "-", "0", "-VS  ", +step, chunk);
   execute(slicer, VECTOR_SIZE-1, -1, -VECTOR_SIZE, -step, chunk);
   error += check(desc, "0", "-", "-VS  ", -step, chunk);
   execute(slicer, 0, VECTOR_SIZE, VECTOR_SIZE/2, +step, chunk);
   error += check(desc, "+", "+", "+VS/2", +step, chunk);
   execute(slicer, VECTOR_SIZE-1, -1, VECTOR_SIZE/2, -step, chunk);
   error += check(desc, "+", "+", "+VS/2", -step, chunk);
   execute(slicer, 0, VECTOR_SIZE, -(VECTOR_SIZE/2), +step, chunk);
   error += check(desc, "-", "+", "-VS/2", +step, chunk);
   execute(slicer, VECTOR_SIZE-1, -1, -(VECTOR_SIZE/2), -step, chunk);
   error += check(desc, "+", "-", "-VS/2", -step, chunk);
   execute(slicer, 0, VECTOR_SIZE, -(2*VECTOR_SIZE), +step, chunk);
   error += check(desc, "-", "-", "-VS  ", +step, chunk);
   execute(slicer, VECTOR_SIZE-1, -1, -(2*VECTOR_SIZE), -step, chunk);
   error += check(desc, "-", "-", "-VS  ", -step, chunk);

   return error;
}

void print_vector()
{
#ifdef EXTRA_VERBOSE
   for (int i = -VECTOR_MARGIN; i < 0; i++) fprintf(stderr,"%d:",A[i]);
   fprintf(stderr,"[");
   for (int i = 0; i < VECTOR_SIZE; i++) fprintf(stderr,"%d:",A[i]);
   fprintf(stderr,"]");
   for (int i = VECTOR_SIZE; i < VECTOR_SIZE+VECTOR_MARGIN; i++ ) fprintf(stderr,"%d:",A[i]);
   fprintf(stderr,"\n");
#endif
}

#endif /* SLICER_FOR_HPP */
