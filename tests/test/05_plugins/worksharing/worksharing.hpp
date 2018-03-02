/*************************************************************************************/
/*      Copyright 2018 Barcelona Supercomputing Center                               */
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

#ifndef WORKSHARING_HPP
#define WORKSHARING_HPP

#if !defined(NUM_ITERS) || \
   !defined(VECTOR_SIZE) || \
   !defined(VECTOR_MARGIN)
#error "Define NUM_ITERS, VECTOR_SIZE and VECTOR_MARGIN before including worksharing.hpp"
#endif
#ifndef NUM_ITERS
#error Define NUM_ITERS before including __FILE__
#endif

#include "nanos.h"
#include "nanos_omp.h"
#include <stdio.h>

enum { STEP_ERR = -17 };

struct nanos_args_0_t {
   nanos_ws_desc_t *wsd_1;
   int step;
   int offset;
};

void execute(nanos_omp_sched_t sched, int lower, int upper, int offset, int step, int chunk);
int check(const char *desc, const char *lower, const char *upper,
      const char *offset, int step, int chunk);
int ws_test(nanos_omp_sched_t sched, const char *desc, int step, int chunk);
void main__loop_1(struct nanos_args_0_t *const args);
void print_vector();

static int I[VECTOR_SIZE+2*VECTOR_MARGIN] = {0};
static int *A = &I[VECTOR_MARGIN];
static int error = 0;

void execute(nanos_omp_sched_t sched, int lower, int upper, int offset, int step, int chunk)
{
   for (int i = 0; i < NUM_ITERS; i++) {
      /* Set up worksharing */
      nanos_ws_desc_t *wsd_1 = NULL;;
      nanos_ws_info_loop_t nanos_setup_info_loop;
      nanos_setup_info_loop.lower_bound = lower + offset;
      nanos_setup_info_loop.upper_bound = upper + offset;
      nanos_setup_info_loop.loop_step = step;
      nanos_setup_info_loop.chunk_size = chunk;
      nanos_ws_t current_ws_policy = nanos_omp_find_worksharing(sched);
      nanos_worksharing_create(&wsd_1, current_ws_policy, (void **)&nanos_setup_info_loop, 0);

      /* Set up Slicer WD */
      static nanos_slicer_t replicate = nanos_find_slicer("replicate");
      nanos_wd_t nanos_wd = 0;
      struct nanos_args_0_t *ol_args = NULL;
      static nanos_smp_args_t smp_ol_main_0_args;
      smp_ol_main_0_args.outline =
         (void (*)(void *))(void (*)(struct nanos_args_0_t *))&main__loop_1;
      nanos_wd_dyn_props_t nanos_wd_dyn_props = {0};
      static nanos_device_t devices[1] = {{ &nanos_smp_factory, &smp_ol_main_0_args }};
      static nanos_wd_props_t props;
      props.mandatory_creation = 1;
      props.tied = 1;
      nanos_create_sliced_wd(&nanos_wd, 1, devices, sizeof(struct nanos_args_0_t),
            __alignof__(struct nanos_args_0_t), (void **)&ol_args, nanos_current_wd(),
            replicate, &props, &nanos_wd_dyn_props, 0, NULL, 0, NULL);
      /* Set up outline arguments */
      (*ol_args).wsd_1 = wsd_1;
      (*ol_args).step = step;
      (*ol_args).offset = -offset;
      /* Submit and wait completion */
      nanos_submit(nanos_wd, 0, 0, 0);
      nanos_wg_wait_completion(nanos_current_wd(), 0);
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

int ws_test(nanos_omp_sched_t sched, const char *desc, int step, int chunk)
{
   int error = 0;

   execute(sched, 0, VECTOR_SIZE, 0, +step, chunk);
   error += check(desc, "0", "+", "+0000",+ step, chunk);
   execute(sched, VECTOR_SIZE-1, -1, 0, -step, chunk);
   error += check(desc, "+", "0", "+0000", -step, chunk);
   execute(sched, 0, VECTOR_SIZE, -VECTOR_SIZE, +step, chunk);
   error += check(desc, "-", "0", "-VS  ", +step, chunk);
   execute(sched, VECTOR_SIZE-1, -1, -VECTOR_SIZE, -step, chunk);
   error += check(desc, "0", "-", "-VS  ", -step, chunk);
   execute(sched, 0, VECTOR_SIZE, VECTOR_SIZE/2, +step, chunk);
   error += check(desc, "+", "+", "+VS/2", +step, chunk);
   execute(sched, VECTOR_SIZE-1, -1, VECTOR_SIZE/2, -step, chunk);
   error += check(desc, "+", "+", "+VS/2", -step, chunk);
   execute(sched, 0, VECTOR_SIZE, -(VECTOR_SIZE/2), +step, chunk);
   error += check(desc, "-", "+", "-VS/2", +step, chunk);
   execute(sched, VECTOR_SIZE-1, -1, -(VECTOR_SIZE/2), -step, chunk);
   error += check(desc, "+", "-", "-VS/2", -step, chunk);
   execute(sched, 0, VECTOR_SIZE, -(2*VECTOR_SIZE), +step, chunk);
   error += check(desc, "-", "-", "-VS  ", +step, chunk);
   execute(sched, VECTOR_SIZE-1, -1, -(2*VECTOR_SIZE), -step, chunk);
   error += check(desc, "-", "-", "-VS  ", -step, chunk);

   return error;
}

void main__loop_1(struct nanos_args_0_t *const args)
{
   nanos_ws_desc_t *wsd_1 = (*args).wsd_1;
   int step = (*args).step;
   int offset = (*args).offset;

   int i;
   nanos_ws_item_loop_t nanos_item_loop;

   nanos_worksharing_next_item(wsd_1, (void**)&nanos_item_loop);
#ifdef VERBOSE
   fprintf(stderr,"First item: [%d..%d:%d/%d]\n",
         nanos_item_loop.lower,
         nanos_item_loop.upper,
         step,
         offset);
#endif

   while (nanos_item_loop.execute)
   {
      if (step > 0)
      {
         for (i = nanos_item_loop.lower; i <= nanos_item_loop.upper; i += step)
         {
            A[i+offset]++;
         }
      }
      else if (step < 0)
      {
         for (i = nanos_item_loop.lower; i >= nanos_item_loop.upper; i += step)
         {
            A[i+offset]++;
         }
      }
      else
      {
         I[0] = STEP_ERR;
      }
      nanos_worksharing_next_item(wsd_1, (void**)&nanos_item_loop);
   }
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

#endif /* WORKSHARING_HPP */
