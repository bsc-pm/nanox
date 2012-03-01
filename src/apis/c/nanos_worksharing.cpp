
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

#include "nanos.h"
#include "worksharing_decl.hpp"

using namespace nanos;

nanos_err_t nanos_worksharing_create( nanos_ws_desc_t **wsd, nanos_ws_t ws, nanos_ws_info_t *info,  bool *b )
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","",NANOS_RUNTIME) ); //FIXME: To register new event

   try {
      *b = ((WorkSharing *) ws)->create( wsd, info );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

nanos_err_t nanos_worksharing_next_item( nanos_ws_desc_t *wsd, nanos_ws_item_t *wsi )
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","",NANOS_RUNTIME) ); //FIXME: To register new event

   try {
      ((WorkSharing *) wsd->ws)->nextItem( wsd, wsi );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;

}

// FIXME: Following code comes from intone project
// FIXME: To remove
#if 0
typedef enum {
	NANOS_WS_SCH_DEFAULT=0,
	NANOS_WS_SCH_STATIC=1,
	NANOS_WS_SCH_DYNAMIC=2,
 	NANOS_WS_SCH_GUIDED=4,
	NANOS_WS_SCH_RUNTIME=8,
	NANOS_WS_ORDERED=16
} nanos_ws_sched_t;

#define NANOS_WS_SCHEDULE_TYPE_MASK	0x00000007

void nanos_ws_begin_for(int *start, int *end, int *step, int *chunk, nanos_ws_sched_t *schedule);
int nanos_ws_next_iters(int *lstart, int *lend, int *last);
void nanos_ws_end_for(int *barrier_needed);

/* ---- */

void nanos_ws_begin_for(int *start, int *end, int *step, int *chunk, nanos_ws_sched_t *schedule);
{
	nth_word_t loop_shared;
	nth_desc_t *myself = NTH_MYSELF;

	assert(myself->player);

	if (*__intone_schedule == NANOS_WS_SCH_RUNTIME){
		*__intone_schedule = nth_data.runtime_sched;
		if (*__intone_chunk == 0){
			*__intone_chunk = nth_data.runtime_chunk;
		}
	}
	if (*__intone_schedule == NANOS_WS_SCH_DEFAULT){
		*__intone_schedule = NANOS_WS_SCH_STATIC;
	}

	loop_shared = !(*__intone_schedule & NANOS_WS_SCH_STATIC);
	if (!myself->player->team) loop_shared = FALSE;

	if (loop_shared){
		myself->player->loop_current = (nth_loop_desc_t*)
			&myself->player->team->loop[myself->player->loop_id];
		nth_spin_lock(&myself->player->loop_current->mutex);
		myself->player->loop_current->shared = TRUE;
	}
	else{
		myself->player->loop_current = (nth_loop_desc_t*)
			&myself->player->loop_local;
		myself->player->loop_current->shared = FALSE;
	}
		
	if (!myself->player->loop_current->init){
		myself->player->loop_current->nreached = 0;
		/* function arguments */
		myself->player->loop_current->start = *__intone_start;
		myself->player->loop_current->end = *__intone_end;
		myself->player->loop_current->step = *__intone_step;
		myself->player->loop_current->chunk = *__intone_chunk;
		myself->player->loop_current->schedule =
			*__intone_schedule & NANOS_WS_SCHEDULE_TYPE_MASK;
		myself->player->loop_current->ordered =
			*__intone_schedule & NANOS_WS_ORDERED;
		if (myself->player->team){
			myself->player->loop_current->nplayers = myself->player->team->ro.d.nplayers;
		}
		else{
			myself->player->loop_current->nplayers = 1;
		}
		myself->player->loop_current->step_abs =
			abs(*__intone_step);
		myself->player->loop_current->step_sign =
			((*__intone_step > 0) ? 1: -1);

		myself->player->loop_current->next_iter = *__intone_start;

/* depends from  schedule type: chunk_real and chunk_remainder */
		switch (myself->player->loop_current->schedule){
			case NANOS_WS_SCH_STATIC:
				if (*__intone_chunk == 0){

					myself->player->loop_current->chunk_real =
						(abs(*__intone_end - *__intone_start)+1+myself->player->loop_current->step_abs-1)
						/ (myself->player->loop_current->nplayers
							* myself->player->loop_current->step_abs);
					myself->player->loop_current->chunk_remainder =
						((abs(*__intone_end - *__intone_start)+1)
						/ myself->player->loop_current->step_abs) % 
							myself->player->loop_current->nplayers;
				}
				else{
					myself->player->loop_current->chunk_real =
						*__intone_chunk;
					myself->player->loop_current->chunk_remainder = 0;
				}
/* compute next_iter (only in static scheduler) */
				myself->player->loop_current->next_iter =
					*__intone_start
					+ (myself->player->loop_current->step_sign
						* myself->player->id
						* myself->player->loop_current->chunk_real
						* myself->player->loop_current->step_abs)
					+  myself->player->loop_current->step_sign
						* myself->player->loop_current->step_abs
						* nth_min(myself->player->id, myself->player->loop_current->chunk_remainder);
				break;
			case NANOS_WS_SCH_DYNAMIC:
				if (*__intone_chunk == 0) *__intone_chunk = 1;
				myself->player->loop_current->chunk_real = *__intone_chunk;
				myself->player->loop_current->chunk_remainder = 0;
				break;
			case NANOS_WS_SCH_GUIDED:
				/* xteruel:FIXME: guided scheduler is not implemented*/
				fprintf(stderr,"Error: Guided scheduler not implemented yet.\n");
				break;
			default:
				fprintf(stderr,
					"%s %d (in__tone_begin_for): "
					"Unknow scheduler type: 0x%x.\n",
					 __FILE__, __LINE__, (int) (myself->player->loop_current->schedule)
				);
				exit(NTH_NANOS_WS_ERROR); 
				break;
		} /* switch(schedule) */
		myself->player->loop_current->init = TRUE;
	}

	if (loop_shared) nth_spin_unlock(&myself->player->loop_current->mutex);

}

void nanos_ws_end_for(int *barrier_needed);
void in__tone_end_for (int *__intone_barrier_needed)
{
	nth_word_t nreached;
	nth_desc_t *myself = NTH_MYSELF;

	if (*__intone_barrier_needed) in__tone_barrier();

	if (myself->player->loop_current->shared){
		nreached = nth_atm_add(&NTH_MYSELF->player->loop_current->nreached, 1)+1;
		if (nreached == NTH_MYSELF->player->loop_current->nplayers){
			NTH_MYSELF->player->loop_current->init = FALSE;
		}
	}
	else{
		NTH_MYSELF->player->loop_current->init = FALSE;
	}

	NTH_MYSELF->player->loop_id =
		(NTH_MYSELF->player->loop_id+1) % NTH_DEFAULT_LOOPS_PER_DESCRIPTOR;

}

int nanos_ws_next_iters(int *lstart, int *lend, int *last);
int in__tone_next_iters (int *__intone_lstart, int *__intone_lend, int *__intone_last)
{
	int rv;

	if (!NTH_MYSELF->player->loop_current->init){
		/* xteruel:FIXME: wait over player->loop_current->init needed  */
		fprintf(stderr,"Error:\n");
		exit(NTH_NANOS_WS_ERROR);
	}

	if (NTH_MYSELF->player->loop_current->shared){
		nth_spin_lock(&NTH_MYSELF->player->loop_current->mutex);
	}

	switch(NTH_MYSELF->player->loop_current->schedule){
		case NANOS_WS_SCH_STATIC:
			/* return value and lstart */
			{
				/* do it this way due some compilers bug */
				int a = 
				(NTH_MYSELF->player->loop_current->step_sign
					* NTH_MYSELF->player->loop_current->next_iter);
				int b = 
				(NTH_MYSELF->player->loop_current->step_sign
					* NTH_MYSELF->player->loop_current->end);
				rv = (a <= b);
			}
			*__intone_lstart = (int) NTH_MYSELF->player->loop_current->next_iter;

			/* next_iter */
			if (NTH_MYSELF->player->loop_current->chunk == 0){
				NTH_MYSELF->player->loop_current->next_iter =
					NTH_MYSELF->player->loop_current->end
					+ NTH_MYSELF->player->loop_current->step;
			}
			else{
				NTH_MYSELF->player->loop_current->next_iter =
					NTH_MYSELF->player->loop_current->next_iter
					+ NTH_MYSELF->player->loop_current->step
					* NTH_MYSELF->player->loop_current->chunk
					* NTH_MYSELF->player->loop_current->nplayers;
			}
			break;
		case NANOS_WS_SCH_DYNAMIC:
			/* return value and lstart */
			{
				/* do it this way due some compilers bug */
				int a = 
				(NTH_MYSELF->player->loop_current->step_sign
					* NTH_MYSELF->player->loop_current->next_iter);
				int b = 
				(NTH_MYSELF->player->loop_current->step_sign
					* NTH_MYSELF->player->loop_current->end);
				rv = (a <= b);
			}
			*__intone_lstart = (int) NTH_MYSELF->player->loop_current->next_iter;
			/* next_iter */
			NTH_MYSELF->player->loop_current->next_iter =
					NTH_MYSELF->player->loop_current->next_iter
					+ NTH_MYSELF->player->loop_current->step
					* NTH_MYSELF->player->loop_current->chunk_real;
			break;
		case NANOS_WS_SCH_GUIDED:
			/* xteruel:FIXME: guided scheduler is not implemented*/
			fprintf(stderr,"Error: Guided scheduler is not implemented.\n");
		default:
			fprintf(stderr,
				"%s %d (in__tone_next_iters): "
				"Unknow scheduler type: 0x%x.\n",
				 __FILE__, __LINE__, (int) (NTH_MYSELF->player->loop_current->schedule)
			);
			exit(NTH_NANOS_WS_ERROR);
	}

	if (NTH_MYSELF->player->loop_current->step > 0){
		*__intone_lend = (int) NTH_MIN(
			*__intone_lstart
			+ (
				(NTH_MYSELF->player->loop_current->chunk_real)
				* NTH_MYSELF->player->loop_current->step
			) 
			+ (NTH_MYSELF->player->loop_current->chunk_remainder > NTH_MYSELF->player->id)
			* NTH_MYSELF->player->loop_current->step
			-1
			,
			NTH_MYSELF->player->loop_current->end
		);
	}
	else{
		*__intone_lend = (int) NTH_MAX(
			*__intone_lstart
			- (
				(NTH_MYSELF->player->loop_current->chunk_real - 1)
				* NTH_MYSELF->player->loop_current->step_abs
			)
			+ (NTH_MYSELF->player->loop_current->chunk_remainder > NTH_MYSELF->player->id)
			* NTH_MYSELF->player->loop_current->step
			,
			NTH_MYSELF->player->loop_current->end
		);
	}
		
	*__intone_last = (int) (*__intone_lend == NTH_MYSELF->player->loop_current->end);

	if (NTH_MYSELF->player->loop_current->shared){
		nth_spin_unlock(&NTH_MYSELF->player->loop_current->mutex);
	}

	if (!rv){
		*__intone_lstart = (int)0;
		*__intone_lend = (int)0;
		*__intone_last = (int)0;
	}

	return (int)rv;
}
#endif
