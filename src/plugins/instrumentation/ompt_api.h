/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#ifndef OMPT_API_H
#define OMPT_API_H
#include "ompt_callbacks.h"

#define OMPT_API

/* callback management */
OMPT_API int ompt_set_callback( /* register a callback for an event */
      ompt_event_t event, /* the event of interest */
      ompt_callback_t callback /* function pointer for the callback */
      );
OMPT_API int ompt_get_callback( /* return the current callback for an event (if any) */
      ompt_event_t event, /* the event of interest */
      ompt_callback_t *callback /* pointer to receive the return value */
      );
/* state inquiry */
OMPT_API int ompt_enumerate_state( /* extract the set of states supported */
      ompt_state_t current_state, /* current state in the enumeration */
      ompt_state_t *next_state, /* next state in the enumeration */
      const char **next_state_name /* string description of next state */
      );
/* thread inquiry */
OMPT_API ompt_thread_id_t ompt_get_thread_id( /* identify the current thread */
      void
      );
OMPT_API ompt_state_t ompt_get_state( /* get the state for a thread */
      ompt_wait_id_t *wait_id /* for wait states: identify what awaited */
      );
OMPT_API void * ompt_get_idle_frame( /* identify the idle frame (if any) for a thread */
      void
      );
/* parallel region inquiry */
OMPT_API ompt_parallel_id_t ompt_get_parallel_id( /* identify a parallel region */
      int ancestor_level /* how many levels the ancestor is removed from the current region */
      );
OMPT_API int ompt_get_parallel_team_size( /* query # threads in a parallel region */
      int ancestor_level /* how many levels the ancestor is removed from the current region */
      );
/* task inquiry */
OMPT_API ompt_task_id_t *ompt_get_task_id( /* identify a task */
      int depth /* how many levels removed from the current task */
      );
OMPT_API ompt_frame_t *ompt_get_task_frame(
      int depth /* how many levels removed from the current task */
      );
#endif
