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

#ifndef OMPT_TYPES_H
#define OMPT_TYPES_H

#include <inttypes.h>

typedef enum {
   /*--- Mandatory Events ---*/
   ompt_event_parallel_begin = 1,                /* parallel create */
   ompt_event_parallel_end = 2,                  /* parallel exit */
   ompt_event_task_begin = 3,                    /* task create */
   ompt_event_task_end = 4,                      /* task destroy */
   ompt_event_thread_begin = 5,                  /* thread begin */
   ompt_event_thread_end = 6,                    /* thread end */
   ompt_event_control = 7,                       /* support control calls */
   ompt_event_runtime_shutdown = 8,              /* runtime shutdown */
   /*--- Optional Events (blame shifting) ---*/
   ompt_event_idle_begin = 9,                    /* begin idle state */
   ompt_event_idle_end = 10,                     /* end idle state */
   ompt_event_wait_barrier_begin = 11,           /* begin wait at barrier */
   ompt_event_wait_barrier_end = 12,             /* end wait at barrier */
   ompt_event_wait_taskwait_begin = 13,          /* begin wait at taskwait */
   ompt_event_wait_taskwait_end = 14,            /* end wait at taskwait */
   ompt_event_wait_taskgroup_begin = 15,         /* begin wait at taskgroup */
   ompt_event_wait_taskgroup_end = 16,           /* end wait at taskgroup */
   ompt_event_release_lock = 17,                 /* lock release */
   ompt_event_release_nest_lock_last = 18,       /* last nest lock release */
   ompt_event_release_critical = 19,             /* critical release */
   ompt_event_release_atomic = 20,               /* atomic release */
   ompt_event_release_ordered = 21,              /* ordered release */
   /*--- Optional Events (synchronous events) --- */
   ompt_event_implicit_task_begin = 22,          /* implicit task create */
   ompt_event_implicit_task_end = 23,            /* implicit task destroy */
   ompt_event_initial_task_begin = 24,           /* initial task create */
   ompt_event_initial_task_end = 25,             /* initial task destroy */
   ompt_event_task_switch = 26,                  /* task switch */
   ompt_event_loop_begin = 27,                   /* task at loop begin */
   ompt_event_loop_end = 28,                     /* task at loop end */
   ompt_event_sections_begin = 29,               /* task at section begin */
   ompt_event_sections_end = 30,                 /* task at section end */
   ompt_event_single_in_block_begin = 31,        /* task at single begin */
   ompt_event_single_in_block_end = 32,          /* task at single end */
   ompt_event_single_others_begin = 33,          /* task at single begin */
   ompt_event_single_others_end = 34,            /* task at single end */
   ompt_event_workshare_begin = 35,              /* task at workshare begin */
   ompt_event_workshare_end = 36,                /* task at workshare end */
   ompt_event_master_begin = 37,                 /* task at master begin */
   ompt_event_master_end = 38,                   /* task at master end */
   ompt_event_barrier_begin = 39,                /* task at barrier begin */
   ompt_event_barrier_end = 40,                  /* task at barrier end */
   ompt_event_taskwait_begin = 41,               /* task at taskwait begin */
   ompt_event_taskwait_end = 42,                 /* task at task wait end */
   ompt_event_taskgroup_begin = 43,              /* task at taskgroup begin */
   ompt_event_taskgroup_end = 44,                /* task at taskgroup end */
   ompt_event_release_nest_lock_prev = 45,       /* prev nest lock release */
   ompt_event_wait_lock = 46,                    /* lock wait */
   ompt_event_wait_nest_lock = 47,               /* nest lock wait */
   ompt_event_wait_critical = 48,                /* critical wait */
   ompt_event_wait_atomic = 49,                  /* atomic wait */
   ompt_event_wait_ordered = 50,                 /* ordered wait */
   ompt_event_acquired_lock = 51,                /* lock acquired */
   ompt_event_acquired_nest_lock_first = 52,     /* 1st nest lock acquired */
   ompt_event_acquired_nest_lock_next = 53,      /* next nest lock acquired */
   ompt_event_acquired_critical = 54,            /* critical acquired */
   ompt_event_acquired_atomic = 55,              /* atomic acquired */
   ompt_event_acquired_ordered = 56,             /* ordered acquired */
   ompt_event_init_lock = 57,                    /* lock init */
   ompt_event_init_nest_lock = 58,               /* nest lock init */
   ompt_event_destroy_lock = 59,                 /* lock destruction */
   ompt_event_destroy_nest_lock = 60,            /* nest lock destruction */
   ompt_event_flush = 61,                        /* after executing flush */
   ompt_event_dependence = 62                    /* when a dependence is found */
} ompt_event_t;

typedef enum {
   /* work states (0..15) */
   ompt_state_work_serial = 0x00, /* working outside parallel */
   ompt_state_work_parallel = 0x01, /* working within parallel */
   ompt_state_work_reduction = 0x02, /* performing a reduction */
   /* idle (16..31) */
   ompt_state_idle = 0x10, /* waiting for work */
   /* overhead states (32..63) */
   ompt_state_overhead = 0x20, /* non-wait overhead */
   /* barrier wait states (64..79) */
   ompt_state_wait_barrier = 0x40, /* generic barrier */
   ompt_state_wait_barrier_implicit = 0x41, /* implicit barrier */
   ompt_state_wait_barrier_explicit = 0x42, /* explicit barrier */
   /* task wait states (80..95) */
   ompt_state_wait_taskwait = 0x50, /* waiting at a taskwait */
   ompt_state_wait_taskgroup = 0x51, /* waiting at a taskgroup */
   /* mutex wait states (96..111) */
   ompt_state_wait_lock = 0x60, /* waiting for lock */
   ompt_state_wait_nest_lock = 0x61, /* waiting for nest lock */
   ompt_state_wait_critical = 0x62, /* waiting for critical */
   ompt_state_wait_atomic = 0x63, /* waiting for atomic */
   ompt_state_wait_ordered = 0x64, /* waiting for ordered */
   /* misc (112.127) */
   ompt_state_undefined = 0x70, /* undefined thread state */
   ompt_state_first = 0x71, /* initial enumeration state */
} ompt_state_t;

typedef uint64_t ompt_thread_id_t;
typedef uint64_t ompt_wait_id_t;
typedef uint64_t ompt_task_id_t;
typedef uint64_t ompt_parallel_id_t;
#if 0
typedef struct ompt_frame_s
{
	void *exit_runtime_frame; /* next frame is user code */
	void *reenter_runtime_frame; /* previous frame is user code */
} ompt_frame_t;
#else
typedef uint64_t ompt_frame_t;
#endif

typedef enum {
   ompt_dependence_raw = 1,
   ompt_dependence_war = 2,
   ompt_dependence_waw = 3
} ompt_dependence_type_t;

#endif
