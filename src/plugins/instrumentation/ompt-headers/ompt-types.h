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

#ifndef OMPT_TYPES_H_INCLUDED
#define OMPT_TYPES_H_INCLUDED

#define OMPT_API                            /* used to mark OMPT functions obtained from     *
                                             * lookup function passed to ompt_initialize     */ 

#define OMPT_TARG_API                       /* used to mark OMPT functions obtained from     *
                                             * lookup function passed to                     *
                                             * ompt_target_get_device_info                   */

typedef uint64_t ompt_thread_id_t;          /* uniquely identifies the thread                */ 

typedef uint64_t ompt_task_id_t;            /* uniquely identifies the task instance         */

typedef uint64_t ompt_parallel_id_t;        /* uniquely identifies the parallel instance     */

typedef uint64_t ompt_wait_id_t;            /* identify what a thread is awaiting            */

typedef uint64_t ompt_target_activity_id_t; /* ID of an activity on a device                 */

typedef uint32_t ompt_lock_type_t;          /* small integer encoding the type of a lock     */

typedef uint32_t ompt_lock_hint_t;          /* as described in the OpenMP language standard  */

typedef uint32_t ompt_bool;                 /* takes the values 0 (false) or 1 (true)        */

typedef void * ompt_target_device_t;        /* opaque object representing a target device    */

typedef uint64_t ompt_target_time_t;        /* raw time value on a device                    */

typedef void * ompt_target_buffer_t;        /* opaque handle for a target buffer             */ 

typedef uint64_t ompt_target_buffer_cursor_t;/* opaque handle for a position in target buffer*/ 

typedef enum ompt_thread_type_e {
  ompt_thread_initial  = 1,
  ompt_thread_worker   = 2,
  ompt_thread_other    = 3
} ompt_thread_type_t;

typedef enum ompt_target_task_type_e {
  ompt_target_task_target     = 1,
  ompt_target_task_enter_data = 2,
  ompt_target_task_exit_data  = 3,
  ompt_target_task_update     = 4
} ompt_target_task_type_t;

typedef enum ompt_native_mon_flags_e {
  ompt_native_data_motion_explicit     = 1,
  ompt_native_data_motion_implicit     = 2,
  ompt_native_kernel_invocation        = 4,
  ompt_native_kernel_execution         = 8,
  ompt_native_driver                   = 16,
  ompt_native_runtime                  = 32,
  ompt_native_overhead                 = 64,
  ompt_native_idleness                 = 128
} ompt_native_mon_flags_t;

typedef enum  ompt_task_type_e {
  ompt_task_serial     = 1,
  ompt_task_implicit   = 2,
  ompt_task_explicit   = 3,
  ompt_task_target     = 4,
  ompt_task_degenerate = 5
} ompt_task_type_t;

typedef enum ompt_invoker_e {
  ompt_invoker_program = 0,         /* program invokes master task  */
  ompt_invoker_runtime = 1          /* runtime invokes master task  */
} ompt_invoker_t;

typedef enum ompt_task_dependence_flag_e {
  // a two bit field for the dependence type
  ompt_task_dependence_type_out   = 1,
  ompt_task_dependence_type_in    = 2,
  ompt_task_dependence_type_inout = 3,
} ompt_task_dependence_flag_t;

typedef struct ompt_task_dependence_s {
  void *variable_addr;
  uint32_t  dependence_flags;
} ompt_task_dependence_t;

typedef enum ompt_target_map_flag_e {
  // a three bit field 0..7 for map type to/from/tofrom/alloc/release/delete
  ompt_target_map_flag_to      = 1,
  ompt_target_map_flag_from    = 2,
  ompt_target_map_flag_tofrom  = 3, 
  ompt_target_map_flag_alloc   = 4,
  ompt_target_map_flag_release = 5, 
  ompt_target_map_flag_delete  = 6, 
  // 7 unused

  // one bit for synchronous/asynchronous
  ompt_target_map_flag_sync = 8,   
} ompt_target_map_flag_t;

typedef struct ompt_frame_s {
  void *exit_runtime_frame;    /* next frame is user code     */
  void *reenter_runtime_frame; /* user frame that reenters the runtime  */
} ompt_frame_t;

#define ompt_hwid_none (-1)
#define ompt_dev_task_none (~0ULL)
#define ompt_time_none (~0ULL)

#define ompt_lock_type_first 0
#define ompt_atomic_type_first 0

#endif /* OMPT_TYPES_H_INCLUDED */
