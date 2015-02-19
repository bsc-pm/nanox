#ifndef OMPT_CALLBACKS_H
#define OMPT_CALLBACKS_H

#include "ompt_types.h"

/* initialization */
typedef void (*ompt_interface_fn_t)( void);
typedef ompt_interface_fn_t (*ompt_function_lookup_t)(
      const char *entry_point /* entry point to look up */
      );

/* threads */
typedef void (*ompt_thread_callback_t) ( /* for thread */
               ompt_thread_id_t thread_id /* ID of thread */
      );
typedef enum ompt_thread_type_e {
   ompt_thread_initial = 1,
   ompt_thread_worker = 2,
   ompt_thread_other = 3
} ompt_thread_type_t;
typedef void (*ompt_thread_type_callback_t) ( /* for thread */
      ompt_thread_type_t thread_type, /* type of thread */
      ompt_thread_id_t thread_id /* ID of thread */
      );
typedef void (*ompt_wait_callback_t) ( /* for wait */
      ompt_wait_id_t wait_id /* wait ID */
      );

/* parallel & workshares */
typedef void (*ompt_parallel_callback_t) ( /* for inside parallel */
      ompt_parallel_id_t parallel_id, /* ID of parallel region */
      ompt_task_id_t task_id /* ID of task */
      );
typedef void (*ompt_new_workshare_callback_t) ( /* for workshares */
      ompt_parallel_id_t parallel_id, /* ID of parallel region */
      ompt_task_id_t task_id, /* ID of task */
      void *workshare_function /* pointer to outlined function */
      );
typedef void (*ompt_new_parallel_callback_t) ( /* for new parallel */
      ompt_task_id_t parent_task_id, /* ID of parent task */
      ompt_frame_t *parent_task_frame, /* frame data of parent task */
      ompt_parallel_id_t parallel_id, /* ID of parallel region */
      uint32_t requested_team_size, /* requested number of threads */
      void *parallel_function /* pointer to outlined function */
      );

/* tasks */
typedef void (*ompt_task_callback_t) ( /* for tasks */
      ompt_task_id_t task_id /* ID of task */
      );
typedef void (*ompt_task_switch_callback_t) ( /* for task switch */
      ompt_task_id_t suspended_task_id, /* ID of suspended task */
      ompt_task_id_t resumed_task_id /* ID of resumed task */
      );

typedef void (*ompt_new_task_callback_t) ( /* for new tasks */
      ompt_task_id_t parent_task_id, /* ID of parent task */
      ompt_frame_t *parent_task_frame, /* frame data for parent task */
      ompt_task_id_t new_task_id, /* ID of created task */
      void *new_task_function /* pointer to outlined function */
      );

/* program */
typedef void (*ompt_control_callback_t) ( /* for control */
      uint64_t command, /* command of control call */
      uint64_t modifier /* modifier of control call */
      );
typedef void (*ompt_callback_t)( /* for shutdown */
      void
      );

typedef void (*ompt_new_dependence_callback_t) ( /* for new dependence instrumentation */
      ompt_task_id_t pred_task_id, /* ID of predecessor task */
      ompt_task_id_t succ_task_id, /* ID of successor task */
      ompt_dependence_type_t type, /* Type of dependence */
      void *data                   /* Pointer to related data */
      );

#endif
