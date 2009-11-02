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

#ifndef _NANOS_H_
#define _NANOS_H_

#include <unistd.h>
#include <stdbool.h>

#ifdef _MERCURIUM_
// define API version
#pragma nanos interface family(master) version(5000)
#endif

// data types

// C++ types hidden as void *
typedef void * nanos_wg_t;
typedef void * nanos_wd_t;
typedef void * nanos_team_t;
typedef void * nanos_thread_t;
typedef void * nanos_sched_t;
typedef void * nanos_lock_t;
typedef void * nanos_dd_t;

// other types
typedef struct {
   void **address;
   struct {
     bool  input: 1;
     bool  output: 1;
     bool  can_rename:1;
   } flags;
   size_t  size;
} nanos_dependence_t;

typedef struct {
   bool mandatory_creation:1;
   bool tied:1;
   bool reserved0:1;
   bool reserved1:1;
   bool reserved2:1;
   bool reserved3:1;
   bool reserved4:1;
   bool reserved5:1;
   nanos_thread_t * tie_to;
   unsigned int priority;
} nanos_wd_props_t;

typedef struct {
   int nthreads;
   void *arch;
} nanos_constraint_t;

typedef enum { NANOS_OK=0,
               NANOS_UNKNOWN_ERR,          // generic error
               NANOS_UNIMPLEMENTED,        // service not implemented
} nanos_err_t;

// TODO: move smp to some dependent part
typedef struct {
   void (*outline) (void *);
} nanos_smp_args_t;

typedef struct {
  void * (*factory) (void *prealloc, void *arg);
  void * arg;
} nanos_device_t;

#ifdef __cplusplus

#define _Bool bool

extern "C" {
#endif
   
// Functions related to WD
nanos_wd_t nanos_current_wd (void);
int nanos_get_wd_id(nanos_wd_t wd);

nanos_err_t nanos_create_wd ( nanos_wd_t *wd, size_t num_devices, nanos_device_t *devices, size_t data_size,
                              void ** data, nanos_wg_t wg, nanos_wd_props_t *props );

nanos_err_t nanos_submit ( nanos_wd_t wd, nanos_dependence_t *deps, nanos_team_t team );

nanos_err_t nanos_create_wd_and_run ( size_t num_devices, nanos_device_t *devices, void * data,
                                      nanos_dependence_t *deps, nanos_wd_props_t *props );

nanos_err_t nanos_set_internal_wd_data ( nanos_wd_t wd, void *data );
nanos_err_t nanos_get_internal_wd_data ( nanos_wd_t wd, void **data );

// Team related functions

nanos_err_t nanos_create_team(nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads,
                              nanos_constraint_t * constraints, bool reuse, nanos_thread_t *info);

nanos_err_t nanos_create_team_mapped (nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads,                                                           unsigned int *mapping);

nanos_err_t nanos_end_team ( nanos_team_t team );

nanos_err_t nanos_team_barrier ( void );

nanos_err_t nanos_single_guard ( bool *);

// sync

nanos_err_t nanos_wg_wait_completation ( nanos_wg_t wg );
nanos_err_t nanos_wait_on_int ( volatile int *p, int condition );
nanos_err_t nanos_wait_on_bool ( volatile _Bool *p, _Bool condition );

nanos_err_t nanos_init_lock ( nanos_lock_t *lock );
nanos_err_t nanos_set_lock (nanos_lock_t lock);
nanos_err_t nanos_unset_lock (nanos_lock_t lock);
nanos_err_t nanos_try_lock ( nanos_lock_t lock, bool *result );
nanos_err_t nanos_destroy_lock ( nanos_lock_t lock );

// system interface
nanos_err_t nanos_get_num_running_tasks ( int *num );

// error handling

void nanos_handle_error ( nanos_err_t err );

// factories
void * nanos_smp_factory( void *prealloc ,void *args);

// utility macros

#define NANOS_SAFE(call) \
do {\
   nanos_err_t err = call;\
   if ( err != NANOS_OK ) nanos_handle_error(err);\
} while (0)

#ifdef __cplusplus
}
#endif

#endif
