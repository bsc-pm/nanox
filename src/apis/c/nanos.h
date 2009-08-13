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
  void * (*factory) (void *arg);
  void * arg;
} nanos_device_t;

#ifdef __cplusplus
extern "C" {
#endif
   
// Functions related to WD
nanos_wd_t nanos_current_wd (void);

nanos_err_t nanos_create_wd ( nanos_wd_t *wd, size_t num_devices, nanos_device_t *devices, size_t data_size,
                              void ** data, nanos_wg_t wg, nanos_wd_props_t *props );

nanos_err_t nanos_submit ( nanos_wd_t wd, nanos_dependence_t *deps, nanos_team_t team );

nanos_err_t nanos_create_wd_and_run ( size_t num_devices, nanos_device_t *devices, void * data,
                                      nanos_dependence_t *deps, nanos_wd_props_t *props );

// Team related functions

nanos_err_t nanos_create_team(nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads,
                              nanos_constraint_t * constraints, bool reuse, nanos_thread_t *info);

nanos_err_t nanos_create_team_mapped (nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads,                                                           unsigned int *mapping);

nanos_err_t nanos_end_team ( nanos_team_t team, bool need_barrier);

nanos_err_t nanos_team_barrier ( void );

// wg

nanos_err_t nanos_wg_wait_completation ( nanos_wg_t wg );

// factories
void * nanos_smp_factory(void *args);

#ifdef __cplusplus
}
#endif

#endif
