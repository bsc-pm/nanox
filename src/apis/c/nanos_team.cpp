#include "nanos.h"

nanos_err_t nanos_create_team(nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads,
                              nanos_constraint_t * constraints, bool reuse, nanos_thread_t *info)
{
   return NANOS_UNIMPLEMENTED;
}

nanos_err_t nanos_create_team_mapped (nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads,
                                      unsigned int *mapping)
{
   return NANOS_UNIMPLEMENTED;
}

nanos_err_t nanos_end_team ( nanos_team_t team, bool need_barrier)
{
   return NANOS_UNIMPLEMENTED;
}

nanos_err_t nanos_team_barrier ( void )
{
   return NANOS_UNIMPLEMENTED;
}