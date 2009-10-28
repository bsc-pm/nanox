#include "nanos.h"
#include "system.hpp"
#include "debug.hpp"

using namespace nanos;

nanos_err_t nanos_create_team(nanos_team_t *team, nanos_sched_t sp, unsigned int *nthreads,
                              nanos_constraint_t * constraints, bool reuse, nanos_thread_t *info)
{
   try {
       if ( *team ) warning("pre-allocated team not supported yet");
       
       ThreadTeam *new_team = sys.createTeam(*nthreads,(SG *)sp,constraints,reuse);
       *team = new_team;
       *nthreads = new_team->size();
       for ( unsigned i = 0; i < new_team->size(); i++ )
          info[i] = (nanos_thread_t) &(*new_team)[i];
   } catch (...) {
      return NANOS_UNKNOWN_ERR;
   }
   
   return NANOS_UNIMPLEMENTED;
}

nanos_err_t nanos_create_team_mapped (nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads,
                                      unsigned int *mapping)
{
   return NANOS_UNIMPLEMENTED;
}

nanos_err_t nanos_end_team ( nanos_team_t team )
{
   return NANOS_UNIMPLEMENTED;
}

nanos_err_t nanos_team_barrier ( void )
{
   return NANOS_UNIMPLEMENTED;
}


