#include "nanos.h"
#include "omp_init.hpp"
#include "nanos_omp.h"
#include "omp_wd_data.hpp"
#include "basethread.hpp"
#include "instrumentationmodule_decl.hpp"

namespace nanos
{
   namespace OpenMP {
      extern nanos_ws_t  ws_plugins[NANOS_OMP_WS_TSIZE];
   };
};

using namespace nanos;
using namespace nanos::OpenMP;

NANOS_API_DEF(nanos_err_t, nanos_omp_set_implicit, ( nanos_wd_t uwd ))
{
    WD *wd = (WD *) uwd;
    wd->setImplicit(true);

    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_omp_single, ( bool *b ))
{
    if ( myThread->getCurrentWD()->isImplicit() ) return nanos_single_guard(b);

    *b=true;
    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_omp_barrier, ( void ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","omp_barrier",NANOS_SYNCHRONIZATION) );

   try {
      WD &wd = *myThread->getCurrentWD();
    
      wd.waitCompletion();
      if ( wd.isImplicit() ) {
         myThread->getTeam()->barrier();
      }
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_ws_t, nanos_omp_find_worksharing,( nanos_omp_sched_t kind ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","omp_find_worksharing",NANOS_SYNCHRONIZATION) );
   return ((OpenMPInterface&)sys.getPMInterface()).findWorksharing(kind);
}

NANOS_API_DEF(nanos_err_t, nanos_omp_get_schedule, ( nanos_omp_sched_t *kind, int *modifier ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","omp_get_schedule",NANOS_RUNTIME) );
   try {
      omp_get_schedule ( reinterpret_cast<omp_sched_t *> (kind), modifier);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}
