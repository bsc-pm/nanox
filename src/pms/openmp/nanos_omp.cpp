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

#include "nanos.h"
#include "os.hpp"
#include "cpuset.hpp"
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

NANOS_API_DEF(void, nanos_omp_get_process_mask, ( nanos_cpu_set_t cpu_set ))
{
   const CpuSet& process_mask = sys.getPMInterface().getCpuProcessMask();
   process_mask.copyTo( static_cast<cpu_set_t*>(cpu_set) );
}

NANOS_API_DEF(int, nanos_omp_set_process_mask, ( const_nanos_cpu_set_t cpu_set ))
{
   bool b = sys.getPMInterface().setCpuProcessMask( static_cast<const cpu_set_t*>(cpu_set) );
   return (b) ? 0 : -1;
}

NANOS_API_DEF(void, nanos_omp_add_process_mask, ( const_nanos_cpu_set_t cpu_set ))
{
   sys.getPMInterface().addCpuProcessMask( static_cast<const cpu_set_t*>(cpu_set) );
}

NANOS_API_DEF(void, nanos_omp_get_active_mask, ( nanos_cpu_set_t cpu_set ))
{
   const CpuSet& active_mask = sys.getCpuActiveMask();
   active_mask.copyTo( static_cast<cpu_set_t*>(cpu_set) );
}

NANOS_API_DEF(int, nanos_omp_set_active_mask, ( const_nanos_cpu_set_t cpu_set ))
{
   bool b = sys.getPMInterface().setCpuActiveMask( static_cast<const cpu_set_t*>(cpu_set) );
   return (b) ? 0 : -1;
}

NANOS_API_DEF(void, nanos_omp_add_active_mask, ( const_nanos_cpu_set_t cpu_set ))
{
   sys.getPMInterface().addCpuActiveMask( static_cast<const cpu_set_t*>(cpu_set) );
}

NANOS_API_DEF(int, nanos_omp_enable_cpu, ( int cpuid ))
{
   try {
      sys.getPMInterface().enableCpu( cpuid );
   } catch ( nanos_err_t e) {
      return e;
   }
   return NANOS_OK;
}

NANOS_API_DEF(int, nanos_omp_disable_cpu, ( int cpuid ))
{
   try {
      sys.getPMInterface().disableCpu( cpuid );
   } catch ( nanos_err_t e) {
      return e;
   }
   return NANOS_OK;
}

NANOS_API_DEF ( int, nanos_omp_get_max_processors, (void ) )
{
   return nanos::OS::getMaxProcessors();
}

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
      if ( sys.getPMInterface().isOmpSs() ) {
         return NANOS_UNIMPLEMENTED;
      }

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

