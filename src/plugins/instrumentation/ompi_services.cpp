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

#include "system.hpp"
#include "smpthread.hpp"
#ifdef GPU_DEV
#include "gpuconfig.hpp"
#endif
#include "network_decl.hpp"

namespace nanos {

extern "C" {

   // Forward function declarations
   unsigned int   nanos_extrae_get_max_threads();
   unsigned int   nanos_ompitrace_get_max_threads();
   unsigned int   nanos_extrae_get_thread_num();
   unsigned int   nanos_ompitrace_get_thread_num();
   void           nanos_extrae_instrumentation_barrier();
   void           nanos_ompitrace_instrumentation_barrier();
   unsigned int   nanos_extrae_node_id();
   unsigned int   nanos_extrae_num_nodes();

   void OMPItrace_neventandcounters (unsigned int count, unsigned int *types, unsigned int *values);

   unsigned int nanos_extrae_get_max_threads ( void )
   {
      return sys.getSMPPlugin()->getNumThreads();
   }

   unsigned int nanos_ompitrace_get_max_threads ( void )
   {
      return nanos_extrae_get_max_threads();
   }

   unsigned int nanos_extrae_get_thread_num ( void )
   { 
      return myThread == NULL ? 0 : myThread->getOsId();
   }

   unsigned int nanos_ompitrace_get_thread_num ( void )
   {
      return nanos_extrae_get_thread_num();
   }

   void nanos_extrae_instrumentation_barrier ( void )
   {
#ifdef CLUSTER_DEV
      sys.getNetwork()->nodeBarrier();
#endif
   }

   void nanos_ompitrace_instrumentation_barrier ( void )
   {
      nanos_extrae_instrumentation_barrier();
   }

   unsigned int nanos_extrae_node_id ( void )
   {
#ifdef CLUSTER_DEV
      return sys.getNetwork()->getNodeNum();
#else
      return 0;
#endif
   }

   unsigned int nanos_extrae_num_nodes ( void )
   {
#ifdef CLUSTER_DEV
      return sys.getNetwork()->getNumNodes();
#else
      return 1;
#endif
   }
}

} // namespace nanos

