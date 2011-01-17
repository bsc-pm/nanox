#include "system.hpp"
#ifdef GPU_DEV
#include "gpuconfig.hpp"
#endif
#include "clusterinfo.hpp"

namespace nanos {

extern "C" {

   void OMPItrace_neventandcounters (unsigned int count, unsigned int *types, unsigned int *values);

   unsigned int nanos_extrae_get_max_threads ( void )
   {
#ifdef GPU_DEV
      return sys.getNumPEs() + nanos::ext::GPUConfig::getGPUCount();
#else

#ifdef CLUSTER_DEV
      return sys.getNumPEs() + nanos::ext::ClusterInfo::getExtraPEsCount();
#else
      return sys.getNumPEs();
#endif

#endif
   }

   unsigned int nanos_extrae_get_thread_num ( void )
   { 
      if ( myThread == NULL ) return 0;
      else return myThread->getId(); 
   }

   void nanos_extrae_instrumentation_barrier ( void )
   {
#ifdef CLUSTER_DEV
      sys.getNetwork()->nodeBarrier();
#endif
   }

   unsigned int nanos_extrae_node_id ( void )
   {
      return sys.getNetwork()->getNodeNum();
   }

   unsigned int nanos_extrae_num_nodes ( void )
   {
      return sys.getNetwork()->getNumNodes();
   }
}

} // namespace nanos

