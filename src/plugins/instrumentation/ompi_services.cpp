#include "system.hpp"
#include "smpthread.hpp"
#ifdef GPU_DEV
#include "gpuconfig.hpp"
#endif
#include "clusterinfo_decl.hpp"

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
#ifdef GPU_DEV
#ifdef CLUSTER_DEV
      /* GPU_DEV & CLUSTER_DEV */
      return sys.getMaxThreads() + nanos::ext::GPUConfig::getGPUCount() + nanos::ext::ClusterInfo::getExtraPEsCount();
#else
      /* GPU_DEV & no CLUSTER_DEV */
      return sys.getMaxThreads() + nanos::ext::GPUConfig::getGPUCount();
#endif
#else
#ifdef CLUSTER_DEV
      /* no GPU_DEV & CLUSTER_DEV */
      return sys.MaxThreads() + nanos::ext::ClusterInfo::getExtraPEsCount();
#else
      /* no GPU_DEV & no CLUSTER_DEV */
      return sys.MaxThreads();
#endif
#endif
   }

   unsigned int nanos_ompitrace_get_max_threads ( void )
   {
      return nanos_extrae_get_max_threads();
   }

   unsigned int nanos_extrae_get_thread_num ( void )
   { 
      if ( myThread == NULL )
         return 0;
      else if ( myThread->getParent() != NULL )
         return myThread->getParent()->getId();
      else
         return myThread->getId(); 
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

