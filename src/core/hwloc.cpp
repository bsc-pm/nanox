#include "hwloc_decl.hpp"
#include "debug.hpp"

#ifdef HWLOC
 #ifdef GPU_DEV
  #include <hwloc/cudart.h>
 #endif
#endif

namespace nanos {

Hwloc::Hwloc() 
#ifdef HWLOC
   : _hwlocTopology( NULL ) , _topologyPath() 
#endif
{
}

Hwloc::~Hwloc() {
}

bool Hwloc::isHwlocAvailable () const
{
#ifdef HWLOC
   return true;
#else
   return false;
#endif
}

void Hwloc::loadHwloc ()
{
#ifdef HWLOC
   // Allocate and initialize topology object.
   hwloc_topology_init( &_hwlocTopology );

   // If the user provided an alternate topology
   //if ( !_topologyPath.empty() )
   //{
   //   int res = hwloc_topology_set_xml( _hwlocTopology, _topologyPath.c_str() );
   //   fatal_cond0( res != 0, "Could not load hwloc topology xml file." );
   //}

   // Enable GPU detection
   hwloc_topology_set_flags( _hwlocTopology, HWLOC_TOPOLOGY_FLAG_IO_DEVICES );

   // Perform the topology detection.
   hwloc_topology_load( _hwlocTopology );
#endif
}


void Hwloc::unloadHwloc ()
{
#ifdef HWLOC
   /* Destroy topology object. */
   hwloc_topology_destroy( _hwlocTopology );
#endif
}


unsigned int Hwloc::getNumaNodeOfCpu ( unsigned int cpu )
{
   int numaNodeId = 0;
#ifdef HWLOC
   loadHwloc();
   hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index( _hwlocTopology, cpu );

   // Now we have the PU object, go find its parent numa node
   hwloc_obj_t numaNode =
      hwloc_get_ancestor_obj_by_type( _hwlocTopology, HWLOC_OBJ_NODE, pu );

   // If the machine is not NUMA
   if ( numaNode != NULL )
   {
      numaNodeId = numaNode->os_index;
   }

   unloadHwloc();
   return numaNodeId;
#else
   return numaNodeId;
#endif
}

void Hwloc::getNumSockets(unsigned int &allowedNodes, int &numSockets, unsigned int &hwThreads) {
#ifdef HWLOC
   loadHwloc();
   numSockets = 0;
   // Nodes that can be seen by hwloc
   allowedNodes = 0;
   // Hardware threads
   hwThreads = 0;

   int depth = hwloc_get_type_depth( _hwlocTopology, HWLOC_OBJ_NODE );
   // If there are NUMA nodes in this machine
   if ( depth != HWLOC_TYPE_DEPTH_UNKNOWN ) {
      //hwloc_const_cpuset_t cpuset = hwloc_topology_get_online_cpuset( _hwlocTopology );
      //allowedNodes = hwloc_get_nbobjs_inside_cpuset_by_type( _hwlocTopology, cpuset, HWLOC_OBJ_NODE );
      //hwThreads = hwloc_get_nbobjs_inside_cpuset_by_type( _hwlocTopology, cpuset, HWLOC_OBJ_PU );
      unsigned nodes = hwloc_get_nbobjs_by_depth( _hwlocTopology, depth );
      //hwloc_cpuset_t set = i

      // For each node, count how many hardware threads there are below.
      for ( unsigned nodeIdx = 0; nodeIdx < nodes; ++nodeIdx )
      {
         hwloc_obj_t node = hwloc_get_obj_by_depth( _hwlocTopology, depth, nodeIdx );
         int localThreads = hwloc_get_nbobjs_inside_cpuset_by_type( _hwlocTopology, node->cpuset, HWLOC_OBJ_PU );
         // Increase hw thread count
         hwThreads += localThreads;
         // If this node has hw threads beneath, increase the number of viewable nodes
         if ( localThreads > 0 ) ++allowedNodes;
      }
      numSockets = nodes;
   }
   // Otherwise, set it to 1
   else {
      allowedNodes = 1; 
      numSockets = 1;
   }
   unloadHwloc();
#else
   numSockets = 0;
   allowedNodes = 0;
#endif
}

unsigned int Hwloc::getNumaNodeOfGpu( unsigned int gpu ) {
   unsigned int node = 0;
#ifdef GPU_DEV
#ifdef HWLOC
   loadHwloc();
   hwloc_obj_t obj = hwloc_cudart_get_device_pcidev ( _hwlocTopology, gpu );
   if ( obj != NULL ) {
      hwloc_obj_t objNode = hwloc_get_ancestor_obj_by_type( _hwlocTopology, HWLOC_OBJ_NODE, obj );
      if ( objNode != NULL ){
         node = objNode->os_index;
      }
   }
   unloadHwloc();
#endif
#endif
   return node;
}

}
