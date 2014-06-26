#ifndef HWLOC_DECL
#define HWLOC_DECL
#include <string>

#ifdef HWLOC
#include <hwloc.h>
#endif

namespace nanos {

class Hwloc {
#ifdef HWLOC
   //! hwloc topology structure
   hwloc_topology_t             _hwlocTopology;
   //! Path to a hwloc topology xml
   std::string                  _topologyPath;
#endif

   public:

   Hwloc();
   ~Hwloc();
   bool isHwlocAvailable() const;

   void loadHwloc();
   void unloadHwloc();
   unsigned int getNumaNodeOfCpu( unsigned int cpu );
   unsigned int getNumaNodeOfGpu( unsigned int gpu );
   void getNumSockets(unsigned int &allowedNodes, int &numSockets, unsigned int &hwThreads);

};

}

#endif /* HWLOC_DECL */
