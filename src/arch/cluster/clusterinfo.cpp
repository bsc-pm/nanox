#include "clusterinfo_decl.hpp"
#include "new_decl.hpp"

using namespace nanos;
using namespace ext;

#define DEFAULT_NODE_MEM (0xc0000000ULL) // 2 Gb of memory
#define MAX_NODE_MEM (0xc0000000ULL) // 2 Gb of memory

unsigned int ClusterInfo::_numSegments = 0;
void ** ClusterInfo::_segmentAddrList = NULL;
size_t * ClusterInfo::_segmentLenList = NULL;
unsigned int ClusterInfo::_extraPEsCount = 0;
std::string ClusterInfo::_conduit;
std::size_t ClusterInfo::_nodeMem = DEFAULT_NODE_MEM;

void ClusterInfo::addSegments( unsigned int numSegments, void **segmentAddr, size_t *segmentSize )
{
   unsigned int idx;
   _numSegments = numSegments;
   _segmentAddrList = new void *[ numSegments ];
   _segmentLenList = new size_t[ numSegments ];

   for ( idx = 0; idx < numSegments; idx += 1)
   {
      _segmentAddrList[ idx ] = segmentAddr[ idx ];
      _segmentLenList[ idx ] = segmentSize[ idx ];
   }
}

void * ClusterInfo::getSegmentAddr( unsigned int idx )
{
   return _segmentAddrList[ idx ];
}

size_t ClusterInfo::getSegmentLen( unsigned int idx )
{
   return _segmentLenList[ idx ];
}

unsigned int ClusterInfo::getExtraPEsCount()
{
   return _extraPEsCount;
}

void ClusterInfo::setExtraPEsCount( unsigned int num)
{
   _extraPEsCount = num;
}

void ClusterInfo::prepare( Config& cfg )
{
   /* Cluster: memory size to be allocated on remote nodes */
   cfg.registerConfigOption ( "node-memory", NEW Config::SizeVar ( _nodeMem ), "Sets the memory size that will be used on each node to send and receive data." );
   cfg.registerArgOption ( "node-memory", "cluster-node-memory" );
   cfg.registerEnvOption ( "node-memory", "NX_CLUSTER_NODE_MEMORY" );
}

std::size_t ClusterInfo::getNodeMem() {
   return _nodeMem;
}
