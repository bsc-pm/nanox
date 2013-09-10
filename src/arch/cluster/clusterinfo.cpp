#include "clusterinfo_decl.hpp"
#include "new_decl.hpp"
#include "system.hpp"
#include "config.hpp"
#include "remoteworkgroup_decl.hpp"

using namespace nanos;
using namespace ext;

#define DEFAULT_NODE_MEM (0x540000000ULL) 
#define MAX_NODE_MEM     (0x540000000ULL) 

unsigned int ClusterInfo::_numSegments = 0;
void ** ClusterInfo::_segmentAddrList = NULL;
std::size_t * ClusterInfo::_segmentLenList = NULL;
unsigned int ClusterInfo::_numPinnedSegments = 0;
void ** ClusterInfo::_pinnedSegmentAddrList = NULL;
std::size_t * ClusterInfo::_pinnedSegmentLenList = NULL;
unsigned int ClusterInfo::_extraPEsCount = 0;
std::string ClusterInfo::_conduit;
std::size_t ClusterInfo::_nodeMem = DEFAULT_NODE_MEM;
bool ClusterInfo::_allocWide = false;
int ClusterInfo::_gpuPresend = 1;
int ClusterInfo::_smpPresend = 1;
System::CachePolicyType ClusterInfo::_cachePolicy = System::DEFAULT;

ClusterInfo::ClusterInfo() {
}

ClusterInfo::~ClusterInfo() {
   if ( _segmentAddrList != NULL )
      delete _segmentAddrList;
   if ( _segmentLenList != NULL )
      delete _segmentLenList;
}

void ClusterInfo::addPinnedSegments( unsigned int numSegments, void **segmentAddr, std::size_t *segmentSize ) {
   unsigned int idx;
   _numPinnedSegments = numSegments;
   _pinnedSegmentAddrList = new void *[ numSegments ];
   _pinnedSegmentLenList = new std::size_t[ numSegments ];

   for ( idx = 0; idx < numSegments; idx += 1)
   {
      _pinnedSegmentAddrList[ idx ] = segmentAddr[ idx ];
      _pinnedSegmentLenList[ idx ] = segmentSize[ idx ];
   }
}

void * ClusterInfo::getPinnedSegmentAddr( unsigned int idx ) {
   return _pinnedSegmentAddrList[ idx ];
}

std::size_t ClusterInfo::getPinnedSegmentLen( unsigned int idx ) {
   return _pinnedSegmentLenList[ idx ];
}

void ClusterInfo::addSegments( unsigned int numSegments, void **segmentAddr, std::size_t *segmentSize ) {
   unsigned int idx;
   _numSegments = numSegments;
   _segmentAddrList = new void *[ numSegments ];
   _segmentLenList = new std::size_t[ numSegments ];

   for ( idx = 0; idx < numSegments; idx += 1)
   {
      _segmentAddrList[ idx ] = segmentAddr[ idx ];
      _segmentLenList[ idx ] = segmentSize[ idx ];
   }
}

void * ClusterInfo::getSegmentAddr( unsigned int idx ) {
   return _segmentAddrList[ idx ];
}

std::size_t ClusterInfo::getSegmentLen( unsigned int idx ) {
   return _segmentLenList[ idx ];
}

unsigned int ClusterInfo::getExtraPEsCount() {
   return _extraPEsCount;
}

void ClusterInfo::setExtraPEsCount( unsigned int num) {
   _extraPEsCount = num;
}

void ClusterInfo::prepare( Config& cfg ) {
   /* Cluster: memory size to be allocated on remote nodes */
   cfg.registerConfigOption ( "node-memory", NEW Config::SizeVar ( _nodeMem ), "Sets the memory size that will be used on each node to send and receive data." );
   cfg.registerArgOption ( "node-memory", "cluster-node-memory" );
   cfg.registerEnvOption ( "node-memory", "NX_CLUSTER_NODE_MEMORY" );

   cfg.registerConfigOption ( "cluster-alloc-wide", NEW Config::FlagOption( _allocWide ), "Allocate full objects.");
   cfg.registerArgOption( "cluster-alloc-wide", "cluster-alloc-wide" );

   cfg.registerConfigOption ( "cluster-gpu-presend", NEW Config::IntegerVar ( _gpuPresend ), "Number of Tasks to be sent to a remote node without waiting waiting any completion (GPU)." );
   cfg.registerArgOption ( "cluster-gpu-presend", "cluster-gpu-presend" );
   cfg.registerEnvOption ( "cluster-gpu-presend", "NX_CLUSTER_GPU_PRESEND" );

   cfg.registerConfigOption ( "cluster-smp-presend", NEW Config::IntegerVar ( _smpPresend ), "Number of Tasks to be sent to a remote node without waiting waiting any completion (SMP)." );
   cfg.registerArgOption ( "cluster-smp-presend", "cluster-smp-presend" );
   cfg.registerEnvOption ( "cluster-smp-presend", "NX_CLUSTER_SMP_PRESEND" );

   System::CachePolicyConfig *cachePolicyCfg = NEW System::CachePolicyConfig ( _cachePolicy );
   cachePolicyCfg->addOption("wt", System::WRITE_THROUGH );
   cachePolicyCfg->addOption("wb", System::WRITE_BACK );
   cachePolicyCfg->addOption("no", System::NONE );
   cfg.registerConfigOption ( "cluster-cache-policy", cachePolicyCfg, "Defines the cache policy for Cluster architectures: write-through / write-back (wb by default)" );
   cfg.registerEnvOption ( "cluster-cache-policy", "NX_CLUSTER_CACHE_POLICY" );
   cfg.registerArgOption( "cluster-cache-policy", "cluster-cache-policy" );
}

void ClusterInfo::setUpCache() {
}

std::size_t ClusterInfo::getNodeMem() {
   return _nodeMem;
}

int ClusterInfo::getSmpPresend() {
   return _smpPresend;
}

int ClusterInfo::getGpuPresend() {
   return _gpuPresend;
}

System::CachePolicyType ClusterInfo::getCachePolicy ( void ) {
   return _cachePolicy;
}

RemoteWorkGroup * ClusterInfo::getRemoteWorkGroup( int archId ) {
   return NEW RemoteWorkGroup( archId );
}

bool ClusterInfo::getAllocWide() {
   return _allocWide;
}
