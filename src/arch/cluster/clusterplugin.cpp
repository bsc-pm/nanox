/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "plugin.hpp"
#include "system.hpp"
#include "gasnetapi_decl.hpp"
#include "clusterplugin_decl.hpp"
#include "remoteworkgroup_decl.hpp"


#define DEFAULT_NODE_MEM (0x542000000ULL) 
#define MAX_NODE_MEM     (0x542000000ULL) 

namespace nanos {
namespace ext {

ClusterPlugin::ClusterPlugin() : Plugin( "Cluster PE Plugin", 1 ), _gasnetApi( *this ),
_numPinnedSegments ( 0 ),
_pinnedSegmentAddrList ( NULL ), _pinnedSegmentLenList ( NULL ), _extraPEsCount ( 0 ), _conduit (""),
_nodeMem ( DEFAULT_NODE_MEM ), _allocWide ( false ), _gpuPresend ( 1 ), _smpPresend ( 1 ),
_cachePolicy ( System::DEFAULT )
{}

void ClusterPlugin::config( Config& cfg )
{
   cfg.setOptionsSection( "Cluster Arch", "Cluster specific options" );
   this->prepare( cfg );
}

void ClusterPlugin::init()
{
   sys.getNetwork()->setAPI(&_gasnetApi);
   sys.getNetwork()->initialize();
   sys.getNetwork()->setGpuPresend(this->getGpuPresend() );
   sys.getNetwork()->setSmpPresend(this->getSmpPresend() );

   this->setExtraPEsCount( 1 ); // We will use 1 paraver thread only to represent the soft-threads and the container. (extrae_get_thread_num must be coded acordingly
   sys.getNetwork()->setExtraPEsCount(this->getExtraPEsCount() );
}

void ClusterPlugin::addPinnedSegments( unsigned int numSegments, void **segmentAddr, std::size_t *segmentSize ) {
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

void * ClusterPlugin::getPinnedSegmentAddr( unsigned int idx ) {
   return _pinnedSegmentAddrList[ idx ];
}

std::size_t ClusterPlugin::getPinnedSegmentLen( unsigned int idx ) {
   return _pinnedSegmentLenList[ idx ];
}

#if 0
void ClusterPlugin::addSegments( unsigned int numSegments, void **segmentAddr, std::size_t *segmentSize ) {
   unsigned int idx;
   _numSegments = numSegments;
   _segmentAddrList = NEW void *[ numSegments ];
   _segmentLenList = NEW std::size_t[ numSegments ];

   for ( idx = 0; idx < numSegments; idx += 1)
   {
      _segmentAddrList[ idx ] = segmentAddr[ idx ];
      _segmentLenList[ idx ] = segmentSize[ idx ];
   }
}

void * ClusterPlugin::getSegmentAddr( unsigned int idx ) {
   return _segmentAddrList[ idx ];
}

std::size_t ClusterPlugin::getSegmentLen( unsigned int idx ) {
   return _segmentLenList[ idx ];
}
#endif

unsigned int ClusterPlugin::getExtraPEsCount() {
   return _extraPEsCount;
}

void ClusterPlugin::setExtraPEsCount( unsigned int num) {
   _extraPEsCount = num;
}

void ClusterPlugin::setUpCache() {
}

std::size_t ClusterPlugin::getNodeMem() {
   return _nodeMem;
}

int ClusterPlugin::getSmpPresend() {
   return _smpPresend;
}

int ClusterPlugin::getGpuPresend() {
   return _gpuPresend;
}

System::CachePolicyType ClusterPlugin::getCachePolicy ( void ) {
   return _cachePolicy;
}

RemoteWorkGroup * ClusterPlugin::getRemoteWorkGroup( int archId ) {
   return NEW RemoteWorkGroup( archId );
}

bool ClusterPlugin::getAllocWide() {
   return _allocWide;
}

void ClusterPlugin::prepare( Config& cfg ) {
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

}
}

nanos::ext::ClusterPlugin NanosXPlugin;

