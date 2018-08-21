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

#include "plugin.hpp"
#include "system.hpp"
#include "gasnetapi_decl.hpp"
#include "clusterplugin_decl.hpp"
#include "clusternode_decl.hpp"
#include "remoteworkdescriptor_decl.hpp"
#include "basethread.hpp"
#include "smpprocessor.hpp"
#ifdef OpenCL_DEV
#include "opencldd.hpp"
#endif
#ifdef GPU_DEV
#include "gpudd.hpp"
#endif
#ifdef FPGA_DEV
#include "fpgadd.hpp"
#endif

#if defined(__SIZEOF_SIZE_T__)
   #if  __SIZEOF_SIZE_T__ == 8

#define DEFAULT_NODE_MEM (0x542000000ULL)
#define MAX_NODE_MEM     (0x542000000ULL)

   #elif __SIZEOF_SIZE_T__ == 4

#define DEFAULT_NODE_MEM (0x40000000UL)
#define MAX_NODE_MEM     (0x40000000UL)

   #else
      #error "Weird"
   #endif
#else
   #error "I need to know the size of a size_t"
#endif


namespace nanos {
namespace ext {

ClusterPlugin::ClusterPlugin() : ArchPlugin( "Cluster PE Plugin", 1 ),
   _gasnetApi( NEW GASNetAPI() ), _numPinnedSegments( 0 ), _pinnedSegmentAddrList( NULL ),
   _pinnedSegmentLenList( NULL ), _extraPEsCount( 0 ), _conduit(""),
   _nodeMem( DEFAULT_NODE_MEM ), _allocFit( false ), _allowSharedThd( false ),
   _unalignedNodeMem( false ), _gpuPresend( 1 ), _smpPresend( 1 ),
   _cachePolicy( System::DEFAULT ), _remoteNodes( NULL ), _cpu( NULL ),
   _clusterThread( NULL ), _gasnetSegmentSize( 0 ) {
}

void ClusterPlugin::config( Config& cfg )
{
   cfg.setOptionsSection( "Cluster Arch", "Cluster specific options" );
   this->prepare( cfg );
}

void ClusterPlugin::init()
{
   _gasnetApi->initialize( sys.getNetwork() );
   //sys.getNetwork()->setAPI(_gasnetApi);
   _gasnetApi->setGASNetSegmentSize( _gasnetSegmentSize );
   _gasnetApi->setUnalignedNodeMemory( _unalignedNodeMem );
   sys.getNetwork()->initialize( _gasnetApi );
   sys.getNetwork()->setGpuPresend(this->getGpuPresend() );
   sys.getNetwork()->setSmpPresend(this->getSmpPresend() );

   unsigned int nodes = _gasnetApi->getNumNodes();

   if ( nodes > 1 ) {
      if ( _gasnetApi->getNodeNum() == 0 ) {
         void *segmentAddr[ nodes ];
         sys.getNetwork()->mallocSlaves( &segmentAddr[ 1 ], _nodeMem );
         segmentAddr[ 0 ] = NULL;

         ClusterNode::ClusterSupportedArchMap supported_archs;
         supported_archs[0] = &getSMPDevice();

         #ifdef GPU_DEV
         supported_archs[1] = &GPU;
         #endif
         #ifdef OpenCL_DEV
         supported_archs[2] = &OpenCLDev;
         #endif
         #ifdef FPGA_DEV
         supported_archs[3] = &FPGA;
         #endif

         const Device * supported_archs_array[supported_archs.size()];
         unsigned int arch_idx = 0;
         for ( ClusterNode::ClusterSupportedArchMap::const_iterator it = supported_archs.begin();
               it != supported_archs.end(); it++ ) {
            supported_archs_array[arch_idx] = it->second;
            arch_idx += 1;
         }

         _remoteNodes = NEW std::vector<nanos::ext::ClusterNode *>(nodes - 1, (nanos::ext::ClusterNode *) NULL);
         unsigned int node_index = 0;
         for ( unsigned int nodeC = 0; nodeC < nodes; nodeC++ ) {
            if ( nodeC != _gasnetApi->getNodeNum() ) {
               memory_space_id_t id = sys.addSeparateMemoryAddressSpace( ext::Cluster, !( getAllocFit() ), 0 );
               SeparateMemoryAddressSpace &nodeMemory = sys.getSeparateMemory( id );
               nodeMemory.setSpecificData( NEW SimpleAllocator( ( uintptr_t ) segmentAddr[ nodeC ], _nodeMem ) );
               nodeMemory.setNodeNumber( nodeC );
               nanos::ext::ClusterNode *node = new nanos::ext::ClusterNode( nodeC, id, supported_archs, supported_archs_array );
               (*_remoteNodes)[ node_index ] = node;
               node_index += 1;
            }
         }
      }
      _cpu = sys.getSMPPlugin()->getLastFreeSMPProcessorAndReserve();
      if ( _cpu ) {
         _cpu->setNumFutureThreads( 1 );
      } else {
         if ( _allowSharedThd ) {
            _cpu = sys.getSMPPlugin()->getLastSMPProcessor();
            if ( !_cpu ) {
               fatal0("Unable to get a cpu to run the cluster thread.");
            }
         } else {
            fatal0("Unable to get a cpu to run the cluster thread. Try using --cluster-allow-shared-thread");
         }
      }
   }
}

//void ClusterPlugin::addPinnedSegments( unsigned int numSegments, void **segmentAddr, std::size_t *segmentSize ) {
//   unsigned int idx;
//   _numPinnedSegments = numSegments;
//   _pinnedSegmentAddrList = new void *[ numSegments ];
//   _pinnedSegmentLenList = new std::size_t[ numSegments ];
//
//   for ( idx = 0; idx < numSegments; idx += 1)
//   {
//      _pinnedSegmentAddrList[ idx ] = segmentAddr[ idx ];
//      _pinnedSegmentLenList[ idx ] = segmentSize[ idx ];
//   }
//}

// void * ClusterPlugin::getPinnedSegmentAddr( unsigned int idx ) const {
//    return _pinnedSegmentAddrList[ idx ];
// }
//
// std::size_t ClusterPlugin::getPinnedSegmentLen( unsigned int idx ) const {
//    return _pinnedSegmentLenList[ idx ];
// }

std::size_t ClusterPlugin::getNodeMem() const {
   return _nodeMem;
}

int ClusterPlugin::getSmpPresend() const {
   return _smpPresend;
}

int ClusterPlugin::getGpuPresend() const {
   return _gpuPresend;
}

System::CachePolicyType ClusterPlugin::getCachePolicy ( void ) const {
   return _cachePolicy;
}

RemoteWorkDescriptor * ClusterPlugin::getRemoteWorkDescriptor( unsigned int nodeId, int archId ) {
   RemoteWorkDescriptor *rwd = NEW RemoteWorkDescriptor( nodeId );
   rwd->_mcontrol.preInit();
   rwd->_mcontrol.initialize( *_cpu );
   return rwd;
}

bool ClusterPlugin::getAllocFit() const {
   return _allocFit;
}

void ClusterPlugin::prepare( Config& cfg ) {
   /* Cluster: memory size to be allocated on remote nodes */
   cfg.registerConfigOption ( "node-memory", NEW Config::SizeVar ( _nodeMem ), "Sets the memory size that will be used on each node to send and receive data." );
   cfg.registerArgOption ( "node-memory", "cluster-node-memory" );
   cfg.registerEnvOption ( "node-memory", "NX_CLUSTER_NODE_MEMORY" );

   cfg.registerConfigOption ( "cluster-alloc-fit", NEW Config::FlagOption( _allocFit ), "Allocate full objects.");
   cfg.registerArgOption( "cluster-alloc-fit", "cluster-alloc-fit" );

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

   cfg.registerConfigOption ( "allow-shared-thread", NEW Config::FlagOption ( _allowSharedThd ), "Allow the cluster thread to share CPU with other threads." );
   cfg.registerArgOption ( "allow-shared-thread", "cluster-allow-shared-thread" );
   cfg.registerEnvOption ( "allow-shared-thread", "NX_CLUSTER_ALLOW_SHARED_THREAD" );

   cfg.registerConfigOption ( "cluster-unaligned-node-memory", NEW Config::FlagOption ( _unalignedNodeMem ), "Do not align node memory." );
   cfg.registerArgOption ( "cluster-unaligned-node-memory", "cluster-unaligned-node-memory" );
   cfg.registerEnvOption ( "cluster-unaligned-node-memory", "NX_CLUSTER_UNALIGNED_NODE_MEMORY" );

   cfg.registerConfigOption ( "gasnet-segment", NEW Config::SizeVar ( _gasnetSegmentSize ), "GASNet segment size." );
   cfg.registerArgOption ( "gasnet-segment", "gasnet-segment-size" );
   cfg.registerEnvOption ( "gasnet-segment", "NX_GASNET_SEGMENT_SIZE" );

}

ProcessingElement * ClusterPlugin::createPE( unsigned id, unsigned uid ){
   return NULL;
}

unsigned ClusterPlugin::getNumThreads() const {
   return 1;
}

void ClusterPlugin::startSupportThreads() {
   if ( _gasnetApi->getNumNodes() > 1 )
   {
      if ( _gasnetApi->getNodeNum() == 0 ) {
         _clusterThread = dynamic_cast<ext::SMPMultiThread *>( &_cpu->startMultiWorker( _gasnetApi->getNumNodes() - 1, (ProcessingElement **) &(*_remoteNodes)[0] ) );
      } else {
         _clusterThread = dynamic_cast<ext::SMPMultiThread *>( &_cpu->startMultiWorker( 0, NULL ) );
         if ( sys.getPMInterface().getInternalDataSize() > 0 )
            _clusterThread->getThreadWD().setInternalData(NEW char[sys.getPMInterface().getInternalDataSize()]);
         //_pmInterface->setupWD( smpRepThd->getThreadWD() );
         //setSlaveParentWD( &mainWD );
         if ( sys.getNumAccelerators() > 0 ) {
            /* This works, but it could happen that the cluster is initialized before the accelerators, and this call could return 0 */
            sys.getNetwork()->enableCheckingForDataInOtherAddressSpaces();
         }

         _gasnetApi->_rwgs = (GASNetAPI::ArchRWDs *) NEW GASNetAPI::ArchRWDs();
         _gasnetApi->_rwgs[0][0] = getRemoteWorkDescriptor(0, 0);
         _gasnetApi->_rwgs[0][1] = getRemoteWorkDescriptor(0, 1);
         _gasnetApi->_rwgs[0][2] = getRemoteWorkDescriptor(0, 2);
         _gasnetApi->_rwgs[0][3] = getRemoteWorkDescriptor(0, 3);
      }
   }
}

void ClusterPlugin::startWorkerThreads( std::map<unsigned int, BaseThread *> &workers ) {
   if ( _gasnetApi->getNodeNum() == 0 )
   {
      if ( _clusterThread ) {
         for ( unsigned int thdIndex = 0; thdIndex < _clusterThread->getNumThreads(); thdIndex += 1 )
         {
            BaseThread *thd = _clusterThread->getThreadVector()[ thdIndex ];
            workers.insert( std::make_pair( thd->getId(), thd ) );
         }
      }
   } else {
      workers.insert( std::make_pair( _clusterThread->getId(), _clusterThread ) );
   }
}

void ClusterPlugin::finalize() {
   if ( _gasnetApi->getNodeNum() == 0 ) {
      //message0("Master: Created " << createdWds << " WDs.");
      //message0("Master: Failed to correctly schedule " << sys.getAffinityFailureCount() << " WDs.");
      int soft_inv = 0;
      int hard_inv = 0;
      unsigned int max_execd_wds = 0;
      if ( _remoteNodes ) {
         for ( unsigned int idx = 0; idx < _remoteNodes->size(); idx += 1 ) {
            soft_inv += sys.getSeparateMemory( (*_remoteNodes)[idx]->getMemorySpaceId() ).getSoftInvalidationCount();
            hard_inv += sys.getSeparateMemory( (*_remoteNodes)[idx]->getMemorySpaceId() ).getHardInvalidationCount();
            max_execd_wds = max_execd_wds >= (*_remoteNodes)[idx]->getExecutedWDs() ? max_execd_wds : (*_remoteNodes)[idx]->getExecutedWDs();
            //message("Memory space " << idx <<  " has performed " << _separateAddressSpaces[idx]->getSoftInvalidationCount() << " soft invalidations." );
            //message("Memory space " << idx <<  " has performed " << _separateAddressSpaces[idx]->getHardInvalidationCount() << " hard invalidations." );
         }
      }
      message0("Cluster Soft invalidations: " << soft_inv);
      message0("Cluster Hard invalidations: " << hard_inv);
      //if ( max_execd_wds > 0 ) {
      //   float balance = ( (float) createdWds) / ( (float)( max_execd_wds * (_separateMemorySpacesCount-1) ) );
      //   message0("Cluster Balance: " << balance );
      //}
   }
}


void ClusterPlugin::addPEs( PEMap &pes ) const {
   if ( _remoteNodes ) {
      std::vector<ClusterNode *>::const_iterator it = _remoteNodes->begin();
      //NOT ANYMORE: it++; //position 0 is null, node 0 does not have a ClusterNode object
      for (; it != _remoteNodes->end(); it++ ) {
         pes.insert( std::make_pair( (*it)->getId(), *it ) );
      }
   }
}

unsigned int ClusterPlugin::getNumPEs() const {
   if ( _remoteNodes ) {
      return _remoteNodes->size();
   } else {
      return 0;
   }
}

unsigned int ClusterPlugin::getMaxPEs() const {
   if ( _remoteNodes ) {
      return _remoteNodes->size();
   } else {
      return 0;
   }
}

unsigned int ClusterPlugin::getNumWorkers() const {
   if ( _remoteNodes ) {
      return _remoteNodes->size();
   } else {
      return 0;
   }
}

unsigned int ClusterPlugin::getMaxWorkers() const {
   if ( _remoteNodes ) {
      return _remoteNodes->size();
   } else {
      return 0;
   }
}

bool ClusterPlugin::unalignedNodeMemory() const {
   return _unalignedNodeMem;
}

// std::size_t ClusterPlugin::getGASNetSegmentSize() const {
//    return _gasnetSegmentSize;
// }


}
}

DECLARE_PLUGIN("arch-cluster",nanos::ext::ClusterPlugin);
