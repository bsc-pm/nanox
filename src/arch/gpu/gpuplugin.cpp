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
#include "archplugin.hpp"
#include "gpuconfig.hpp"
#include "gpudd.hpp"
#include "gpumemoryspace_decl.hpp"
#include "gpuprocessor.hpp"
#include "gpumemoryspace_decl.hpp"
#include "smpprocessor.hpp"
#include "system_decl.hpp"
#include <fstream>
#include <sstream>
#include <vector>

namespace nanos {
namespace ext {

class GPUPlugin : public ArchPlugin
{
   std::vector<ext::GPUProcessor *> *_gpus;
   std::vector<ext::GPUThread *>    *_gpuThreads;
   public:
      GPUPlugin() : ArchPlugin( "GPU PE Plugin", 1 )
         , _gpus( NULL )
         , _gpuThreads( NULL )
      {}

      void config( Config& cfg )
      {
         GPUConfig::prepare( cfg );
      }

      void init()
      {
         GPUConfig::apply();
         _gpus = NEW std::vector<nanos::ext::GPUProcessor *>(nanos::ext::GPUConfig::getGPUCount(), (nanos::ext::GPUProcessor *) NULL); 
         _gpuThreads = NEW std::vector<nanos::ext::GPUThread *>(nanos::ext::GPUConfig::getGPUCount(), (nanos::ext::GPUThread *) NULL); 
         for ( int gpuC = 0; gpuC < nanos::ext::GPUConfig::getGPUCount() ; gpuC++ ) {
            memory_space_id_t id = sys.addSeparateMemoryAddressSpace( ext::GPU, nanos::ext::GPUConfig::getAllocWide(), 0 );
            SeparateMemoryAddressSpace &gpuMemory = sys.getSeparateMemory( id );
            gpuMemory.setAcceleratorNumber( sys.getNewAcceleratorId() );

            ext::GPUMemorySpace *gpuMemSpace = NEW ext::GPUMemorySpace();
            gpuMemory.setSpecificData( gpuMemSpace );

            unsigned int node = getNumaNodeOfGPU( gpuC );

            ext::SMPProcessor *core = sys.getSMPPlugin()->getFreeSMPProcessorByNUMAnodeAndReserve(node);
            if ( core == NULL ) {
               core = sys.getSMPPlugin()->getLastFreeSMPProcessorAndReserve();
               if ( core == NULL ) {
                  core = sys.getSMPPlugin()->getLastSMPProcessor();
                  if ( core == NULL ) {
                     fatal0("Unable to get a core to run the GPU thread.");
                  }
                  warning0("Unable to get an exclusive cpu to run the CPU thread. The thread will run on PE " << core->getId() << " and share the cpu");
               }
               if (node != core->getNumaNode()) {
                  warning0("Unable to get a cpu on numa node " << node << " to run the CPU thread. Will run on numa node "<< core->getNumaNode());
               }
            }
            core->setNumFutureThreads( core->getNumFutureThreads() + 1 );
            
            //bool reserved;
            //unsigned pe = sys.reservePE( numa, node, reserved );
            //
            //if ( numa ) {
            //   verbose( "Reserving node " << node << " for GPU " << i << ", returned pe " << pe << ( reserved ? " (exclusive)" : " (shared)") );
            //}
            //else {
            //   verbose( "Reserving for GPU " << i << ", returned pe " << pe << ( reserved ? " (exclusive)" : " (shared)") );
            //}

            ext::GPUProcessor *gpu = NEW nanos::ext::GPUProcessor( gpuC, id, core, *gpuMemSpace );
            (*_gpus)[gpuC] = gpu;
         }
      }
      
      virtual unsigned getNumHelperPEs() const
      {
         return GPUConfig::getGPUCount();
      }

      virtual unsigned getNumThreads() const
      {
            return GPUConfig::getGPUCount();
      }

      int getNumaNodeOfGPU( int gpuId ) {
         // Is NUMA info is available
         int node = -1;
         if ( sys._hwloc.isHwlocAvailable() )
         {
            node = sys._hwloc.getNumaNodeOfGpu( gpuId );
         }
         else
         {
            // Warning: Linux specific:
#if defined( CUDA_VERSION ) && (CUDA_VERSION < 4010 )
            // This depends on the cuda driver, we are currently NOT linking against it.
            //int domainId, busId, deviceId;
            //cuDeviceGetAttribute( &domainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device);
            //cuDeviceGetAttribute( &busId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device);
            //cuDeviceGetAttribute( &deviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);
            //std::stringstream ssDevice;
            //ssDevice << std::hex << std::setfill( '0' ) << std::setw( 4 ) << domainId << ":" << std::setw( 2 ) << busId << ":" << std::setw( 2 ) << deviceId << ".0";
            //strcpy( pciDevice, ssDevice.str().c_str() );
#elif defined( CUDART_VERSION ) && ( CUDART_VERSION >= 4010 )
            char pciDevice[20]; // 13 min

            cudaDeviceGetPCIBusId( pciDevice, 20, gpuId );

            // This is common code for cuda 4.0 and 4.1
            std::stringstream ss;
            ss << "/sys/bus/pci/devices/" << pciDevice << "/numa_node";
            std::ifstream fNode( ss.str().c_str() );
            if ( fNode.good() )
               fNode >> node;
            fNode.close();
#endif

         }
         // Fallback / safety measure
         if ( node < 0 || sys.getSMPPlugin()->getNumSockets() == 1 ) {
            node = 0;
            // As we don't have NUMA info, don't request an specific node
         }
         return node;
      }
            
#if 0
      virtual void createBindingList()
      {
         /* As we now how many devices we have and how many helper threads we
          * need, reserve a PE for them */
         for ( int i = 0; i < GPUConfig::getGPUCount(); ++i )
         {
            int node = -1;
            // Is NUMA info is available
            bool numa = true;
            if ( sys.getSMPPlugin()->isHwlocAvailable() )
            {
#ifdef HWLOC
               hwloc_topology_t topology = ( hwloc_topology_t ) sys.getSMPPlugin()->getHwlocTopology();
               
               hwloc_obj_t obj = hwloc_cudart_get_device_pcidev ( topology, i );
               if ( obj != NULL ) {
                  hwloc_obj_t objNode = hwloc_get_ancestor_obj_by_type( topology, HWLOC_OBJ_NODE, obj );
                  if ( objNode != NULL ){
                     node = objNode->os_index;
                  }
               }
#endif
            }
            else
            {
               // Warning: Linux specific:
#if CUDA_VERSION < 4010
               // This depends on the cuda driver, we are currently NOT linking against it.
               //int domainId, busId, deviceId;
               //cuDeviceGetAttribute( &domainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device);
               //cuDeviceGetAttribute( &busId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device);
               //cuDeviceGetAttribute( &deviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);
               //std::stringstream ssDevice;
               //ssDevice << std::hex << std::setfill( '0' ) << std::setw( 4 ) << domainId << ":" << std::setw( 2 ) << busId << ":" << std::setw( 2 ) << deviceId << ".0";
               //strcpy( pciDevice, ssDevice.str().c_str() );
#else
               char pciDevice[20]; // 13 min

               NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( GPUUtils::NANOS_GPU_CUDA_GET_PCI_BUS_EVENT );
               cudaDeviceGetPCIBusId( pciDevice, 20, i );
               NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

               // This is common code for cuda 4.0 and 4.1
               std::stringstream ss;
               ss << "/sys/bus/pci/devices/" << pciDevice << "/numa_node";
               std::ifstream fNode( ss.str().c_str() );
               if ( fNode.good() )
                  fNode >> node;
               fNode.close();
#endif

            }
            // Fallback / safety measure
            if ( node < 0 || sys.getSMPPlugin()->getNumSockets() == 1 ) {
               node = 0;
               // As we don't have NUMA info, don't request an specific node
               numa = false;
            }
            
            bool reserved;
            unsigned pe = sys.reservePE( numa, node, reserved );
            
            if ( numa ) {
               verbose( "Reserving node " << node << " for GPU " << i << ", returned pe " << pe << ( reserved ? " (exclusive)" : " (shared)") );
            }
            else {
               verbose( "Reserving for GPU " << i << ", returned pe " << pe << ( reserved ? " (exclusive)" : " (shared)") );
            }
            // Now add this node to the binding list
            addBinding( pe );
         }
      }
#endif

      virtual PE* createPE( unsigned id, unsigned uid )
      {
//<<<<<<< HEAD
         //jbueno: disabled verbose( "Calling getBinding for id " << id << ", result: " << getBinding( id ) );
         //jbueno: disabled PE* pe = NEW GPUProcessor( getBinding( id ) , id );
         //jbueno: disabled pe->setNUMANode( sys.getNodeOfPE( pe->getId() ) );
         //jbueno: disabled return pe;
         //return NULL;
//=======         
        //pe->setNUMANode( sys.getNodeOfPE( pe->getId() ) );
//        memory_space_id_t mid = sys.getNewSeparateMemoryAddressSpaceId();
//        SeparateMemoryAddressSpace *gpumemory = NEW SeparateMemoryAddressSpace( mid, nanos::ext::GPU, nanos::ext::GPUConfig::getAllocWide());
//        gpumemory->setNodeNumber( 0 );
//        //ext::OpenCLMemorySpace *oclmemspace = NEW ext::OpenCLMemorySpace();
//        //oclmemory->setSpecificData( oclmemspace );
//        sys.addSeparateMemory(mid,gpumemory);
//        
//        ext::GPUMemorySpace *gpuMemSpace = NEW ext::GPUMemorySpace();
//        gpumemory->setSpecificData( gpuMemSpace );
//        nanos::ext::GPUProcessor *gpuPE = NEW nanos::ext::GPUProcessor( getBinding(id), id, uid, mid, *gpuMemSpace );
//
//        gpuPE->setNUMANode( sys.getNodeOfPE( gpuPE->getId() ) ); 
//        return gpuPE;
         return NULL;
      }

//      virtual void boot() {
//   int gpuC;
//   for ( gpuC = 0; gpuC < nanos::ext::GPUConfig::getGPUCount() ; gpuC++ ) {
//      SeparateMemoryAddressSpace *gpuMemory = sys.createNewSeparateMemoryAddressSpace( GPU, false );
//      gpuMemory->setNodeNumber( 0 );
//      ext::GPUMemorySpace *gpuMemSpace = NEW ext::GPUMemorySpace();
//      gpuMemory->setSpecificData( gpuMemSpace );
//      
//      nanos::ext::GPUProcessor *gpuPE = NEW nanos::ext::GPUProcessor( gpuC, *gpuMemSpace );
//      _pes.push_back( gpuPE );
//      BaseThread *gpuThd = &gpuPE->startWorker();
//      _workers.push_back( gpuThd );
//      _masterGpuThd = ( _masterGpuThd == NULL ) ? gpuThd : _masterGpuThd;
//   }
//         
//      }

      virtual void addPEs( PEMap &pes ) const
      {
         for ( std::vector<GPUProcessor *>::const_iterator it = _gpus->begin(); it != _gpus->end(); it++ ) {
            pes.insert( std::make_pair( (*it)->getId(), *it ) );
         }
      }

      virtual void addDevices( DeviceList &devices ) const
      {
         if ( !_gpus->empty() ) {
            std::vector<const Device *> const &pe_archs = ( *_gpus->begin() )->getDeviceTypes();
            for ( std::vector<const Device *>::const_iterator it = pe_archs.begin();
                  it != pe_archs.end(); it++ ) {
               devices.insert( *it );
            }
         }
      }

      virtual void startSupportThreads() {
         for ( unsigned int gpuC = 0; gpuC < _gpus->size(); gpuC += 1 ) {
            GPUProcessor *gpu = (*_gpus)[gpuC];
            (*_gpuThreads)[gpuC] = (ext::GPUThread *) &gpu->startGPUThread();
         }
      }

      virtual void startWorkerThreads( std::map<unsigned int, BaseThread *> &workers ) {
         for ( std::vector<GPUThread *>::iterator it = _gpuThreads->begin(); it != _gpuThreads->end(); it++ ) {
            workers.insert( std::make_pair( (*it)->getId(), *it ) );
         }
      }

      virtual unsigned int getNumPEs() const {
         return _gpus->size();
      }
      virtual unsigned int getMaxPEs() const {
         return _gpus->size();
      }
      virtual unsigned int getNumWorkers() const {
         return _gpus->size();
      }
      virtual unsigned int getMaxWorkers() const {
         return _gpus->size();
      }

      virtual void finalize() {
         if ( _gpuThreads->size() ) {
            int soft_inv = 0;
            int hard_inv = 0;
            for ( unsigned int idx = 0; idx < _gpus->size(); idx += 1 ) {
               soft_inv += sys.getSeparateMemory( (*_gpus)[idx]->getMemorySpaceId() ).getSoftInvalidationCount();
               hard_inv += sys.getSeparateMemory( (*_gpus)[idx]->getMemorySpaceId() ).getHardInvalidationCount();
            }
            message0("GPUs Soft invalidations: " << soft_inv);
            message0("GPUs Hard invalidations: " << hard_inv);
         }
      }

};

}
}

DECLARE_PLUGIN("arch-gpu",nanos::ext::GPUPlugin);
