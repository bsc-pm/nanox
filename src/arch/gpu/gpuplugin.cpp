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
#include "archplugin.hpp"
#include "gpuconfig.hpp"
#include "gpuprocessor.hpp"
#include "system_decl.hpp"
#include <fstream>
#include <sstream>

#ifdef HWLOC
#include <hwloc.h>
#include <hwloc/cudart.h>
#endif

namespace nanos {
namespace ext {

class GPUPlugin : public ArchPlugin
{
   public:
      GPUPlugin() : ArchPlugin( "GPU PE Plugin", 1 ) {}

      void config( Config& cfg )
      {
         GPUConfig::prepare( cfg );
      }

      void init()
      {
         GPUConfig::apply();
      }
      
      virtual unsigned getNumHelperPEs() const
      {
         return GPUConfig::getGPUCount();
      }

      virtual unsigned getNumPEs() const
      {
         return GPUConfig::getGPUCount();
      }

      virtual unsigned getNumThreads() const
      {
            return GPUConfig::getGPUCount();
      }
            
      virtual void createBindingList()
      {
         /* As we now how many devices we have and how many helper threads we
          * need, reserve a PE for them */
         for ( int i = 0; i < GPUConfig::getGPUCount(); ++i )
         {
            int node = -1;
            // Is NUMA info is available
            bool numa = true;
            if ( sys.isHwlocAvailable() )
            {
#ifdef HWLOC
               hwloc_topology_t topology = ( hwloc_topology_t ) sys.getHwlocTopology();
               
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

               cudaDeviceGetPCIBusId( pciDevice, 20, i );

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
            if ( node < 0 || sys.getNumSockets() == 1 ) {
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

      virtual PE* createPE( unsigned id, unsigned uid )
      {
         verbose( "Calling getBinding for id " << id << ", result: " << getBinding( id ) );
         PE* pe = NEW GPUProcessor( getBinding( id ) , id, uid );
         pe->setNUMANode( sys.getNodeOfPE( pe->getId() ) );
         return pe;
      }
};

}
}

DECLARE_PLUGIN("arch-gpu",nanos::ext::GPUPlugin);
