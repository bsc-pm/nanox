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
               char pciDevice[20]; // 13 min
               cudaDeviceGetPCIBusId( pciDevice, 20, i );
               std::stringstream ss;
               ss << "/sys/bus/pci/devices/" << pciDevice << "/numa_node";
               std::ifstream fNode( ss.str().c_str() );
               if ( fNode.good() )
                  fNode >> node;
               fNode.close();

            }
            // Fallback / safety measure
            if ( node < 0 || sys.getNumSockets() == 1 )
               node = sys.getNumSockets() - 1;
            
            bool reserved;
            unsigned pe = sys.reservePE( node, reserved );
            
            verbose( "Reserving node " << node << " for GPU " << i << ", returned pe " << pe << ( reserved ? " (exclusive)" : " (shared)") );
            // Now add this node to the binding list
            addBinding( pe );
         }
      }

      virtual PE* createPE( unsigned id )
      {
         //verbose( "Calling getBinding for id " << id << ", result: " << getBinding( id ) );
         //PE* pe = NEW GPUProcessor( getBinding( id ) , id );
         //pe->setNUMANode( sys.getNodeOfPE( pe->getId() ) );
         //return pe;
         return NULL;
      }
};

}
}

DECLARE_PLUGIN("arch-gpu",nanos::ext::GPUPlugin);
