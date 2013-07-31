
/*************************************************************************************/
/*      Copyright 2013 Barcelona Supercomputing Center                               */
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

#include "debug.hpp"
#include "openclconfig.hpp"
#include "os.hpp"
#include "processingelement_fwd.hpp"
#include "plugin.hpp"
#include "archplugin.hpp"
#include "openclprocessor.hpp"

#include <dlfcn.h>

using namespace nanos;
using namespace nanos::ext;

namespace nanos {
namespace ext {

class OpenCLPlugin : public ArchPlugin
{
   
private:
    
   static std::string _devTy;
  // All found devices.
  static std::map<cl_device_id, cl_context> _devices;

   friend class OpenCLConfig;
   
public:
   OpenCLPlugin() : ArchPlugin( "OpenCL PE Plugin", 1 ) { }

   ~OpenCLPlugin() { }

   void config( Config &cfg )
   {
      // Select the device to use.
      cfg.registerConfigOption( "opencl-device-type",
                                NEW Config::StringVar( _devTy ),
                                "Defines the OpenCL device type to use "
                                "(ALL, CPU, GPU, ACCELERATOR)" );
      cfg.registerEnvOption( "opencl-device-type", "NX_OPENCL_DEVICE_TYPE" );
      cfg.registerArgOption( "opencl-device-type", "opencl-device-type" );
   
      OpenCLConfig::prepare( cfg );
   }

   void init()
   {
      OpenCLConfig::apply(_devTy,_devices);
   }
   
   /*virtual unsigned getPEsInNode( unsigned node ) const
   {
      // TODO: make it work correctly
      // If it is the last node, assign
      //if ( node == ( sys.getNumSockets() - 1 ) )
   }*/
   
   virtual unsigned getNumHelperPEs() const
   {
      return OpenCLConfig::getOpenCLDevicesCount();
   }

   virtual unsigned getNumPEs() const
   {
      return OpenCLConfig::getOpenCLDevicesCount();
   }
   
   virtual unsigned getNumThreads() const
   {
      return OpenCLConfig::getOpenCLDevicesCount();
   }
   
   virtual void createBindingList()
   {
      /* As we now how many devices we have and how many helper threads we
       * need, reserve a PE for them */
      for ( unsigned i = 0; i < OpenCLConfig::getOpenCLDevicesCount(); ++i )
      {
         // As we don't have NUMA info, don't request an specific node
         bool numa = false;
         // TODO: if HWLOC is available, use it.
         int node = sys.getNumSockets() - 1;
         bool reserved;
         unsigned pe = sys.reservePE( numa, node, reserved );
         
         // Now add this node to the binding list
         addBinding( pe );
      }
   }

   virtual PE* createPE( unsigned id, unsigned uid )
   {
      PE * pe = NEW OpenCLProcessor( getBinding( id ) , id, uid );
      pe->setNUMANode( sys.getNodeOfPE( pe->getId() ) );
      return pe;
   }
};

std::string OpenCLPlugin::_devTy = "ALL";
// All found devices.
std::map<cl_device_id, cl_context> OpenCLPlugin::_devices;
} // End namespace ext.
} // End namespace nanos.


DECLARE_PLUGIN("arch-opencl",nanos::ext::OpenCLPlugin);

