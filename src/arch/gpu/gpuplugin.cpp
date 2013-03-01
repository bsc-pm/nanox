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
#include "system_decl.hpp"
#include "gpuprocessor.hpp"

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
      
      /*virtual unsigned getPEsInNode( unsigned node ) const
      {
         // TODO: make it work correctly
         // If it is the last node, assign
         //if ( node == ( sys.getNumSockets() - 1 ) )
      }*/
      
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
            // TODO: if HWLOC is available, use it.
            int node = sys.getNumSockets() - 1;
            unsigned pe = sys.reservePE( node );
            // Now add this node to the binding list
            addBinding( pe );
         }
      }

      virtual PE* createPE( unsigned id )
      {
         return NEW GPUProcessor( getBinding( id ) , id );
      }
};

}
}

DECLARE_PLUGIN("arch-gpu",nanos::ext::GPUPlugin);
