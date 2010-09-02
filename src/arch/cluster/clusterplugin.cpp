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
#include "gasnetapi.hpp"


namespace nanos {
namespace ext {

class ClusterPlugin : public Plugin
{
   GasnetAPI _gasnetApi;
   public:
      ClusterPlugin() : Plugin( "Cluster PE Plugin", 1 ) {}

      virtual void config( Config& config )
      {
         config.setOptionsSection( "Cluster Arch", "Cluster specific options" );
         //config.registerConfigOption ( "num-gpus", new Config::IntegerVar( _numGPUs ),
         //                              "Defines the maximum number of GPUs to use" );
         //config.registerArgOption ( "num-gpus", "gpus" );
         //config.registerEnvOption ( "num-gpus", "NX_GPUS" );
      }

      virtual void init()
      {
         sys.getNetwork()->setAPI(&_gasnetApi);
      }
};

}
}

nanos::ext::ClusterPlugin NanosXPlugin;

