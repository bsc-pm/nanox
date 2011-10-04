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
#include "clusterinfo_decl.hpp"


namespace nanos {
namespace ext {

class ClusterPlugin : public Plugin
{
   GASNetAPI _gasnetApi;
   public:
      ClusterPlugin() : Plugin( "Cluster PE Plugin", 1 ) {}

      virtual void config( Config& cfg )
      {
         cfg.setOptionsSection( "Cluster Arch", "Cluster specific options" );
         ClusterInfo::prepare( cfg );
      }

      virtual void init()
      {
         sys.getNetwork()->setAPI(&_gasnetApi);
         sys.getNetwork()->initialize();

         //if (sys.getNetwork()->getNodeNum() == 0)
         //{
            //ClusterInfo::setExtraPEsCount( sys.getNetwork()->getNumNodes() ); // we have num_nodes-1 "soft" threads, and 1 "container" thread
            ClusterInfo::setExtraPEsCount( 1 ); // We will use 1 paraver thread only to represent the soft-threads and the container. (extrae_get_thread_num must be coded acordingly
         //}
      }
};

}
}

nanos::ext::ClusterPlugin NanosXPlugin;

