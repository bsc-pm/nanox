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
#include "clusterdd.hpp"
#include "clusterprocessor.hpp"
#include "system.hpp"
#include "os.hpp"

#include <gasnet.h>

namespace nanos {
namespace ext {

PE * clusterProcessorFactory ( int id )
{
   return new ClusterProcessor( id );
}

void am_handler(void)
{
}

class ClusterPlugin : public Plugin
{
   private:
      int _numNodes; 

   public:
      ClusterPlugin() : Plugin( "Cluster PE Plugin", 1 ), _numNodes( -1 ) {}

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

          int my_argc = OS::getArgc();
          char **my_argv = OS::getArgv();

          gasnet_handlerentry_t htable[] = {{ 203, am_handler}};

          gasnet_init(&my_argc, &my_argv);
          gasnet_attach(htable, 1, 0, 0);
          gasnet_barrier_notify(0,GASNET_BARRIERFLAG_ANONYMOUS);            
          gasnet_barrier_wait(0,GASNET_BARRIERFLAG_ANONYMOUS); 		
#if 0
         // Find out how many CUDA-capable GPUs the system has
         int totalCount, device, deviceCount = 0;
         struct cudaDeviceProp gpuProperties;

         cudaError_t cudaErr = cudaGetDeviceCount( &totalCount );

         if ( cudaErr != cudaSuccess ) {
            totalCount = 0;
         }

         // Machines with no GPUs can still report one emulation device
         for ( devices = 0; devices < totalCount; devices++ ) {
            cudaGetDeviceProperties( &gpuProperties, device );
            if ( gpuProperties.major != 9999 ) {
               // 9999 means emulation only
               deviceCount++;
            }
         }

         //displayAllGPUsProperties();

         // Check if the user has set a different number of GPUs to use
         if ( _numGPUs >= 0 ) {
            deviceCount = std::min( _numGPUs, deviceCount );
         }

         GPUDD::_gpuCount = deviceCount;
#endif

      }
};
}
}

nanos::ext::ClusterPlugin NanosXPlugin;

