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
#include "gpuprocessor.hpp"
#include "gpudd.hpp"
#include "system.hpp"

namespace nanos {
namespace ext {

PE * gpuProcessorFactory ( int id )
{
   return new GPUProcessor( id );
}

class GPUPlugin : public Plugin
{

   public:
      GPUPlugin() : Plugin( "GPU PE Plugin",1 ) {}

      virtual void config( Config& config )
      {
         config.setOptionsSection( "GPU Arch", "GPU specific options" );
         GPUProcessor::prepareConfig( config );
         GPUDD::prepareConfig( config );
      }

      virtual void init() {
         //sys.setHostFactory( gpuProcessorFactory );
      }
};
}
}

nanos::ext::GPUPlugin NanosXPlugin;

