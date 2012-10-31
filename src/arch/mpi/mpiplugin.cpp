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
#include "mpiprocessor.hpp"
#include "mpidd.hpp"
#include "system.hpp"

namespace nanos {
namespace ext {

nanos::PE * mpiProcessorFactory ( int id );

nanos::PE * mpiProcessorFactory ( int id )
{
   return new MPIProcessor( id );
}

class MPIPlugin : public Plugin
{

   public:
      MPIPlugin() : Plugin( "MPI PE Plugin",1 ) {}

      virtual void config ( Config& cfg )
      {
         cfg.setOptionsSection( "MPI Arch", "MPI specific options" );
         MPIProcessor::prepareConfig( cfg );
         MPIDD::prepareConfig( cfg );
      }

      virtual void init() {
         sys.setHostFactory( mpiProcessorFactory );
      }
};
}
}

DECLARE_PLUGIN("arch-mpi",nanos::ext::MPIPlugin);
