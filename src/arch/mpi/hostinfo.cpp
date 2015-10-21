
#include "hostinfo.hpp"
#include "mpiprocessor_decl.hpp"

using namespace nanos::mpi;
using namespace nanos::ext;

HostInfo const& HostInfo::defaultSettings() {
   static bool initialized = false;
   int mpi_initialized;
   MPI_Initialized( &mpi_initialized );

   if( !initialized && mpi_initialized == 1 ) {
      // Set common default properties for spawn MPI_Info
      MPIProcessor::_defaultHostInfo.initialize();

      // - In case the MPI implementation supports tpp (threads per process) key...
      if( MPIProcessor::_workers_per_process > 0 ) {
          MPIProcessor::_defaultHostInfo.set( "tpp", MPIProcessor::_workers_per_process );
      }
      // In case the MPI implementation supports nodetype (cluster/booster) key...
      MPIProcessor::_defaultHostInfo.set( "nodetype", MPIProcessor::_mpiNodeType );
      initialized = true;
   }
   return MPIProcessor::_defaultHostInfo;
}
