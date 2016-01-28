
#include "hostinfo.hpp"
#include "mpiprocessor_decl.hpp"

using namespace nanos::mpi;
using namespace nanos::ext;

HostInfo HostInfo::defaultSettings() {
	int mpi_initialized;
	MPI_Initialized( &mpi_initialized );

	if( mpi_initialized == 1 ) {
		HostInfo defaultValue;
		// Set common default properties for spawn MPI_Info
		// - In case the MPI implementation supports tpp (threads per process) key...
		if( MPIProcessor::_workers_per_process > 0 ) {
			defaultValue.set( "tpp", MPIProcessor::_workers_per_process );
		}
		// In case the MPI implementation supports nodetype (cluster/booster) key...
		defaultValue.set( "nodetype", MPIProcessor::_mpiNodeType );

		return defaultValue;
	} else {
		return HostInfo(MPI_INFO_NULL);
	}
}
