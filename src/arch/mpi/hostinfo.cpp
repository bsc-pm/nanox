
#include "hostinfo.hpp"
#include "mpiprocessor_decl.hpp"

using namespace nanos::mpi;
using namespace nanos::ext;

// Set common default properties for spawn MPI_Info
void HostInfo::defaultSettings() {
	int mpi_initialized;
	MPI_Initialized( &mpi_initialized );

        assert(mpi_initialized);

        // - In case the MPI implementation supports tpp (threads per process) key...
        if( MPIProcessor::_workers_per_process > 0 )
            set( "tpp", MPIProcessor::_workers_per_process );

        // In case the MPI implementation supports nodetype (cluster/booster) key...
        set( "nodetype", MPIProcessor::_mpiNodeType );

        // Disable parricide (i.e., to prevent an aborting child to kill its parents)
        // This key is specific from ParaStation MPI*value,*value,
        set( "parricide", "disabled" );
}
