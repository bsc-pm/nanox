/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

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
