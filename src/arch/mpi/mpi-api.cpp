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

#include "nanos-mpi.h"
#include "nanos.h"
#include "mpiprocessor.hpp"
#include "system.hpp"
#include "mpidd.hpp"

using namespace nanos;

NANOS_API_DEF(void *, nanos_mpi_factory, (void *args)) {
    nanos_mpi_args_t *mpi = (nanos_mpi_args_t *) args;
    return (void *) new ext::MPIDD(mpi->outline,mpi->_assignedComm,mpi->_assignedRank);
}

NANOS_API_DEF(nanos_err_t, DEEP_Booster_free, (MPI_Comm *intercomm)) {
    try {
        nanos::ext::MPIProcessor::DEEP_Booster_free(intercomm, -1);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, DEEP_Booster_free_single, (MPI_Comm *intercomm, int rank)) {
    try {
        nanos::ext::MPIProcessor::DEEP_Booster_free(intercomm, rank);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, DEEP_Booster_alloc, (MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm)) {
    try {
        sys.DEEP_Booster_alloc(comm, number_of_spawns, intercomm);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, setMpiFilename, (char * new_name)) {
    try {
        nanos::ext::MPIProcessor::setMpiFilename(new_name);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_MPI_Init, (int* argc, char ***argv)) {
    try {
        nanos::ext::MPIProcessor::nanos_MPI_Init(argc, argv);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}

NANOS_API_DEF(int, nanos_MPI_Send_taskinit, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm)) {
        return nanos::ext::MPIProcessor::nanos_MPI_Send_taskinit(buf,count,datatype,dest,comm);
}
NANOS_API_DEF(int, nanos_MPI_Recv_taskinit, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm, MPI_Status *status)){
        return nanos::ext::MPIProcessor::nanos_MPI_Recv_taskinit(buf,count,datatype,dest,comm,status); 
}
NANOS_API_DEF(int, nanos_MPI_Send_taskend, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm)){
            return nanos::ext::MPIProcessor::nanos_MPI_Send_taskend(buf,count,datatype,dest,comm);
}
NANOS_API_DEF(int, nanos_MPI_Recv_taskend, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm, MPI_Status *status)){
        return nanos::ext::MPIProcessor::nanos_MPI_Recv_taskend(buf,count,datatype,dest,comm,status);
}
NANOS_API_DEF(int, nanos_MPI_Send_datastruct, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm)){
        return nanos::ext::MPIProcessor::nanos_MPI_Send_datastruct(buf,count,datatype,dest,comm);
}
NANOS_API_DEF(int, nanos_MPI_Recv_datastruct, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm, MPI_Status *status)){
        return nanos::ext::MPIProcessor::nanos_MPI_Recv_datastruct(buf,count,datatype,dest,comm,status);
}