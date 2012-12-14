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

#include "mpi.h"
#include "nanos-mpi.h"
#include "nanos.h"
#include "mpiprocessor.hpp"
#include "system.hpp"
#include "mpidd.hpp"
#include <string.h>


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

//NANOS_API_DEF(nanos_err_t, DEEP_Booster_alloc_hostfile, (MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm, char* hosts)) {
//    try {
//        sys.DEEP_Booster_alloc(comm, number_of_spawns, intercomm, hosts);
//    } catch (...) {
//        return NANOS_UNKNOWN_ERR;
//    }
//
//    return NANOS_OK;
//}

//NANOS_API_DEF(nanos_err_t, DEEP_Booster_alloc_hostlist, (MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm, char* hosts, char* exec_name)) {
//    try {
//        sys.DEEP_Booster_alloc(comm, number_of_spawns, intercomm, hosts, exec_name);
//    } catch (...) {
//        return NANOS_UNKNOWN_ERR;
//    }
//
//    return NANOS_OK;
//}
//
NANOS_API_DEF(nanos_err_t, DEEP_Booster_alloc, (MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm)) {
    try {
        sys.DEEP_Booster_alloc(comm, number_of_spawns, intercomm);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, setMpiExename, (char * new_name)) {
    try {
        nanos::ext::MPIProcessor::setMpiExename(new_name);
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

NANOS_API_DEF(MPI_Datatype, ompss_get_mpi_type, (char* type)) {
    MPI_Datatype result = MPI_DATATYPE_NULL;
    if (strcmp(type, "__mpitype_ompss_char") == 0) {
        result = MPI_CHAR;
    } else if (strcmp(type, "__mpitype_ompss_wchar_t") == 0) {
        result = MPI_WCHAR;
    } else if (strcmp(type, "__mpitype_ompss_signed_short") == 0) {
        result = MPI_SHORT;
    } else if (strcmp(type, "__mpitype_ompss_signed_int") == 0) {
        result = MPI_INT;
    } else if (strcmp(type, "__mpitype_ompss_signed_long") == 0) {
        result = MPI_LONG;
    } else if (strcmp(type, "__mpitype_ompss_signed_char") == 0) {
        result = MPI_SIGNED_CHAR;
    } else if (strcmp(type, "__mpitype_ompss_unsigned_char") == 0) {
        result = MPI_UNSIGNED_CHAR;
    } else if (strcmp(type, "__mpitype_ompss_unsigned_short") == 0) {
        result = MPI_UNSIGNED_SHORT;
    } else if (strcmp(type, "__mpitype_ompss_unsigned_int") == 0) {
        result = MPI_UNSIGNED;
    } else if (strcmp(type, "__mpitype_ompss_float") == 0) {
        result = MPI_FLOAT;
    } else if (strcmp(type, "__mpitype_ompss_double") == 0) {
        result = MPI_DOUBLE;
    } else if (strcmp(type, "__mpitype_ompss_long_double") == 0) {
        result = MPI_LONG_DOUBLE;
    } else if (strcmp(type, "__mpitype_ompss_bool") == 0) {
        //result=MPI_BOOL;
    } else if (strcmp(type, "__mpitype_ompss_byte") == 0) {
        result = MPI_BYTE;
    } else if (strcmp(type, "__mpitype_ompss_unsigned_long") == 0) {
        result = MPI_UNSIGNED_LONG;
    } else if (strcmp(type, "__mpitype_ompss_unsigned_long_long") == 0) {
        result = MPI_UNSIGNED_LONG_LONG;
    }
    return result;
}