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
#include "mpiprocessor.hpp"
#include "system.hpp"
#include "mpidd.hpp"
#include <string.h>


using namespace nanos;

NANOS_API_DEF(void *, nanos_mpi_factory, (void *args)) {
    nanos_mpi_args_t *mpi = (nanos_mpi_args_t *) args;
    return (void *) NEW ext::MPIDD(mpi->outline,mpi->assignedComm,mpi->assignedRank);
}

void deep_booster_free (MPI_Comm *intercomm) {
    nanos::ext::MPIProcessor::DEEP_Booster_free(intercomm, -1);
}

void deep_booster_free_ (MPI_Comm *intercomm) {
    nanos::ext::MPIProcessor::DEEP_Booster_free(intercomm, -1);
}

void deep_booster_free_single (MPI_Comm *intercomm, int rank) {
    nanos::ext::MPIProcessor::DEEP_Booster_free(intercomm, rank);
}

void deep_booster_free_single_ (MPI_Comm *intercomm, int* rank) {
    nanos::ext::MPIProcessor::DEEP_Booster_free(intercomm, *rank);
}

void deep_booster_alloc(MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm) {
    nanos::ext::MPIProcessor::DEEPBoosterAlloc(comm, number_of_hosts, process_per_host, intercomm, 0);
}

void deep_booster_alloc_(MPI_Comm* comm, int* number_of_hosts, int* process_per_host, MPI_Comm *intercomm) {
    nanos::ext::MPIProcessor::DEEPBoosterAlloc(*comm, *number_of_hosts, *process_per_host, intercomm, 0);
}


void deep_booster_alloc_offset (MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, int offset) {
        nanos::ext::MPIProcessor::DEEPBoosterAlloc(comm, number_of_hosts, process_per_host, intercomm, offset);
}

void deep_booster_alloc_offset_ (MPI_Comm* comm, int* number_of_hosts, int* process_per_host, MPI_Comm* intercomm, int* offset) {
        nanos::ext::MPIProcessor::DEEPBoosterAlloc(*comm, *number_of_hosts,*process_per_host, intercomm, *offset);
}

NANOS_API_DEF(nanos_err_t, set_mpi_exename, (char * new_name)) {
    try {
        nanos::ext::MPIProcessor::setMpiExename(new_name);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }
    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_mpi_init, (int* argc, char ***argv)) {
    try {
        nanos::ext::MPIProcessor::nanosMPIInit(argc, argv);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}


NANOS_API_DEF(nanos_err_t, nanos_mpi_init_thread, (int* argc, char ***argv, int required, int *provided)) {
    try {
        nanos::ext::MPIProcessor::nanosMPIInit(argc, argv);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}

NANOS_API_DEF(void, nanos_mpi_initf, (void)) {
    nanos::ext::MPIProcessor::nanosMPIInit(0, 0);
}

int nanos_mpi_finalize (void) {
    nanos::ext::MPIProcessor::nanosMPIFinalize();
    return NANOS_OK;
}

void nanos_mpi_finalizef_ (void) {
    nanos::ext::MPIProcessor::nanosMPIFinalize();
}

NANOS_API_DEF(int, nanos_mpi_send_taskinit, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm)) {
        return nanos::ext::MPIProcessor::nanosMPISendTaskinit(buf,count,datatype,dest,comm);
}
NANOS_API_DEF(int, nanos_mpi_recv_taskinit, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm, MPI_Status *status)){
        return nanos::ext::MPIProcessor::nanosMPIRecvTaskinit(buf,count,datatype,dest,comm,status); 
}
NANOS_API_DEF(int, nanos_mpi_send_taskend, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm)){
        return nanos::ext::MPIProcessor::nanosMPISendTaskend(buf,count,datatype,dest,comm);
}
NANOS_API_DEF(int, nanos_mpi_recv_taskend, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm, MPI_Status *status)){
        return nanos::ext::MPIProcessor::nanosMPIRecvTaskend(buf,count,datatype,dest,comm,status);
}
NANOS_API_DEF(int, nanos_mpi_send_datastruct, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm)){
        return nanos::ext::MPIProcessor::nanosMPISendDatastruct(buf,count,datatype,dest,comm);
}
NANOS_API_DEF(int, nanos_mpi_recv_datastruct, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm, MPI_Status *status)){
        return nanos::ext::MPIProcessor::nanosMPIRecvDatastruct(buf,count,datatype,dest,comm,status);
}

NANOS_API_DEF(int, nanos_mpi_type_create_struct, ( int count, int array_of_blocklengths[],  
        MPI_Aint array_of_displacements[], MPI_Datatype array_of_types[], MPI_Datatype *newtype)){
    return nanos::ext::MPIProcessor::nanosMPITypeCreateStruct(count,array_of_blocklengths,array_of_displacements, array_of_types,newtype);
}

NANOS_API_DEF(MPI_Datatype, ompss_get_mpi_type, (const char* type)) {
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
        result = MPI_LOGICAL;
    } else if (strcmp(type, "__mpitype_ompss_byte") == 0) {
        result = MPI_BYTE;
    } else if (strcmp(type, "__mpitype_ompss_unsigned_long") == 0) {
        result = MPI_UNSIGNED_LONG;
    } else if (strcmp(type, "__mpitype_ompss_unsigned_long_long") == 0) {
        result = MPI_UNSIGNED_LONG_LONG;
    }
    return result;
}

NANOS_API_DEF(int, nanos_mpi_get_parent, (MPI_Comm* parent_out)){
    return MPI_Comm_get_parent(parent_out);
}

NANOS_API_DEF(int, ompss_mpi_get_function_index_host, (void* func_pointer)){
    return nanos::ext::MPIProcessor::ompssMpiGetFunctionIndexHost(func_pointer);
}