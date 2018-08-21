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

#include "nanos-mpi.h"
#include "mpiremotenode.hpp"
#include "system.hpp"
#include "mpidd.hpp"

#include <mpi.h>
#include <string.h>

using namespace nanos;

NANOS_API_DEF(void *, nanos_mpi_factory, (void *args)) {
    nanos_mpi_args_t *mpi = (nanos_mpi_args_t *) args;
    return (void *) NEW ext::MPIDD(mpi->outline,mpi->assignedComm,mpi->assignedRank);
}

NANOS_API_DEF(void *, nanos_mpi_fortran_factory, (void *args)) {
    nanos_mpi_args_t *mpi = (nanos_mpi_args_t *) args;
    //Fortran will write a 4byte integer into assignedComm, so this is "safe"
    int val=0+(uintptr_t)mpi->assignedComm;
    //We dont want to translate null/0 MPI_Comm (it may be MPI_COMM_WORLD in
    // some implementations, but it makes no sense in this place)
    MPI_Comm aComm;
    if (val==0) aComm=0;
    else aComm=MPI_Comm_f2c((MPI_Fint)val);
    return (void *) NEW ext::MPIDD(mpi->outline,aComm,mpi->assignedRank);
}

void deep_booster_free (MPI_Comm *intercomm) {
    nanos::ext::MPIRemoteNode::DEEP_Booster_free(intercomm, -1);
}

void deep_booster_free_ (MPI_Fint *intercomm) {
    MPI_Comm c_comm=MPI_Comm_f2c(*intercomm);
    nanos::ext::MPIRemoteNode::DEEP_Booster_free(&c_comm, -1);
}

void deep_booster_free_single (MPI_Comm *intercomm, int rank) {
    nanos::ext::MPIRemoteNode::DEEP_Booster_free(intercomm, rank);
}

void deep_booster_free_single_ (MPI_Fint *intercomm, int* rank) {
    MPI_Comm c_comm=MPI_Comm_f2c(*intercomm);
    nanos::ext::MPIRemoteNode::DEEP_Booster_free(&c_comm, *rank);
}

void deep_booster_alloc(MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm) {
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(comm, number_of_hosts, process_per_host, intercomm, true, NULL, 0, NULL);
}

void deep_booster_alloc_(MPI_Fint* comm, int* number_of_hosts, int* process_per_host, MPI_Fint *intercomm) {
    MPI_Comm c_comm=MPI_Comm_f2c(*comm);
    MPI_Comm dummy_intercomm;
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(c_comm, *number_of_hosts, *process_per_host, &dummy_intercomm, true, NULL, 0, NULL); 
    MPI_Fint f_intercomm=MPI_Comm_c2f(dummy_intercomm);
    *intercomm=f_intercomm;
}


void deep_booster_alloc_offset (MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, int offset) {
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(comm, number_of_hosts, process_per_host, intercomm, true, NULL, offset, NULL);
}

void deep_booster_alloc_offset_ (MPI_Fint* comm, int* number_of_hosts, int* process_per_host, MPI_Fint* intercomm, int* offset) {
    MPI_Comm c_comm=MPI_Comm_f2c(*comm);
    MPI_Comm dummy_intercomm;
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(c_comm, *number_of_hosts,*process_per_host, &dummy_intercomm, true, NULL, *offset, NULL);
    MPI_Fint f_intercomm=MPI_Comm_c2f(dummy_intercomm);
    *intercomm=f_intercomm;
}

void deep_booster_alloc_list(MPI_Comm comm, int number_of_hosts, int* process_per_host_list, MPI_Comm *intercomm) {
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(comm, number_of_hosts, 0, intercomm, true, NULL, 0, process_per_host_list);
}

void deep_booster_alloc_list_(MPI_Fint* comm, int* number_of_hosts, int* process_per_host_list,MPI_Fint *intercomm) {
    MPI_Comm c_comm=MPI_Comm_f2c(*comm);
    MPI_Comm dummy_intercomm;
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(c_comm, *number_of_hosts, 0, &dummy_intercomm, true, NULL, 0, process_per_host_list);
    MPI_Fint f_intercomm=MPI_Comm_c2f(dummy_intercomm);
    *intercomm=f_intercomm;
}

void deep_booster_alloc_nonstrict(MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, int* provided) {
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(comm, number_of_hosts, process_per_host, intercomm, false, provided, 0, NULL);
}

void deep_booster_alloc_nonstrict_(MPI_Fint* comm, int* number_of_hosts, int* process_per_host, MPI_Fint *intercomm, int* provided) {
    MPI_Comm c_comm=MPI_Comm_f2c(*comm);
    MPI_Comm dummy_intercomm;
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(c_comm, *number_of_hosts, *process_per_host, &dummy_intercomm, false, provided, 0, NULL); 
    MPI_Fint f_intercomm=MPI_Comm_c2f(dummy_intercomm);
    *intercomm=f_intercomm;
}


void deep_booster_alloc_offset_nonstrict(MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, int offset, int* provided) {
        nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(comm, number_of_hosts, process_per_host, intercomm, false, provided, offset, NULL);
}

void deep_booster_alloc_offset_nonstrict_(MPI_Fint* comm, int* number_of_hosts, int* process_per_host, MPI_Fint* intercomm, int* offset, int* provided) {
    MPI_Comm c_comm=MPI_Comm_f2c(*comm);
    MPI_Comm dummy_intercomm;
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(c_comm, *number_of_hosts,*process_per_host, &dummy_intercomm, false, provided, *offset, NULL);
    MPI_Fint f_intercomm=MPI_Comm_c2f(dummy_intercomm);
    *intercomm=f_intercomm;
}

void deep_booster_alloc_list_nonstrict(MPI_Comm comm, int number_of_hosts, int* process_per_host_list, MPI_Comm *intercomm, int* provided) {
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(comm, number_of_hosts, 0, intercomm, false, provided, 0, process_per_host_list);
}

void deep_booster_alloc_list_nonstrict_(MPI_Fint* comm, int* number_of_hosts, int* process_per_host_list,MPI_Fint *intercomm, int* provided) {
    MPI_Comm c_comm=MPI_Comm_f2c(*comm);
    MPI_Comm dummy_intercomm;
    nanos::ext::MPIRemoteNode::DEEPBoosterAlloc(c_comm, *number_of_hosts, 0, &dummy_intercomm, false, provided, 0, process_per_host_list);
    MPI_Fint f_intercomm=MPI_Comm_c2f(dummy_intercomm);
    *intercomm=f_intercomm;
}


NANOS_API_DEF(nanos_err_t, nanos_mpi_init, (int* argc, char ***argv)) {
    try {
        nanos::ext::MPIRemoteNode::nanosMPIInit(argc, argv, MPI_THREAD_MULTIPLE, 0);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}


NANOS_API_DEF(nanos_err_t, nanos_mpi_init_thread, (int* argc, char ***argv, int required, int *provided)) {
    try {
        nanos::ext::MPIRemoteNode::nanosMPIInit(argc, argv, required, provided);
    } catch (...) {
        return NANOS_UNKNOWN_ERR;
    }

    return NANOS_OK;
}

NANOS_API_DEF(void, nanos_mpi_initf, (void)) {
    nanos::ext::MPIRemoteNode::nanosMPIInit(0, 0,MPI_THREAD_MULTIPLE,0);
}

int nanos_mpi_finalize (void) {
    nanos::ext::MPIRemoteNode::nanosMPIFinalize();
    return NANOS_OK;
}

void nanos_mpi_finalizef_ (void) {
    nanos::ext::MPIRemoteNode::nanosMPIFinalize();
}

NANOS_API_DEF(int, nanos_mpi_send_taskinit, (void *buf, int count, int dest, MPI_Comm comm)) {
        return nanos::ext::MPIRemoteNode::nanosMPISendTaskInit(buf,count,dest,comm);
}

NANOS_API_DEF(int, nanos_mpi_send_taskend, (void *buf, int count, int disconnect, MPI_Comm comm)){
        return nanos::ext::MPIRemoteNode::nanosMPISendTaskEnd(buf,count,MPI_INT,disconnect,comm);
}
NANOS_API_DEF(int, nanos_mpi_send_datastruct, (void *buf, int count, MPI_Datatype* datatype, int dest, MPI_Comm comm)){
        return nanos::ext::MPIRemoteNode::nanosMPISendDatastruct(buf,count,*datatype,dest,comm);
}
NANOS_API_DEF(int, nanos_mpi_recv_datastruct, (void *buf, int count, MPI_Datatype* datatype, int dest, MPI_Comm comm)){
        return nanos::ext::MPIRemoteNode::nanosMPIRecvDatastruct(buf,count,*datatype,dest,comm,MPI_STATUS_IGNORE);
}

NANOS_API_DEF(int, nanos_mpi_type_create_struct, ( int count, int array_of_blocklengths[],  
        MPI_Aint array_of_displacements[], MPI_Datatype array_of_types[], MPI_Datatype **newtype, int taskId)){
    return nanos::ext::MPIRemoteNode::nanosMPITypeCreateStruct(count, array_of_blocklengths,
            array_of_displacements, array_of_types, newtype, taskId);
}

NANOS_API_DEF(int, nanos_mpi_type_get_struct, ( int taskId, MPI_Datatype **newtype )){
    nanos::ext::MPIRemoteNode::nanosMPITypeCacheGet( taskId, newtype );
    return NANOS_OK;
}

NANOS_API_DEF(MPI_Datatype, ompss_get_mpi_type, (int type)) {
    MPI_Datatype result = MPI_DATATYPE_NULL;
    switch(type) {
        case mpitype_ompss_char: 
            result = MPI_CHAR; 
            break;
        case mpitype_ompss_wchar_t:
            result = MPI_WCHAR;
            break;
        case mpitype_ompss_signed_short:
            result = MPI_SHORT;
            break;
        case mpitype_ompss_signed_int:
            result = MPI_INT;
            break;
        case mpitype_ompss_signed_long:
            result = MPI_LONG;
            break;
        case mpitype_ompss_signed_char:
            result = MPI_SIGNED_CHAR;
            break;
        case mpitype_ompss_unsigned_char:
            result = MPI_UNSIGNED_CHAR;
            break;
        case mpitype_ompss_unsigned_short:
            result = MPI_UNSIGNED_SHORT;
            break;
        case mpitype_ompss_unsigned_int:
            result = MPI_UNSIGNED;
            break;
        case mpitype_ompss_float:
            result = MPI_FLOAT;
            break;
        case mpitype_ompss_double:
            result = MPI_DOUBLE;
            break;
        case mpitype_ompss_long_double:
            result = MPI_LONG_DOUBLE;
            break;
        case mpitype_ompss_bool:
            result = MPI_LOGICAL;
            break;
        case mpitype_ompss_byte:
            result = MPI_BYTE;
            break;
        case mpitype_ompss_unsigned_long:
            result = MPI_UNSIGNED_LONG;
            break;
        case mpitype_ompss_unsigned_long_long:
            result = MPI_UNSIGNED_LONG_LONG;
            break;
        case mpitype_ompss_signed_long_long:
            result = MPI_LONG_LONG;
            break;
        default:
        break;
    }
    return result;
}

NANOS_API_DEF(int, nanos_mpi_get_parent, (MPI_Comm* parent_out)){
    return MPI_Comm_get_parent(parent_out);
}

NANOS_API_DEF(int, ompss_mpi_get_function_index_host, (void* func_pointer)){
    return nanos::ext::MPIRemoteNode::ompssMpiGetFunctionIndexHost(func_pointer);
}

NANOS_API_DEF(int, ompss_mpi_get_function_index_dev, (void* func_pointer)){
    return nanos::ext::MPIRemoteNode::ompssMpiGetFunctionIndexDevice(func_pointer);
}
