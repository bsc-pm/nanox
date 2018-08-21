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

#ifndef _NANOS_MPI_H_
#define _NANOS_MPI_H_

#include <mpi.h>
#include "nanos-int.h" 
#include "nanos_error.h"

#define MPI_Init nanos_mpi_init
#define MPI_Init_thread nanos_mpi_init_thread
#define MPI_Finalize nanos_mpi_finalize


#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        void (*outline) (void *);
        MPI_Comm assignedComm;
        int assignedRank;
    } nanos_mpi_args_t;

    //MPI
    NANOS_API_DECL(void *, nanos_mpi_factory, (void *args));
    NANOS_API_DECL(void *, nanos_mpi_fortran_factory, (void *args));
    
#define NANOS_MPI_DESC( args ) { nanos_mpi_factory, &( args ) } 
#define NANOS_MPI_FORTRAN_DESC( args ) { nanos_mpi_fortran_factory, &( args ) } 
    
    //This functions can may be called by the user (_ are subroutines in fortran...)
    void deep_booster_alloc (MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm);
    void deep_booster_alloc_ (MPI_Fint* comm, int* number_of_hosts, int* process_per_host, MPI_Fint* intercomm);
    void deep_booster_alloc_offset (MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, int offset);
    void deep_booster_alloc_offset_ (MPI_Fint* comm, int* number_of_hosts, int* process_per_host, MPI_Fint* intercomm, int* offset);   
    void deep_booster_alloc_list(MPI_Comm comm, int number_of_hosts, int* process_per_host_list, MPI_Comm *intercomm);
    void deep_booster_alloc_list_(MPI_Fint* comm, int* number_of_hosts, int* process_per_host_list,MPI_Fint *intercomm);  
    
    void deep_booster_alloc_nonstrict (MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, int* provided);
    void deep_booster_alloc_nonstrict_ (MPI_Fint* comm, int* number_of_hosts, int* process_per_host, MPI_Fint* intercomm, int* provided);
    void deep_booster_alloc_offset_nonstrict (MPI_Comm comm, int number_of_hosts, int process_per_host, MPI_Comm *intercomm, int offset, int* provided);
    void deep_booster_alloc_offset_nonstrict_ (MPI_Fint* comm, int* number_of_hosts, int* process_per_host, MPI_Fint* intercomm, int* offset, int* provided);   
    void deep_booster_alloc_list_nonstrict (MPI_Comm comm, int number_of_hosts, int* process_per_host_list, MPI_Comm *intercomm, int* provided);
    void deep_booster_alloc_list_nonstrict_ (MPI_Fint* comm, int* number_of_hosts, int* process_per_host_list,MPI_Fint *intercomm, int* provided);
    
    
    void deep_booster_free (MPI_Comm *intercomm);
    void deep_booster_free_ (MPI_Fint *intercomm);
    void deep_booster_free_single (MPI_Comm *intercomm, int rank);
    void deep_booster_free_single_ (MPI_Fint *intercomm, int* rank);
    int nanos_mpi_finalize(void);
    void nanos_mpi_finalizef_(void);
    
    //Called by user but no need to do an special interface for fortran
    NANOS_API_DECL(nanos_err_t, nanos_mpi_init, (int* argc, char*** argv));
    NANOS_API_DECL(nanos_err_t, nanos_mpi_init_thread, (int* argc, char*** argv, int required, int *provided));
    NANOS_API_DECL(void, nanos_mpi_initf, (void));
    
    NANOS_API_DECL(int, nanos_mpi_send_taskinit, (void *buf, int count, int dest, MPI_Comm comm));
    NANOS_API_DECL(int, nanos_mpi_send_taskend, (void *buf, int count, int disconnect, MPI_Comm comm));
    NANOS_API_DECL(int, nanos_mpi_send_datastruct, (void *buf, int count, MPI_Datatype* datatype, int dest, MPI_Comm comm));
    NANOS_API_DECL(int, nanos_mpi_recv_datastruct, (void *buf, int count, MPI_Datatype* datatype, int dest, MPI_Comm comm));       
    NANOS_API_DECL(int, nanos_mpi_type_create_struct, ( int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[],  
            MPI_Datatype array_of_types[], MPI_Datatype **newtype, int taskId));
    NANOS_API_DECL(int, nanos_mpi_type_get_struct, ( int taskId, MPI_Datatype **newtype));
    NANOS_API_DECL(MPI_Datatype, ompss_get_mpi_type, (int type));    
    NANOS_API_DECL(int, nanos_mpi_get_parent, (MPI_Comm* parent_out));    
    NANOS_API_DECL(int, ompss_mpi_get_function_index_host, (void* func_pointer));
    NANOS_API_DECL(int, ompss_mpi_get_function_index_dev, (void* func_pointer));

#ifdef __cplusplus
}
#endif

enum OmpSsMPIType {
    mpitype_ompss_char = 0,
    mpitype_ompss_wchar_t = 1,
    mpitype_ompss_signed_short = 2,
    mpitype_ompss_signed_int = 3,
    mpitype_ompss_signed_long = 4,
    mpitype_ompss_signed_char = 5,
    mpitype_ompss_unsigned_char = 6,
    mpitype_ompss_unsigned_short = 7,
    mpitype_ompss_unsigned_int = 8,
    mpitype_ompss_float = 9,
    mpitype_ompss_double = 10,
    mpitype_ompss_long_double = 11,
    mpitype_ompss_bool = 12,
    mpitype_ompss_byte = 13,
    mpitype_ompss_unsigned_long = 14,
    mpitype_ompss_unsigned_long_long = 15,
    mpitype_ompss_signed_long_long = 16
};

#endif
