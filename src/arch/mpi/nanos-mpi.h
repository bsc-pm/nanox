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

#ifndef _NANOS_MPI_H_
#define _NANOS_MPI_H_

#include "mpi.h"
#include "nanos.h" 
#define MPI_Init nanos_MPI_Init


#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        void (*outline) (void *);
        MPI_Comm _assignedComm;
        int _assignedRank;
    } nanos_mpi_args_t;

    //MPI
    NANOS_API_DECL(void *, nanos_mpi_factory, (void *args));
    
#define NANOS_MPI_DESC( args ) { nanos_mpi_factory, &( args ) } 

    NANOS_API_DECL(nanos_err_t, DEEP_Booster_alloc, (MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm));
//    NANOS_API_DECL(nanos_err_t, DEEP_Booster_alloc_hostfile, (MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm, char* hosts));
//    NANOS_API_DECL(nanos_err_t, DEEP_Booster_alloc_hostlist, (MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm, char* hosts, char* exec_file));
    NANOS_API_DECL(nanos_err_t, DEEP_Booster_free, (MPI_Comm *intercomm));
    NANOS_API_DECL(nanos_err_t, DEEP_Booster_free_single, (MPI_Comm *intercomm, int rank));
    NANOS_API_DECL(nanos_err_t, DEEP_Booster_free_size, (MPI_Comm *intercomm, int size));
    NANOS_API_DECL(nanos_err_t, setMpiExename, (char* new_name));
    NANOS_API_DECL(nanos_err_t, nanos_MPI_Init, (int* argc, char*** argv));
    NANOS_API_DECL(nanos_err_t, nanos_set_MPI_control_pointers, (int* file_mask, int mask, unsigned int* file_namehash, unsigned int* file_size));
    //Syncs pointers so device file has exactly the same order than host file
    NANOS_API_DECL(nanos_err_t, nanos_sync_dev_pointers, (int* file_mask, int mask, unsigned int* file_namehash, unsigned int* file_size,
            unsigned int* task_per_file,void (*ompss_mpi_func_pointers_dev[])()));
    NANOS_API_DECL(int, nanos_MPI_Send_taskinit, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm));
    NANOS_API_DECL(int, nanos_MPI_Recv_taskinit, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm, MPI_Status *status));
    NANOS_API_DECL(int, nanos_MPI_Send_taskend, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm));
    NANOS_API_DECL(int, nanos_MPI_Recv_taskend, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm, MPI_Status *status));
    NANOS_API_DECL(int, nanos_MPI_Send_datastruct, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm));
    NANOS_API_DECL(int, nanos_MPI_Recv_datastruct, (void *buf, int count, MPI_Datatype datatype, int dest, MPI_Comm comm, MPI_Status *status));    
    NANOS_API_DECL(MPI_Datatype, ompss_get_mpi_type, (char* type));
    


#ifdef __cplusplus
}
#endif

////Protected code only needed by mercurium compilation phases, workaround for use-define-after-preprocess "bug"
//#ifdef _MERCURIUM_MPI_
//
////Mercurium converts some types to their longer type.
////For example, shorts are ints, floats are double...
//
//enum OmpSsMPIType {
//    __mpitype_ompss_char = MPI_CHAR,
//    __mpitype_ompss_wchar_t = MPI_WCHAR,
//    __mpitype_ompss_signed_short = MPI_INT,
//    __mpitype_ompss_signed_int = MPI_INT,
//    __mpitype_ompss_signed_long = MPI_LONG,
//    __mpitype_ompss_signed_char = MPI_SIGNED_CHAR,
//    __mpitype_ompss_unsigned_char = MPI_UNSIGNED_CHAR,
//    __mpitype_ompss_unsigned_short = MPI_UNSIGNED_SHORT,
//    __mpitype_ompss_unsigned_int = MPI_UNSIGNED,
//    __mpitype_ompss_unsigned_long = MPI_UNSIGNED_LONG,
//    __mpitype_ompss_unsigned_long_long = MPI_UNSIGNED_LONG_LONG,
//    __mpitype_ompss_float = MPI_DOUBLE,
//    __mpitype_ompss_double = MPI_DOUBLE,
//    __mpitype_ompss_long_double = MPI_LONG_DOUBLE,
//    //Intel mpi boolean
//#ifdef MPI_C_BOOL
//    __mpitype_ompss_bool = MPI_C_BOOL,
//#endif
//    //MPI Standard boolean
//#ifdef MPI_BOOL
//    __mpitype_ompss_bool = MPI_BOOL,
//#endif
//    __mpitype_ompss_byte = MPI_BYTE
//};
//#endif


#endif
