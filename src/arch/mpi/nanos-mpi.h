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


#include "nanos.h" 
#include "mpi.h"


#ifdef __cplusplus
extern "C" {
#endif
    
    typedef struct {
        void (*outline) (void *);
    } nanos_mpi_args_t;

    //MPI
    NANOS_API_DECL(void *, nanos_mpi_factory, (void *args));
#define NANOS_MPI_DESC( args ) { nanos_mpi_factory, &( args ) } 

NANOS_API_DECL(nanos_err_t, DEEP_Booster_alloc, ( MPI_Comm comm, int number_of_spawns, MPI_Comm *intercomm ));
NANOS_API_DECL(nanos_err_t, setMpiFilename, ( char* new_name ));

#ifdef __cplusplus
}
#endif

//Protected code only needed by mercurium compilation phases, workaround for use-define-after-preprocess "bug"
#ifdef _MERCURIUM_MPI_

#include "mpi.h"

//Mercurium converts some types to their longer type.
//For example, shorts are ints, floats are double...
enum OmpSsMPIType {
    __mpitype_ompss_char = MPI_CHAR,
    __mpitype_ompss_wchar_t = MPI_WCHAR,
    __mpitype_ompss_signed_short = MPI_INT,
    __mpitype_ompss_signed_int = MPI_INT,
    __mpitype_ompss_signed_long = MPI_LONG,
    __mpitype_ompss_signed_char = MPI_SIGNED_CHAR,
    __mpitype_ompss_unsigned_char = MPI_UNSIGNED_CHAR,
    __mpitype_ompss_unsigned_short = MPI_UNSIGNED_SHORT,
    __mpitype_ompss_unsigned_int = MPI_UNSIGNED,
    __mpitype_ompss_unsigned_long = MPI_UNSIGNED_LONG,
    __mpitype_ompss_float = MPI_DOUBLE,
    __mpitype_ompss_double = MPI_DOUBLE,
    __mpitype_ompss_long_double = MPI_LONG_DOUBLE,
    //Intel mpi boolean
#ifdef MPI_C_BOOL
    __mpitype_ompss_bool = MPI_C_BOOL,
#endif
    //MPI Standard boolean
#ifdef MPI_BOOL
    __mpitype_ompss_bool = MPI_BOOL,
#endif
    __mpitype_ompss_byte = MPI_BYTE
};

//MPI_Datatype ompss_get_mpi_type(char* type){
//    MPI_Datatype result;
//    if (type == "__mpitype_ompss_char"){
//        result=MPI_CHAR;
//    } else if (type == "__mpitype_ompss_wchar_t"){
//        result=MPI_WCHAR;
//    } else if (type == "__mpitype_ompss_signed_short"){
//        result=MPI_SHORT;
//    } else if (type == "__mpitype_ompss_signed_int"){
//        result=MPI_INT;
//    } else if (type == "__mpitype_ompss_signed_long"){
//        result=MPI_LONG;
//    } else if (type == "__mpitype_ompss_signed_char"){
//        result=MPI_SIGNED_CHAR;
//    } else if (type == "__mpitype_ompss_unsigned_char"){
//        result=MPI_UNSIGNED_CHAR;
//    } else if (type == "__mpitype_ompss_unsigned_short"){
//        result=MPI_UNSIGNED_SHORT;
//    } else if (type == "__mpitype_ompss_unsigned_int"){
//        result=MPI_UNSIGNED;
//    } else if (type == "__mpitype_ompss_float"){
//        result=MPI_FLOAT;
//    } else if (type == "__mpitype_ompss_double"){
//        result=MPI_DOUBLE;
//    } else if (type == "__mpitype_ompss_long_double"){
//        result=MPI_LONG_DOUBLE;
//    //} else if (type == "__mpitype_ompss_bool"){
//    //  result=MPI_BOOL;
//    } else if (type == "__mpitype_ompss_byte"){
//        result=MPI_BYTE;
//    }
//    return result;
//}
#endif //END ifndef _MERCURIUM_MPI_


#endif
