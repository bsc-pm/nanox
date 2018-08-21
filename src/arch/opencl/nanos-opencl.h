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

#ifndef NANOS_OPENCL_H
#define	NANOS_OPENCL_H

//#include "CL/opencl.h"
#include "nanos-int.h" 
#include "nanos_error.h"
#ifdef	__cplusplus
extern "C" {
#endif
    
typedef struct {
    void (*outline) (void *);
} nanos_opencl_args_t;

//MPI
NANOS_API_DECL(void *, nanos_opencl_factory, (void *args));
    
#define NANOS_OCL_DESC( args ) { nanos_opencl_factory, &( args ) } 

NANOS_API_DECL(void*, nanos_create_current_kernel, (const char* kernel_name, const char* opencl_code, const char* compiler_opts)); 
NANOS_API_DECL(nanos_err_t,nanos_opencl_set_bufferarg, (void* opencl_kernel, int arg_num, const void* pointer));
NANOS_API_DECL(nanos_err_t,nanos_opencl_set_arg, (void* opencl_kernel, int arg_num, size_t size, const void* pointer));
NANOS_API_DECL(nanos_err_t,nanos_exec_kernel, (void* opencl_kernel, int work_dim, size_t* ndr_local_size, size_t* ndr_global_size));
NANOS_API_DECL(nanos_err_t,nanos_profile_exec_kernel, (void* opencl_kernel, int work_dim, size_t* ndr_global_size));

#ifndef _MF03
unsigned int nanos_get_opencl_num_devices (void);
#endif
void * ompss_opencl_malloc ( size_t size );
void ompss_opencl_free ( void * address ) ;
void nanos_get_opencl_num_devices_ (int* numret);
// Deprecated
__attribute__( (deprecated) )void * nanos_malloc_opencl( size_t size );
__attribute__( (deprecated) )void nanos_free_opencl( void * address );

NANOS_API_DECL(void, nanos_opencl_allocate_fortran, ( ptrdiff_t size, void* ptr )); // ptr is a void **
NANOS_API_DECL(void, nanos_opencl_deallocate_fortran, ( void * address ));

#ifdef _MERCURIUM_OPENCL_
//unsigned get_work_dim();
//unsigned get_global_id(unsigned int arg0);
//unsigned get_global_size(unsigned int arg0);
//unsigned get_local_size(unsigned int arg0);
//unsigned get_local_id(unsigned int arg0);
//unsigned get_num_groups(unsigned int arg0);
//unsigned get_group_id(unsigned int arg0);
//unsigned get_global_offset(unsigned int arg0);
////We dont care about types inside struct, we want compiler to accept float4.x....
//struct __opencl2__ { char x, y,s0,s1,lo,hi,even,odd; };
//struct __opencl4__ {char x,y,w,z,s0,s1,s2,s3,lo,hi,even,odd; };
//struct __opencl8__ {char s0,s1,s2,s3,s4,s5,s6,s7,lo,hi,even,odd;};
//struct _opencl16__ {char s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sA,sb,sB,sc,sC,sd,sD,se,sE,sf,sF,lo,hi,even,odd;};
//typedef struct __opencl2__ float2,char2,short2,int2,half2,double2,uint2,ulong2,ushort2,uchar2;
//typedef struct __opencl4__ float4,char4,short4,int4,half4,double4,uint4,ulong4,ushort4,uchar4;
//typedef struct __opencl8__ float8,char8,short8,int8,half8,double8,uint8,ulong8,ushort8,uchar8;
//typedef struct __opencl16__ float16,char16,short16,int16,half16,double16,uint16,ulong16,ushort16,uchar16;
//typedef cl_float2 float2;
//typedef cl_float4 float4;
#endif

#ifdef	__cplusplus
}
#endif

#endif	/* NANOS_OpenCL_H */

