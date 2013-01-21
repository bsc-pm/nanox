#ifndef NANOS_OPENCL_H
#define	NANOS_OPENCL_H

#include "nanos.h" 
#ifdef	__cplusplus
extern "C" {
#endif
    
typedef struct {
    void (*outline) (void *);
} nanos_opencl_args_t;

//MPI
NANOS_API_DECL(void *, nanos_opencl_factory, (void *args));
    
#define NANOS_MPI_DESC( args ) { nanos_opencl_factory, &( args ) } 
//TODO: ADD
    //void* nanos_get_program(char* kernel_name);
    //nanos_create_current_kernel(char* kernel_name);
    //nanos_ocl_set_bufferarg(void* ocl_kernel, arg_num, void* pointer);
    //nanos_ocl_set_arg(void* ocl_kernel, arg_num, size_t data_size, void* pointer);
    //nanos_exec_kernel(void* ocl_kernel, int work_dim, size_t* ndr_offset, size_t* ndr_local_size, size_t* ndr_global_size);
NANOS_API_DECL(void*, nanos_get_ocl_program, (char* ocl_code, char* compiler_opts));
NANOS_API_DECL(void*, nanos_create_current_kernel, (char* kernel_name, void* program_pointer));
NANOS_API_DECL(nanos_err_t,nanos_ocl_set_bufferarg, (void* ocl_kernel, int arg_num, void* pointer));
NANOS_API_DECL(nanos_err_t,nanos_ocl_set_arg, (void* ocl_kernel, int arg_num, size_t size, void* pointer));
NANOS_API_DECL(nanos_err_t,nanos_exec_kernel, (void* ocl_kernel, int work_dim, size_t* ndr_offset, size_t* ndr_local_size, size_t* ndr_global_size));

enum {
    OMPSS_SEEK_END=SEEK_END,OMPSS_SEEK_SET=SEEK_SET
};

#ifdef _MERCURIUM_OPENCL_
unsigned get_work_dim();
unsigned get_global_id(unsigned int arg0);
unsigned get_global_size(unsigned int arg0);
unsigned get_local_size(unsigned int arg0);
unsigned get_local_id(unsigned int arg0);
unsigned get_num_groups(unsigned int arg0);
unsigned get_group_id(unsigned int arg0);
unsigned get_global_offset(unsigned int arg0);
//We dont care about types inside struct, we want compiler to accept float4.x....
struct __ocl2__ { char x, y,s0,s1,lo,hi,even,odd; };
struct __ocl4__ {char x,y,w,z,s0,s1,s2,s3,lo,hi,even,odd; };
struct __ocl8__ {char s0,s1,s2,s3,s4,s5,s6,s7,lo,hi,even,odd;};
struct _ocl16__ {char s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sA,sb,sB,sc,sC,sd,sD,se,sE,sf,sF,lo,hi,even,odd;};
typedef struct __ocl2__ float2,char2,short2,int2,half2,double2,uint2,ulong2,ushort2,uchar2;
typedef struct __ocl4__ float4,char4,short4,int4,half4,double4,uint4,ulong4,ushort4,uchar4;
typedef struct __ocl8__ float8,char8,short8,int8,half8,double8,uint8,ulong8,ushort8,uchar8;
typedef struct __ocl16__ float16,char16,short16,int16,half16,double16,uint16,ulong16,ushort16,uchar16;
#endif

#ifdef	__cplusplus
}
#endif

#endif	/* NANOS_OCL_H */

