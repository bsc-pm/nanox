
#ifndef _NANOS_OCL_WRAPPER
#define _NANOS_OCL_WRAPPER

// To view all APIs.
//#define CL_USE_DEPRECATED_OPENCL_1_0_APIS

#include <CL/opencl.h>

#define OCL_API(S) \
   extern __typeof__(S) *p_ ## S;
#include "oclapi.def"
#undef OCL_API

#endif // _NANOS_OCL_WRAPPER
