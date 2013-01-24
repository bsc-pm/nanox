
#ifndef _NANOS_OpenCL_WRAPPER
#define _NANOS_OpenCL_WRAPPER

// To view all APIs.
//#define CL_USE_DEPRECATED_OpenCL_1_0_APIS

#include <CL/opencl.h>

#define OpenCL_API(S) \
   extern __typeof__(S) *p_ ## S;
#include "openclapi.def"
#undef OpenCL_API

#endif // _NANOS_OpenCL_WRAPPER
