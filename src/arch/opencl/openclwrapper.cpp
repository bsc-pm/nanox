
#include "debug.hpp"
#include "openclwrapper.hpp"

#include <string>

#include <dlfcn.h>

#define OpenCL_API(S) \
  __typeof__(S) *p_ ## S = NULL;
#include "openclapi.def"
#undef OpenCL_API
