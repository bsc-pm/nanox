
#include "debug.hpp"
#include "oclwrapper.hpp"

#include <string>

#include <dlfcn.h>

#define OCL_API(S) \
  __typeof__(S) *p_ ## S = NULL;
#include "oclapi.def"
#undef OCL_API
