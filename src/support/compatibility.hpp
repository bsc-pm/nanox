#ifndef _NANOS_COMPATIBILITY_HPP
#define _NANOS_COMPATIBILITY_HPP

// compiler issues

#if __GXX_EXPERIMENTAL_CXX0X__

#include <unordered_map>

namespace TR1 = std;

#else

#include <tr1/unordered_map>

namespace TR1 = std::tr1;

#endif


#endif

