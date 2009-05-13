#ifndef _NANOS_XSTRING
#define _NANOS_XSTRING

#include <sstream>
#include <string>

namespace nanos {

template <class T>
inline std::string toString (const T& t)
{
   std::stringstream ss;
   ss << t;
   return ss.str();
}

}

#endif

