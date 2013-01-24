
#include "opencldd.hpp"
#include "opencldevice.hpp"
#include "openclconfig.hpp"

using namespace nanos;
using namespace nanos::ext;

OpenCLDevice nanos::ext::OpenCLDev( "OPENCL" );

OpenCLDD * OpenCLDD::copyTo ( void *toAddr )
{
   OpenCLDD *dd = new ( toAddr ) OpenCLDD( *this );
   return dd;
}