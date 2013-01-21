
#include "ocldd.hpp"
#include "ocldevice.hpp"
#include "oclconfig.hpp"

using namespace nanos;
using namespace nanos::ext;

OCLDevice nanos::ext::OCLDev( "OCL" );

OpenCLDD * OpenCLDD::copyTo ( void *toAddr )
{
   OpenCLDD *dd = new ( toAddr ) OpenCLDD( *this );
   return dd;
}