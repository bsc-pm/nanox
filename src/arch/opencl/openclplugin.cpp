
#include "debug.hpp"
#include "openclconfig.hpp"
#include "os.hpp"
#include "plugin.hpp"

#include <dlfcn.h>

using namespace nanos;
using namespace nanos::ext;

namespace nanos {
namespace ext {

class OpenCLPlugin : public Plugin
{
public:
   OpenCLPlugin() : Plugin( "OpenCL PE Plugin", 1 ) { }

   ~OpenCLPlugin() { }

   void config( Config &cfg )
   {
      OpenCLConfig::prepare( cfg );
   }

   void init()
   {
      OpenCLConfig::apply();
   }

};

} // End namespace ext.
} // End namespace nanos.


DECLARE_PLUGIN("arch-opencl",nanos::ext::OpenCLPlugin);

