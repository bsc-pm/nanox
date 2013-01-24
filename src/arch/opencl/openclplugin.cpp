
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
   OpenCLPlugin() : Plugin( "OpenCL PE Plugin", 1 ),
                 _libOpenCL( NULL ) { }

   ~OpenCLPlugin() { dlclose( _libOpenCL ); }

   void config( Config &cfg )
   {
      loadLibOpenCL();
      OpenCLConfig::prepare( cfg );
   }

   void init()
   {
      OpenCLConfig::apply();
   }

private:
   void loadLibOpenCL();

   void tryOpenHostOpenCL() { tryOpenHostOpenCL( "" ); }

   void tryOpenHostOpenCL( const std::string &path );

private:
   void *_libOpenCL;

};

} // End namespace ext.
} // End namespace nanos.

void OpenCLPlugin::loadLibOpenCL()
{
   std::string paths = OS::getEnvironmentVariable( "LD_LIBRARY_PATH" );

   // Try looking in default library search path.
   if( paths.empty() )
      tryOpenHostOpenCL();

   // Try looking in user-specified paths.
   else for( size_t i = 0,
                    e = paths.size(),
                    j;
                    i != e && !_libOpenCL;
                    i = j + ( j != e ) )
   {
      j = paths.find( ':', i );

      if( j == std::string::npos )
         j = e;

      tryOpenHostOpenCL( paths.substr( i, j - i ) );
   }

   // Library not found.
   if( !_libOpenCL )
      fatal0( "Cannot open libOpenCL" );

   #define OpenCL_API(S)                                     \
      {                                                   \
      void *sym = dlsym( _libOpenCL, #S );                \
      p_ ## S = reinterpret_cast<__typeof__(S) *>( sym ); \
         if( !p_ ## S ) {                                 \
            std::string msg = "Cannot locate symbol ";    \
            msg += #S;                                    \
            fatal0( msg.c_str() );                        \
         }                                                \
      }
   #include "openclapi.def"
   #undef OpenCL_API
}

void OpenCLPlugin::tryOpenHostOpenCL( const std::string &path )
{
   std::string libOpenCLPath = path +
                               ( path.empty() ? "" : "/" ) +
                               "libOpenCL.so";

   _libOpenCL = dlopen( libOpenCLPath.c_str(), RTLD_LAZY );
   if( !_libOpenCL )
      return;

   if( !dlsym( _libOpenCL, "vclIdToken" ) )
     return;

   dlclose( _libOpenCL );
   _libOpenCL = NULL;
}

DECLARE_PLUGIN("arch-opencl",nanos::ext::OpenCLPlugin);

