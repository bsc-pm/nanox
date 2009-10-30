#include "debug.hpp"
#include "plugin.hpp"
#include "os.hpp"

using namespace nanos;

std::string PluginManager::pluginsDir( LIBDIR );
std::vector<Plugin *> PluginManager::activePlugins;

bool PluginManager::load ( const char *name )
{
   std::string dlname;
   void * handler;

   dlname = "libnanox-";
   dlname += name;
   handler = OS::loadDL( pluginsDir,dlname );

   if ( !handler ) {
      warning0 ( "plugin error=" << OS::dlError( handler ) );
      return false;
   }

   Plugin *plugin = ( Plugin * ) OS::dlFindSymbol( handler, "NanosXPlugin" );

   if ( !plugin ) {
      warning0 ( "plugin error=" << OS::dlError( handler ) );
      return false;
   }

   plugin->init();

   return true;
}
