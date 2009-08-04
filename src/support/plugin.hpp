#ifndef _NANOS_PLUGIN
#define _NANOS_PLUGIN

#include <string>
#include <vector>

namespace nanos
{

   class Plugin
   {

      private:
         std::string name;
         int    version;
         void  *handler;

      public:
         Plugin(std::string &_name, int _version) : name(_name),version(_version) {}
         Plugin(const char *_name, int _version) : name(_name),version(_version) {}
         virtual void init() {};
         virtual void fini() {};
   };

   class PluginManager
   {

      private:
         static std::string pluginsDir;
         static std::vector<Plugin *> activePlugins;

      public:

         static void setDirectory ( const char *dir ) {
            pluginsDir = dir;
         }

         static void setDirectory ( const std::string & dir ) {
            pluginsDir = dir;
         }

         static bool load ( const char *plugin_name );
         static bool load ( const std::string &plugin_name ) { return load(plugin_name.c_str()); };
   };

}

#endif
