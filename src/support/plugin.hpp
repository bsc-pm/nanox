/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

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
         Plugin( std::string &_name, int _version ) : name( _name ),version( _version ) {}

         Plugin( const char *_name, int _version ) : name( _name ),version( _version ) {}

         virtual ~Plugin() {}

         virtual void init() {}

         virtual void fini() {}

         const std::string & getName() const { return name; }

         int getVersion() const { return version; }
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

         static const std::string &getDirectory () { return pluginsDir; }

         static bool load ( const char *plugin_name );
         static bool load ( const std::string &plugin_name ) { return load( plugin_name.c_str() ); };
   };

}

#endif
