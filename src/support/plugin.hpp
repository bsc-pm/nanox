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
         std::string    _name;
         int            _version;
         void  *        _handler;

      public:
         Plugin( std::string &_name, int _version ) : _name( _name ),_version( _version ) {}

         Plugin( const char *_name, int _version ) : _name( _name ),_version( _version ) {}

         virtual ~Plugin() {}

         virtual void init() {}

         virtual void fini() {}

         const std::string & getName() const { return _name; }

         int getVersion() const { return _version; }
   };

   class PluginManager
   {

      private:
         typedef std::vector<Plugin *>    PluginList;
         static std::string               _pluginsDir;
         static PluginList                _activePlugins;

      public:

         static void setDirectory ( const char *dir ) {  _pluginsDir = dir;  }
         static void setDirectory ( const std::string & dir ) { _pluginsDir = dir;  }

         static const std::string &getDirectory () { return _pluginsDir; }

         static bool load ( const char *plugin_name );
         static bool load ( const std::string &plugin_name ) { return load( plugin_name.c_str() ); };
   };

}

#endif
