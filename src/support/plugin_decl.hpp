/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef _NANOS_PLUGIN_DECL
#define _NANOS_PLUGIN_DECL

#include <string>
#include "config_decl.hpp"
#include "compatibility.hpp"

namespace nanos {

   class Plugin
   {

      private:
         const char *  	_name;
         int            _version;
         void  *        _handler;

         Plugin ( const Plugin & );
         const Plugin operator= ( const Plugin & );
         
      public:
         Plugin( const char *name, int version ) : _name( name ), _version( version ) {}

         virtual ~Plugin() {}

         virtual void config(Config &cfg) {}
         
         virtual bool configurable() { return true; }
         virtual void init() {}

         virtual void fini() {}

         const char * getName() const;

         int getVersion() const;
   };

   class PluginManager
   {
      public:
         typedef TR1::unordered_map<std::string, Plugin *> PluginMap;

      private:
         PluginMap   _availablePlugins;
         PluginMap   _activePlugins;

         explicit PluginManager ( PluginManager & );
         const PluginManager operator= ( const PluginManager & );

      public:
         PluginManager() {}
         ~PluginManager() {}

         void init();

         bool isPlugin ( const char *name );
         bool isPlugin ( const std::string &name );

         void registerPlugin ( const char *name, Plugin &plugin );

         bool load ( const char *plugin_name, const bool init=true );
         bool load ( const std::string &plugin_name, const bool init=true );
         Plugin* loadAndGetPlugin ( const char *plugin_name, const bool init=true );
         Plugin* loadAndGetPlugin ( const std::string &plugin_name, const bool init=true );
         void unloadPlugins();
   };

} // namespace nanos

#endif
