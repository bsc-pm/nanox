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

#include "debug.hpp"
#include "plugin.hpp"
#include "os.hpp"
#include "config.hpp"

using namespace nanos;

void PluginManager::init()
{
}

bool PluginManager::isPlugin ( const char *name )
{
   std::string dlname;
   void * handler;

   dlname = "libnanox-";
   dlname += name;
   handler = OS::loadDL( "",dlname );

   if ( !handler ) {
      warning0 ( "plugin error=" << OS::dlError( handler ) );
      return false;
   }

   Plugin *plugin = ( Plugin * ) OS::dlFindSymbol( handler, "NanosXPlugin" );

   return plugin != NULL;
}

void PluginManager::registerPlugin ( Plugin & plugin )
{
   _availablePlugins[plugin.getName().c_str()] = &plugin;
}

bool PluginManager::load ( const char *name, const bool initPlugin )
{
   return loadAndGetPlugin ( name, initPlugin ) != NULL;
}

Plugin * PluginManager::loadAndGetPlugin( const char *name, const bool initPlugin )
{
   std::string dlname;
   void * handler;

   Plugin * plugin = NULL;
   PluginMap::iterator it;
   
   if ( (it = _availablePlugins.find(name)) != _availablePlugins.end() )
   {
     plugin = it->second;
     
   } else {

     dlname = "libnanox-";
     dlname += name;
     handler = OS::loadDL( "",dlname );

     if ( !handler ) {
       warning0 ( "plugin error=" << OS::dlError( handler ) );
       return NULL;
     }

     plugin = ( Plugin * ) OS::dlFindSymbol( handler, "NanosXPlugin" );

     if ( !plugin ) {
       warning0 ( "plugin error=" << OS::dlError( handler ) );
       return NULL;
     }
      
   }

   Config config;
   plugin->config(config);
   config.init();

   if ( initPlugin )
      plugin->init();

   _activePlugins[plugin->getName().c_str()] = plugin;

   return plugin;
}
