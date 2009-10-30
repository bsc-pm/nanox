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

using namespace nanos;

std::string PluginManager::_pluginsDir( LIBDIR );
PluginManager::PluginList PluginManager::_activePlugins;

bool PluginManager::load ( const char *name )
{
   std::string dlname;
   void * handler;

   dlname = "libnanox-";
   dlname += name;
   handler = OS::loadDL( _pluginsDir,dlname );

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
