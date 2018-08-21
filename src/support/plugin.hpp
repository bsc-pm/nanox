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

#ifndef _NANOS_PLUGIN
#define _NANOS_PLUGIN

#include "config.hpp"
#include "plugin_decl.hpp"
#include "smartpointer.hpp"

#include <string>

namespace nanos {

inline const char * Plugin::getName() const
{
   return _name;
}

inline int Plugin::getVersion() const
{
   return _version;
}

inline bool PluginManager::isPlugin ( const std::string &name )
{
   return isPlugin( name.c_str() );
}

inline bool PluginManager::load ( const std::string &plugin_name, const bool initPlugin )
{
   return load( plugin_name.c_str(), initPlugin );
}

inline Plugin* PluginManager::loadAndGetPlugin ( const std::string &plugin_name, const bool initPlugin )
{
   return loadAndGetPlugin( plugin_name.c_str(), initPlugin );
}

} // namespace nanos

#ifdef PIC 
#define DECLARE_PLUGIN(name,type)     \
   extern "C" {                       \
      nanos::Plugin * NanosXPluginFactory(); \
   }                                  \
   nanos::Plugin * NanosXPluginFactory() {   \
      static nanos::unique_pointer<type> plugin; \
      if( !plugin ) {                 \
         plugin.reset(new type());    \
      }                               \
      return plugin.get();            \
   }
#else
#define INITX {_registerPlugin, NULL} 
#define DECLARE_PLUGIN(name,type) \
       static void _registerPlugin (void *arg); \
       static void _registerPlugin (void *arg) { \
          static nanos::unique_pointer<type> plugin; \
          if( !plugin ) {                 \
             plugin.reset(new type());    \
          }                               \
          nanos::sys.registerPlugin(name, *plugin ); \
       }                                  \
       LINKER_SECTION(nanos_init, nanos_init_desc_t, INITX);
#endif

#endif
