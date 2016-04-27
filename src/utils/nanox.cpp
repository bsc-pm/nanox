/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#include "os.hpp"
#include "plugin.hpp"
#include "config.hpp"
#include "system.hpp"
#include <string>
#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <stdlib.h>
#include <list>
#include "compatibility.hpp"
#include "basethread.hpp"

using namespace nanos;

// Keep the plugin name list here
typedef std::pair<std::string,Plugin*> PluginInfo;
typedef std::list<PluginInfo> PluginList;
PluginList* pluginNames;

namespace nanos {
	class NoneInterface : public PMInterface
	{
		public:
			virtual void start () {}
			virtual int getInternalDataSize() const { return 0; }
			virtual int getInternalDataAlignment() const { return 1; }
			virtual void initInternalData( void *data ) {}
			virtual void setupWD( WD &wd ) {}
			virtual int getMaxThreads() const { return 0; }
			virtual void setNumThreads( int nthreads ) {}
			virtual void setNumThreads_globalState ( int nthreads ) {}
			virtual bool setCpuProcessMask( const CpuSet& cpu_set ) { return true; }
			const CpuSet& getCpuProcessMask() const
			{
				return sys.getCpuProcessMask();
			}

			virtual void addCpuProcessMask( const CpuSet& cpu_set ) {}
			virtual bool setCpuActiveMask( const CpuSet& cpu_set ) { return true; }
			const CpuSet& getCpuActiveMask() const
			{
				return sys.getCpuActiveMask();
			}
			virtual void addCpuActiveMask( const CpuSet& cpu_set ) {}
			virtual PMInterface::Interfaces getInterface() const { return PMInterface::None; }
	};

	namespace PMInterfaceType
	{
		int * ssCompatibility = 0;
		void set_interface_cb( void * );
		void set_interface_cb( void * p  )
		{
			sys.setPMInterface(NEW nanos::NoneInterface());
		}
		void (*set_interface)( void * ) = set_interface_cb;
	}
}

void utilInit ( void * );

void utilInit ( void * ) 
{
   struct dirent **namelist;
   int n;
   
   // Initialise here
   pluginNames = new PluginList();
   
   n = scandir( PLUGIN_DIR, &namelist, 0, alphasort );

   if ( n < 0 )
      perror( "scandir" );
   else {
      while ( n-- ) {
         std::string name( namelist[n]->d_name );    
         void    *handle;
         int     *iptr;
         if ( name.compare(0,9,"libnanox-") == 0 && name.compare( name.size()-3,3,".so" ) == 0 ) {
            //Check if the library has the symbol NanosXPlugin
            handle = dlopen( (std::string(PLUGIN_DIR) + std::string("/") + name).c_str(), RTLD_LOCAL | RTLD_LAZY);
            iptr = (int *)dlsym(handle, "NanosXPluginFactory");
            if (iptr!=NULL){
                name.erase( name.size()-3 );
                name.erase( 0, std::string("libnanox-").size() );

                Plugin *plugin = sys.loadAndGetPlugin( name );

                if ( plugin != NULL ) {
                   size_t separator = name.find( "-" );
                   std::string module = name.substr( 0, separator );
                   // The option name of the plugin (i.e. sched-priority -> priority)
                   std::string pluginName = name.substr( separator + 1 );

                   sys.setValidPlugin( module, pluginName );

                   pluginNames->push_back( PluginInfo( name, plugin ) );

                }
            }
         }

         free( namelist[n] );
      }

      free( namelist );
   }
   
   sys.setDelayedStart(true);
}

#define INIT_NULL { utilInit, 0 }                                                                                                                                         
LINKER_SECTION(nanos_init, nanos_init_desc_t , INIT_NULL) 

int main (int argc, char* argv[])
{
   bool listVersion = false;
   bool listHelp = false;
   bool listPlugins = false;
   std::string arg;

   for ( int i=1; i < argc; i++ ) {
      arg = std::string( argv[i] );

      if ( arg.compare( "--help" ) == 0 ) {
         listHelp = true;
      } else if ( arg.compare( "--list-modules" ) == 0 ) {
         listPlugins = true;
      } else if ( arg.compare( "--version" ) == 0 ) {
         listVersion = true;
      } else {
         std::cout << "usage: " << argv[0] << " [--version] [--help] [--list-modules]" << std::endl;
         exit(0);
      }
   } 

   if ( !listPlugins && !listHelp && !listVersion) {
      std::cout << "usage: " << argv[0] << " [--version] [--help] [--list-modules]" << std::endl;
      exit(0);
   }

   if ( listVersion ) {
      std::cout << PACKAGE << " " << VERSION << " (" << NANOX_BUILD_VERSION << ")" <<  std::endl;
      std::cout << "Configured with: " << NANOX_CONFIGURE_ARGS << std::endl;
   }

   if ( listPlugins ){
      std::cout << "Nanox runtime library available plugins at '" << PLUGIN_DIR << "':" << std::endl;
   
      for ( PluginList::const_iterator it = pluginNames->begin();
           it != pluginNames->end(); ++it )
      {
         const std::string name = it->first;
         const Plugin* plugin = it->second;
         
         std::cout << "   " << name << " - " << plugin->getName() << " - version " << plugin->getVersion() << std::endl;
      }
   }

   if ( listHelp ) {
      if ( listPlugins )
         std::cout << std::endl;

      std::cout << "Nanos++ runtime library version " << VERSION << "." << std::endl;
      std::cout << std::endl;
      std::cout << "The runtime configuration can be set using arguments added" << std::endl;
      std::cout << "to the NX_ARGS environment variable or through specialised" << std::endl;
      std::cout << "environment variables. As an example of NX_ARGS, to execute" << std::endl;
      std::cout << "with verbose mode and 4 worker threads the NX_ARGS environment" << std::endl;
      std::cout << "variable should be: 'NX_ARGS=\"--smp-workers=4 --verbose\"'." << std::endl;
      std::cout << std::endl;
      std::cout << "All NX_ARGS and env vars are listed below." << std::endl;
      std::cout << std::endl;

      std::cout << Config::getNanosHelp();
   }
   
   delete pluginNames;

   return 0;
}
