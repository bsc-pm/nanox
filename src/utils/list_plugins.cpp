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

#include "os.hpp"
#include "plugin.hpp"
#include <string>
#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <stdlib.h>

using namespace nanos;


int main ()
{

   struct dirent **namelist;
   int n;

   n = scandir( PluginManager::getDirectory().c_str(), &namelist, 0, alphasort );

   if ( n < 0 )
      perror( "scandir" );
   else {
      while ( n-- ) {
         std::string name( namelist[n]->d_name );

         if ( name.compare( 0,9,"libnanox-" ) != 0 ) continue;

         if ( name.compare( name.size()-3,3,".so" ) == 0 ) {
            name.erase( name.size()-3 );

            void * handler = OS::loadDL( PluginManager::getDirectory(),name );

            if ( !handler ) continue;

            Plugin * plugin = ( Plugin * ) OS::dlFindSymbol( handler, "NanosXPlugin" );

            if ( !plugin ) continue;

            std::cout << name << " - " << plugin->getName() << " - version " << plugin->getVersion() << std::endl;
         }

         free( namelist[n] );
      }

      free( namelist );
   }

}
