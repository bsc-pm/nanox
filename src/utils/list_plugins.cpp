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

   n = scandir(PluginManager::getDirectory().c_str(), &namelist, 0, alphasort);
   if (n < 0)
     perror("scandir");
   else {
     while (n--) {
       std::string name(namelist[n]->d_name);

       if ( name.compare(0,9,"libnanox-") != 0) continue;

       if ( name.compare(name.size()-3,3,".so") == 0 ) {
            name.erase(name.size()-3);

            void * handler = OS::loadDL(PluginManager::getDirectory(),name);
            if ( !handler ) continue;

            Plugin * plugin = ( Plugin * ) OS::dlFindSymbol( handler, "NanosXPlugin" );
            if ( !plugin ) continue;
            
            std::cout << name << " - " << plugin->getName() << " - version " << plugin->getVersion() << std::endl;
       }
       free(namelist[n]);
     }
     free(namelist);
   }

}