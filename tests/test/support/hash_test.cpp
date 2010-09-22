/*
<testinfo>
test_generator=gens/mixed-generator
</testinfo>
*/

#include "config.hpp"
#include "nanos.h"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include "threadteam.hpp"
#include <string.h>
#include "list.hpp"
#include "hashmap.hpp"
#include <list>
#include <algorithm>
#include <iomanip>

using namespace std;
using namespace nanos;
using namespace nanos::ext;

WD *mainWD;
/*      key, val      */
HashMap<int, int, false, 4, Hash<int> > _map;

/*! works only with bf scheduler */
void barrier_code ( void * )
{
   int i, its=0;
   // PARALLEL tests
   if ( mainWD == myThread->getCurrentWD()) {
      // 4 items per list
      for (i=0; i < 16; i++) {
         _map[i] = i;
      }
      if ( _map.find(8) == NULL ) {
         std::cout << "ERRPOR: added element not found" << std::endl;
         abort();
      }
   }

   nanos_team_barrier();

   // Main erases one element other threads read it 100 times
   if ( mainWD == myThread->getCurrentWD()) {
      std::cout << "passed first barrier" << std::endl;
      for (i=0; i < 100; i++);
      while (!_map.erase(8))
         its++;
      std::cout << "Main needed " << std::setbase(10) << its << " iterations to erase the element." << std::endl;
   } else {
      while ( _map.find(8) != NULL )
         its++;
      std::cout <<  myThread->getCurrentWD() << " readed element " << std::setbase(10) << (int)its << " times until it got erased." << std::endl;
      std::cout.flush();
   }
   nanos_team_barrier();

}

int main (int argc, char **argv)
{
   int i, size;


   if ( _map.find(0) != NULL ) {
      std::cout << "Error, empty map finds elements" << std::endl;
      exit(1);
   }

   int& val = _map[0];

   if ( _map.find(0) == NULL ) {
      std::cout << "Error, map doesn't find element in it" << std::endl;
      exit(1);
   }

   if ( !_map.erase(0) ) {
      std::cout << "Error, map doesn't erase element in it" << std::endl;
      exit(1);
   }

   if ( _map.find(0) != NULL ) {
      std::cout << "Error, map finds erased element in it" << std::endl;
      exit(1);
   }

   _map[0] = 4;
   if ( *(_map.find(0)) != 4 ) {
      std::cout << "Error, element had unexpected value" << std::endl;
      exit(1);
   }

   for (i=0; i < 16; i++) {
      _map[i] = i;
   }

   if ( _map.findAndReference(8) == NULL ) {
      std::cout << "Error: element to reference not found" << std::endl;
      exit(1);
   }

   std::list<int> unrefs;
   _map.listUnreferencedKeys(unrefs);

   if (unrefs.empty()) {
      std::cout << "Error: unreferenced elements list must not be empty" << std::endl;
      exit(1);
   }

   std::list<int>::iterator it = std::find(unrefs.begin(),unrefs.end(),8);
   if ( it != unrefs.end() ) {
      std::cout << "Error: unreferenced elements list contains a referenced element's key: " << *(it) << std::endl;
      exit(1);
   }

   // delete references in the map before the threaded test
   _map.deleteReference(8);

   cout << "start threaded tests" << endl;
   //all threads perform a barrier , before the barrier they will freely access the list
   ThreadTeam &team = *myThread->getTeam();
   size = team.size();
   for ( unsigned i = 1; i < team.size(); i++ ) {
          WD * wd = new WD(new SMPDD(barrier_code));
          wd->tieTo(team[i]);
          sys.submit(*wd);
   }
   usleep(100);

   WD *wd = myThread->getCurrentWD();
   wd->tieTo(*myThread);

   mainWD = wd;
   barrier_code(NULL);

   cout << "end" << endl;
}
