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

/*
<testinfo>
test_generator="gens/core-generator -a \"--gpus=0\""
</testinfo>
*/

#include "config.hpp"
#include <iostream>
#include "nanos.h"
#include "smpprocessor.hpp"
#include "system.hpp"
#include "threadteam.hpp"
#include <string.h>
#include "list.hpp"
#include "hashmap.hpp"
#include <list>
#include <algorithm>
#include <iomanip>
#include <unistd.h>

using namespace std;
using namespace nanos;
using namespace nanos::ext;

WD *mainWD;
/*      key, val      */
typedef std::pair<int, int> PairKey;
typedef PairHash<int, int> PairHashType;
HashMap<PairKey, int, false, 4, PairHashType > _map;

void barrier_code ( void * );

/*! works only with bf scheduler */
void barrier_code ( void * )
{
   int i, its=0;
   // PARALLEL tests
   if ( mainWD == getMyThreadSafe()->getCurrentWD()) {
      // 4 items per list
      for (i=0; i < 16; i++) {
         PairKey key = std::make_pair(i, i);
         _map[key] = i;
      }
      if ( _map.find(std::make_pair(8, 8)) == NULL ) {
         std::cout << "ERROR: added element not found" << std::endl;
         abort();
      }
   }

   nanos_team_barrier();

   // Main erases one element other threads read it 100 times
   if ( mainWD == getMyThreadSafe()->getCurrentWD()) {
      std::cout << "passed first barrier" << std::endl;
      for (i=0; i < 100; i++) {}
      while (!_map.erase(std::make_pair(8, 8)))
         its++;
      std::cout << "Main needed " << std::setbase(10) << its << " iterations to erase the element." << std::endl;
   } else {
      while ( _map.find(std::make_pair(8, 8)) != NULL )
         its++;
      std::cout <<  getMyThreadSafe()->getCurrentWD() << " read element " << std::setbase(10) << (int)its << " times until it got erased." << std::endl;
      std::cout.flush();
   }
   nanos_team_barrier();

}

int main (int argc, char **argv)
{
   unsigned int i;


   PairKey key0 = std::make_pair(0, 0);
   if ( _map.find(key0) != NULL ) {
      std::cout << "Error, empty map finds elements" << std::endl;
      exit(1);
   }

   int& val = _map[key0];

   if ( _map.find(key0) == NULL ) {
      std::cout << "Error, map doesn't find element in it, value was: " << val << std::endl;
      exit(1);
   }

   if ( !_map.erase(key0) ) {
      std::cout << "Error, map doesn't erase element in it" << std::endl;
      exit(1);
   }

   if ( _map.find(key0) != NULL ) {
      std::cout << "Error, map finds erased element in it" << std::endl;
      exit(1);
   }

   _map[key0] = 4;
   if ( *(_map.find(key0)) != 4 ) {
      std::cout << "Error, element had unexpected value" << std::endl;
      exit(1);
   }

   for (i=0; i < 16; i++) {
      PairKey key = std::make_pair(i, i);
      _map[key] = i;
   }

   if ( _map.findAndReference(std::make_pair(8, 8)) == NULL ) {
      std::cout << "Error: element to reference not found" << std::endl;
      exit(1);
   }

   HashMap<PairKey, int, false, 4, PairHashType >::KeyList unrefs;
   _map.listUnreferencedKeys(unrefs);

   if (unrefs.empty()) {
      std::cout << "Error: unreferenced elements list must not be empty" << std::endl;
      exit(1);
   }

#if 0 // 'find()' method is giving problems with a pair hash key
   //HashMap<PairKey, int, false, 4, PairHashType >::KeyList::iterator it = std::find(unrefs.begin(),unrefs.end(),8);
   HashMap<PairKey, int, false, 4, PairHashType >::KeyList::iterator it = unrefs.find(std::make_pair(8, 8));
   if ( it != unrefs.end() ) {
      std::cout << "Error: unreferenced elements list contains a referenced element's pair key: <"
            << (it->second.first) << ", " << (it->second.second) << ">" << std::endl;
      exit(1);
   }
#endif

   // delete references in the map before the threaded test
   _map.deleteReference(std::make_pair(8, 8));

   cout << "start threaded tests" << endl;
   //all threads perform a barrier , before the barrier they will freely access the list
   ThreadTeam &team = *getMyThreadSafe()->getTeam();
   for ( i = 1; i < team.size(); i++ ) {
          WD * wd = new WD(new SMPDD(barrier_code));
          wd->tieTo(team[i]);
          sys.submit(*wd);
   }
   usleep(100);

   WD *wd = getMyThreadSafe()->getCurrentWD();
   wd->tieTo(*getMyThreadSafe());

   mainWD = wd;
   barrier_code(NULL);

   cout << "end" << endl;
}
