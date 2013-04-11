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

#include <utility>
#include "dependenciesdomain.hpp"
#include "commutationdepobj.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "dataaccess.hpp"


namespace nanos {

Atomic<int> DependenciesDomain::_atomicSeed( 0 );
Atomic<int> DependenciesDomain::_tasksInGraph( 0 );
Lock DependenciesDomain::_lock;

using namespace dependencies_domain_internal;

void DependenciesDomain::increaseTasksInGraph()
{
   NANOS_INSTRUMENT(lock();)
   NANOS_INSTRUMENT(int tasks = ++_tasksInGraph;)
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("graph-size");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents(1, &key, (nanos_event_value_t *) &tasks );)
   NANOS_INSTRUMENT(unlock();)
}

void DependenciesDomain::decreaseTasksInGraph()
{
   NANOS_INSTRUMENT(lock();)
   NANOS_INSTRUMENT(int tasks = --_tasksInGraph;)
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("graph-size");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents(1, &key, (nanos_event_value_t *) &tasks );)
   NANOS_INSTRUMENT(unlock();)
}


//void DependenciesDomain::dump(std::string const &function, std::string const &file, size_t line)
//{
//   static int version = 0;
//   
//   version++;
//   std::ostringstream filename;
//   
//   filename << "rt-";
//   filename.width(4);
//   filename.fill('0');
//   filename << version;
//   filename << "-domain-";
//   filename.width(16);
//   filename << std::hex << (void *)this << ".dot";
//   
//   std::cout << function << " " << file << ":" << line << ": Dumping Region Tree " << filename.str() << std::endl;
//   std::ofstream output(filename.str().c_str());
//   output << _regionMap;
//   output.close();
//}

//unsigned int DependenciesDomain::getNumReaders ( Region const &region )
//{
//   unsigned int result = 0;
//   SyncRecursiveLockBlock lock1( _instanceLock );
//   RegionMap::iterator_list_t subregions;
//   _regionMap.find( region, /* out */subregions );
//   
//      std::cerr << __FUNCTION__;
//   for (
//      RegionMap::iterator_list_t::iterator it = subregions.begin();
//      it != subregions.end();
//      it++
//   ) {
//      RegionMap::iterator &accessor = *it;
//      RegionStatus &regionStatus = *accessor;
//      
//      if ( !regionStatus.getReaders().empty() ) {
//      std::cerr << " r ";
//      for (RegionStatus::DependableObjectList::iterator rit = regionStatus.getReaders().begin(); rit != regionStatus.getReaders().end(); rit++ ) {
//         std::cerr << (*rit)->getWD()->getId() << " ";
//      }
//      result += regionStatus.getReaders().size();
//      }
//      
//   }
//      std::cerr << " total:  " << result << std::endl;
//   return result;
//}

//#include "regiondirectory.hpp"

//unsigned int DependenciesDomain::getNumAllReaders ( )
//{
//   nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
//   Region region = NewRegionDirectory::build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
//   unsigned int result = 0;
//   SyncRecursiveLockBlock lock1( _instanceLock );
//   RegionMap::iterator_list_t subregions;
//   _regionMap.find( region, /* out */subregions );
//   std::set<DependableObject *> objects;
//   
//      std::cerr << __FUNCTION__;
//   for (
//      RegionMap::iterator_list_t::iterator it = subregions.begin();
//      it != subregions.end();
//      it++
//   ) {
//      RegionMap::iterator &accessor = *it;
//      RegionStatus &regionStatus = *accessor;
//      
//      if ( !regionStatus.getReaders().empty() ) {
//      std::cerr << " r ";
//      for (RegionStatus::DependableObjectList::iterator rit = regionStatus.getReaders().begin(); rit != regionStatus.getReaders().end(); rit++ ) {
//         objects.insert( *rit );
//         std::cerr << (*rit)->getWD()->getId() << " ";
//      }
//      }
//      
//   }
//   result = objects.size();
//      std::cerr << " total:  " << result << std::endl;
//   return result;
//}
} // namespace nanos
