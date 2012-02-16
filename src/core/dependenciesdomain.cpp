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

using namespace nanos;

Atomic<int> DependenciesDomain::_atomicSeed( 0 );
Atomic<int> DependenciesDomain::_tasksInGraph( 0 );
Lock DependenciesDomain::_lock;

using namespace dependencies_domain_internal;

inline void DependenciesDomain::submitDependableObjectDataAccess ( DependableObject &depObj, Target const &target, AccessType const &accessType, SchedulePolicySuccessorFunctor* callback )
{
   if ( accessType.commutative ) {
      if ( !( accessType.input && accessType.output ) || depObj.waits() ) {
         fatal( "Commutation task must be inout" );
      }
   }
   
   // gmiranda: this lock is only required in lookupDependency
   //SyncRecursiveLockBlock lock1( _instanceLock );
   //typedef std::set<RegionMap::iterator> subregion_set_t;
   // TODO (gmiranda): replace this by a call to findAndPopulate
#if 0
   typedef RegionMap::iterator_list_t subregion_set_t;
   subregion_set_t subregions;

   RegionMap::iterator wholeRegion = _regionMap.findAndPopulate( target, /* out */subregions );
   if ( !wholeRegion.isEmpty() ) {
      subregions.push_back(wholeRegion);
   }
   
   for (
      subregion_set_t::iterator it = subregions.begin();
      it != subregions.end();
      it++
   ) {
      RegionMap::iterator &accessor = *it;
      RegionStatus &regionStatus = *accessor;
      regionStatus.hold(); // This is necessary since we may trigger a removal in finalizeReduction
   }
   
   for (
      subregion_set_t::iterator it = subregions.begin();
      it != subregions.end();
      it++
   ) {
#endif
   //DepsMap::iterator it = _addressDependencyMap.find( target );
   //if ( it != _addressDependencyMap.end() ) {
      
      //MappedType &status = *it->second;
      MappedType &status = *lookupDependency( target );
      //! TODO (gmiranda): enable this if required
      //status.hold(); // This is necessary since we may trigger a removal in finalizeReduction
      
      if ( accessType.commutative ) {
         submitDependableObjectCommutativeDataAccess( depObj, target, accessType, status, callback );
      } else if ( accessType.input && accessType.output ) {
         submitDependableObjectInoutDataAccess( depObj, target, accessType, status, callback );
      } else if ( accessType.input ) {
         submitDependableObjectInputDataAccess( depObj, target, accessType, status, callback );
      } else if ( accessType.output ) {
         submitDependableObjectOutputDataAccess( depObj, target, accessType, status, callback );
      } else {
         fatal( "Invalid dara access" );
      }
      
      //! TODO (gmiranda): renable this if required
      //status.unhold();
   //}
   
   if ( !depObj.waits() && !accessType.commutative ) {
      if ( accessType.output ) {
         depObj.addWriteTarget( target );
      } else if (accessType.input /* && !accessType.output && !accessType.commutative */ ) {
         depObj.addReadTarget( target );
      }
   }
}

template<typename iterator>
void DependenciesDomain::submitDependableObjectInternal ( DependableObject& depObj, iterator begin, iterator end, SchedulePolicySuccessorFunctor* callback )
{
   depObj.setId ( _lastDepObjId++ );
   depObj.init();
   depObj.setDependenciesDomain( this );

   // Object is not ready to get its dependencies satisfied
   // so we increase the number of predecessors to permit other dependableObjects to free some of
   // its dependencies without triggering the "dependenciesSatisfied" method
   depObj.increasePredecessors();

   std::list<DataAccess *> filteredDeps;
   for ( iterator it = begin; it != end; it++ ) {
      DataAccess& newDep = (*it);
      bool found = false;
      // For every dependency processed earlier
      for ( std::list<DataAccess *>::iterator current = filteredDeps.begin(); current != filteredDeps.end(); current++ ) {
         DataAccess* currentDep = *current;
         if ( newDep.getDepAddress()  == currentDep->getDepAddress() )
         {
            // Both dependencies use the same address, put them in common
            currentDep->setInput( newDep.isInput() || currentDep->isInput() );
            currentDep->setOutput( newDep.isOutput() || currentDep->isOutput() );
            found = true;
            break;
         }
      }
      if ( !found ) {
         filteredDeps.push_back(&newDep);
      }
   }
   
   // This list is needed for waiting
   std::list<uint64_t> flushDeps;
   
   for ( std::list<DataAccess *>::iterator it = filteredDeps.begin(); it != filteredDeps.end(); it++ ) {
      DataAccess &dep = *(*it);
      
      Target target = dep.getDepAddress();
      AccessType const &accessType = dep.flags;
      
      submitDependableObjectDataAccess( depObj, target, accessType, callback );
      flushDeps.push_back( (uint64_t) target );
   }
   
   // To keep the count consistent we have to increase the number of tasks in the graph before releasing the fake dependency
   increaseTasksInGraph();

   depObj.submitted();

   // now everything is ready
   if ( depObj.decreasePredecessors() > 0 )
      depObj.wait( flushDeps );
}

//template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, Dependency* begin, Dependency* end, SchedulePolicySuccessorFunctor* callback );
//template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, std::vector<Dependency>::iterator begin, std::vector<Dependency>::iterator end, SchedulePolicySuccessorFunctor* callback );
template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, DataAccess* begin, DataAccess* end, SchedulePolicySuccessorFunctor* callback );
template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, std::vector<DataAccess>::iterator begin, std::vector<DataAccess>::iterator end, SchedulePolicySuccessorFunctor* callback );


void DependenciesDomain::increaseTasksInGraph()
{
   NANOS_INSTRUMENT(lock();)
   NANOS_INSTRUMENT(int tasks = ++_tasksInGraph;)
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("graph-size");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) tasks );)
   NANOS_INSTRUMENT(unlock();)
}

void DependenciesDomain::decreaseTasksInGraph()
{
   NANOS_INSTRUMENT(lock();)
   NANOS_INSTRUMENT(int tasks = --_tasksInGraph;)
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("graph-size");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) tasks );)
   NANOS_INSTRUMENT(unlock();)
}

