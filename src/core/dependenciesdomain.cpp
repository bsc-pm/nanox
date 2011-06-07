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
#include "regiontree.hpp"
#include "regionbuilder.hpp"
#include "regionstatus.hpp"
#include <iostream>
#include <fstream>
#include <sstream>


namespace nanos {

Atomic<int> DependenciesDomain::_atomicSeed( 0 );
Atomic<int> DependenciesDomain::_tasksInGraph( 0 );
Lock DependenciesDomain::_lock;


namespace dependencies_domain_internal {
   class AccessType: public nanos_access_type_internal_t {
   public:
      AccessType()
         {
            input = 0;
            output = 0;
            can_rename = 0;
            commutative = 0;
         }
      
      AccessType(nanos_access_type_internal_t const &accessType)
         {
            input = accessType.input;
            output = accessType.output;
            can_rename = accessType.can_rename;
            commutative = accessType.commutative;
         }
      
      AccessType const &operator|=(nanos_access_type_internal_t const &accessType)
         {
            input |= accessType.input;
            output |= accessType.output;
            can_rename |= accessType.can_rename;
            commutative |= accessType.commutative;
            
            return *this;
         }
      friend std::ostream &operator<<( std::ostream &o, nanos::AccessType const &accessType);
   };


   inline std::ostream & operator<<( std::ostream &o, nanos::AccessType const &accessType)
   {
      if ( accessType.input && accessType.output ) {
         if ( accessType.commutative ) {
            o << "RED";
         } else {
            o << "INOUT";
         }
      } else if ( accessType.input && !accessType.commutative ) {
         o << "IN";
      } else if ( accessType.output && !accessType.commutative ) {
         o << "OUT";
      } else {
         o << "ERR";
      }
      return o;
   }
} // namespace dependencies_domain_internal


using namespace dependencies_domain_internal;


#if 0
static void dumpAccessTypeRegionTree(RegionTree<AccessType> const &accessTypeRegionTree, std::string const &file = "", size_t line = 0)
{
   static int version = 0;
   
   version++;
   std::ostringstream filename;
   
   filename << "at-rt-";
   filename.width(4);
   filename.fill('0');
   filename << version;
   filename << "-domain-";
   filename.width(16);
   filename << std::hex << (void *)&accessTypeRegionTree << ".dot";
   
   std::cout << file << ":" << line << ": Dumping Access Type Tree " << filename.str() << std::endl;
   std::ofstream output(filename.str().c_str());
   output << accessTypeRegionTree;
   output.close();
}
#endif


inline void DependenciesDomain::finalizeReduction( RegionStatus &regionStatus, Region const &region )
{
   CommutationDO *commDO = regionStatus.getCommDO();
   if ( commDO != NULL ) {
      regionStatus.setCommDO( NULL );

      // This ensures that even if commDO's dependencies are satisfied
      // during this step, lastWriter will be reseted 
      DependableObject *lw = regionStatus.getLastWriter();
      if ( commDO->increasePredecessors() == 0 ) {
         // We increased the number of predecessors but someone just decreased them to 0
         // that will execute finished and we need to wait for the lastWriter to be deleted
         if ( lw == commDO ) {
            while ( regionStatus.getLastWriter() != NULL ) {}
         }
      }
      commDO->addWriteRegion( region );
      regionStatus.setLastWriter( *commDO );
      commDO->resetReferences();
      commDO->decreasePredecessors();
   }
}


inline void DependenciesDomain::dependOnLastWriter( DependableObject &depObj, RegionStatus const &regionStatus )
{
   DependableObject *lastWriter = regionStatus.getLastWriter();
   if ( lastWriter != NULL ) {
      SyncLockBlock lck( lastWriter->getLock() );
      if ( regionStatus.getLastWriter() == lastWriter ) {
         if ( lastWriter->addSuccessor( depObj ) ) {
            depObj.increasePredecessors();
         }
      }
   }
}


inline void DependenciesDomain::dependOnReadersAndSetAsWriter( DependableObject &depObj, RegionStatus &regionStatus, Region const &region )
{
   RegionStatus::DependableObjectList &readersList = regionStatus.getReaders();
   SyncLockBlock lock4( regionStatus.getReadersLock() );
   for ( RegionStatus::DependableObjectList::iterator i = readersList.begin(); i != readersList.end(); i++) {
      DependableObject * predecessorReader = *i;
      SyncLockBlock lock5(predecessorReader->getLock());
      if ( predecessorReader->addSuccessor( depObj ) ) {
         depObj.increasePredecessors();
      }
      // WaR dependency
#if 0
      debug (" DO_ID_" << predecessorReader->getId() << " [style=filled label=" << predecessorReader->getDescription() << " color=" << "red" << "];");
      debug (" DO_ID_" << predecessorReader->getId() << "->" << "DO_ID_" << depObj.getId() << "[color=red];");
#endif
   }
   
   regionStatus.flushReaders();
   if ( !depObj.waits() ) {
      // set depObj as writer of dependencyObject
      depObj.addWriteRegion( region );
      regionStatus.setLastWriter( depObj );
   }
}


inline void DependenciesDomain::addAsReader( DependableObject &depObj, RegionStatus &regionStatus )
{
   SyncLockBlock lock3( regionStatus.getReadersLock() );
   regionStatus.setReader( depObj );
}


inline void DependenciesDomain::submitDependableObjectCommutativeDataAccess ( DependableObject &depObj, Region const &region, AccessType const &accessType, RegionStatus &regionStatus )
{
   CommutationDO *initialCommDO = NULL;
   CommutationDO *commDO = regionStatus.getCommDO();
   
   /* FIXME: this must be atomic */

   if ( commDO == NULL ) {
      commDO = new CommutationDO( region );
      commDO->setDependenciesDomain( this );
      commDO->increasePredecessors();
      regionStatus.setCommDO( commDO );
      commDO->addWriteRegion( region );
   } else {
      if ( commDO->increasePredecessors() == 0 ) {
         commDO = new CommutationDO( region );
         commDO->setDependenciesDomain( this );
         commDO->increasePredecessors();
         regionStatus.setCommDO( commDO );
         commDO->addWriteRegion( region );
      }
   }

   if ( regionStatus.hasReaders() ) {
      initialCommDO = new CommutationDO( region );
      initialCommDO->setDependenciesDomain( this );
      initialCommDO->increasePredecessors();
      // add dependencies to all previous reads using a CommutationDO
      RegionStatus::DependableObjectList &readersList = regionStatus.getReaders();
      {
         SyncLockBlock lock1( regionStatus.getReadersLock() );

         for ( RegionStatus::DependableObjectList::iterator i = readersList.begin(); i != readersList.end(); i++) {
            DependableObject * predecessorReader = *i;
            {
               SyncLockBlock lock2( predecessorReader->getLock() );
               if ( predecessorReader->addSuccessor( *initialCommDO ) ) {
                  initialCommDO->increasePredecessors();
               }
            }
         }
         regionStatus.flushReaders();
      }
      initialCommDO->addWriteRegion( region );
      // Replace the lastWriter with the initial CommutationDO
      regionStatus.setLastWriter( *initialCommDO );
   }
   
   // Add the Commutation object as successor of the current DO (depObj)
   depObj.addSuccessor( *commDO );
   
   // assumes no new readers added concurrently
   dependOnLastWriter( depObj, regionStatus );

   // The dummy predecessor is to make sure that initialCommDO does not execute 'finished'
   // while depObj is being added as its successor
   if ( initialCommDO != NULL ) {
      initialCommDO->decreasePredecessors();
   }

   dependOnReadersAndSetAsWriter( depObj, regionStatus, region );
}


inline void DependenciesDomain::submitDependableObjectInoutDataAccess ( DependableObject &depObj, Region const &region, AccessType const &accessType, RegionStatus &regionStatus )
{
   finalizeReduction( regionStatus, region );
   
   dependOnLastWriter( depObj, regionStatus );
   dependOnReadersAndSetAsWriter( depObj, regionStatus, region );
}


inline void DependenciesDomain::submitDependableObjectInputDataAccess ( DependableObject &depObj, Region const &region, AccessType const &accessType, RegionStatus &regionStatus )
{
   finalizeReduction( regionStatus, region );
   dependOnLastWriter( depObj, regionStatus );

   if ( !depObj.waits() ) {
      addAsReader( depObj, regionStatus );
   }
}


inline void DependenciesDomain::submitDependableObjectOutputDataAccess ( DependableObject &depObj, Region const &region, AccessType const &accessType, RegionStatus &regionStatus )
{
   finalizeReduction( regionStatus, region );
   
   // assumes no new readers added concurrently
   if ( !regionStatus.hasReaders() ) {
      dependOnLastWriter( depObj, regionStatus );
   }

   dependOnReadersAndSetAsWriter( depObj, regionStatus, region );
}


inline void DependenciesDomain::submitDependableObjectDataAccess ( DependableObject &depObj, Region const &region, AccessType const &accessType )
{
   if ( accessType.commutative ) {
      if ( !( accessType.input && accessType.output ) || depObj.waits() ) {
         fatal( "Commutation task must be inout" );
      }
   }
   
   
   SyncRecursiveLockBlock lock1( _instanceLock );
   //typedef std::set<RegionMap::iterator> subregion_set_t;
   typedef RegionMap::iterator_list_t subregion_set_t;
   subregion_set_t subregions;
   RegionMap::iterator wholeRegion = _regionMap.findAndPopulate( region, /* out */subregions );
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
      RegionMap::iterator &accessor = *it;
      
      Region const &subregion = accessor.getRegion();
      RegionStatus &regionStatus = *accessor;
      
      if ( accessType.commutative ) {
         submitDependableObjectCommutativeDataAccess( depObj, subregion, accessType, regionStatus );
      } else if ( accessType.input && accessType.output ) {
         submitDependableObjectInoutDataAccess( depObj, subregion, accessType, regionStatus );
      } else if ( accessType.input ) {
         submitDependableObjectInputDataAccess( depObj, subregion, accessType, regionStatus );
      } else if ( accessType.output ) {
         submitDependableObjectOutputDataAccess( depObj, subregion, accessType, regionStatus );
      } else {
         fatal( "Invalid dara access" );
      }
      
      regionStatus.unhold();
      // Just in case
      if ( regionStatus.isEmpty() ) {
         accessor.erase();
      }
   }
   
   if ( !depObj.waits() && !accessType.commutative ) {
      if ( accessType.output ) {
         depObj.addWriteRegion( region );
      } else if (accessType.input /* && !accessType.output && !accessType.commutative */ ) {
         depObj.addReadRegion( region );
      }
   }
}


template<typename const_iterator>
void DependenciesDomain::submitDependableObjectInternal ( DependableObject& depObj, const_iterator begin, const_iterator end )
{
   depObj.setId ( _lastDepObjId++ );
   depObj.init();
   depObj.setDependenciesDomain( this );

   // Object is not ready to get its dependencies satisfied
   // so we increase the number of predecessors to permit other dependableObjects to free some of
   // its dependencies without triggering the "dependenciesSatisfied" method
   depObj.increasePredecessors();

   // This tree is used for coalescing the accesses to avoid duplicates
   typedef RegionTree<AccessType> access_type_region_tree_t;
   typedef access_type_region_tree_t::iterator_list_t::iterator access_type_accessor_iterator_t;
   access_type_region_tree_t accessTypeRegionTree;
   for ( const_iterator it = begin; it != end; it++ ) {
      DataAccess const &dataAccess = *it;
      
      // Find out the displacement due to the lower bounds and correct it in the address
      size_t base = 1UL;
      size_t displacement = 0L;
      for (short dimension = 0; dimension < dataAccess.dimension_count; dimension++) {
         nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];
         displacement = displacement + dimensionData.lower_bound * base;
         base = base * dimensionData.size;
      }
      size_t address = (size_t)dataAccess.address + displacement;
      
      // Build the Region
      
      // First dimension is base 1
      size_t additionalContribution = 0UL; // Contribution of the previous dimensions (necessary due to alignment issues)
      Region region = RegionBuilder::build(address, 1UL, dataAccess.dimensions[0].accessed_length, additionalContribution);
      
      // Add the bits corresponding to the rest of the dimensions (base the previous one)
      base = 1 * dataAccess.dimensions[0].size;
      for (short dimension = 1; dimension < dataAccess.dimension_count; dimension++) {
         nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];
         
         region |= RegionBuilder::build(address, base, dimensionData.accessed_length, additionalContribution);
         base = base * dimensionData.size;
      }
      
      typename access_type_region_tree_t::iterator_list_t accessors;
      access_type_region_tree_t::iterator exactAccessor = accessTypeRegionTree.findAndPopulate(region, accessors);
      if (!exactAccessor.isEmpty()) {
         accessors.push_back(exactAccessor);
      }
      for (access_type_accessor_iterator_t it2 = accessors.begin(); it2 != accessors.end(); it2++) {
         access_type_region_tree_t::iterator accessor = *it2;
         (*accessor) |= dataAccess.flags;
      }
   }
   
   // This list is needed for waiting
   std::list<Region> regions;
   
   {
      typename access_type_region_tree_t::iterator_list_t accessors;
      accessTypeRegionTree.find ( Region(0UL, 0UL), accessors );
      for (access_type_accessor_iterator_t it = accessors.begin(); it != accessors.end(); it++) {
         access_type_region_tree_t::iterator accessor = *it;
         Region region = accessor.getRegion();
         AccessType const &accessType = *accessor;
         
         submitDependableObjectDataAccess( depObj, region, accessType );
         regions.push_back(region);
      }
   }
      
   // To keep the count consistent we have to increase the number of tasks in the graph before releasing the fake dependency
   increaseTasksInGraph();

   depObj.submitted();

   // now everything is ready
   if ( depObj.decreasePredecessors() > 0 )
      depObj.wait( regions );
}


void DependenciesDomain::deleteLastWriter ( DependableObject &depObj, Region const &region )
{
   SyncRecursiveLockBlock lock1( _instanceLock );
   RegionMap::iterator_list_t subregions;
   _regionMap.find( region, /* out */subregions );
   
   for (
      RegionMap::iterator_list_t::iterator it = subregions.begin();
      it != subregions.end();
      it++
   ) {
      RegionMap::iterator &accessor = *it;
      RegionStatus &regionStatus = *accessor;
      
      regionStatus.deleteLastWriter(depObj);
      
      if ( regionStatus.isEmpty() && !regionStatus.isOnHold( ) ) {
         accessor.erase( );
      }
   }
}


void DependenciesDomain::deleteReader ( DependableObject &depObj, Region const &region )
{
   SyncRecursiveLockBlock lock1( _instanceLock );
   RegionMap::iterator_list_t subregions;
   _regionMap.find( region, /* out */subregions );
   
   for (
      RegionMap::iterator_list_t::iterator it = subregions.begin();
      it != subregions.end();
      it++
   ) {
      RegionMap::iterator &accessor = *it;
      RegionStatus &regionStatus = *accessor;
      
      {
         SyncLockBlock lock2( regionStatus.getReadersLock() );
         regionStatus.deleteReader(depObj);
      }
            
      if ( regionStatus.isEmpty() && !regionStatus.isOnHold( ) ) {
         accessor.erase( );
      }
   }
}


void DependenciesDomain::removeCommDO ( CommutationDO *commDO, Region const &region )
{
   SyncRecursiveLockBlock lock1( _instanceLock );
   RegionMap::iterator_list_t subregions;
   _regionMap.find( region, /* out */subregions );
   
   for (
      RegionMap::iterator_list_t::iterator it = subregions.begin();
      it != subregions.end();
      it++
   ) {
      RegionMap::iterator &accessor = *it;
      RegionStatus &regionStatus = *accessor;
      
      if ( regionStatus.getCommDO ( ) == commDO ) {
         regionStatus.setCommDO ( 0 );
      }
      
      if ( regionStatus.isEmpty() && !regionStatus.isOnHold( ) ) {
         accessor.erase( );
      }
   }
}


template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, DataAccess const *begin, DataAccess const *end );
template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, std::vector<DataAccess>::const_iterator begin, std::vector<DataAccess>::const_iterator end );

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


void DependenciesDomain::dump(std::string const &function, std::string const &file, size_t line)
{
   static int version = 0;
   
   version++;
   std::ostringstream filename;
   
   filename << "rt-";
   filename.width(4);
   filename.fill('0');
   filename << version;
   filename << "-domain-";
   filename.width(16);
   filename << std::hex << (void *)this << ".dot";
   
   std::cout << function << " " << file << ":" << line << ": Dumping Region Tree " << filename.str() << std::endl;
   std::ofstream output(filename.str().c_str());
   output << _regionMap;
   output.close();
}


} // namespace nanos
