/*************************************************************************************/
/*      Copyright 2012 Barcelona Supercomputing Center                               */
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

#include "basedependenciesdomain.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"
#include "address.hpp"
#include "compatibility.hpp"

namespace nanos {
   namespace ext {

      class PlainDependenciesDomain : public BaseDependenciesDomain
      {
         private:
            typedef TR1::unordered_map<Address::TargetType, TrackableObject*> DepsMap; /**< Maps addresses to Trackable objects */
            
         private:
            DepsMap _addressDependencyMap; /**< Used to track dependencies between DependableObject */
         private:
            /*! \brief Looks for the dependency's address in the domain and returns the trackableObject associated.
             *  \param dep Dependency to be checked.
             *  \sa Dependency TrackableObject
             */
            TrackableObject* lookupDependency ( const Address& target )
            {
               TrackableObject* status = NULL;
               
               DepsMap::iterator it = _addressDependencyMap.find( target() ); 
               if ( it == _addressDependencyMap.end() ) {
                  // Lock this so we avoid problems when concurrently calling deleteLastWriter
                  SyncRecursiveLockBlock lock1( getInstanceLock() );
                  status = NEW TrackableObject();
                  _addressDependencyMap.insert( std::make_pair( target(), status ) );
               } else {
                  status = it->second;
               }
               
               return status;
            }
         protected:
            /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
             *  \param depObj DependableObject to be added to the domain.
             *  \param begin Iterator to the start of the list of dependencies to be associated to the Dependable Object.
             *  \param end Iterator to the end of the mentioned list.
             *  \param callback A function to call when a WD has a successor [Optional].
             *  \sa Dependency DependableObject TrackableObject
             */
            template<typename iterator>
            void submitDependableObjectInternal ( DependableObject &depObj, iterator begin, iterator end, SchedulePolicySuccessorFunctor* callback )
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

                  // if address == NULL, just ignore it
                  if ( newDep.getDepAddress() == NULL ) continue;
                  
                  bool found = false;
                  // For every dependency processed earlier
                  for ( std::list<DataAccess *>::iterator current = filteredDeps.begin(); current != filteredDeps.end(); current++ ) {
                     DataAccess* currentDep = *current;
                     if ( newDep.getDepAddress()  == currentDep->getDepAddress() ) {
                        // Both dependencies use the same address, put them in common
                        currentDep->setInput( newDep.isInput() || currentDep->isInput() );
                        currentDep->setOutput( newDep.isOutput() || currentDep->isOutput() );
                        found = true;
                        break;
                     }
                  }

                  if ( !found ) filteredDeps.push_back(&newDep);
               }
               
               // This list is needed for waiting
               std::list<uint64_t> flushDeps;
               
               for ( std::list<DataAccess *>::iterator it = filteredDeps.begin(); it != filteredDeps.end(); it++ ) {
                  DataAccess &dep = *(*it);
                  
                  Address target = dep.getDepAddress();
                  AccessType const &accessType = dep.flags;
                  
                  submitDependableObjectDataAccess( depObj, target, accessType, callback );
                  flushDeps.push_back( (uint64_t) target() );
               }
               
               // To keep the count consistent we have to increase the number of tasks in the graph before releasing the fake dependency
               increaseTasksInGraph();
            
               depObj.submitted();
            
               // now everything is ready
               depObj.decreasePredecessors();
               //Flush the object
               depObj.wait( flushDeps );
            }
            /*! \brief Adds a region access of a DependableObject to the domains dependency system.
             *  \param depObj target DependableObject
             *  \param target accessed memory address
             *  \param accessType kind of region access
             *  \param callback Function to call if an immediate predecessor is found.
             */
            void submitDependableObjectDataAccess( DependableObject &depObj, Address const &target, AccessType const &accessType, SchedulePolicySuccessorFunctor* callback )
            {
               if ( accessType.concurrent || accessType.commutative ) {
                  if ( !( accessType.input && accessType.output ) || depObj.waits() ) {
                     fatal( "Commutation/concurrent task must be inout" );
                  }
               }
               
               if ( accessType.concurrent && accessType.commutative ) {
                  fatal( "Task cannot be concurrent AND commutative" );
               }
               
               TrackableObject &status = *lookupDependency( target );
               //! TODO (gmiranda): enable this if required
               //status.hold(); // This is necessary since we may trigger a removal in finalizeReduction
               
               if ( accessType.concurrent || accessType.commutative ) {
                  submitDependableObjectCommutativeDataAccess( depObj, target, accessType, status, callback );
               } else if ( accessType.input && accessType.output ) {
                  submitDependableObjectInoutDataAccess( depObj, target, accessType, status, callback );
               } else if ( accessType.input ) {
                  submitDependableObjectInputDataAccess( depObj, target, accessType, status, callback );
               } else if ( accessType.output ) {
                  submitDependableObjectOutputDataAccess( depObj, target, accessType, status, callback );
               } else {
                  fatal( "Invalid data access" );
               }
               
               if ( !depObj.waits() && !accessType.concurrent && !accessType.commutative ) {
                  if ( accessType.output ) {
                     depObj.addWriteTarget( target );
                  } else if (accessType.input ) {
                     depObj.addReadTarget( target );
                  }
               }
            }
            
            inline void deleteLastWriter ( DependableObject &depObj, BaseDependency const &target )
            {
               const Address& address( dynamic_cast<const Address&>( target ) );
               SyncRecursiveLockBlock lock1( getInstanceLock() );
               DepsMap::iterator it = _addressDependencyMap.find( address() );
               
               if ( it != _addressDependencyMap.end() ) {
                  TrackableObject &status = *it->second;
                  
                  status.deleteLastWriter(depObj);
               }
            }
            
            
            inline void deleteReader ( DependableObject &depObj, BaseDependency const &target )
            {
               const Address& address( dynamic_cast<const Address&>( target ) );
               SyncRecursiveLockBlock lock1( getInstanceLock() );
               DepsMap::iterator it = _addressDependencyMap.find( address() );
               
               if ( it != _addressDependencyMap.end() ) {
                  TrackableObject &status = *it->second;
                  
                  {
                     SyncLockBlock lock2( status.getReadersLock() );
                     status.deleteReader(depObj);
                  }
               }
            }
            
            inline void removeCommDO ( CommutationDO *commDO, BaseDependency const &target )
            {
               const Address& address( dynamic_cast<const Address&>( target ) );
               SyncRecursiveLockBlock lock1( getInstanceLock() );
               DepsMap::iterator it = _addressDependencyMap.find( address() );
               
               if ( it != _addressDependencyMap.end() ) {
                  TrackableObject &status = *it->second;
                  
                  if ( status.getCommDO ( ) == commDO ) {
                     status.setCommDO ( 0 );
                  }
               }
            }

         public:
            PlainDependenciesDomain() : BaseDependenciesDomain(), _addressDependencyMap() {}
            PlainDependenciesDomain ( const PlainDependenciesDomain &depDomain )
               : BaseDependenciesDomain( depDomain ),
               _addressDependencyMap ( depDomain._addressDependencyMap ) {}
            
            ~PlainDependenciesDomain()
            {
               for ( DepsMap::iterator it = _addressDependencyMap.begin(); it != _addressDependencyMap.end(); it++ ) {
                  delete it->second;
               }
            }
            
            /*!
             *  \note This function cannot be implemented in
             *  BaseDependenciesDomain since it calls a template function,
             *  and they cannot be virtual.
             */
            inline void submitDependableObject ( DependableObject &depObj, std::vector<DataAccess> &deps, SchedulePolicySuccessorFunctor* callback )
            {
               submitDependableObjectInternal ( depObj, deps.begin(), deps.end(), callback );
            }
            
            /*!
             *  \note This function cannot be implemented in
             *  BaseDependenciesDomain since it calls a template function,
             *  and they cannot be virtual.
             */
            inline void submitDependableObject ( DependableObject &depObj, size_t numDeps, DataAccess* deps, SchedulePolicySuccessorFunctor* callback )
            {
               submitDependableObjectInternal ( depObj, deps, deps+numDeps, callback );
            }

            bool haveDependencePendantWrites ( void *addr )
            {
               DepsMap::iterator it = _addressDependencyMap.find( addr ); 
               if ( it == _addressDependencyMap.end() ) {
                  return false;
               } else {
                  TrackableObject* status = it->second;
                  DependableObject *lastWriter = status->getLastWriter();
                  return (lastWriter != NULL);
               }
            }
            
         
      };
      
      template void PlainDependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, DataAccess* begin, DataAccess* end, SchedulePolicySuccessorFunctor* callback );
      template void PlainDependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, std::vector<DataAccess>::iterator begin, std::vector<DataAccess>::iterator end, SchedulePolicySuccessorFunctor* callback );
      
      /*! \brief Default plugin implementation.
       */
      class PlainDependenciesManager : public DependenciesManager
      {
         public:
            PlainDependenciesManager() : DependenciesManager("Nanos plain dependencies domain") {}
            virtual ~PlainDependenciesManager () {}
            
            /*! \brief Creates a default dependencies domain.
             */
            DependenciesDomain* createDependenciesDomain () const
            {
               return NEW PlainDependenciesDomain();
            }
      };
  
      class NanosDepsPlugin : public Plugin
      {
            
         public:
            NanosDepsPlugin() : Plugin( "Nanos++ plain dependencies management plugin",1 )
            {
            }

            virtual void config ( Config &cfg )
            {
            }

            virtual void init()
            {
               sys.setDependenciesManager(NEW PlainDependenciesManager());
            }
      };

   }
}

DECLARE_PLUGIN("deps-plain",nanos::ext::NanosDepsPlugin);
