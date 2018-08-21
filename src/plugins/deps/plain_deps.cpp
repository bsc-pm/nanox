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

            //! \brief Clear current dependencies domain
            //!
            //! This function should be called withing a thread safe area. It is, when other
            //! tasks can not update the domain: after a taskwait and before any task submission.
            void clearDependenciesDomain ( void )
            {
               _addressDependencyMap.clear(); 
            }

            //! \brief Looks for the dependency's address, returns the trackableObject associated
            //! \param dep Dependency to be checked.
            //! \sa Dependency TrackableObject
            TrackableObject* lookupDependency ( const Address& target )
            {
               TrackableObject* status = NULL;
               
               DepsMap::iterator it = _addressDependencyMap.find( target() ); 

               if ( it == _addressDependencyMap.end() ) {
                   status = NEW TrackableObject();
                   {
                      // Lock this so we avoid problems when concurrently calling deleteLastWriter
                      // due this function will also chase also the map
                      SyncRecursiveLockBlock lock1( getInstanceLock() );
                      _addressDependencyMap.insert( std::make_pair( target(), status ) );
                   }
               } else {
                  status = it->second;
               }
               
               return status;
            }
         protected:
            //! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
            //! \param depObj DependableObject to be added to the domain.
            //! \param begin Iterator to the start of the list of dependencies to be associated to the Dependable Object.
            //! \param end Iterator to the end of the mentioned list.
            //! \param callback A function to call when a WD has a successor [Optional].
            //! \sa Dependency DependableObject TrackableObject
            template<typename iterator>
            void submitDependableObjectInternal ( DependableObject &depObj, iterator begin, iterator end,
                                                  SchedulePolicySuccessorFunctor* callback )
            {
               // Initializing several properties of the depObject
               depObj.setId ( _lastDepObjId++ );
               depObj.init();
               depObj.setDependenciesDomain( this );
            
               // Object is not ready to get its dependencies satisfied, so we increase the
               // number of predecessors to permit other dependableObjects to free some of
               // its dependencies without triggering the "dependenciesSatisfied" method.
               depObj.increasePredecessors();
            
               // flushDeps will be needed for waiting (see decreasePredecessors)
               std::list<uint64_t> flushDeps;

               // Iterate from begin to end, just to handle each data access
               for ( iterator it = begin; it != end; it++ ) {
                  DataAccess &dep = (*it);
                  Address target = dep.getDepAddress();

                  // if address == NULL, just ignore it
                  if ( target() == NULL ) continue;
                  AccessType const &accessType = dep.flags;

                  submitDependableObjectDataAccess( depObj, target, accessType, callback );
                  flushDeps.push_back( (uint64_t) target() );
               }
               
               // Calling scheduler policy "atCreate"
               sys.getDefaultSchedulePolicy()->atCreate( depObj );
               
               // To Task In Graph count consistent before releasing the fake dependency
               increaseTasksInGraph();
            
               depObj.submitted();
            
               // Now everything is ready, release fake dependency
               depObj.decreasePredecessors( &flushDeps, NULL, false, true );
            }

            //! \brief Adds a region access of a DependableObject to the domains dependency system.
            //! \param depObj target DependableObject
            //! \param target accessed memory address
            //! \param accessType kind of region access
            //! \param callback Function to call if an immediate predecessor is found.
            void submitDependableObjectDataAccess( DependableObject &depObj, Address const &target,
                                                   AccessType const &accessType, SchedulePolicySuccessorFunctor* callback )
            {

               ensure(!(accessType.concurrent && accessType.commutative),"Task cannot be concurrent AND commutative");

               TrackableObject &status = *lookupDependency( target );

               if ( status.getLastWriter() == &depObj ) return;

               if ( accessType.concurrent || accessType.commutative ) {
                  ensure(accessType.input && accessType.output,"Commutative & concurrent must be inout");
                  ensure(!depObj.waits(), "Commutative & concurrent should not wait" );
                  submitDependableObjectCommutativeDataAccess( depObj, target, accessType, status, callback );
               } else if ( accessType.output && accessType.input ) {
                  submitDependableObjectInoutDataAccess( depObj, target, accessType, status, callback );
                  // We don't add as write target depObj.addWriteTarget(), due this op is done internally
                  // in basedependencyregion as part of finding a writer. This same mechanism will be
                  // used by commutative and concurrent access to summarize dependences
                  if ( !depObj.waits() ) depObj.addReadTarget( target );
               } else if ( accessType.output ) {
                  // We don't add as write target depObj.addWriteTarget(), see comment above
                  submitDependableObjectOutputDataAccess( depObj, target, accessType, status, callback );
               } else if ( accessType.input  ) {
                  submitDependableObjectInputDataAccess( depObj, target, accessType, status, callback );
                  if ( !depObj.waits() ) depObj.addReadTarget( target );
               } else {
                  fatal( "Invalid data access" );
               }

            }
            
            inline void deleteLastWriter ( DependableObject &depObj, BaseDependency const &target )
            {
               const Address& address( static_cast<const Address&>( target ) );
               SyncRecursiveLockBlock lock1( getInstanceLock() );
               DepsMap::iterator it = _addressDependencyMap.find( address() );
               
               if ( it != _addressDependencyMap.end() ) {
                  TrackableObject &status = *it->second;
                  
                  status.deleteLastWriter(depObj);
               }
            }
            
            
            inline void deleteReader ( DependableObject &depObj, BaseDependency const &target )
            {
               const Address& address( static_cast<const Address&>( target ) );
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
               const Address& address( static_cast<const Address&>( target ) );
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
            void finalizeAllReductions ( void )
            {
               DepsMap::iterator it; 
               for ( it = _addressDependencyMap.begin(); it != _addressDependencyMap.end(); it++ ) {
                  TrackableObject& status = *( it->second );
                  Address::TargetType target = it->first;
                  CommutationDO *commDO = status.getCommDO();
                  if ( commDO != NULL ) {
                     status.setCommDO( NULL );
                     status.setLastWriter( *commDO );

                     TaskReduction *tr = myThread->getCurrentWD()->getTaskReduction( (const void *) target );
                     if ( tr != NULL ) {
                        if ( myThread->getCurrentWD()->getDepth() == tr->getDepth() )
							commDO->setTaskReduction( tr );
                     }

                     commDO->resetReferences();

                     //! Finally decrease dummy dependence added in createCommutationDO
                     std::list<uint64_t> flushDeps;
                     commDO->decreasePredecessors( &flushDeps, NULL, false, false ); 
                  }
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
