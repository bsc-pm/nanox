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

#ifndef _NANOS_DEPENDENCIES_DOMAIN
#define _NANOS_DEPENDENCIES_DOMAIN
#include <stdlib.h>
#include <map>
#include <list>
#include <vector>
#include "dependenciesdomain_decl.hpp"
#include "atomic.hpp"
#include "dependableobject.hpp"
#include "trackableobject.hpp"
#include "dataaccess_decl.hpp"
#include "compatibility.hpp"


using namespace nanos;

inline DependenciesDomain::~DependenciesDomain ( )
{
   for ( DepsMap::iterator it = _addressDependencyMap.begin(); it != _addressDependencyMap.end(); it++ ) {
      delete it->second;
   }
}

inline void DependenciesDomain::submitDependableObject ( DependableObject &depObj, std::vector<DataAccess> &deps, SchedulePolicySuccessorFunctor* callback )
{
   submitDependableObjectInternal ( depObj, deps.begin(), deps.end(), callback );
}

inline void DependenciesDomain::submitDependableObject ( DependableObject &depObj, size_t numDeps, DataAccess* deps, SchedulePolicySuccessorFunctor* callback )
{
   submitDependableObjectInternal ( depObj, deps, deps+numDeps, callback );
}

inline RecursiveLock& DependenciesDomain::getInstanceLock()
{
   return _instanceLock;
}

inline Lock& DependenciesDomain::getLock()
{
   return _lock;
}

inline void DependenciesDomain::lock ( )
{
   _lock.acquire();
   memoryFence();
}

inline void DependenciesDomain::unlock ( )
{
   memoryFence();
   _lock.release();
}

inline const std::string & DependenciesManager::getName () const
{
   return _name;
}

inline DependenciesDomain::MappedType* DependenciesDomain::lookupDependency ( const Target& target )
{
   MappedType* status = NULL;
   //Target address = dep.getDepAddress();
   
   DepsMap::iterator it = _addressDependencyMap.find( target ); 
   if ( it == _addressDependencyMap.end() ) {
      // Lock this so we avoid problems when concurrently calling deleteLastWriter
      SyncRecursiveLockBlock lock1( _instanceLock );
      status = NEW MappedType( target );
      _addressDependencyMap.insert( std::make_pair( target, status ) );
   } else {
      status = it->second;
   }
   
   return status;
}

inline void DependenciesDomain::finalizeReduction( MappedType &status,  const Target& target )
{
   CommutationDO *commDO = status.getCommDO();
   if ( commDO != NULL ) {
      status.setCommDO( NULL );

      // This ensures that even if commDO's dependencies are satisfied
      // during this step, lastWriter will be reseted 
      DependableObject *lw = status.getLastWriter();
      if ( commDO->increasePredecessors() == 0 ) {
         // We increased the number of predecessors but someone just decreased them to 0
         // that will execute finished and we need to wait for the lastWriter to be deleted
         if ( lw == commDO ) {
            while ( status.getLastWriter() != NULL ) {}
         }
      }
      commDO->addWriteTarget( target );
      status.setLastWriter( *commDO );
      commDO->resetReferences();
      commDO->decreasePredecessors();
   }
}

inline void DependenciesDomain::dependOnLastWriter( DependableObject &depObj, MappedType const & status, SchedulePolicySuccessorFunctor* callback )
{
   DependableObject *lastWriter = status.getLastWriter();
   if ( lastWriter != NULL ) {
      SyncLockBlock lck( lastWriter->getLock() );
      if ( status.getLastWriter() == lastWriter ) {
         if ( lastWriter->addSuccessor( depObj ) ) {
            depObj.increasePredecessors();
            if ( callback != NULL ) {
               debug( "Calling callback" );
               ( *callback )( lastWriter, &depObj );
            }
         }
      }
   }
}

inline void DependenciesDomain::dependOnReadersAndSetAsWriter( DependableObject &depObj, MappedType &status, Target const &target, SchedulePolicySuccessorFunctor* callback )
{
   MappedType::DependableObjectList &readersList = status.getReaders();
   SyncLockBlock lock4( status.getReadersLock() );
   for ( MappedType::DependableObjectList::iterator i = readersList.begin(); i != readersList.end(); i++) {
      DependableObject * predecessorReader = *i;
      SyncLockBlock lock5(predecessorReader->getLock());
      if ( predecessorReader->addSuccessor( depObj ) ) {
         depObj.increasePredecessors();
         if ( callback != NULL ) {
            debug( "Calling callback" );
            ( *callback )( predecessorReader, &depObj );
         }
      }
      // WaR dependency
#if 0
      debug (" DO_ID_" << predecessorReader->getId() << " [style=filled label=" << predecessorReader->getDescription() << " color=" << "red" << "];");
      debug (" DO_ID_" << predecessorReader->getId() << "->" << "DO_ID_" << depObj.getId() << "[color=red];");
#endif
   }
   
   status.flushReaders();
   if ( !depObj.waits() ) {
      // set depObj as writer of dependencyObject
      depObj.addWriteTarget( target );
      status.setLastWriter( depObj );
   }
}

inline void DependenciesDomain::addAsReader( DependableObject &depObj, MappedType &status )
{
   SyncLockBlock lock3( status.getReadersLock() );
   status.setReader( depObj );
}

inline void DependenciesDomain::submitDependableObjectCommutativeDataAccess ( DependableObject &depObj, Target const &target, AccessType const &accessType, MappedType &status, SchedulePolicySuccessorFunctor* callback )
{
   CommutationDO *initialCommDO = NULL;
   CommutationDO *commDO = status.getCommDO();
   
   /* FIXME: this must be atomic */

   if ( commDO == NULL ) {
      commDO = new CommutationDO( target );
      commDO->setDependenciesDomain( this );
      commDO->increasePredecessors();
      status.setCommDO( commDO );
      commDO->addWriteTarget( target );
   } else {
      if ( commDO->increasePredecessors() == 0 ) {
         commDO = new CommutationDO( target );
         commDO->setDependenciesDomain( this );
         commDO->increasePredecessors();
         status.setCommDO( commDO );
         commDO->addWriteTarget( target );
      }
   }

   if ( status.hasReaders() ) {
      initialCommDO = new CommutationDO( target );
      initialCommDO->setDependenciesDomain( this );
      initialCommDO->increasePredecessors();
      // add dependencies to all previous reads using a CommutationDO
      MappedType::DependableObjectList &readersList = status.getReaders();
      {
         SyncLockBlock lock1( status.getReadersLock() );

         for ( MappedType::DependableObjectList::iterator i = readersList.begin(); i != readersList.end(); i++) {
            DependableObject * predecessorReader = *i;
            {
               SyncLockBlock lock2( predecessorReader->getLock() );
               if ( predecessorReader->addSuccessor( *initialCommDO ) ) {
                  initialCommDO->increasePredecessors();
               }
            }
         }
         status.flushReaders();
      }
      initialCommDO->addWriteTarget( target );
      // Replace the lastWriter with the initial CommutationDO
      status.setLastWriter( *initialCommDO );
   }
   
   // Add the Commutation object as successor of the current DO (depObj)
   depObj.addSuccessor( *commDO );
   
   // assumes no new readers added concurrently
   dependOnLastWriter( depObj, status, callback );

   // The dummy predecessor is to make sure that initialCommDO does not execute 'finished'
   // while depObj is being added as its successor
   if ( initialCommDO != NULL ) {
      initialCommDO->decreasePredecessors();
   }

   dependOnReadersAndSetAsWriter( depObj, status, target, callback );
}

inline void DependenciesDomain::submitDependableObjectInoutDataAccess ( DependableObject &depObj, Target const &target, AccessType const &accessType, MappedType &status, SchedulePolicySuccessorFunctor* callback )
{
   finalizeReduction( status, target );
   
   dependOnLastWriter( depObj, status, callback );
   dependOnReadersAndSetAsWriter( depObj, status, target, callback );
}

inline void DependenciesDomain::submitDependableObjectInputDataAccess ( DependableObject &depObj, Target const &target, AccessType const &accessType, MappedType &status, SchedulePolicySuccessorFunctor* callback )
{
   finalizeReduction( status, target );
   dependOnLastWriter( depObj, status, callback );

   if ( !depObj.waits() ) {
      addAsReader( depObj, status );
   }
}

inline void DependenciesDomain::submitDependableObjectOutputDataAccess ( DependableObject &depObj, Target const &target, AccessType const &accessType, MappedType &status, SchedulePolicySuccessorFunctor* callback )
{
   finalizeReduction( status, target );
   
   // assumes no new readers added concurrently
   if ( !status.hasReaders() ) {
      dependOnLastWriter( depObj, status, callback );
   }

   dependOnReadersAndSetAsWriter( depObj, status, target, callback );
}

inline void DependenciesDomain::deleteLastWriter ( DependableObject &depObj, Target const &target )
{
   SyncRecursiveLockBlock lock1( _instanceLock );
   DepsMap::iterator it = _addressDependencyMap.find( target );
   
   if ( it != _addressDependencyMap.end() ) {
      MappedType &status = *it->second;
      
      status.deleteLastWriter(depObj);
   }
}


inline void DependenciesDomain::deleteReader ( DependableObject &depObj, Target const &target )
{
   SyncRecursiveLockBlock lock1( _instanceLock );
   DepsMap::iterator it = _addressDependencyMap.find( target );
   
   if ( it != _addressDependencyMap.end() ) {
      MappedType &status = *it->second;
      
      {
         SyncLockBlock lock2( status.getReadersLock() );
         status.deleteReader(depObj);
      }
   }
}

inline void DependenciesDomain::removeCommDO ( CommutationDO *commDO, Target const &target )
{
   SyncRecursiveLockBlock lock1( _instanceLock );
   DepsMap::iterator it = _addressDependencyMap.find( target );
   
   if ( it != _addressDependencyMap.end() ) {
      MappedType &status = *it->second;
      
      if ( status.getCommDO ( ) == commDO ) {
         status.setCommDO ( 0 );
      }
   }
}

#endif

