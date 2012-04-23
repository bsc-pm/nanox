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

#ifndef _NANOS_BASE_DEPENDENCIES_DOMAIN
#define _NANOS_BASE_DEPENDENCIES_DOMAIN

#include "basedependenciesdomain_decl.hpp"
#include "debug.hpp"
#include "schedule_decl.hpp"
#include "system.hpp"


namespace nanos {

inline void BaseDependenciesDomain::finalizeReduction( TrackableObject &status, const BaseDependency& target )
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

inline void BaseDependenciesDomain::dependOnLastWriter( DependableObject &depObj, TrackableObject const &status, SchedulePolicySuccessorFunctor* callback )
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

inline void BaseDependenciesDomain::dependOnReaders( DependableObject &depObj, TrackableObject &status, BaseDependency const &target, SchedulePolicySuccessorFunctor* callback )
{
   TrackableObject::DependableObjectList &readersList = status.getReaders();
   SyncLockBlock lock4( status.getReadersLock() );
   for ( TrackableObject::DependableObjectList::iterator i = readersList.begin(); i != readersList.end(); i++) {
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
}

inline void BaseDependenciesDomain::setAsWriter( DependableObject &depObj, TrackableObject &status, BaseDependency const &target )
{
   {
      SyncLockBlock lock3( status.getReadersLock() );
      status.flushReaders();
   }
   if ( !depObj.waits() ) {
      // set depObj as writer of dependencyObject
      depObj.addWriteTarget( target );
      status.setLastWriter( depObj );
   }
}

inline void BaseDependenciesDomain::dependOnReadersAndSetAsWriter( DependableObject &depObj, TrackableObject &status, BaseDependency const &target, SchedulePolicySuccessorFunctor* callback )
{
   dependOnReaders(depObj, status, target, callback);
   setAsWriter( depObj, status, target );
}

inline void BaseDependenciesDomain::addAsReader( DependableObject &depObj, TrackableObject &status )
{
   SyncLockBlock lock3( status.getReadersLock() );
   status.setReader( depObj );
}

inline CommutationDO * BaseDependenciesDomain::setUpTargetCommutationDependableObject ( BaseDependency const &target, TrackableObject &status )
{
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
   
   return commDO;
}


inline CommutationDO * BaseDependenciesDomain::setUpInitialCommutationDependableObject ( BaseDependency const &target, TrackableObject &status )
{
   CommutationDO *initialCommDO = NULL;
   
   if ( status.hasReaders() ) {
      initialCommDO = new CommutationDO( target );
      initialCommDO->setDependenciesDomain( this );
      initialCommDO->increasePredecessors();
      
      // add dependencies to all previous reads using a CommutationDO
      dependOnReaders( *initialCommDO, status, target, NULL );
      {
         SyncLockBlock lock3( status.getReadersLock() );
         status.flushReaders();
      }
      initialCommDO->addWriteTarget( target );
      
      // Replace the lastWriter with the initial CommutationDO
      status.setLastWriter( *initialCommDO );
   }
   
   return initialCommDO;
}


inline void BaseDependenciesDomain::submitDependableObjectCommutativeDataAccess ( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback )
{
   // NOTE: Do not change the order
   CommutationDO *initialCommDO = setUpInitialCommutationDependableObject( target, status );
   CommutationDO *commDO = setUpTargetCommutationDependableObject( target, status );
   
   // Add the Commutation object as successor of the current DO (depObj)
   depObj.addSuccessor( *commDO );
   
   // assumes no new readers added concurrently
   dependOnLastWriter( depObj, status, callback );

   // The dummy predecessor is to make sure that initialCommDO does not execute 'finished'
   // while depObj is being added as its successor
   if ( initialCommDO != NULL ) {
      initialCommDO->decreasePredecessors();
   }
}

inline void BaseDependenciesDomain::submitDependableObjectInoutDataAccess ( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback )
{
   finalizeReduction( status, target );
   dependOnLastWriter( depObj, status, callback );
   dependOnReadersAndSetAsWriter( depObj, status, target, callback );
}

inline void BaseDependenciesDomain::submitDependableObjectInputDataAccess ( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback )
{
   finalizeReduction( status, target );
   dependOnLastWriter( depObj, status, callback );

   if ( !depObj.waits() ) {
      addAsReader( depObj, status );
   }
}

inline void BaseDependenciesDomain::submitDependableObjectOutputDataAccess ( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback )
{
   finalizeReduction( status, target );
   
   // assumes no new readers added concurrently
   if ( !status.hasReaders() ) {
      dependOnLastWriter( depObj, status, callback );
   }

   dependOnReadersAndSetAsWriter( depObj, status, target, callback );
}

}

#endif
