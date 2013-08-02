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

inline void BaseDependenciesDomain::dependOnLastWriter ( DependableObject &depObj, TrackableObject const &status,
                                                         BaseDependency const &target, SchedulePolicySuccessorFunctor* callback )
{
   DependableObject *lastWriter = status.getLastWriter();
   if ( lastWriter != NULL ) {
      SyncLockBlock lck( lastWriter->getLock() );
      if ( status.getLastWriter() == lastWriter ) {

         // new instrument event: dependence lastWriter -> depObj
         NANOS_INSTRUMENT ( WorkDescriptor *wd_sender = (WorkDescriptor *) lastWriter->getRelatedObject(); )
         NANOS_INSTRUMENT ( WorkDescriptor *wd_receiver = (WorkDescriptor *) depObj.getRelatedObject(); )
         NANOS_INSTRUMENT ( if ( wd_sender && wd_receiver ) { )
            NANOS_INSTRUMENT ( nanos_event_value_t Values[3]; )
            NANOS_INSTRUMENT ( Values[0] = ( ((nanos_event_value_t) wd_sender->getId()) << 32 ) + wd_receiver->getId(); )
            NANOS_INSTRUMENT ( Values[1] = ((nanos_event_value_t) 0); )
            NANOS_INSTRUMENT ( Values[2] = ((nanos_event_value_t) target.getAddress() ); )
            NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(3, _insKeyDeps, Values); )
         NANOS_INSTRUMENT ( } )

         if ( lastWriter->addSuccessor( depObj ) ) {
            // new dependence lastWriter -> depObj
            depObj.increasePredecessors();
            if ( callback != NULL ) {
               ( *callback )( lastWriter, &depObj );
            }
         }
      }
   }
}

inline void BaseDependenciesDomain::dependOnReaders( DependableObject &depObj, TrackableObject &status, BaseDependency const &target,
                                                     SchedulePolicySuccessorFunctor* callback )
{
   TrackableObject::DependableObjectList &readersList = status.getReaders();
   SyncLockBlock lock4( status.getReadersLock() );
   for ( TrackableObject::DependableObjectList::iterator i = readersList.begin(); i != readersList.end(); i++) {
      DependableObject * predecessorReader = *i;
      SyncLockBlock lock5(predecessorReader->getLock());

      // new instrument event: dependence predecessorReader -> depObj
      NANOS_INSTRUMENT ( WorkDescriptor *wd_sender = (WorkDescriptor *) predecessorReader->getRelatedObject(); )
      NANOS_INSTRUMENT ( WorkDescriptor *wd_receiver = (WorkDescriptor *) depObj.getRelatedObject(); )
      NANOS_INSTRUMENT ( if ( wd_sender && wd_receiver ) { )
         NANOS_INSTRUMENT ( nanos_event_value_t Values[3]; )
         NANOS_INSTRUMENT ( Values[0] = ( ((nanos_event_value_t) wd_sender->getId()) << 32 ) + wd_receiver->getId(); )
         NANOS_INSTRUMENT ( Values[1] = ((nanos_event_value_t) 1); )
         NANOS_INSTRUMENT ( Values[2] = ((nanos_event_value_t) target.getAddress() ); )
         NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(3, _insKeyDeps, Values); )
      NANOS_INSTRUMENT ( } )

      if ( predecessorReader->addSuccessor( depObj ) ) {
         // new dependence predecessorReader -> depObj
         depObj.increasePredecessors();
         if ( callback != NULL ) {
            ( *callback )( predecessorReader, &depObj );
         }
      }
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

inline void BaseDependenciesDomain::dependOnReadersAndSetAsWriter( DependableObject &depObj, TrackableObject &status, BaseDependency const &target,
                                                                   SchedulePolicySuccessorFunctor* callback )
{
   dependOnReaders( depObj, status, target, callback );
   setAsWriter( depObj, status, target );
}

inline void BaseDependenciesDomain::addAsReader( DependableObject &depObj, TrackableObject &status )
{
   SyncLockBlock lock3( status.getReadersLock() );
   status.setReader( depObj );
}

inline CommutationDO * BaseDependenciesDomain::createCommutationDO( BaseDependency const &target, AccessType const &accessType, TrackableObject &status )
{
   CommutationDO *commDO = new CommutationDO( target, accessType.commutative );
   commDO->setDependenciesDomain( this );
   commDO->increasePredecessors();
   status.setCommDO( commDO );
   commDO->addWriteTarget( target );
   return commDO;
}

inline CommutationDO * BaseDependenciesDomain::setUpTargetCommutationDependableObject ( BaseDependency const &target, AccessType const &accessType, TrackableObject &status )
{
   CommutationDO *commDO = status.getCommDO();

   /* FIXME: this must be atomic */

   if ( commDO == NULL ) {
      commDO = createCommutationDO( target, accessType, status );
   }
   else if ( commDO->isCommutative() != accessType.commutative ) {
      // Make sure that old CommutationDO is of same type, or finalize it and create a new
      finalizeReduction( status, target );
      commDO = createCommutationDO( target, accessType, status );
   }
   else if ( commDO->increasePredecessors() == 0 ) {
      commDO = createCommutationDO( target, accessType, status );
   }

   return commDO;
}


inline CommutationDO * BaseDependenciesDomain::setUpInitialCommutationDependableObject ( BaseDependency const &target, AccessType const &accessType, TrackableObject &status )
{
   CommutationDO *initialCommDO = NULL;

   if ( status.hasReaders() ) {
      initialCommDO = new CommutationDO( target, accessType.commutative );
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


inline void BaseDependenciesDomain::submitDependableObjectCommutativeDataAccess ( DependableObject &depObj,
   BaseDependency const &target, AccessType const &accessType, TrackableObject &status,
   SchedulePolicySuccessorFunctor* callback )
{
   // NOTE: Do not change the order
   CommutationDO *initialCommDO = setUpInitialCommutationDependableObject( target, accessType, status );
   CommutationDO *commDO = setUpTargetCommutationDependableObject( target, accessType, status );

   // Add the Commutation object as successor of the current DO (depObj)
   depObj.addSuccessor( *commDO );

   // assumes no new readers added concurrently
   dependOnLastWriter( depObj, status, target, callback );

   // The dummy predecessor is to make sure that initialCommDO does not execute 'finished'
   // while depObj is being added as its successor
   if ( initialCommDO != NULL ) {
      initialCommDO->decreasePredecessors();
   }
}

inline void BaseDependenciesDomain::submitDependableObjectInoutDataAccess ( DependableObject &depObj,
   BaseDependency const &target, AccessType const &accessType, TrackableObject &status,
   SchedulePolicySuccessorFunctor* callback )
{
   finalizeReduction( status, target );
   dependOnLastWriter( depObj, status, target, callback );
   dependOnReadersAndSetAsWriter( depObj, status, target, callback );
}

inline void BaseDependenciesDomain::submitDependableObjectInputDataAccess ( DependableObject &depObj,
   BaseDependency const &target, AccessType const &accessType, TrackableObject &status,
   SchedulePolicySuccessorFunctor* callback )
{
   finalizeReduction( status, target );
   dependOnLastWriter( depObj, status, target, callback );

   if ( !depObj.waits() ) {
      addAsReader( depObj, status );
   }
}

inline void BaseDependenciesDomain::submitDependableObjectOutputDataAccess ( DependableObject &depObj,
   BaseDependency const &target, AccessType const &accessType, TrackableObject &status,
   SchedulePolicySuccessorFunctor* callback )
{
   finalizeReduction( status, target );

   // assumes no new readers added concurrently
   if ( !status.hasReaders() ) {
      dependOnLastWriter( depObj, status, target, callback );
   }

   dependOnReadersAndSetAsWriter( depObj, status, target, callback );
}

}

#endif

