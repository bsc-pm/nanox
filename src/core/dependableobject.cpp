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

#include "dependableobject.hpp"
#include "instrumentation.hpp"
#include "system.hpp"
#include "basethread.hpp"
#include <alloca.h>

using namespace nanos;

void DependableObject::finished ( )
{
   if ( --_references == 0) {
      DependableObject& depObj = *this;

      // This step guarantees that any Object that wants to add depObj as a successor has done it
      // before we continue or, alternatively, won't do it.
      TargetVector const &outs = depObj.getWrittenTargets();
      DependenciesDomain *domain = depObj.getDependenciesDomain();
      if ( domain != 0 && outs.size() > 0 ) {
         SyncRecursiveLockBlock lock1( domain->getInstanceLock() ); // This is needed here to avoid a dead-lock
         SyncLockBlock lock2( depObj.getLock() );
         for ( unsigned int i = 0; i < outs.size(); i++ ) {
            BaseDependency const &target = *outs[i];
            
            domain->deleteLastWriter ( depObj, target );
         }
      }
      
      //  Delete depObj from all trackableObjects it reads 
      if ( domain != 0 ) {
         DependableObject::TargetVector const &reads = depObj.getReadTargets();
         for ( DependableObject::TargetVector::const_iterator it = reads.begin(); it != reads.end(); it++ ) {
            BaseDependency const & target = *(*it);
            
            domain->deleteReader ( depObj, target );
         }
      }

      DependableObject::DependableObjectVector &succ = depObj.getSuccessors();

      //*(myThread->_file) << "Successors for wd " << this->getWD()->getId() << " : " << ( this->getWD()->getDescription() != NULL ? this->getWD()->getDescription() : "[no description]" ) << " { " ;
      //for ( DependableObject::DependableObjectVector::iterator it = succ.begin(); it != succ.end(); it++ ) {
      //   WD *wd = (*it)->getWD();
      //   if ( wd != NULL ) {
      //      *(myThread->_file) << "[" << wd->getId() << " : "<< ( wd->getDescription() != NULL ? wd->getDescription() : "[no description]" ) << " ]";
      //   } else {
      //      *(myThread->_file) << "[null succ]";
      //   }
      //}
      //*(myThread->_file) << " }" << std::endl;

      // See if it's worth batch releasing.
      // The idea here is to prevent initialising the vector unless we will
      // use it.
      if ( succ.size() > 1 )
      {
         // Construct list of successors to be released immediately.
         // Allocate as many elements as we have in the successor list
         WD** immediateSucc = (WD**) alloca( sizeof(WD*) * succ.size() );
         WD** pIS = immediateSucc;
         
         for ( DependableObject::DependableObjectVector::iterator it = succ.begin(); it != succ.end(); it++ ) {
            NANOS_INSTRUMENT ( instrument ( *(it->second) ); ) 
            // If this dependable object can't be released in batch
            if ( !it->second->canBeBatchReleased() )
            {
               it->second->decreasePredecessors( NULL, this, false, false );
               continue;
            }
            
            // Release this Dependable Object without triggering submission
            DependableObject& dSucc = *it->second;
            // Decrease predecessors
            int numPred = dSucc.decreasePredecessors( NULL, this, true, false );
            
            // If after decreasing the predecessors it's not 0, fatal_cond
            fatal_cond( numPred != 0, "Num predecessors is not 0" );
            
            // dependenciesSatisfied code
            dSucc.dependenciesSatisfiedNoSubmit();
            
            // Convert to WD*
            WD* wd = (WD*) dSucc.getRelatedObject();
            fatal_cond( wd == NULL, "Cannot cast the related object to WD" );

            if ( this->getWD() != NULL ) {
               wd->predecessorFinished( this->getWD() );
            }
            
            *pIS++ = wd ;
         }

         size_t numImmediate = pIS - immediateSucc;
      
         // Batch submit and counter decrement
         if ( numImmediate > 0 ){
            DependenciesDomain::decreaseTasksInGraph( numImmediate );
            Scheduler::submit( immediateSucc, numImmediate );
         }
      }
      else 
      {
         for ( DependableObject::DependableObjectVector::iterator it = succ.begin(); it != succ.end(); it++ ) {
            NANOS_INSTRUMENT ( instrument ( *it->second ); )
            it->second->decreasePredecessors( NULL, this, false, false );
         }
      }
   }
}

void DependableObject::releaseReadDependencies ()
{
   DependableObject& depObj = *this;
   DependableObject::DependableObjectVector &succ = depObj.getSuccessors();

   // This step guarantees that any Object that wants to add depObj as a successor has done it
   // before we continue or, alternatively, won't do it.
   DependenciesDomain *domain = depObj.getDependenciesDomain();
   DependableObject::TargetVector const &reads = depObj.getReadTargets();
   DependableObject::TargetVector const &writes = depObj.getWrittenTargets();
   //  Delete depObj from all trackableObjects it reads 
   if ( domain != 0 ) {
      for ( DependableObject::TargetVector::const_iterator it = reads.begin(); it != reads.end(); it++ ) {
         BaseDependency const & target = *(*it);
         //if ( target.getAddress() == addr ) {    
            domain->deleteReader ( depObj, target );
         //}
      }
   }

   //Decrease predecessor for sucessor tasks
   //Only decrease if they are NOT writing or reading something that we write
   for ( DependableObject::DependableObjectVector::iterator currSucessorIt = succ.begin(); currSucessorIt != succ.end(); ) {
      DependableObject::TargetVector const &sucessorWrites = currSucessorIt->second->getWrittenTargets();
      DependableObject::TargetVector const &sucessorReads = currSucessorIt->second->getReadTargets();
      bool canRemovePredecessor=true;
      for ( DependableObject::TargetVector::const_iterator itCurrWrites = writes.begin(); itCurrWrites != writes.end() && canRemovePredecessor; itCurrWrites++ ) {
         BaseDependency const & currWrite = *(*itCurrWrites);
         for ( DependableObject::TargetVector::const_iterator it = sucessorWrites.begin(); it != sucessorWrites.end() && canRemovePredecessor; it++ ) {
            BaseDependency const & succWrite = *(*it);            
            //If address is the "released" or dependency is different, we can remove predecessor
            canRemovePredecessor &= ( !currWrite.overlap(succWrite) );
         }

         for ( DependableObject::TargetVector::const_iterator it = sucessorReads.begin(); it != sucessorReads.end() && canRemovePredecessor; it++ ) {
            BaseDependency const & succRead = *(*it);            
            //If address is the "released" or dependency is different, we can remove predecessor
            canRemovePredecessor &= ( !currWrite.overlap(succRead) );
         }
      }
      if (canRemovePredecessor) {
         //DependenciesDomain::decreaseTasksInGraph();
         NANOS_INSTRUMENT ( instrument ( *currSucessorIt->second ); ) 
         currSucessorIt->second->decreasePredecessors( NULL, this, false, false );
         succ.erase(currSucessorIt++);
      }
      else 
      {
         currSucessorIt++;
      }
   }
}


bool DependableObject::canBeBatchReleased ( ) const
{
   return false;
}

DependableObject * DependableObject::releaseImmediateSuccessor ( DependableObjectPredicate &condition, bool keepDeps )
{
   DependableObject * found = NULL;

   DependableObject::DependableObjectVector &succ = getSuccessors();
   DependableObject::DependableObjectVector incorrectlyErased;

   {
      SyncLockBlock lock( this->getLock() );
      // NOTE: it gets incremented in the erase
      for ( DependableObject::DependableObjectVector::iterator it = succ.begin(); it != succ.end(); ) {
         // Is this an immediate successor? 
         if ( it->second->numPredecessors() == 1 && condition(*it->second) && !(it->second->waits()) ) {
            if (it->second->isSubmitted()) {
               // remove it
               found = it->second;
               unsigned int wdId = it->first;
               succ.erase(it++);
               if ( found->numPredecessors() != 1 ) {
                  incorrectlyErased.insert( std::make_pair( wdId, found ) );
                  found = NULL;
               } else {
                  NANOS_INSTRUMENT ( instrument ( *found ); )

                  DependenciesDomain::decreaseTasksInGraph();

                  if ( found->getWD() != NULL ) {
                     found->getWD()->predecessorFinished( this->getWD() );
                  }
                  if ( keepDeps ) {
                     // This means that the WD related to this DO does not need to be submitted,
                     // because someone else will do it
                     // Keep the dependency to signal when the WD can actually be run respecting dependencies
                     found->disableSubmission();
                     succ.insert( std::make_pair( wdId, found ) );
                  } else {
                     // We have removed the successor, so we need to decrease its predecessors
                     found->decreasePredecessors( NULL, this, true, false );
                  }

                  //*(myThread->_file) << "Immediate successor for wd " << this->getWD()->getId() << " : " <<
                  //   ( this->getWD()->getDescription() != NULL ? this->getWD()->getDescription() : "[no description]" ) <<
                  //   " { " << found->getWD()->getId() << 
                  //   " : " << ( found->getWD()->getDescription() != NULL ? found->getWD()->getDescription() : "[no description]" ) <<
                  //   " }" << std::endl;

                  break;
               }
            } else {
               it++;
            }
         } else {
            it++;
         }
      }
      for ( DependableObject::DependableObjectVector::iterator it = incorrectlyErased.begin(); it != incorrectlyErased.end(); it++) {
         succ.insert(*it);
      }
   }
   return found;
}
