
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

#include "dependableobject.hpp"
#include "regionstatus.hpp"
#include "instrumentation.hpp"
#include "system.hpp"

using namespace nanos;

void DependableObject::finished ( )
{
   if ( --_references == 0) {
      DependableObject& depObj = *this;
      // This step guarantees that any Object that wants to add depObj as a successor has done it
      // before we continue or, alternatively, won't do it.
//<<<<<<< HEAD
//      DependableObject::TrackableObjectVector &outs = depObj.getOutputObjects();
//      if ( !outs.empty() ) {
//         {
//            SyncLockBlock lock( depObj.getLock() );
//            for ( unsigned int i = 0; i < outs.size(); i++ ) {
//               outs[i]->deleteLastWriter(depObj);
//            }
//=======
      DependableObject::RegionContainer const &outs = depObj.getWrittenRegions();
      DependenciesDomain *domain = depObj.getDependenciesDomain();
      if ( domain != 0 && outs.size() > 0 ) {
         SyncRecursiveLockBlock lock1( domain->getInstanceLock() ); // This is needed here to avoid a dead-lock
         SyncLockBlock lock2( depObj.getLock() );
         for ( unsigned int i = 0; i < outs.size(); i++ ) {
            Region const &region = outs[i];
            
            domain->deleteLastWriter ( depObj, region );
//>>>>>>> regions
         }
      }
      
      //  Delete depObj from all trackableObjects it reads 
      if ( domain != 0 ) {
         DependableObject::RegionContainer const &reads = depObj.getReadRegions();
         for ( DependableObject::RegionContainer::const_iterator it = reads.begin(); it != reads.end(); it++ ) {
            Region const & region = *it;
            
            domain->deleteReader ( depObj, region );
         }
      }

      DependableObject::DependableObjectVector &succ = depObj.getSuccessors();
      for ( DependableObject::DependableObjectVector::iterator it = succ.begin(); it != succ.end(); it++ ) {

         NANOS_INSTRUMENT ( instrument ( *(*it) ); ) 

         (*it)->decreasePredecessors( this );
      }
   }
}

DependableObject * DependableObject::releaseImmediateSuccessor ( DependableObjectPredicate &condition )
{
   DependableObject * found = NULL;

   DependableObject::DependableObjectVector &succ = getSuccessors();
   DependableObject::DependableObjectVector incorrectlyErased;

   {
      SyncLockBlock lock( this->getLock() );
      // NOTE: it gets incremented in the erase
      for ( DependableObject::DependableObjectVector::iterator it = succ.begin(); it != succ.end(); ) {
         // Is this an immediate successor? 
         if ( (*it)->numPredecessors() == 1 && condition(**it) && !((*it)->waits()) ) {
            // remove it
            found = *it;
            if ((*it)->isSubmitted()) {
               succ.erase(it++);
               if ( found->numPredecessors() != 1 ) {
                  incorrectlyErased.insert( found );
                  found = NULL;
               } else {
                  NANOS_INSTRUMENT ( instrument ( *found ); )
                  DependenciesDomain::decreaseTasksInGraph();
                  break;
               }
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
