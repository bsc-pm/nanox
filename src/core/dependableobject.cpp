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

#include "dependableobject.hpp"
#include "trackableobject.hpp"
#include "instrumentation.hpp"
#include "system.hpp"

using namespace nanos;

void DependableObject::finished ( )
{
   if ( --_references == 0) {
      DependableObject& depObj = *this;
      // This step guarantees that any Object that wants to add depObj as a successor has done it
      // before we continue or, alternatively, won't do it.
      DependableObject::TrackableObjectVector &outs = depObj.getOutputObjects();
      if (outs.size() > 0) {
         depObj.lock();
         for ( unsigned int i = 0; i < outs.size(); i++ ) {
            outs[i]->deleteLastWriter(depObj);
         }
         depObj.unlock();
      }
      
      //  Delete depObj from all trackableObjects it reads 
      DependableObject::TrackableObjectVector &reads = depObj.getReadObjects();
      for ( DependableObject::TrackableObjectVector::iterator it = reads.begin(); it != reads.end(); it++ ) {
         TrackableObject* readObject = *it;
         readObject->lockReaders();
         readObject->deleteReader(depObj);
         readObject->unlockReaders();
      }

      NANOS_INSTRUMENT ( void * predObj = getRelatedObject(); )

      DependableObject::DependableObjectVector &succ = depObj.getSuccessors();
      for ( DependableObject::DependableObjectVector::iterator it = succ.begin(); it != succ.end(); it++ ) {

         NANOS_INSTRUMENT ( void * succObj = (*it)->getRelatedObject(); )
         NANOS_INSTRUMENT ( instrument ( predObj, succObj ); ) 

         (*it)->decreasePredecessors();
      }
   }
}

DependableObject * DependableObject::releaseImmediateSuccessor ( void )
{
   DependableObject * found = NULL;

   DependableObject::DependableObjectVector &succ = getSuccessors();
   for ( DependableObject::DependableObjectVector::iterator it = succ.begin(); it != succ.end(); it++ ) {
      // Is this an immediate successor? 
      if ( (*it)->numPredecessors() == 1 ) {
        // remove it
        found = *it;
        this->lock();
        succ.erase(it);
        this->unlock();
        if ( found->numPredecessors() != 1 ) {
           this->lock();
           succ.insert( found );
           this->unlock();
        } else {
           break;
        }
      }
   }

   return found;
}
