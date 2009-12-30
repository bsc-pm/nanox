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
#include "debug.hpp"
#include "system.hpp"
#include <iostream>

using namespace nanos;

Atomic<int> DependenciesDomain::_atomicSeed( 0 );

TrackableObject* DependenciesDomain::lookupDependency ( const Dependency& dep )
{
   TrackableObject *trackableObject = NULL;
   
   DepsMap::iterator it = _addressDependencyMap.find(*(dep.getAddress())); 
   if ( it == _addressDependencyMap.end() ) {
      trackableObject = new TrackableObject(*dep.getAddress());
      _addressDependencyMap.insert( std::make_pair(*(dep.getAddress()), trackableObject) );
   } else {
      trackableObject = it->second;
   }
   
   return trackableObject;
}

template<typename iterator>
void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, iterator begin, iterator end )
{
   depObj.setId ( _lastDepObjId++ );

   depObj.init();
   
   // Object is not ready to get its dependencies satisfied
   // so we increase the number of predecessor to permit other dependableObjects to free some of
   // its dependencies without triggerinf the "dependenciesSatisfied" method
   depObj.increasePredecessors();

   std::list<Dependency *> filteredDeps;
   for ( iterator it = begin; it != end; it++ ) {
      Dependency* newDep = &(*it);
      bool found = false;
      for ( std::list<Dependency *>::iterator current = filteredDeps.begin(); current != filteredDeps.end(); current++ ) {
         Dependency* currentDep = *current;
         if ( *(newDep->getAddress()) == *(currentDep->getAddress()) ) {
            // Both dependencies use the same address, put them in common
            currentDep->setInput( newDep->isInput() || currentDep->isInput() );
            currentDep->setOutput( newDep->isOutput() || currentDep->isOutput() );
            found = true;
            break;
         }
      }
      if ( !found ) {
         filteredDeps.push_back(newDep);
      }
   }

   for ( std::list<Dependency *>::iterator it = filteredDeps.begin(); it != filteredDeps.end(); it++ ) {
      Dependency &dep = *(*it);

      // TODO for renaming, remember that the trackableObject keeps the address for the dependency but not
      // where this address is stored for the DependableObject that will use it. This last address is not the same
      // for all DependableObjects so it needs to be stored somewhere accessible when the dependableObject will use
      // the storage to change it if renaming happened.
      TrackableObject * dependencyObject = lookupDependency( dep );

      // assumes no new readers added concurrently
      if ( dep.isInput() || (dep.isOutput() && !(dependencyObject->hasReaders()) ) ) {
         DependableObject *lastWriter = dependencyObject->getLastWriter();
         
         if ( lastWriter != NULL ) {
            lastWriter->lock();
            if ( dependencyObject->getLastWriter() == lastWriter ) {
               depObj.increasePredecessors();
               lastWriter->addSuccessor( depObj );
               if ( ( !(dep.isOutput()) || dep.isInput() ) ) {
                  // RaW dependency
                  debug (" DO_ID_" << lastWriter->getId() << "->" << "DO_ID_" << depObj.getId() << "[color=green];");
               } else {
                  // WaW dependency
                  debug (" DO_ID_" << lastWriter->getId() << "->" << "DO_ID_" << depObj.getId() << "[color=blue];");
               }
            }
            lastWriter->unlock();
         }
      }

      // only for non-inout dependencies
      if ( dep.isInput() && !( dep.isOutput() ) && !( depObj.waits() ) ) {
         depObj.addReadObject( dependencyObject );

         dependencyObject->lockReaders();
         dependencyObject->setReader( depObj );
         dependencyObject->unlockReaders();
      }

      if ( dep.isOutput() ) {
         // add dependencies to all previous reads
         TrackableObject::DependableObjectList &readersList = dependencyObject->getReaders();
         dependencyObject->lockReaders();

         for ( TrackableObject::DependableObjectList::iterator it = readersList.begin(); it != readersList.end(); it++) {
            DependableObject * predecessorReader = *it;
            predecessorReader->addSuccessor( depObj );
            depObj.increasePredecessors();
            // WaR dependency
            debug (" DO_ID_" << predecessorReader->getId() << "->" << "DO_ID_" << depObj.getId() << "[color=red];");
         }
         dependencyObject->flushReaders();

         dependencyObject->unlockReaders();
         
         if ( !depObj.waits() ) {
            // set depObj as writer of dependencyObject
            depObj.addOutputObject( dependencyObject );
            dependencyObject->setLastWriter( depObj );
         }
      }

   }

   if ( sys.getVerbose() ) return;

   // now everything is ready
   depObj.decreasePredecessors();

   depObj.wait();
}

template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, Dependency* begin, Dependency* end );
template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, std::vector<Dependency>::iterator begin, std::vector<Dependency>::iterator end );

void DependenciesDomain::finished ( DependableObject &depObj )
{

   if ( sys.getVerbose() ) return;

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
      
   DependableObject::DependableObjectVector &succ = depObj.getSuccessors();
   for ( unsigned int i = 0; i < succ.size(); i++ ) {
      succ[i]->decreasePredecessors();
   }
}

