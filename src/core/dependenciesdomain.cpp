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

using namespace nanos;

/*! \brief Looks for the dependency's address in the domain and returns the trackableObject associated.
 *  \param dep Dependency to be checked.
 *  \sa Dependency TrackableObject
 */
TrackableObject* DependenciesDomain::lookupDependency ( const Dependency& dep )
{
   TrackableObject *trackableObject = NULL;
   
   DepsMap::iterator it = _addressDependencyMap.find(*(dep.getAddress())); 
   if ( it == _addressDependencyMap.end() ) {
      trackableObject = new TrackableObject(dep.getAddress());
      _addressDependencyMap.insert( std::make_pair(*(dep.getAddress()), trackableObject) );
   } else {
      trackableObject = it->second;
   }
   
   return trackableObject;
}

/*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
 *  \param depObj DependableObject to be added to the domain.
 *  \param begin Iterator to the start of the list of dependencies to be associated to the Dependable Object.
 *  \param end Iterator to the end of the mentioned list.
 *  \sa Dependency DependableObject TrackableObject
 */
template<typename iterator>
void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, iterator begin, iterator end )
{
   depObj.setId ( _lastDepObjId++ );
   
   // Object is not ready to get its dependencies satisfied
   // so we increase the number of predecessor to permit other dependableObjects to free some of
   // its dependencies without triggerinf the "dependenciesSatisfied" method
   depObj.increasePredecessors();
   
   for ( iterator it = begin; it != end; it++ ) {
      const Dependency &dep = *it;
      TrackableObject * dependencyObject = lookupDependency( dep );

      if ( dep.isInput() ) {
         DependableObject *lastWriter = dependencyObject->getLastWriter();
         
         if ( lastWriter != NULL ) {
            lastWriter->lock();
            if ( dependencyObject->getLastWriter() == lastWriter ) {
               depObj.increasePredecessors();
               lastWriter->addSuccessor( depObj );
            }
            lastWriter->unlock();
         }
      }

      // only for non-inout dependencies
      if ( dep.isInput() && !( dep.isOutput() ) ) {
         depObj.addReadObject( dependencyObject );
         dependencyObject->setReader( depObj );
      }

      if ( dep.isOutput() ) {
         // add dependencies to all previous reads

         TrackableObject::DependableObjectList &readersList = dependencyObject->getReaders();
         dependencyObject->lockReaders();

         for ( TrackableObject::DependableObjectList::iterator it = readersList.begin(); it != readersList.end(); it++) {
            DependableObject * predecessorReader = *it;
            predecessorReader->addSuccessor( depObj );
            depObj.increasePredecessors();
         }
         dependencyObject->flushReaders();

         dependencyObject->unlockReaders();
         
         // set depObj as writer of dependencyObject
         depObj.addOutputObject( dependencyObject );
         dependencyObject->setLastWriter( depObj );
      }
   }
   
   // now everything is ready
   depObj.decreasePredecessors();
   
}

template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, Dependency* begin, Dependency* end );
template void DependenciesDomain::submitDependableObjectInternal ( DependableObject &depObj, const std::vector<Dependency>::const_iterator begin, const std::vector<Dependency>::const_iterator end );

/*! \brief Dependable Object depObj is finished and its outgoing dependencies are removed.
 *  \param desObj Dependable Object that finished
 *  \sa DependableObject
 */
void DependenciesDomain::finished ( DependableObject &depObj )
{
   //! This step guarantees that any Object that wants to add depObj as a successor has done it
   //! before we continue or, alternatively, won't do it.
   depObj.lock();
   DependableObject::TrackableObjectVector &outs = depObj.getOutputObjects();
   for ( unsigned int i = 0; i < outs.size(); i++ ) {
      outs[i]->deleteLastWriter(depObj);
   }
   depObj.unlock();
   
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
