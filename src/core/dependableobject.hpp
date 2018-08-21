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

#ifndef _NANOS_DEPENDABLE_OBJECT
#define _NANOS_DEPENDABLE_OBJECT
#include <stdlib.h>
#include <list>
#include <set>
#include <vector>

#include "atomic.hpp"
#include "lock.hpp"

#include "dependableobject_decl.hpp"
#include "basedependency_decl.hpp"
#include "workdescriptor_decl.hpp"
#include "system_decl.hpp"

#include "dataaccess.hpp"
#include "functors.hpp"

namespace nanos {

inline DependableObject::~DependableObject ( )
{
   {
      SyncLockBlock lock( this->getLock() );
      for ( DependableObjectVector::iterator it = _predecessors.begin(); it != _predecessors.end(); it++ ) {
         it->second->deleteSuccessor( *this );
      }
   }

   std::for_each(_outputObjects.begin(),_outputObjects.end(),deleter<BaseDependency>);
   std::for_each(_readObjects.begin(),_readObjects.end(),deleter<BaseDependency>);
}

inline DependableObject::DependableObject ( const DependableObject &depObj )
   : _id(), _numPredecessors(), _references(), _predecessors(), _successors(), _domain(),
   _outputObjects(), _readObjects(), _objectLock(), _submitted( false ),
   _needsSubmission( false ), _wd(), _schedulerData( NULL ), _num(), _lss()
{
   LockBlock lock( depObj._objectLock );
   _id = depObj._id;
   _numPredecessors = depObj._numPredecessors;
   _references = depObj._references;
   _predecessors = depObj._predecessors;
   _successors = depObj._successors;
   _domain = depObj._domain;
   _wd = depObj._wd;
   _num = depObj._num;
   _lss = depObj._lss;
}

inline const DependableObject & DependableObject::operator= ( const DependableObject &depObj )
{
   if ( this == &depObj ) return *this;

   DoubleLockBlock lock( _objectLock, depObj._objectLock );
   _id = depObj._id;
   _numPredecessors = depObj._numPredecessors;
   _references = depObj._references;
   _predecessors = depObj._predecessors;
   _successors = depObj._successors;
   _domain = depObj._domain;
   _outputObjects = depObj._outputObjects;
   _submitted = depObj._submitted;
   _needsSubmission = depObj._needsSubmission;
   _wd = depObj._wd;
   _num = depObj._num;
   return *this;
}

inline bool DependableObject::waits ( )
{
   return false;
}

inline unsigned long DependableObject::getDescription ( )
{
   return 0;
}

inline void * DependableObject::getRelatedObject ( )
{
   return NULL;
}

inline const void * DependableObject::getRelatedObject ( ) const
{
   return NULL;
}

inline void DependableObject::setId ( unsigned int id )
{
   _id = id;
}

inline unsigned int DependableObject::getId () const
{
   return _id;
}

inline int DependableObject::increasePredecessors ( )
{
   return _numPredecessors++;
}

inline int DependableObject::decreasePredecessors ( std::list<uint64_t> const * flushDeps,
      DependableObject * finishedPred, bool batchRelease, bool blocking )
{
   int numPred;

   if ( sys.getPredecessorLists() ) {
      LockBlock lock( _objectLock );
      numPred = --_numPredecessors;
      decreasePredecessorsInLock( finishedPred, numPred );
   } else {
      numPred = --_numPredecessors;
   }

   if ( numPred == 0 && !batchRelease ) {
      dependenciesSatisfied( );
   }

   return numPred;
}

inline void DependableObject::decreasePredecessorsInLock ( DependableObject * finishedPred,
       int numPred )
{
   if ( finishedPred != NULL ) {
      if ( getWD() != NULL && finishedPred->getWD() != NULL ) {
         getWD()->predecessorFinished( finishedPred->getWD() );
      }

      //remove the predecessor from the list!
      if ( _predecessors.size() != 0 ) {
         unsigned int wdId = finishedPred->getWD() == NULL ? 0 : finishedPred->getWD()->getId();
         DependableObjectVector::iterator it = _predecessors.find( std::make_pair( wdId, finishedPred ) );
         if ( it != _predecessors.end() )
            _predecessors.erase( it );
      }
   }

   if ( numPred == 0 && !_predecessors.empty() ) {
      _predecessors.clear();
   }
}

inline int DependableObject::numPredecessors () const
{
   return _numPredecessors.value();
}

inline DependableObject::DependableObjectVector & DependableObject::getPredecessors ( )
{
   return _predecessors;
}

inline DependableObject::DependableObjectVector & DependableObject::getSuccessors ( )
{
   return _successors;
}

inline bool DependableObject::addPredecessor ( DependableObject &depObj )
{
   // Avoiding create cycles in dependence graph
   if ( this == &depObj ) return false;

   bool inserted = _predecessors.insert (
         std::make_pair( depObj.getWD() == NULL ? 0 : depObj.getWD()->getId(), &depObj)
         ).second;

   return inserted;
}

inline bool DependableObject::addSuccessor ( DependableObject &depObj )
{
   // Avoiding create cycles in dependence graph
   if ( this == &depObj ) return false;

   if ( sys._preSchedule ) {
      if (depObj._num < _num + 1) {
         depObj._num = _num + 1;

         if ( sys.getPredecessorLists() ) {
            SyncLockBlock lock( depObj._objectLock );
            for ( DependableObjectVector::const_iterator it = depObj._predecessors.begin();
                  it != depObj._predecessors.end(); it++ ) {
               int value = (it->second->_lss == -1 ) ? depObj._num - 1 : (it->second->_lss < depObj._num - 1 ? depObj._num - 1 : it->second->_lss );
               it->second->_lss = value;
            }
         }
      }
      if ( _lss == -1 ) {
         _lss = depObj._num - 1;
      } else if ( depObj._num < _lss ) {
         _lss = depObj._num - 1;
      }
   }

   //Maintain the list of predecessors
   if ( sys.getPredecessorLists() ) {
      SyncLockBlock lock( depObj._objectLock );
      depObj.addPredecessor( *this );
   }

   sys.getDefaultSchedulePolicy()->atSuccessor( depObj, *this );

   return _successors.insert ( std::make_pair( depObj.getWD() == NULL ? 0 : depObj.getWD()->getId(), &depObj ) ).second;
}

inline bool DependableObject::deleteSuccessor ( DependableObject *depObj )
{
   return _successors.erase( std::make_pair( depObj->getWD() == NULL ? 0 : depObj->getWD()->getId(), depObj ) ) > 0;
}

inline bool DependableObject::deleteSuccessor ( DependableObject &depObj )
{
   return deleteSuccessor( &depObj );
}

inline DependenciesDomain * DependableObject::getDependenciesDomain ( ) const
{
   return _domain;
}

inline void DependableObject::setDependenciesDomain ( DependenciesDomain *dependenciesDomain )
{
   _domain = dependenciesDomain;
}

inline void DependableObject::addWriteTarget ( BaseDependency const &outObj )
{
   _outputObjects.push_back ( outObj.clone() );
}

inline DependableObject::TargetVector const & DependableObject::getWrittenTargets ( )
{
   return _outputObjects;
}

inline void DependableObject::addReadTarget ( BaseDependency const &readObj )
{
   _readObjects.push_back( readObj.clone() );
}

inline DependableObject::TargetVector const & DependableObject::getReadTargets ( )
{
   return _readObjects;
}

inline void DependableObject::increaseReferences()
{
   _references++;
}

inline void DependableObject::resetReferences()
{
   _references = 1;
}


inline bool DependableObject::isSubmitted()
{
   return
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
      __atomic_load_n(&_submitted, __ATOMIC_ACQUIRE)
#else
      _submitted
#endif
      ;
}

inline void DependableObject::submitted()
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   __atomic_store_n(&_submitted, true, __ATOMIC_RELEASE);
#else
   _submitted = true;
#endif
   enableSubmission();
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
#else
   memoryFence();
#endif
}

inline bool DependableObject::needsSubmission() const
{
   return _needsSubmission;
}

inline void DependableObject::enableSubmission()
{
   _needsSubmission = true;
}

inline void DependableObject::disableSubmission()
{
   _needsSubmission = false;
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   __atomic_store_n(&_submitted, false, __ATOMIC_RELEASE);
#else
   _submitted = false;
   memoryFence();
#endif
}

inline Lock& DependableObject::getLock()
{
   return _objectLock;
}

inline void DependableObject::setWD( WorkDescriptor *wd )
{
   _wd = wd;
}

inline WorkDescriptor * DependableObject::getWD( void ) const
{
   return _wd;
}

inline DOSchedulerData* DependableObject::getSchedulerData ( )
{
   return _schedulerData;
}

inline void DependableObject::setSchedulerData ( DOSchedulerData* scData)
{
        _schedulerData = scData;
}

} // namespace nanos

#endif
