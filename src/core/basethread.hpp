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

#ifndef _BASE_THREAD_ELEMENT
#define _BASE_THREAD_ELEMENT

#include "workdescriptor_fwd.hpp"
#include "atomic.hpp"
#include "processingelement.hpp"
#include "debug.hpp"
#include "schedule_fwd.hpp"
#include "threadteam_fwd.hpp"
#include "basethread_decl.hpp"
#include "atomic.hpp"
#include "system.hpp"

namespace nanos
{
   inline TeamData::~TeamData ()
   {
      delete _schedData;
   }
   
   // atomic access
   inline void BaseThread::lock () { _mlock++; }
 
   inline void BaseThread::unlock () { _mlock--; }
 
   inline void BaseThread::stop() { _mustStop = true; }
 
   // set/get methods
   inline void BaseThread::setCurrentWD ( WD &current ) { _currentWD = &current; }
 
   inline WD * BaseThread::getCurrentWD () const { return _currentWD; }
 
   inline WD & BaseThread::getThreadWD () const { return _threadWD; }
 
   inline void BaseThread::resetNextWD () { _nextWD = NULL; }
 
   inline bool BaseThread::setNextWD ( WD *next ) { 
      debug("Set next WD as: " << next << ":??" << " @ thread " << _id );
      return compareAndSwap( &_nextWD, (WD *) NULL, next);
   }
 
   inline bool BaseThread::reserveNextWD ( void ) { 
      return compareAndSwap( &_nextWD, (WD *) NULL, (WD *) 1);
   }
 
   inline bool BaseThread::setReservedNextWD ( WD *next ) { 
      debug("Set next WD as: " << next << ":??" << " @ thread " << _id );
      return compareAndSwap( &_nextWD, (WD *) 1, next);
   }
 
   inline WD * BaseThread::getNextWD () const
   { 
      /* First copy value to avoid race conditions */
      WD * retWD = _nextWD;
      if ( retWD == (WD *) 1 ) return NULL;
      return retWD;
   }
 
   // team related methods
   inline void BaseThread::reserve() { _hasTeam = 1; }
 
   inline void BaseThread::enterTeam( TeamData *data )
   { 
      if ( data != NULL ) _teamData = data;
      else _teamData = _nextTeamData;
      _hasTeam=1;
   }
 
   inline bool BaseThread::hasTeam() const { return _hasTeam; }
 
   inline void BaseThread::leaveTeam()
   {
      if ( _teamData ) 
      {
         TeamData *td = _teamData;
         _teamData = _teamData->getParentTeamData();
         _hasTeam = _teamData != NULL;
         delete td;
      }
   }
 
   inline ThreadTeam * BaseThread::getTeam() const { return _teamData ? _teamData->getTeam() : NULL; }
 
   inline TeamData * BaseThread::getTeamData() const { return _teamData; }

   inline void BaseThread::setNextTeamData( TeamData * td) { _nextTeamData = td; }

 
   //! Returns the id of the thread inside its current team 
   inline int BaseThread::getTeamId() const { return _teamData->getId(); }
 
   inline bool BaseThread::isStarted () const { return _started; }
 
   inline bool BaseThread::isRunning () const { return _started && !_mustStop; }
 
   inline ProcessingElement * BaseThread::runningOn() const { return _pe; }
 
   inline int BaseThread::getId() { return _id; }
 
   inline int BaseThread::getCpuId() { return runningOn()->getId(); }
 
   inline Allocator & BaseThread::getAllocator() { return _allocator; }
   /*! \brief Rename the basethread
   */
   inline void BaseThread::rename ( const char *name )
   {
     _name = name;
   }
 
   /*! \brief Get BaseThread name
   */
   inline const std::string & BaseThread::getName ( void ) const
   {
      return _name;
   }
 
   /*! \brief Get BaseThread description
   */
   inline const std::string & BaseThread::getDescription ( void ) 
   {
     if ( _description.empty() ) {
 
        /* description name */
        _description = getName();
        _description.append("-");
 
        /* adding device type */
        _description.append( _pe->getDeviceType().getName() );
        _description.append("-");
 
        /* adding global id */
        _description.append( toString<int>(getId()) );
     }
 
     return _description;
   }

}

#endif
