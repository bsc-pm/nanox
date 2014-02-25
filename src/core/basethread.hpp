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
   
   inline bool TeamData::isStarring ( void ) const { return _star; }

   inline void TeamData::setStar ( bool v ) { _star = v; }

   inline void TeamData::setCreator ( bool value ) { _creator = value; }

   inline bool TeamData::isCreator ( void ) const { return _creator; }

   inline nanos_ws_desc_t *TeamData::getTeamWorkSharingDescriptor( BaseThread *thread, bool *b )
   {
      nanos_ws_desc_t *next = NULL, *myNext = NULL;

      *b = false;

      // If current WorkDescriptor in not implicit, 
      if ( thread->getCurrentWD()->isImplicit() == false ) {
         if ( _team == NULL ) fatal("Asking for team WorkSharing with no associated team.");
         next = NEW nanos_ws_desc_t();
         *b = true;
         return next;
      }

      if ( _wsDescriptor ) {
         // Having a previous _wsDescriptor
         if ( _wsDescriptor->next ) {
            next = _wsDescriptor->next;
         } else {
            myNext = NEW nanos_ws_desc_t();
            myNext->ws = NULL;
            if ( compareAndSwap( &(_wsDescriptor->next), (nanos_ws_desc_t *) NULL, myNext) ) {
               next = myNext;
               *b = true;
            } else {
               next = _wsDescriptor->next;
               delete myNext;
            }
         }
      } else if ( _team ) {
         // With no previous _wsDescriptor but having a team
         nanos_ws_desc_t **teamNext = _team->getWorkSharingDescriptorAddr();
         if ( *teamNext ) {
            next = *teamNext;
         } else {
            myNext = NEW nanos_ws_desc_t();
            myNext->ws = NULL;
            if ( compareAndSwap( teamNext, (nanos_ws_desc_t *) NULL, myNext) )
            {
               next = myNext;
               *b = true;
            } else {
               next = *teamNext;
               delete myNext;
            }
         }
      } else {
         // With no previous _wsDescriptor neither team
         // next = NEW nanos_ws_desc_t();
         fatal("Asking for team WorkSharing with no associated team.");
      }

      _wsDescriptor = next;

      return next;
   }

   inline BaseThread::BaseThread ( WD &wd, ProcessingElement *creator ) :
      _id( sys.nextThreadId() ), _maxPrefetch( 1 ), _status( ), _pe( creator ), _mlock( ),
      _threadWD( wd ), _currentWD( NULL), _nextWDs( ),
      _teamData( NULL ), _nextTeamData( NULL ),
      _name( "Thread" ), _description( "" ), _allocator( ) { }

   inline bool BaseThread::isMainThread ( void ) const { return _status.is_main_thread; }

   inline void BaseThread::setMainThread ( bool v ) { _status.is_main_thread = v; }

   inline void BaseThread::joined ( void ) { _status.has_joined = true; }

   // atomic access
   inline void BaseThread::lock () { _mlock++; }
 
   inline void BaseThread::unlock () { _mlock--; }
 
   inline void BaseThread::stop() { _status.must_stop = true; }

   inline void BaseThread::sleep() { _status.must_sleep = true; }

   inline void BaseThread::wakeup() { _status.must_sleep = false; }
   
   inline void BaseThread::pause ()
   {
      // If the thread was already paused, do nothing
      if ( _status.is_paused ) return;

      // Otherwise, notify this change
      _status.is_paused = true;
      sys.pausedThread();
   }
   
   inline void BaseThread::unpause ()
   {
      // If the thread was already unpaused, do nothing
      if ( !_status.is_paused ) return;

      // Otherwise, notify this change
      _status.is_paused = false;
      sys.unpausedThread();
   }
 
   // set/get methods
   inline void BaseThread::setCurrentWD ( WD &current ) { _currentWD = &current; }
 
   inline WD * BaseThread::getCurrentWD () const { return _currentWD; }
 
   inline WD & BaseThread::getThreadWD () const { return _threadWD; }
 
   inline int BaseThread::getMaxPrefetch () const { return ( int ) _maxPrefetch; }

   inline void BaseThread::setMaxPrefetch ( int max ) { _maxPrefetch = (unsigned short) max; }

   inline bool BaseThread::canPrefetch () const { return _nextWDs.size() < _maxPrefetch; }

   inline bool BaseThread::hasNextWD () const { return !_nextWDs.empty(); }
 
   // team related methods
   inline void BaseThread::reserve() { _status.has_team = true; }
 
   inline void BaseThread::enterTeam( TeamData *data )
   { 
      lock();
      if ( data != NULL ) _teamData = data;
      else _teamData = _nextTeamData;
      _status.has_team = true;
      unlock();
   }
 
   inline bool BaseThread::hasTeam() const { return _status.has_team; }
 
   inline void BaseThread::leaveTeam()
   {
      ensure( this == myThread, "thread is not leaving team by itself" );
      if ( _teamData ) 
      {
         TeamData *td = _teamData;
         debug( "removing thread " << this << " with id " << toString<int>(getTeamId()) << " from " << _teamData->getTeam() );

         size_t final_size = td->getTeam()->removeThread( getTeamId() );
         if ( final_size == td->getTeam()->getFinalSize() ) td->getTeam()->setStable(true);
         _teamData = _teamData->getParentTeamData();
         _status.has_team = _teamData != NULL;
         delete td;
      }
   }

   inline ThreadTeam * BaseThread::getTeam() const { return _teamData ? _teamData->getTeam() : NULL; }
 
   inline TeamData * BaseThread::getTeamData() const { return _teamData; }

   inline void BaseThread::setNextTeamData( TeamData * td) { _nextTeamData = td; }

   inline nanos_ws_desc_t *BaseThread::getLocalWorkSharingDescriptor( void ) { return &_wsDescriptor; }

   inline nanos_ws_desc_t *BaseThread::getTeamWorkSharingDescriptor( bool *b )
   {
      if ( _teamData ) return _teamData->getTeamWorkSharingDescriptor ( this,  b );
      else return NULL;
   }
 
   //! Returns the id of the thread inside its current team 
   inline int BaseThread::getTeamId() const { return _teamData->getId(); }
 
   inline bool BaseThread::isStarted () const { return _status.has_started; }
 
   inline bool BaseThread::isRunning () const { return _status.has_started && !_status.must_stop; }

   inline bool BaseThread::isSleeping () const { return _status.must_sleep; }
   
   inline bool BaseThread::isTeamCreator () const { return _teamData->isCreator(); } 

   inline void BaseThread::wait ( void ) { _status.is_waiting = true; }

   inline void BaseThread::resume ( void ) {_status.is_waiting = false; }

   inline bool BaseThread::isWaiting () const { return _status.is_waiting; }

   inline bool BaseThread::isPaused () const { return _status.is_paused; }
 
   inline ProcessingElement * BaseThread::runningOn() const { return _pe; }
   
   inline void BaseThread::setRunningOn(ProcessingElement* element) { _pe=element; }
 
   inline int BaseThread::getId() const { return _id; }
 
   inline int BaseThread::getCpuId() const { return _pe->getId(); }
 
   inline bool BaseThread::isStarring ( const ThreadTeam *t ) const
   {
      if ( _teamData && t == _teamData->getTeam() ) return _teamData->isStarring();
      else if ( _nextTeamData && t == _nextTeamData->getTeam() ) return _nextTeamData->isStarring();
      return false;
   }

   inline void BaseThread::setStar ( bool v ) { if ( _teamData ) _teamData->setStar ( v ); }

   inline Allocator & BaseThread::getAllocator() { return _allocator; }

   inline void BaseThread::rename ( const char *name ) { _name = name; }
 
   inline const std::string & BaseThread::getName ( void ) const { return _name; }
 
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
