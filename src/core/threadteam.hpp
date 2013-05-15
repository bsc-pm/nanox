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

#ifndef _NANOS_THREAD_TEAM_H
#define _NANOS_THREAD_TEAM_H
#include "threadteam_decl.hpp"
#include "atomic.hpp"
#include "debug.hpp"
#include "system.hpp"

using namespace nanos;

inline ThreadTeam::ThreadTeam ( int maxThreads, SchedulePolicy &policy, ScheduleTeamData *data,
                                Barrier &barrierImpl, ThreadTeamData & ttd, ThreadTeam * parent )
                              : _size(0), _starSize(0), _idleThreads( 0 ), _numTasks( 0 ), _barrier(barrierImpl),
                                _singleGuardCount( 0 ), _schedulePolicy( policy ),
                                _scheduleData( data ), _threadTeamData( ttd ), _parent( parent ),
                                _level( parent == NULL ? 0 : parent->getLevel() + 1 ), _creatorId(-1),
                                _wsDescriptor(NULL), _redList()
{
      _threads = NEW BaseThread *[maxThreads];
}

inline ThreadTeam::~ThreadTeam ()
{
   ensure(size() == 0, "Destroying non-empty team!");
   delete[] _threads;
   delete &_barrier;
   delete _scheduleData;
   delete &_threadTeamData;
}

inline unsigned ThreadTeam::size() const
{
   return _size.value();
}

inline void ThreadTeam::init ()
{
   _barrier.init( size() );
   _threadTeamData.init( _parent );
}

inline void ThreadTeam::resized ()
{
   // TODO
   _barrier.resize(size());
}

inline BaseThread & ThreadTeam::getThread ( int i ) const
{
   return *_threads[i];
}

inline BaseThread & ThreadTeam::getThread ( int i )
{
   return *_threads[i];
}

inline const BaseThread & ThreadTeam::operator[]  ( int i ) const
{
   return getThread(i);
}

inline BaseThread & ThreadTeam::operator[]  ( int i )
{
   return getThread(i);
}

inline unsigned ThreadTeam::addThread ( BaseThread *thread, bool star, bool creator )
{
   unsigned id = _size++;
   _threads[id] =  thread;
   if ( star ) _starSize++;
   if ( creator ) {
      _creatorId = (int) id;
   }
   return id;
}

inline void ThreadTeam::removeThread ( unsigned id )
{
   _threads[id] = 0;
   _size--;
}

inline nanos_ws_desc_t  *ThreadTeam::getWorkSharingDescriptor( void ) { return _wsDescriptor; }

inline nanos_ws_desc_t **ThreadTeam::getWorkSharingDescriptorAddr( void ) { return &_wsDescriptor; }

inline void ThreadTeam::barrier()
{
   _barrier.barrier( myThread->getTeamId() );
}

inline ScheduleTeamData * ThreadTeam::getScheduleData() const
{
   return _scheduleData;
}

inline SchedulePolicy & ThreadTeam::getSchedulePolicy() const
{
   return _schedulePolicy;
}

inline ThreadTeamData & ThreadTeam::getThreadTeamData() const
{
   return _threadTeamData;
}

inline ThreadTeam * ThreadTeam::getParent() const
{
   return _parent;
}

inline int ThreadTeam::getLevel() const
{
   return _level;
}

inline int ThreadTeam::getCreatorId() const
{
   return _creatorId;
}

inline unsigned ThreadTeam::getNumStarringThreads( void ) const
{
   return _starSize.value();
}

inline unsigned ThreadTeam::getStarringThreads( BaseThread **list_of_threads ) const
{
   unsigned i,nThreadsQuery = 0;
   for ( i = 0; i < _size.value(); i++ ) {
      if ( _threads[i]->isStarring( this ) ) {
         list_of_threads[nThreadsQuery++] = _threads[i];
      }
   }
   return nThreadsQuery;
}

inline unsigned ThreadTeam::getNumSupportingThreads( void ) const
{
   return _size.value() - _starSize.value();
}

inline unsigned ThreadTeam::getSupportingThreads( BaseThread **list_of_threads ) const
{
   unsigned i,nThreadsQuery = 0;
   for ( i = 0; i < _size.value(); i++ ) {
      if ( !_threads[i]->isStarring( this ) ) {
         list_of_threads[nThreadsQuery++] = _threads[i];
      }
   }
   return nThreadsQuery;
}

inline void ThreadTeam::createReduction( nanos_reduction_t *red ) { _redList.push_front( red ); }

inline void ThreadTeam::computeVectorReductions ( void )
{
   nanos_reduction_t *red;
   ReductionList::iterator it;
   for ( it = _redList.begin(); it != _redList.end(); it++) {
      red = *it;
      if ( red->vop ) {
         red->vop( this->size(), red->original, red->privates );
      } else {
         unsigned i;
         for ( i = 0; i < this->size(); i++ ) {
         }
      }
   }
}

inline void *ThreadTeam::getReductionPrivateData ( void* s )
{
   ReductionList::iterator it;
   for ( it = _redList.begin(); it != _redList.end(); it++) {
      if ((*it)->original == s) return (*it)->privates;
   }
   return NULL;
}

#endif
