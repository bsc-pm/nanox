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

#ifndef _NANOS_THREAD_TEAM_H
#define _NANOS_THREAD_TEAM_H
#include "threadteam_decl.hpp"
#include "atomic.hpp"
#include "lock.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "task_reduction.hpp"

namespace nanos {

inline ThreadTeam::ThreadTeam ( int maxThreads, SchedulePolicy &policy, ScheduleTeamData *data,
                                Barrier &barrierImpl, ThreadTeamData & ttd, ThreadTeam * parent )
                              : _threads(), _idList(), _expectedThreads(), _starSize(0), _idleThreads( 0 ),
                                _numTasks( 0 ), _barrier(barrierImpl),
                                _singleGuardCount( 0 ), _schedulePolicy( policy ),
                                _scheduleData( data ), _threadTeamData( ttd ), _parent( parent ),
                                _level( parent == NULL ? 0 : parent->getLevel() + 1 ), _creatorId(-1),
                                _wsDescriptor(NULL), _redList(), _lock()
{ }

inline ThreadTeam::~ThreadTeam ()
{
   ensure(size() == 0, "Destroying non-empty team!");
   delete &_barrier;
   delete _scheduleData;
   delete &_threadTeamData;
}

inline void ThreadTeam::lock()
{
   _lock.acquire();
}

inline void ThreadTeam::unlock()
{
   _lock.release();
}

inline unsigned ThreadTeam::size() const
{
   return _threads.size();
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

inline const BaseThread & ThreadTeam::getThread ( int i ) const
{
   // Return the i-th valid element in _threads
   int j = 0;
   ThreadTeamList::const_iterator it = _threads.begin();
   for ( it = _threads.begin(); it != _threads.end(); ++it, ++j ) {
      if ( i == j ) {
         return *(it->second);
      }
   }

   // If we didn't returned during the loop, return last thread
   ThreadTeamList::const_reverse_iterator last = _threads.rbegin();
   return *(last->second);
}

inline BaseThread & ThreadTeam::getThread ( int i )
{
   // Return the i-th valid element in _threads
   int j = 0;
   ThreadTeamList::iterator it = _threads.begin();
   for ( it = _threads.begin(); it != _threads.end(); ++it, ++j ) {
      if ( i == j ) {
         return *(it->second);
      }
   }

   // If we didn't returned during the loop, return last thread
   ThreadTeamList::reverse_iterator last = _threads.rbegin();
   return *(last->second);
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
   unsigned id;
   {
      LockBlock Lock( _lock );
      for ( id = 0; id < _idList.size(); id++) if ( _idList[id] == false ) break;
      _threads[id] = thread;
      _idList[id] = true;
      _expectedThreads.insert( thread );
      _barrier.resize( _expectedThreads.size() );
   }
   if ( star ) _starSize++;
   if ( creator ) {
      _creatorId = (int) id;
   }
   return id;
}

inline size_t ThreadTeam::removeThread ( unsigned id )
{
   LockBlock Lock( _lock );
   _threads.erase( id );
   _idList[id] = false;
   return ( _threads.size() );
}

inline BaseThread * ThreadTeam::popThread ( )
{
   BaseThread * thread;
   {
      // \todo It will be better to use _threads/_idList[] idiom
      LockBlock Lock( _lock );
      ThreadTeamList::iterator last = _threads.end();
      ThreadTeamIdList::iterator lastId = _idList.end();
      --last;
      --lastId;
      thread = last->second;
      lastId->second = false;
      _threads.erase( last );
   }
   return thread;
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
   unsigned nThreadsQuery = 0;
   ThreadTeamList::const_iterator it;
   BaseThread *thread;

   for ( it = _threads.begin(); it != _threads.end(); it++ ) {
      thread = it->second;
      if ( thread->isStarring( this ) ) {
         list_of_threads[nThreadsQuery++] = thread;
      }
   }
   return nThreadsQuery;
}

inline unsigned ThreadTeam::getNumSupportingThreads( void ) const
{
   return size() - _starSize.value();
}

inline unsigned ThreadTeam::getSupportingThreads( BaseThread **list_of_threads ) const
{
   unsigned nThreadsQuery = 0;
   ThreadTeamList::const_iterator it;
   BaseThread *thread;

   for ( it = _threads.begin(); it != _threads.end(); it++ ) {
      thread = it->second;
      if ( !thread->isStarring( this ) ) {
         list_of_threads[nThreadsQuery++] = thread;
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
         char *privates = reinterpret_cast<char*>(red->privates);
         for ( i = 0; i < this->size(); i++ ) {
             char* current = privates + i * red->element_size;
             red->bop(red->original, current, red->num_scalars);
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

inline nanos_reduction_t *ThreadTeam::getReduction ( void* s )
{
   ReductionList::iterator it;
   for ( it = _redList.begin(); it != _redList.end(); it++) {
      if ((*it)->original == s) return *it;
   }

   return NULL;
}

inline size_t ThreadTeam::getFinalSize ( void ) const { return _expectedThreads.size(); }

inline void ThreadTeam::addExpectedThread( BaseThread *thread )
{
   LockBlock Lock( _lock );
   _expectedThreads.insert( thread );
   _barrier.resize( _expectedThreads.size() );
}

inline void ThreadTeam::removeExpectedThread( BaseThread *thread )
{
   LockBlock Lock( _lock );
   _expectedThreads.erase( thread );
   _barrier.resize( _expectedThreads.size() );
}

} // namespace nanos

#endif
