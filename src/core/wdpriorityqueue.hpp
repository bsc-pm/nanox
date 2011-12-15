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

#ifndef _NANOS_LIB_WDPRIORITYQUEUE
#define _NANOS_LIB_WDPRIORITYQUEUE

#include "wdpriorityqueue_decl.hpp"
#include "schedule.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "atomic.hpp"
#include "wddeque.hpp"

using namespace nanos;

inline bool WDPriorityQueue::empty ( void ) const
{
   return _dq.empty();
}
inline size_t WDPriorityQueue::size() const
{
   return _nelems;
}

inline void WDPriorityQueue::push ( WorkDescriptor *wd )
{
   wd->setMyQueue( this );
   {
      LockBlock lock( _lock );
      //_dq.push( wd );
      // Find where to insert the wd
      BaseContainer::iterator it = std::upper_bound( _dq.begin(), _dq.end(), wd, WDPriorityComparison() );
      _dq.insert( it, wd );
      int tasks = ++( sys.getSchedulerStats()._readyTasks );
      increaseTasksInQueues(tasks);
      memoryFence();
   }
}

inline WorkDescriptor * WDPriorityQueue::pop ( BaseThread *thread )
{
  return popWithConstraints<NoConstraints>(thread);
}


inline bool WDPriorityQueue::removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next )
{
  return removeWDWithConstraints<NoConstraints>(thread,toRem,next);
}

// Only ensures tie semantics
template <typename Constraints>
inline WorkDescriptor * WDPriorityQueue::popWithConstraints ( BaseThread *thread )
{
   WorkDescriptor *found = NULL;

   if ( _dq.empty() )
      return NULL;

   {
      LockBlock lock( _lock );

      memoryFence();
   
      if ( !_dq.empty() ) {
         WDPriorityQueue::BaseContainer::iterator it;
   
         for ( it = _dq.begin(); it != _dq.end() ; it++ ) {
            WD &wd = *(WD *)*it; 
            if ( Scheduler::checkBasicConstraints( wd, *thread) && Constraints::check(wd,*thread)) {
               if ( wd.dequeue( &found ) ) {
                  _dq.erase( it );
                  int tasks = --(sys.getSchedulerStats()._readyTasks);
                  decreaseTasksInQueues(tasks);
               }
               break;
            }
         }
      }
   
      if ( found != NULL ) found->setMyQueue( NULL );
   
   }

   ensure( !found || !found->isTied() || found->isTiedTo() == thread, "" );

   return found;
}


template <typename Constraints>
inline bool WDPriorityQueue::removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next )
{
   if ( _dq.empty() ) return false;

   if ( !Scheduler::checkBasicConstraints( *toRem, *thread) || !Constraints::check(*toRem, *thread) ) return false;

   *next = NULL;
   WDPriorityQueue::BaseContainer::iterator it;

   {
      LockBlock lock( _lock );

      memoryFence();

      if ( !_dq.empty() && toRem->getMyQueue() == this ) {
         for ( it = _dq.begin(); it != _dq.end(); it++ ) {
            if ( *it == toRem ) {
               if ( ( *it )->dequeue( next ) ) {
                  _dq.erase( it );
                  int tasks = --(sys.getSchedulerStats()._readyTasks);
                  decreaseTasksInQueues(tasks);
               }
               (*next)->setMyQueue( NULL );
               return true;
            }
         }
      }
   }

   return false;
}

inline void WDPriorityQueue::increaseTasksInQueues( int tasks )
{
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("num-ready");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) tasks );)
   _nelems++;
}

inline void WDPriorityQueue::decreaseTasksInQueues( int tasks )
{
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("num-ready");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) tasks );)
   _nelems--;
}

#endif

