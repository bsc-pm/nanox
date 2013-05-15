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

#ifndef _NANOS_LIB_WDDEQUE
#define _NANOS_LIB_WDDEQUE

#include "wddeque_decl.hpp"
#include "schedule.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "atomic.hpp"

using namespace nanos;

inline bool WDDeque::empty ( void ) const
{
   return _dq.empty();
}
inline size_t WDDeque::size() const
{
   return _nelems;
}

inline void WDDeque::push_front ( WorkDescriptor *wd )
{
   wd->setMyQueue( this );
   {
      LockBlock lock( _lock );
      _dq.push_front( wd );
      int tasks = ++( sys.getSchedulerStats()._readyTasks );
      increaseTasksInQueues(tasks);
      memoryFence();
   }
}

inline void WDDeque::push_back ( WorkDescriptor *wd )
{
   wd->setMyQueue( this );
   {
      LockBlock lock( _lock );
      _dq.push_back( wd );
      int tasks = ++( sys.getSchedulerStats()._readyTasks );
      increaseTasksInQueues(tasks);
      memoryFence();
   }
}

struct NoConstraints
{
#ifdef CLUSTER_DEV
   //static inline bool check ( WD &wd, BaseThread &thread ) { return ( !thread.isCluster() || wd.getDepth() == 1 ) ; }
   //static inline bool check ( WD &wd, BaseThread &thread ) { return (sys.getNetwork()->getNodeNum() > 0) || ( !thread.isCluster() && !( wd.getDepth() == 1  && !(thread.getId() == 0) )) || ( thread.isCluster() && wd.getDepth() == 1) ; }
   static inline bool check ( WD &wd, BaseThread const &thread ) { return true; }
#else
   static inline bool check ( WD &wd, BaseThread const &thread ) { return true; }
#endif
};

inline WorkDescriptor * WDDeque::pop_front ( BaseThread *thread )
{
  return popFrontWithConstraints<NoConstraints>(thread);
}

inline WorkDescriptor * WDDeque::pop_back ( BaseThread *thread )
{
  return popBackWithConstraints<NoConstraints>(thread);
}

inline bool WDDeque::removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next )
{
  return removeWDWithConstraints<NoConstraints>(thread,toRem,next);
}

template <typename Constraints>
inline WorkDescriptor * WDDeque::popFrontWithConstraints ( BaseThread const *thread )
{
   WorkDescriptor *found = NULL;

   if ( _dq.empty() )
      return NULL;

   {
      LockBlock lock( _lock );

      memoryFence();

      if ( !_dq.empty() ) {
         WDDeque::BaseContainer::iterator it;

         for ( it = _dq.begin() ; it != _dq.end(); it++ ) {
            WD &wd = *(WD *)*it;
            if ( Scheduler::checkBasicConstraints( wd, *thread) && Constraints::check(wd,*thread) ) {
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


// Only ensures tie semantics
template <typename Constraints>
inline WorkDescriptor * WDDeque::popBackWithConstraints ( BaseThread const *thread )
{
   WorkDescriptor *found = NULL;

   if ( _dq.empty() )
      return NULL;

   {
      LockBlock lock( _lock );

      memoryFence();

      if ( !_dq.empty() ) {
         WDDeque::BaseContainer::reverse_iterator rit;

         for ( rit = _dq.rbegin(); rit != _dq.rend() ; rit++ ) {
            WD &wd = *(WD *)*rit;
            if ( Scheduler::checkBasicConstraints( wd, *thread) && Constraints::check(wd,*thread)) {
               if ( wd.dequeue( &found ) ) {
                  _dq.erase( ( ++rit ).base() );
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
inline bool WDDeque::removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next )
{
   if ( _dq.empty() ) return false;

   if ( !Scheduler::checkBasicConstraints( *toRem, *thread) || !Constraints::check(*toRem, *thread) ) return false;

   *next = NULL;
   WDDeque::BaseContainer::iterator it;

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

inline void WDDeque::increaseTasksInQueues( int tasks )
{
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("num-ready");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) tasks );)
   _nelems++;
}

inline void WDDeque::decreaseTasksInQueues( int tasks )
{
   NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("num-ready");)
   NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvent( key, (nanos_event_value_t) tasks );)
   _nelems--;
}

inline void WDDeque::transferElemsFrom( WDDeque &dq )
{
   LockBlock lock( _lock );
   memoryFence();
   WDDeque::BaseContainer::iterator it;

   while( !dq._dq.empty() )
   {
      _dq.push_back( dq._dq.front() );
      dq._dq.pop_front();
   }
   _nelems = dq._nelems;
   dq._nelems = 0;
   memoryFence();
}

/***********
 * WDDeque *
 ***********/

inline bool WDLFQueue::empty ( void ) const
{
   volatile WDNode *head;
   volatile WDNode *next;

   while (1) {
      head = _head;
      next = NANOS_ABA_PTR(head)->next();
      if ( head == _head ) {
         if ( NANOS_ABA_PTR(_head) == NANOS_ABA_PTR(_tail) ) {
            if ( NANOS_ABA_PTR(next) == NULL ) return true;
            else return false;
         } else {
            return false;
         }
      }
   }
}

inline size_t WDLFQueue::size() const
{
   fatal0("Calling size method is not allowed using WDLFQueue's"); /*XXX*/
}

inline void WDLFQueue::push_front ( WorkDescriptor *wd )
{
   fatal0("Calling push_front method is not allowed using WDLFQueue's"); /*XXX*/
}

inline void WDLFQueue::push_back( WorkDescriptor *wd )
{
   volatile WDNode *tail;
   volatile WDNode *next;
   WDNode *node = new WDNode( wd, NULL);

   while (1) {
      tail = _tail;
      next = (WDNode *) NANOS_ABA_PTR(tail)->next_value();
      if ( tail == _tail ) {
         if ( NANOS_ABA_PTR(next) == NULL ) {
            if ( compareAndSwap( (void **) NANOS_ABA_PTR(tail)->next_addr(), (void *) next, (void *) NANOS_ABA_COMPOSE(node, next)) ){
               break;
            }
         } else {
            compareAndSwap ( (void **) &_tail, (void *) tail, (void *) NANOS_ABA_COMPOSE(next, tail) );
         }
      }
   }
   /* int tasks = */ ++( sys.getSchedulerStats()._readyTasks );
   compareAndSwap( (void **) &_tail, (void *) tail, (void *) NANOS_ABA_COMPOSE(node,tail) );
}

inline void WDLFQueue::push_back_node ( WDNode *node )
{
   volatile WDNode *tail;
   volatile WDNode *next;
   node->resetNext();                                                        // Set next pointer of node to NULL

   while (1) {                                                               // Keep trying until push is done
      tail = _tail;                                                          // Reading _tail (ptr & ctr)
      next = (WDNode *) NANOS_ABA_PTR(tail)->next_value();                   // Read next ( ptr & ctr)
      if ( tail == _tail ) {                                                 // Are tail and next consistents?
         if ( NANOS_ABA_PTR(next) == NULL ) {                                // Was _tail pointing to the last node?
            if ( compareAndSwap( (void **) NANOS_ABA_PTR(tail)->next_addr(),
                                 (void *) next,
                                 (void *) NANOS_ABA_COMPOSE(node, next)) ){  // Try to link node at the end of the linked list
               break;                                                        // Push is already done
            }
         } else {                                                            // _tail is not pointing to the last node
            compareAndSwap ( (void **) &_tail,
                             (void *) tail,
                             (void *) NANOS_ABA_COMPOSE(next, tail) );       // Try to swing _tail to the next node
         }
      }
   }
   /* int tasks = */ ++( sys.getSchedulerStats()._readyTasks );
   compareAndSwap( (void **) &_tail,
                   (void *) tail,
                   (void *) NANOS_ABA_COMPOSE(node,tail) );                  // Push is already done. Try to swing _tail to the
                                                                             // inserted node
}

inline WorkDescriptor * WDLFQueue::pop_front ( BaseThread *thread )
{
   WorkDescriptor *wd  = NULL;
   WorkDescriptor *swd = NULL;
   volatile WDNode *head;
   volatile WDNode *tail;
   volatile WDNode *next;

   while (1){
      head = _head;                                                         // Read head ( ptr & ctr)
      tail = _tail;                                                         // Read tail ( ptr & ctr)
      next = (volatile WDNode *) NANOS_ABA_PTR(head)->next();               // Read head_ptr->next
      if ( head == _head) {                                                 // Are head, tail and next consistents?
         if ( NANOS_ABA_PTR(head) == NANOS_ABA_PTR(tail) ) {                // Is queue empty or tail falling behind?
            if ( next == NULL) return NULL;                                 // Is queue empty? return NULL
            compareAndSwap( (void **) &_tail,
                            (void *) tail,
                            (void *) NANOS_ABA_COMPOSE(next,tail) );        // Tail is falling behind. Try to advance
         } else if ( next != NULL ) {                                       // extra check, may be not necessary
            wd = next->wd();                                                // read value before CAS, otherwise next can be freed
            if ( compareAndSwap( (void **) &_head,
                                 (void *) head,
                                 (void *) NANOS_ABA_COMPOSE(next,head)) ) { // Try to swing _head to next node
              /*int tasks =*/ --(sys.getSchedulerStats()._readyTasks);
              //decreaseTasksInQueues(tasks);
              if ( Scheduler::checkBasicConstraints( *wd, *thread) /* && Constraints::check(wd,*thread) FIXME*/ ) {
                 if ( !wd->dequeue( &swd ) ) {
                    NANOS_ABA_PTR(head)->setWD(wd);
                    push_back_node( (WDNode *) NANOS_ABA_PTR(head) );
                 }
                 else delete NANOS_ABA_PTR(head);
                 break;
              } else {
                 NANOS_ABA_PTR(head)->setWD(wd);
                 push_back_node( (WDNode *) NANOS_ABA_PTR(head) );
              }
            }
         }
      }
   }
   return swd;
}

inline WorkDescriptor * WDLFQueue::pop_back ( BaseThread *thread )
{
   fatal0("Calling pop_back method is not allowed using WDLFQueue's"); /*XXX*/
}

inline bool WDLFQueue::removeWD( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next )
{
   fatal0("Calling removeWD method is not allowed using WDLFQueue's");
   return false;
}

#if 0
         template <typename Constraints>
         WorkDescriptor * popFrontWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         WorkDescriptor * popBackWithConstraints ( BaseThread *thread );
         template <typename Constraints>
         bool removeWDWithConstraints( BaseThread *thread, WorkDescriptor *toRem, WorkDescriptor **next );
#endif

inline WDPriorityQueue::WDPriorityQueue( bool optimise, bool fifo ) : _dq(), _lock(), _optimise( optimise ), _fifo( fifo )
{
}

inline bool WDPriorityQueue::empty ( void ) const
{
   return _dq.empty();
}
inline size_t WDPriorityQueue::size() const
{
   return _nelems;
}

inline void WDPriorityQueue::insertOrdered( WorkDescriptor *wd )
{
   // Find where to insert the wd
   BaseContainer::iterator it;
   
   if ( _fifo ) {
      // #637: Insert at the back if possible
      if ( ( _optimise ) && ( ( _dq.empty() ) || ( _dq.back()->getPriority() >= wd->getPriority() ) ) )
         it = _dq.end();
      else
         it = std::upper_bound( _dq.begin(), _dq.end(), wd, WDPriorityComparison() );
   }
   else {
      // #637: Insert at the front if possible
      if ( ( _optimise ) && ( ( _dq.empty() ) || ( _dq.front()->getPriority() < wd->getPriority() ) ) )
         it = _dq.begin();
      else {
         it = std::lower_bound( _dq.begin(), _dq.end(), wd, WDPriorityComparison() );
      }
   }
   
   _dq.insert( it, wd );
}

inline void WDPriorityQueue::push ( WorkDescriptor *wd )
{
   wd->setMyQueue( this );
   {
      LockBlock lock( _lock );
      insertOrdered( wd );
      int tasks = ++( sys.getSchedulerStats()._readyTasks );
      increaseTasksInQueues(tasks);
      memoryFence();
   }
}

/*!
 * \brief This method is not supported.
 * \see WDPriorityQueue::push
 */
inline void WDPriorityQueue::push_back ( WorkDescriptor *wd )
{
   fatal( "Method not implemented" );
}

/*!
 * \brief This method is not supported.
 * \see WDPriorityQueue::push
 */
inline void WDPriorityQueue::push_front ( WorkDescriptor *wd )
{
   fatal( "Method not implemented" );
}


inline WorkDescriptor * WDPriorityQueue::pop ( BaseThread *thread )
{
   return popWithConstraints<NoConstraints>(thread);
}

/*!
 * \see WDPriority::pop()
 * TODO (gmiranda): Discuss if pop_back and _front are useful.
 */
inline WorkDescriptor * WDPriorityQueue::pop_back ( BaseThread *thread )
{
   fatal( "Method not implemented" );
}

/*!
 * \see WDPriority::pop()
 */
inline WorkDescriptor * WDPriorityQueue::pop_front ( BaseThread *thread )
{
   fatal( "Method not implemented" );
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
         for ( it = _dq.begin(); it != _dq.end() ; ++it ) {
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

inline bool WDPriorityQueue::reorderWD( WorkDescriptor *wd )
{
   LockBlock l( _lock );
   
   // Find the WD
   BaseContainer::iterator it =
      find( _dq.begin(), _dq.end(), wd );

   // If the WD was not found, return false
   if( it == _dq.end() ){
      return false;
   }

   // Otherwise, reorder it
   _dq.erase( it );
   insertOrdered( wd );

   return true;
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

