/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#include "threadteam.hpp"
#include "atomic.hpp"
#include "debug.hpp"
#include "allocator.hpp"
#include "memtracker.hpp"

using namespace nanos;


bool ThreadTeam::singleGuard( int local )
{
   if ( local <= _singleGuardCount ) return false;
   
   return compareAndSwap( &_singleGuardCount, local-1, local );
}

bool ThreadTeam::enterSingleBarrierGuard( int local )
{
   if ( local <= _singleGuardCount ) return false;
   
   return compareAndSwap( &_singleGuardCount, local-2, local-1 );

}

void ThreadTeam::releaseSingleBarrierGuard( void )
{
   _singleGuardCount++;
}

void ThreadTeam::waitSingleBarrierGuard( int local )
{
   while ( local > _singleGuardCount ) { memoryFence(); }
}

void ThreadTeam::cleanUpReductionList( void ) 
{
   nanos_reduction_t *red;
   while ( !_redList.empty() ) {
      red = _redList.back();
      red->cleanup( red->descriptor ); 
      if ( red->descriptor != red->privates ) 
      { 
#if defined(NANOS_DEBUG_ENABLED) && defined(NANOS_MEMTRACKER_ENABLED)
         nanos::getMemTracker().deallocate( red->descriptor ); 
#elif !defined(NANOS_DISABLE_ALLOCATOR)
         nanos::getAllocator().deallocate ( red->descriptor ); 
#else
         free( red->descriptor );
#endif
      }
#if defined(NANOS_DEBUG_ENABLED) && defined(NANOS_MEMTRACKER_ENABLED)
      nanos::getMemTracker().deallocate( (void *) red );
#elif !defined(NANOS_DISABLE_ALLOCATOR)
      nanos::getAllocator().deallocate ( (void *) red );
#else
      free ( (void*) red );
#endif
      _redList.pop_back();
   }
}

void ThreadTeam::registerTaskReduction( void *p_orig, void *p_dep, size_t p_size, void (*p_init)( void *, void *), void (*p_reducer)( void *, void * ), void (*p_reducer_orig_var)( void *, void * ) )
{
   LockBlock Lock( _lockTaskReductions );

   //! Check if orig is already registered
   task_reduction_list_t::iterator it;
   for ( it = _taskReductions.begin(); it != _taskReductions.end(); it++) {
      if ( (*it)->have( p_orig, 0 ) ) break;
   }

   NANOS_ARCHITECTURE_PADDING_SIZE(p_size);
   if ( it == _taskReductions.end() ) {
      _taskReductions.push_front( new TaskReduction( p_orig, p_dep, p_init, p_reducer, p_reducer_orig_var, p_size, _finalSize.value(), myThread->getCurrentWD()->getDepth() ) );
   }
}

void ThreadTeam::removeTaskReduction( void *p_orig )
{
   LockBlock Lock( _lockTaskReductions );

   //! Check if orig is already registered
   task_reduction_list_t::iterator it;
   for ( it = _taskReductions.begin(); it != _taskReductions.end(); it++) {
      if ( (*it)->have( p_orig, 0 ) ) break;
   }

   if ( it != _taskReductions.end() ) _taskReductions.erase( it );
}

void * ThreadTeam::getTaskReductionThreadStorage( void *p_orig, size_t id )
{
   LockBlock Lock( _lockTaskReductions );

   //! Check if orig is already registered
   task_reduction_list_t::iterator it;
   for ( it = _taskReductions.begin(); it != _taskReductions.end(); it++) {
      void *ptr = (*it)->have( p_orig, id );
      if ( ptr != NULL ) return ptr;
   }
   return NULL;
}

TaskReduction * ThreadTeam::getTaskReduction( const void *p_dep )
{
   LockBlock Lock( _lockTaskReductions );

   //! Check if orig is already registered
   task_reduction_list_t::iterator it;
   for ( it = _taskReductions.begin(); it != _taskReductions.end(); it++) {
      void *ptr = (*it)->have_dependence( p_dep, 0 );
      if ( ptr != NULL ) return (*it);
   }
   return NULL;
}

