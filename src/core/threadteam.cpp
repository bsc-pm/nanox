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

#include "threadteam.hpp"
#include "atomic.hpp"
#include "lock.hpp"
#include "debug.hpp"
#include "allocator.hpp"
#include "memtracker.hpp"

using namespace nanos;


bool ThreadTeam::singleGuard( int local )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   // Double check of locks is an antipattern
#else
   if ( local <= _singleGuardCount ) return false;
#endif
   
   return compareAndSwap( &_singleGuardCount, local-1, local );
}

bool ThreadTeam::enterSingleBarrierGuard( int local )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   // Double check of locks is an antipattern
#else
   if ( local <= _singleGuardCount ) return false;
#endif
   
   return compareAndSwap( &_singleGuardCount, local-2, local-1 );

}

void ThreadTeam::releaseSingleBarrierGuard( void )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   __atomic_fetch_add(&_singleGuardCount, 1, __ATOMIC_ACQ_REL);
#else
   _singleGuardCount++;
#endif
}

void ThreadTeam::waitSingleBarrierGuard( int local )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   while ( local > __atomic_load_n(&_singleGuardCount, __ATOMIC_ACQUIRE) ) { }
#else
   while ( local > _singleGuardCount ) { memoryFence(); }
#endif
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
#elif defined(NANOS_ENABLE_ALLOCATOR)
         nanos::getAllocator().deallocate ( red->descriptor ); 
#else
         free( red->descriptor );
#endif
      }
#if defined(NANOS_DEBUG_ENABLED) && defined(NANOS_MEMTRACKER_ENABLED)
      nanos::getMemTracker().deallocate( (void *) red );
#elif defined(NANOS_ENABLE_ALLOCATOR)
      nanos::getAllocator().deallocate ( (void *) red );
#else
      free ( (void*) red );
#endif
      _redList.pop_back();
   }
}

bool ThreadTeam::isStable ( void )
{
   LockBlock Lock( _lock );

   if ( _expectedThreads.find( myThread ) == _expectedThreads.end() ) {
      // Current thread does not belong to the team, switch to first team member
      ensure( !_expectedThreads.empty(), "Team has no members" );
      BaseThread *target_thread = *_expectedThreads.begin();
      sys.getThreadManager()->unblockThread( target_thread );
      Scheduler::switchToThread( target_thread );
   }

   bool is_stable = _threads.size() == _expectedThreads.size();
   if ( is_stable ) {
      // If first condition is met, we keep looking:
      // _threads is a std::map, we need to construct
      // a std::set to compare with linear cost
      ThreadSet threads_set;
      ThreadTeamList::const_iterator it;
      for ( it = _threads.begin(); it != _threads.end(); ++it ) {
         threads_set.insert( (it->second) );
      }
      is_stable = _expectedThreads == threads_set;
   }
   return is_stable;
}
