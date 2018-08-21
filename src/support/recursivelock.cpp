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

#include "atomic.hpp"
#include "lock.hpp"
#include "basethread.hpp"
#include "recursivelock_decl.hpp"

using namespace nanos;

RecursiveLock::state_t RecursiveLock::operator* () const
{
   return state_;
}

RecursiveLock::state_t RecursiveLock::getState () const
{
   return state_;
}

void RecursiveLock::operator++ ( int )
{
   acquire( );
}

void RecursiveLock::operator-- ( int )
{
   release( );
}

void RecursiveLock::acquire ( )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   if ( __atomic_load_n(&_holderThread, __ATOMIC_ACQUIRE) == getMyThreadSafe() )
   {
      __atomic_add_fetch(&_recursionCount, 1, __ATOMIC_ACQ_REL);
      return;
   }

   while (__atomic_exchange_n( &state_, NANOS_LOCK_BUSY, __ATOMIC_ACQ_REL) == NANOS_LOCK_BUSY ) { }

   __atomic_store_n(&_holderThread, getMyThreadSafe(), __ATOMIC_RELEASE);
   __atomic_add_fetch(&_recursionCount, 1, __ATOMIC_ACQ_REL);

#else
   if ( _holderThread == getMyThreadSafe() )
   {
      _recursionCount++;
      return;
   }
   
spin:
   while ( state_ == NANOS_LOCK_BUSY ) {}

   if ( __sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) goto spin;

   _holderThread = getMyThreadSafe();
   _recursionCount++;
#endif
}

bool RecursiveLock::tryAcquire ( )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   if ( __atomic_load_n(&_holderThread, __ATOMIC_ACQUIRE) == getMyThreadSafe() )
   {
      __atomic_add_fetch(&_recursionCount, 1, __ATOMIC_ACQ_REL);
      return true;
   }

   if ( __atomic_load_n(&state_, __ATOMIC_ACQUIRE) == NANOS_LOCK_FREE )
   {
      if ( __atomic_exchange_n( &state_, NANOS_LOCK_BUSY, __ATOMIC_ACQ_REL ) == NANOS_LOCK_BUSY )
      {
         return false;
      }
      else
      {
         __atomic_store_n(&_holderThread, getMyThreadSafe(), __ATOMIC_RELEASE);
         __atomic_add_fetch(&_recursionCount, 1, __ATOMIC_ACQ_REL);
         return true;
      }
   }
   else
   {
      return false;
   }
#else
   if ( _holderThread == getMyThreadSafe() ) {
      _recursionCount++;
      return true;
   }
   
   if ( state_ == NANOS_LOCK_FREE ) {
      if ( __sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) return false;
      else
      {
         _holderThread = getMyThreadSafe();
         _recursionCount++;
         return true;
      }
   } else return false;
#endif
}

void RecursiveLock::release ( )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   if ( __atomic_sub_fetch(&_recursionCount, 1, __ATOMIC_ACQ_REL) == 0 )
   {
      _holderThread = 0UL;
      __atomic_store_n(&state_, NANOS_LOCK_FREE, __ATOMIC_RELEASE);
   }
#else
   ensure(_recursionCount > 0, "double release lock");
   _recursionCount--;
   if ( _recursionCount == 0UL )
   {
      _holderThread = 0UL;
      __sync_lock_release( &state_ );
   }
#endif
}

RecursiveLockBlock::RecursiveLockBlock ( RecursiveLock & lock ) : _lock(lock)
{
   acquire();
}

RecursiveLockBlock::~RecursiveLockBlock ( )
{
   release();
}

void RecursiveLockBlock::acquire()
{
   _lock++;
}

void RecursiveLockBlock::release()
{
   _lock--;
}

SyncRecursiveLockBlock::SyncRecursiveLockBlock ( RecursiveLock & lock ) : RecursiveLockBlock(lock)
{
   nanos::memoryFence();
}

SyncRecursiveLockBlock::~SyncRecursiveLockBlock ( )
{
   nanos::memoryFence();
}

