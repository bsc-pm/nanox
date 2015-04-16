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
}

bool RecursiveLock::tryAcquire ( )
{
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
}

void RecursiveLock::release ( )
{
   _recursionCount--;
   if ( _recursionCount == 0UL )
   {
      _holderThread = 0UL;
      __sync_lock_release( &state_ );
   }
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

