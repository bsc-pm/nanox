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

#ifndef _NANOS_LOCK
#define _NANOS_LOCK

#include "lock_decl.hpp"
#include "atomic.hpp"

using namespace nanos;

inline Lock::state_t Lock::operator* () const
{
   return state_;
}

inline Lock::state_t Lock::getState () const
{
   return state_;
}

inline void Lock::operator++ ( int val )
{
   acquire();
}

inline void Lock::operator-- ( int val )
{
   release();
}

inline void Lock::acquire ( void )
{
   if ( (state_ == NANOS_LOCK_FREE) &&  !__sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) return;

   // Disabling lock instrumentation; do not remove follow code which can be reenabled for testing purposes
   // NANOS_INSTRUMENT( InstrumentState inst(NANOS_ACQUIRING_LOCK) )

spin:

   while ( state_ == NANOS_LOCK_BUSY ) {}

   if ( __sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) goto spin;

   // NANOS_INSTRUMENT( inst.close() )
}

inline void Lock::acquire_noinst ( void )
{
spin:
   while ( state_ == NANOS_LOCK_BUSY ) {}
   if ( __sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) goto spin;
}

inline bool Lock::tryAcquire ( void )
{
   if ( state_ == NANOS_LOCK_FREE ) {
      if ( __sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) return false;
      else return true;
   } else return false;
}

inline void Lock::release ( void )
{
   __sync_lock_release( &state_ );
}

inline LockBlock::LockBlock ( Lock & lock ) : _lock(lock)
{
   acquire();
}

inline LockBlock::~LockBlock ( )
{
   release();
}

inline void LockBlock::acquire()
{
   _lock++;
}

inline void LockBlock::release()
{
   _lock--;
}

inline LockBlock_noinst::LockBlock_noinst ( Lock & lock ) : _lock(lock)
{
   acquire();
}

inline LockBlock_noinst::~LockBlock_noinst ( )
{
   release();
}

inline void LockBlock_noinst::acquire()
{
   _lock.acquire_noinst();
}

inline void LockBlock_noinst::release()
{
   _lock.release();
}

inline SyncLockBlock::SyncLockBlock ( Lock & lock ) : LockBlock(lock)
{
   memoryFence();
}

inline SyncLockBlock::~SyncLockBlock ( )
{
   memoryFence();
}

#endif
