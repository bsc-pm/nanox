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

#ifndef _NANOS_LOCK
#define _NANOS_LOCK

#include "atomic.hpp"
#include "lock_decl.hpp"

namespace nanos {

inline Lock::state_t Lock::operator* () const
{
   return getState();
}

inline Lock::state_t Lock::getState () const
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   return __atomic_load_n(&state_, __ATOMIC_ACQUIRE);
#else
   return state_;
#endif
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
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   acquire_noinst();
#else
   if ( (state_ == NANOS_LOCK_FREE) &&  !__sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) return;

   // Disabling lock instrumentation; do not remove follow code which can be reenabled for testing purposes
   // NANOS_INSTRUMENT( InstrumentState inst(NANOS_ACQUIRING_LOCK) )
spin:
   while ( state_ == NANOS_LOCK_BUSY ) {}

   if ( __sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) goto spin;

   // NANOS_INSTRUMENT( inst.close() )
#endif
}

inline void Lock::lock()
{
   acquire();
}

inline void Lock::acquire_noinst ( void )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   while (__atomic_exchange_n( &state_, NANOS_LOCK_BUSY, __ATOMIC_ACQ_REL) == NANOS_LOCK_BUSY ) { }
#else
spin:
   while ( state_ == NANOS_LOCK_BUSY ) {}
   if ( __sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) goto spin;
#endif
}

inline bool Lock::tryAcquire ( void )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   if (__atomic_load_n(&state_, __ATOMIC_ACQUIRE) == NANOS_LOCK_FREE)
   {
      if (__atomic_exchange_n(&state_, NANOS_LOCK_BUSY, __ATOMIC_ACQ_REL) == NANOS_LOCK_BUSY)
         return false;
      else // will return NANOS_LOCK_FREE
         return true;
   }
   else
   {
      return false;
   }
#else
   if ( state_ == NANOS_LOCK_FREE ) {
      if ( __sync_lock_test_and_set( &state_,NANOS_LOCK_BUSY ) ) return false;
      else return true;
   } else return false;
#endif
}

inline bool Lock::try_lock()
{
   return tryAcquire();
}

inline void Lock::release ( void )
{
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
   __atomic_store_n(&state_, 0, __ATOMIC_RELEASE);
#else
   __sync_lock_release( &state_ );
#endif
}

inline void Lock::unlock()
{
   release();
}

inline bool operator== ( const Lock& lhs, const Lock& rhs )
{
   return &lhs.state_ == &rhs.state_;
}

inline bool operator!= ( const Lock& lhs, const Lock& rhs )
{
   return &lhs.state_ != &rhs.state_;
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

inline DoubleLockBlock::DoubleLockBlock ( Lock & lock1, Lock & lock2 )
   : _lock1(lock1), _lock2(lock2)
{
   if ( _lock1 == _lock2 ) {
      _lock1.acquire();
   } else {
      while ( true ) {
         // We alternate lock ordering to avoid excessive
         // locking and unlocking if only one Lock is available

         _lock1.acquire();
         if ( _lock2.tryAcquire() ) {
            break;
         }
         _lock1.release();

         _lock2.acquire();
         if ( _lock1.tryAcquire() ) {
            break;
         }
         _lock2.release();
      }
   }
}

inline DoubleLockBlock::~DoubleLockBlock ( )
{
   if ( _lock1 == _lock2 ) {
      _lock1.release();
   } else {
      _lock1.release();
      _lock2.release();
   }
}

} // namespace nanos

#endif
