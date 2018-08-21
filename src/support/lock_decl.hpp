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

#ifndef _NANOS_LOCK_DECL
#define _NANOS_LOCK_DECL

#include "nanos-int.h"

namespace nanos {

   class Lock : public nanos_lock_t
   {
      private:
         typedef nanos_lock_state_t state_t;

         // disable copy constructor and assignment operator
         Lock( const Lock &lock );
         const Lock & operator= ( const Lock& );

      public:
         // constructor
         Lock( state_t init=NANOS_LOCK_FREE ) : nanos_lock_t( init ) {};

         // destructor
         ~Lock() {}

         void acquire();

         void acquire_noinst();

         // compatibility
         void lock();

         bool tryAcquire();

         // compatibility
         bool try_lock();

         void release();

         // compatibility
         void unlock();

         state_t operator* () const;

         state_t getState () const;

         void operator++ ( int val );

         void operator-- ( int val );

         friend bool operator== ( const Lock& lhs, const Lock& rhs );

         friend bool operator!= ( const Lock& lhs, const Lock& rhs );
   };

   class LockBlock
   {
     private:
       Lock & _lock;

       // disable copy-constructor
       explicit LockBlock ( const LockBlock & );

     public:
       LockBlock ( Lock & lock );
       ~LockBlock ( );

       void acquire();
       void release();
   };

   class LockBlock_noinst
   {
     private:
       Lock & _lock;

       // disable copy-constructor
       explicit LockBlock_noinst ( const LockBlock_noinst & );

     public:
       LockBlock_noinst ( Lock & lock );
       ~LockBlock_noinst ( );

       void acquire();
       void release();
   };

   class SyncLockBlock : public LockBlock
   {
     private:
       // disable copy-constructor
       explicit SyncLockBlock ( const SyncLockBlock & );

     public:
       SyncLockBlock ( Lock & lock );
       ~SyncLockBlock ( );
   };

   class DoubleLockBlock
   {
      private:
         Lock & _lock1;
         Lock & _lock2;

         // disable copy-constructor
         explicit DoubleLockBlock ( const DoubleLockBlock & );

      public:
         DoubleLockBlock ( Lock & lock1, Lock & lock2 );
         ~DoubleLockBlock ( );
   };

} // namespace nanos

#endif
