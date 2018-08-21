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

#ifndef _NANOS_RECURSIVELOCK_DECL
#define _NANOS_RECURSIVELOCK_DECL

#include "nanos-int.h"

namespace nanos {

   /** \brief Forwared declaration required by RecursiveLock */
   class BaseThread;

   class RecursiveLock : public nanos_lock_t
   {

      private:
         BaseThread *_holderThread;
         std::size_t _recursionCount;
         
         typedef nanos_lock_state_t state_t;

         // disable copy constructor and assignment operator
         RecursiveLock( const RecursiveLock &lock );
         const RecursiveLock & operator= ( const RecursiveLock& );

      public:
         // constructor
         RecursiveLock( state_t init=NANOS_LOCK_FREE )
            : nanos_lock_t( init ), _holderThread( 0 ), _recursionCount( 0UL )
         {};

         // destructor
         ~RecursiveLock() {}

         void acquire ( void );
         bool tryAcquire ( void );
         void release ( void );

         state_t operator* () const;

         state_t getState () const;

         void operator++ ( int );

         void operator-- ( int );
   };

   class RecursiveLockBlock
   {
     private:
       RecursiveLock & _lock;

       // disable copy-constructor
       explicit RecursiveLockBlock ( const RecursiveLock & );

     public:
       RecursiveLockBlock ( RecursiveLock & lock );
       ~RecursiveLockBlock ( );

       void acquire();
       void release();
   };

   class SyncRecursiveLockBlock : public RecursiveLockBlock
   {
     private:
       // disable copy-constructor
       explicit SyncRecursiveLockBlock ( const SyncRecursiveLockBlock & );

     public:
       SyncRecursiveLockBlock ( RecursiveLock & lock );
       ~SyncRecursiveLockBlock ( );
   };

} // namespace nanos

#endif
