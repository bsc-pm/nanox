/*************************************************************************************/
/*      Copyright 2010 Barcelona Supercomputing Center                               */
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

#include "omp.h"
#include <stdlib.h>
#include <stdio.h>

extern "C"
{

   struct __omp_lock {
      int lock;
   };

   enum { UNLOCKED = -1, INIT, LOCKED };
   void omp_init_lock( omp_lock_t *arg ) {

      struct __omp_lock *lock = ( struct __omp_lock * )arg;
      lock->lock = UNLOCKED;
   }

   void omp_destroy_lock( omp_lock_t *arg ) {

      struct __omp_lock *lock = ( struct __omp_lock * )arg;
      lock->lock = INIT;
   }

   void omp_set_lock( omp_lock_t *arg ) {

      struct __omp_lock *lock = ( struct __omp_lock * )arg;

      if ( lock->lock == UNLOCKED ) {
         lock->lock = LOCKED;
      } else if ( lock->lock == LOCKED ) {
         fprintf( stderr,
                  "error: deadlock in using lock variable\n" );
         exit( 1 );
      } else {
         fprintf( stderr, "error: lock not initialized\n" );
         exit( 1 );
      }

   }

   void omp_unset_lock( omp_lock_t *arg ) {

      struct __omp_lock *lock = ( struct __omp_lock * )arg;

      if ( lock->lock == LOCKED ) {
         lock->lock = UNLOCKED;
      } else if ( lock->lock == UNLOCKED ) {
         fprintf( stderr, "error: lock not set\n" );
         exit( 1 );
      } else {
         fprintf( stderr, "error: lock not initialized\n" );
         exit( 1 );
      }
   }

   int omp_test_lock( omp_lock_t *arg ) {

      struct __omp_lock *lock = ( struct __omp_lock * )arg;

      if ( lock->lock == UNLOCKED ) {
         lock->lock = LOCKED;
         return 1;
      } else if ( lock->lock == LOCKED ) {
         return 0;
      } else {
         fprintf( stderr, "error: lock not initialized\n" );
         exit( 1 );
      }
   }

   struct __omp_nest_lock {
      short owner;
      short count;
   };

   enum { NOOWNER = -1, MASTER = 0 };
   void omp_init_nest_lock( omp_nest_lock_t *arg ) {

      struct __omp_nest_lock *nlock=( struct __omp_nest_lock * )arg;
      nlock->owner = NOOWNER;
      nlock->count = 0;
   }

   void omp_destroy_nest_lock( omp_nest_lock_t *arg ) {

      struct __omp_nest_lock *nlock=( struct __omp_nest_lock * )arg;
      nlock->owner = NOOWNER;
      nlock->count = UNLOCKED;
   }

   void omp_set_nest_lock( omp_nest_lock_t *arg ) {

      struct __omp_nest_lock *nlock=( struct __omp_nest_lock * )arg;

      if ( nlock->owner == MASTER && nlock->count >= 1 ) {
         nlock->count++;
      } else if ( nlock->owner == NOOWNER && nlock->count == 0 ) {
         nlock->owner = MASTER;
         nlock->count = 1;
      } else {
         fprintf( stderr,
                  "error: lock corrupted or not initialized\n" );
         exit( 1 );
      }
   }

   void omp_unset_nest_lock( omp_nest_lock_t *arg ) {

      struct __omp_nest_lock *nlock=( struct __omp_nest_lock * )arg;

      if ( nlock->owner == MASTER && nlock->count >= 1 ) {
         nlock->count--;

         if ( nlock->count == 0 ) {
            nlock->owner = NOOWNER;
         }
      } else if ( nlock->owner == NOOWNER && nlock->count == 0 ) {
         fprintf( stderr, "error: lock not set\n" );
         exit( 1 );
      } else {
         fprintf( stderr,
                  "error: lock corrupted or not initialized\n" );
         exit( 1 );
      }
   }

   int omp_test_nest_lock( omp_nest_lock_t *arg ) {

      struct __omp_nest_lock *nlock=( struct __omp_nest_lock * )arg;
      omp_set_nest_lock( arg );
      return nlock->count;
   }
}

