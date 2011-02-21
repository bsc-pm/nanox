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

#ifndef _NANOS_LIB_QUEUE
#define _NANOS_LIB_QUEUE

#include <queue>
#include "atomic.hpp"
#include "debug.hpp"

namespace nanos
{

// FIX: implement own queue without coherence problems? lock-free?

   template<typename T> class Queue
   {

      private:
         typedef std::queue<T>   BaseContainer;
         Lock                    _qLock;
         BaseContainer           _q;

         // disable copy constructor and assignment operator
         Queue( Queue &orig );
         const Queue & operator= ( const Queue &orig );

      public:
         // constructors
         Queue() {}

         // destructor
         ~Queue() {}

         void push( T data );
         T    pop ( void );
         bool try_pop ( T& result );
   };

   template<typename T> void Queue<T>::push ( T data )
   {
      {
         LockBlock lock( _qLock );
         _q.push( data );
         memoryFence();
      }
   }

   template<typename T> T Queue<T>::pop ( void )
   {

   spin:

      while ( _q.empty() ) memoryFence();

      // not empty
      {
         LockBlock lock( _qLock );

         if ( !_q.empty() ) {
            T tmp = _q.front();
            _q.pop();
            return tmp;
         }

      }

      goto spin;
   }

   template<typename T> bool Queue<T>::try_pop ( T& result )
   {
      bool found = false;

      if ( _q.empty() ) return false;

      memory_fence();

      {
         LockBlock lock( _qLock );

         if ( !_q.empty() ) {
            result = _q.front();
            _q.pop();
            found = true;
         }

      }

      return found;
   }

};

#endif

