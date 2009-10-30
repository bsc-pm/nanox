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

#ifndef _NANOS_LIB_WDDEQUE
#define _NANOS_LIB_WDDEQUE

#include <list>
#include "atomic.hpp"
#include "debug.hpp"
#include "workdescriptor.hpp"
#include "basethread.hpp"

namespace nanos
{

   class SchedulePredicate
   {

      public:
         virtual bool operator() ( WorkDescriptor *wd );

         virtual ~SchedulePredicate() {}
   };


   class WDDeque
   {

      private:
         typedef std::list<WorkDescriptor *> deque_t;

         deque_t	dq;
         Lock 	lock;

         //TODO: copy constructor, assignment
         WDDeque ( const WDDeque & );
         const WDDeque & operator= ( const WDDeque & );

      public:
         WDDeque() {}

         ~WDDeque() {}

         void push_front ( WorkDescriptor *wd );
         void push_back( WorkDescriptor *wd );
         WorkDescriptor * pop_front ( BaseThread *thread );
         WorkDescriptor * pop_front ( BaseThread *thread, SchedulePredicate &predicate );
         WorkDescriptor * pop_back ( BaseThread *thread );
         WorkDescriptor * pop_back ( BaseThread *thread, SchedulePredicate &predicate );

         bool removeWD( WorkDescriptor * toRem );
   };

   inline void WDDeque::push_front ( WorkDescriptor *wd )
   {
      wd->setMyQueue( this );
      lock++;
      //dq.push_back(wd);
      dq.push_front( wd ); //correct: push_back in push_front?
      memory_fence();
      lock--;
   }

   inline void WDDeque::push_back ( WorkDescriptor *wd )
   {
      wd->setMyQueue( this );
      lock++;
      dq.push_back( wd );
      memory_fence();
      lock--;
   }

// Only ensures tie semantics
   inline WorkDescriptor * WDDeque::pop_front ( BaseThread *thread )
   {
      WorkDescriptor *found = NULL;

      if ( dq.empty() )
         return NULL;

      lock++;

      memory_fence();

      if ( !dq.empty() ) {
         WDDeque::deque_t::iterator it;

         for ( it = dq.begin() ; it != dq.end(); it++ ) {
            if ( !( *it )->isTied() || ( *it )->isTiedTo() == thread ) {
               found = *it;
               dq.erase( it );
               break;
            }
         }
      }

      if ( found != NULL ) {found->setMyQueue( NULL );}

      lock--;

      ensure( !found || !found->isTied() || found->isTiedTo() == thread, "" );

      return found;
   }


// Only ensures tie semantics
   inline WorkDescriptor * WDDeque::pop_back ( BaseThread *thread )
   {
      WorkDescriptor *found = NULL;

      if ( dq.empty() )
         return NULL;

      lock++;

      memory_fence();

      if ( !dq.empty() ) {
         WDDeque::deque_t::reverse_iterator rit;

         rit = dq.rbegin();

         while ( rit != dq.rend() ) {
            if ( !( *rit )->isTied() || ( *rit )->isTiedTo() == thread ) {
               found = *rit;
               dq.erase( ( ++rit ).base() );
               break;
            }

            rit++;
         }
      }

      if ( found != NULL ) {found->setMyQueue( NULL );}

      lock--;

      ensure( !found || !found->isTied() || found->isTiedTo() == thread, "" );

      return found;
   }




   inline bool WDDeque::removeWD( WorkDescriptor * toRem )
   {
      if ( dq.empty() )
         return false;

      lock++;

      memory_fence();

      if ( !dq.empty() && toRem->getMyQueue() == this ) {
         WDDeque::deque_t::iterator it;

         for ( it = dq.begin(); it != dq.end(); it++ ) {
            if ( *it == toRem ) {
               dq.erase( it );
               toRem->setMyQueue( NULL );

               lock--;
               return true;
            }
         }
      }

      lock--;

      return false;
   }


   inline WorkDescriptor * WDDeque::pop_front ( BaseThread *thread, SchedulePredicate &predicate )
   {
      WorkDescriptor *found = NULL;

      if ( dq.empty() )
         return NULL;

      lock++;

      memory_fence();

      if ( !dq.empty() ) {
         WDDeque::deque_t::iterator it;

         for ( it = dq.begin() ; it != dq.end(); it++ ) {
            if ( ( !( *it )->isTied() || ( *it )->isTiedTo() == thread ) && ( predicate( *it ) == true ) ) {
               found = *it;
               dq.erase( it );
               break;
            }
         }
      }


      if ( found != NULL ) {found->setMyQueue( NULL );}

      lock--;

      ensure( !found || !found->isTied() || found->isTiedTo() == thread, "" );

      return found;
   }



// Also ensures that the passed predicate is verified on the returned element
   inline WorkDescriptor * WDDeque::pop_back ( BaseThread *thread, SchedulePredicate &predicate )
   {
      WorkDescriptor *found = NULL;

      if ( dq.empty() )
         return NULL;

      lock++;

      memory_fence();

      if ( !dq.empty() ) {
         WDDeque::deque_t::reverse_iterator rit;

         rit = dq.rbegin();

         while ( rit != dq.rend() ) {
            if ( ( !( *rit )->isTied() || ( *rit )->isTiedTo() == thread )  && ( predicate( *rit ) == true ) ) {
               found = *rit;
               dq.erase( ( ++rit ).base() );
               break;
            }

            rit++;
         }
      }

      if ( found != NULL ) {found->setMyQueue( NULL );}

      lock--;

      ensure( !found || !found->isTied() || found->isTiedTo() == thread, "" );

      return found;
   }

}

#endif

