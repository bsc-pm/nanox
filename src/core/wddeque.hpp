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

#include "wddeque_decl.hpp"

using namespace nanos;

inline bool WDDeque::empty ( void ) const
{
   return _dq.empty();
}

inline void WDDeque::push_front ( WorkDescriptor *wd )
{
   wd->setMyQueue( this );
   _lock++;
   //dq.push_back(wd);
   _dq.push_front( wd ); //correct: push_back in push_front?
   memoryFence();
   _lock--;
}

inline void WDDeque::push_back ( WorkDescriptor *wd )
{
   wd->setMyQueue( this );
   _lock++;
   _dq.push_back( wd );
   memoryFence();
   _lock--;
}

// Only ensures tie semantics
inline WorkDescriptor * WDDeque::pop_front ( BaseThread *thread )
{
   WorkDescriptor *found = NULL;

   if ( _dq.empty() )
      return NULL;

   _lock++;

   memoryFence();

   if ( !_dq.empty() ) {
      WDDeque::BaseContainer::iterator it;

      for ( it = _dq.begin() ; it != _dq.end(); it++ ) {
         if ( !(*it)->canRunIn(*thread->runningOn()) ) continue;
         if ( !( *it )->isTied() || ( *it )->isTiedTo() == thread ) {
            if ( (((WD*)( *it ))->dequeue( &found )) == true ) _dq.erase( it );
            break;
         }
      }
   }

   if ( found != NULL ) {found->setMyQueue( NULL );}

   _lock--;

   ensure( !found || !found->isTied() || found->isTiedTo() == thread, "" );

   return found;
}


// Only ensures tie semantics
inline WorkDescriptor * WDDeque::pop_back ( BaseThread *thread )
{
   WorkDescriptor *found = NULL;

   if ( _dq.empty() )
      return NULL;

   _lock++;

   memoryFence();

   if ( !_dq.empty() ) {
      WDDeque::BaseContainer::reverse_iterator rit;

      for ( rit = _dq.rbegin(); rit != _dq.rend() ; rit++ ) {
         if ( !(*rit)->canRunIn(*thread->runningOn()) ) continue;
         if ( !( *rit )->isTied() || ( *rit )->isTiedTo() == thread ) {
            if ( (( *rit )->dequeue( &found )) == true ) _dq.erase( ( ++rit ).base() );
            break;
         }
      }
   }

   if ( found != NULL ) {found->setMyQueue( NULL );}

   _lock--;

   ensure( !found || !found->isTied() || found->isTiedTo() == thread, "" );

   return found;
}


inline bool WDDeque::removeWD( BaseThread *thread, WorkDescriptor * toRem )
{
   if ( _dq.empty() )
      return false;

   if ( toRem->isTied() && toRem->isTiedTo() != thread )
      return false;

   if ( !toRem->canRunIn(*thread->runningOn()) )
      return false;

   _lock++;

   memoryFence();

   if ( !_dq.empty() && toRem->getMyQueue() == this ) {
      WDDeque::BaseContainer::iterator it;

      for ( it = _dq.begin(); it != _dq.end(); it++ ) {
         if ( *it == toRem ) {
            _dq.erase( it );
            toRem->setMyQueue( NULL );

            _lock--;
            return true;
         }
      }
   }

   _lock--;

   return false;
}


inline WorkDescriptor * WDDeque::pop_front ( BaseThread *thread, SchedulePredicate &predicate )
{
   WorkDescriptor *found = NULL;

   if ( _dq.empty() )
      return NULL;

   _lock++;

   memoryFence();

   if ( !_dq.empty() ) {
      WDDeque::BaseContainer::iterator it;

      for ( it = _dq.begin() ; it != _dq.end(); it++ ) {
         if ( !(*it)->canRunIn(*thread->runningOn()) ) continue;
         if ( ( !( *it )->isTied() || ( *it )->isTiedTo() == thread ) && ( predicate( *it ) == true ) ) {
            if ( (( *it )->dequeue( &found )) == true ) _dq.erase( it );
            break;
         }
      }
   }


   if ( found != NULL ) {found->setMyQueue( NULL );}

   _lock--;

   ensure( !found || !found->isTied() || found->isTiedTo() == thread, "" );

   return found;
}



// Also ensures that the passed predicate is verified on the returned element
inline WorkDescriptor * WDDeque::pop_back ( BaseThread *thread, SchedulePredicate &predicate )
{
   WorkDescriptor *found = NULL;

   if ( _dq.empty() )
      return NULL;

   _lock++;

   memoryFence();

   if ( !_dq.empty() ) {
      WDDeque::BaseContainer::reverse_iterator rit;

      for ( rit = _dq.rbegin(); rit != _dq.rend() ; rit++ ) {
         if ( !(*rit)->canRunIn(*thread->runningOn()) ) continue;
         if ( ( !( *rit )->isTied() || ( *rit )->isTiedTo() == thread )  && ( predicate( *rit ) == true ) ) {
            if ( (( *rit )->dequeue( &found )) == true ) _dq.erase( ( ++rit ).base() );
            break;
         }
      }
   }

   if ( found != NULL ) {found->setMyQueue( NULL );}

   _lock--;

   ensure( !found || !found->isTied() || found->isTiedTo() == thread, "" );

   return found;
}

#endif

