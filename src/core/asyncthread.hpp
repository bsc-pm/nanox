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

#ifndef _ASYNC_THREAD_ELEMENT
#define _ASYNC_THREAD_ELEMENT

#include "asyncthread_decl.hpp"

//#include "workdescriptor_fwd.hpp"
//#include "atomic.hpp"
//#include "processingelement.hpp"
//#include "debug.hpp"
//#include "schedule_fwd.hpp"
//#include "threadteam_fwd.hpp"
//#include "atomic.hpp"
//#include "system.hpp"

namespace nanos
{

inline void AsyncThread::checkEvents()
{
   int i = 0, max = _pendingEventsCounter;
   for ( GenericEventList::iterator it = _pendingEvents.begin(); it != _pendingEvents.end(); it++ ) {
      GenericEvent * evt = *it;
      if ( evt->isRaised() ) {
         evt->setCompleted();
         // Move to next step if WD's event is raised
         while ( evt->hasNextAction() ) {
            Action * action = evt->getNextAction();
            action->run();
            delete action;
         }
      }

      i++;
      if ( i == max ) break;
   }

   // Delete completed events
   while ( _pendingEventsCounter && _pendingEvents.front()->isCompleted() ) {
      _pendingEvents.pop_front();
      _pendingEventsCounter--;
   }
}

inline void AsyncThread::checkEvents( WD * wd )
{
   int i = 0, max = _pendingEventsCounter;
   for ( GenericEventList::iterator it = _pendingEvents.begin(); it != _pendingEvents.end(); it++ ) {
      GenericEvent * evt = *it;
      if ( evt->getWD() == wd ) {
         if ( evt->isRaised() ) {
            evt->setCompleted();
            // Move to next step if WD's event is raised
            while ( evt->hasNextAction() ) {
               Action * action = evt->getNextAction();
               action->run();
               delete action;
            }
         }
      }

      i++;
      if ( i == max ) break;
   }

   // Delete completed events
   while ( _pendingEventsCounter && _pendingEvents.front()->isCompleted() ) {
      _pendingEvents.pop_front();
      _pendingEventsCounter--;
   }
}


inline bool AsyncThread::canGetWork()
{
   return BaseThread::canGetWork() && ( int ) _runningWDsCounter < getMaxPrefetch();
}

}

#endif //_ASYNC_THREAD_ELEMENT
