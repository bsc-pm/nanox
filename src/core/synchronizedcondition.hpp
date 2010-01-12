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

#ifndef _NANOS_SYNCRHONIZED_CONDITION
#define _NANOS_SYNCRHONIZED_CONDITION

#include "synchronizedcondition_decl.hpp"
#include "basethread.hpp"
#include "schedule.hpp"

using namespace nanos;

template <class _T>
void SynchronizedCondition< _T>::wait()
{
   int spins=100; // Has this to be configurable??

   myThread->getCurrentWD()->setSyncCond( this );

   while ( !_conditionChecker.checkCondition() ) {
      BaseThread *thread = getMyThreadSafe();
      WD * current = thread->getCurrentWD();
      current->setIdle();

      spins--;
      if ( spins == 0 ) {
         lock();
         if ( !( _conditionChecker.checkCondition() ) ) {
            addWaiter( current );

            WD *next = thread->getSchedulingGroup()->atBlock ( thread );

/*            if ( next ) {
               sys._numReady--;
            } */

            if ( !next )
               next = thread->getSchedulingGroup()->getIdle ( thread );
         
	    if ( next ) {
               current->setBlocked();
               thread->switchTo ( next ); // how do we unlock here?
            }
            else {
               unlock();
               thread->yield(); // TODO
            }
         } else {
            unlock();
         }
      spins = 100;
      }
   }
   myThread->getCurrentWD()->setReady();
   myThread->getCurrentWD()->setSyncCond( NULL );
}

template <class _T>
void SynchronizedCondition< _T>::signal()
{
   lock();
     while ( hasWaiters() ) {
        WD* wd = getAndRemoveWaiter();
        if ( wd->isBlocked() ) {
           wd->setReady();
           Scheduler::queue( *wd );
        }
     }
   unlock(); 
}

#endif

