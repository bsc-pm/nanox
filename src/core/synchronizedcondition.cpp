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

#include "synchronizedcondition.hpp"
#include "basethread.hpp"
#include "schedule.hpp"

using namespace nanos;

void SynchronizedCondition::wait()
{
   int spins=100; // Has this to be configurable??

   // FIXME use exceptions
   if ( _conditionChecker == NULL)
      warning("Synchronized condition has no ConditionChecker for wait");

   myThread->getCurrentWD()->setSyncCond( this );

   while ( !_conditionChecker->checkCondition() ) {
      BaseThread *thread = getMyThreadSafe();
      WD * current = thread->getCurrentWD();
      current->setIdle();

      spins--;
      if ( spins == 0 ) {
         lock();
         if ( !( _conditionChecker->checkCondition() ) ) {
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

void SynchronizedCondition::signal()
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

