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

#include "schedule.hpp"
#include "processingelement.hpp"
#include "basethread.hpp"
#include "system.hpp"

using namespace nanos;



void Scheduler::submit ( WD &wd )
{
   // TODO: increase ready count

   sys._taskNum++;

   debug ( "submitting task " << wd.getId() );
   WD *next = myThread->getSchedulingGroup()->atCreation ( myThread, wd );

   if ( next ) {
      switchTo ( next );
   }
}

void Scheduler::exit ( void )
{
   // TODO: Support WD running on lended stack
   // Cases:
   // The WD was running on its own stack, switch to a new one
   // The WD was running on a thread stack, exit to the loop

   // At this point the WD work is done, so we mark it as such and look for other work to do
   // Deallocation doesn't happen here because:
   // a) We are still running in the WD stack
   // b) Resources can potentially be reused by the next WD
   myThread->getCurrentWD()->done();
   sys._taskNum--;

   WD *next = myThread->getSchedulingGroup()->atExit ( myThread );

   if ( next ) {
      sys._numReady--;
   }

   if ( !next ) {
      next = myThread->getSchedulingGroup()->getIdle ( myThread );
   }

   if ( next ) {
      myThread->exitTo ( next, exitHelper );
   }

   fatal ( "No more tasks to execute!" );
}

void Scheduler::idle ()
{
   sys.getInstrumentor()->enterIdle();

   // This function is run always by the same BaseThread so we don't need to use getMyThreadSafe
   BaseThread *thread = myThread;

   thread->getCurrentWD()->setIdle();

   sys._idleThreads++;

   while ( thread->isRunning() ) {
      if ( thread->getSchedulingGroup() ) {
         WD *next = thread->getSchedulingGroup()->atIdle ( thread );

         if ( next ) {
            sys._numReady--;
         }

         if ( !next )
            next = thread->getSchedulingGroup()->getIdle ( thread );

         if ( next ) {
            sys._idleThreads--;
            sys._numTasksRunning++;
            switchTo ( next );
            sys._numTasksRunning--;
            sys._idleThreads++;
         }
      }
   }

   thread->getCurrentWD()->setReady();

   sys._idleThreads--;

   verbose ( "Working thread finishing" );
   sys.getInstrumentor()->leaveIdle();
}

void Scheduler::queue ( WD &wd )
{
   if ( wd.isIdle() )
      myThread->getSchedulingGroup()->queueIdle ( myThread, wd );
   else {
      myThread->getSchedulingGroup()->queue ( myThread, wd );
      sys._numReady++;
   }
}

void SchedulingGroup::init ( int groupSize )
{
   _group.reserve ( groupSize );
}

void SchedulingGroup::addMember ( BaseThread &thread )
{
   SchedulingData *data = createMemberData ( thread );

   data->setSchId ( getSize() );
   thread.setScheduling ( this, data );
   _group.push_back( data );
}

void SchedulingGroup::removeMember ( BaseThread &thread )
{
//TODO
}

void SchedulingGroup::queueIdle ( BaseThread *thread, WD &wd )
{
   _idleQueue.push_back ( &wd );
}

WD * SchedulingGroup::getIdle ( BaseThread *thread )
{
   return _idleQueue.pop_front ( thread );
}

void Scheduler::switchHelper (WD *oldWD, WD *newWD, void *arg)
{
   GenericSyncCond *syncCond = oldWD->getSyncCond();
   if ( syncCond != NULL ) {
      oldWD->setBlocked();
      syncCond->unlock();
   } else {
      Scheduler::queue( *oldWD );
   }
   myThread->switchHelperDependent(oldWD, newWD, arg);
   
   sys.getInstrumentor()->wdSwitch( oldWD, newWD );
   myThread->setCurrentWD( *newWD );
}

void Scheduler::exitHelper (WD *oldWD, WD *newWD, void *arg)
{
   myThread->exitHelperDependent(oldWD, newWD, arg);
   sys.getInstrumentor()->wdExit( oldWD, newWD );
   delete oldWD;
   myThread->setCurrentWD( *newWD );
}

void Scheduler::inlineWork ( WD *wd )
{
   // run it in the current frame
   WD *oldwd = myThread->getCurrentWD();

   GenericSyncCond *syncCond = oldwd->getSyncCond();
   if ( syncCond != NULL ) {
      syncCond->unlock();
   }

   sys.getInstrumentor()->wdSwitch( oldwd, wd );

   myThread->setCurrentWD( *wd );
   myThread->inlineWorkDependent(*wd);

   sys.getInstrumentor()->wdSwitch( wd, oldwd );
   wd->done();

   myThread->setCurrentWD( *oldwd );
}

void Scheduler::switchTo ( WD *to )
{
   if (!to->started())
      to->start();

   if ( myThread->runningOn()->supportsUserLevelThreads() )
      myThread->switchTo( to, switchHelper );
   else 
      inlineWork(to);
}

#if 0

void Scheduler::waitOnCondition ( GenericSyncCond &condition )
{
   int spins=100; // FIXME: this has to be configurable (see #147)

   myThread->getCurrentWD()->setSyncCond( condition );
   
   while ( !condition.check() ) {
      BaseThread *thread = getMyThreadSafe();
      WD * current = thread->getCurrentWD();
      current->setIdle();

      spins--;
      if ( spins == 0 ) {
         condition.lock();
         if ( !( condition.check() ) ) {
            condition.addWaiter( current );

            WD *next = thread->getSchedulingGroup()->atBlock ( thread );

            if ( next ) {
               sys._numReady--;
            } 

            if ( next ) {
               thread->switchTo ( next );
            }
            else {
               condition.unlock();
               thread->yield();
            }
         } else {
            condition.unlock();
         }
         spins = 100;
      }
   }
   myThread->getCurrentWD()->setReady();
   myThread->getCurrentWD()->setSyncCond( NULL );
}

#endif