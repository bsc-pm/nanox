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

#include "schedule.hpp"
#include "atomic.hpp"
#include "processingelement.hpp"
#include "basethread.hpp"
#include "smpthread.hpp"
#include "system.hpp"
#include "config.hpp"
#include "synchronizedcondition.hpp"
#include "instrumentationmodule_decl.hpp"
#include "os.hpp"
#include "wddeque.hpp"
#include "smpthread.hpp"
#include "nanos-int.h"

#include <iostream>

using namespace nanos;

void SchedulerConf::config (Config &cfg)
{
   cfg.setOptionsSection ( "Core [Scheduler]", "Policy independent scheduler options"  );

   cfg.registerConfigOption ( "num-spins", NEW Config::UintVar( _numSpins ), "Set number of spins on Idle before yield (default = 1)" );
   cfg.registerArgOption ( "num-spins", "spins" );

   cfg.registerConfigOption ( "num-checks", NEW Config::UintVar( _numChecks ), "Set number of checks before Idle (default = 1)" );
   cfg.registerArgOption ( "num-checks", "checks" );

   cfg.registerConfigOption ( "num-steal", NEW Config::PositiveVar( _numStealAfterSpins ), "Try to steal every so spins (default = 1)" );
   cfg.registerArgOption ( "num-steal", "spins-steal" );

   cfg.registerConfigOption ( "hold-tasks", NEW Config::FlagOption( _holdTasks ), "Do not submit tasks until a taskwait is reached." );
   cfg.registerArgOption ( "hold-tasks", "hold-tasks" );
}

void Scheduler::submit ( WD &wd, bool force_queue )
{
   if ( sys.getSchedulerConf().getHoldTasksEnabled() && 
         sys.getNetwork()->getNodeNum() == 0) {
      WD *current = myThread->getCurrentWD();
      WD *wd_to_submit = &wd;
      current->addPresubmittedWDs( 1, &wd_to_submit );
   } else {
      _submit( wd, force_queue );
   }
}

void Scheduler::_submit ( WD &wd, bool force_queue )
{
   NANOS_INSTRUMENT ( InstrumentState inst(NANOS_SCHEDULING, true) );
   BaseThread *mythread = myThread;

   debug ( "submitting task " << wd.getId() << " " << ( wd.getDescription() != NULL ? wd.getDescription() : "") << " team: " << mythread->getTeam() << " this thread is " << mythread );

   wd.submitted();
   wd.setReady();

   /* handle tied tasks */
   BaseThread *wd_tiedto = wd.isTiedTo();
   if ( wd.isTied() && wd_tiedto != mythread ) {
      if ( wd_tiedto->getTeam() == NULL ) {
         wd_tiedto->addNextWD( &wd );
      } else {
         wd_tiedto->getTeam()->getSchedulePolicy().queue( wd_tiedto, wd );
      }
      return;
   }

   /* By checking if there's available WDs in the queue before queuing the current one,
    * we ensure that we only trigger a wakeup if at least the current thread will not remain
    * idle after the WD submission. */
   ThreadManager *const thread_manager = sys.getThreadManager();
   if ( thread_manager->isGreedy()
         && mythread->getTeam()->getSchedulePolicy().testDequeue() ) {
      thread_manager->acquireOne();
   }

   /* handle tasks which cannot run in current thread */
   if ( force_queue || !mythread->runningOn()->canRun( wd ) ) {
     /* We have to avoid work-first scheduler to return this kind of tasks, so we enqueue
      * it in our scheduler system. Global ready task queue will take care about task/thread
      * architecture, while local ready task queue will wait until stealing. */
      mythread->getTeam()->getSchedulePolicy().queue( mythread, wd );

      return;
   }

   //! \todo (#581): move this to the upper if
   if ( !sys.getSchedulerConf().getSchedulerEnabled() ) {
      // Pause this thread
      mythread->pause();
      // Scheduler stopped, queue work.
      mythread->getTeam()->getSchedulePolicy().queue( mythread, wd );

      return;
   }
   // The thread is not paused, mark it as so
   myThread->unpause();
   // And go on
   WD *next = getMyThreadSafe()->getTeam()->getSchedulePolicy().atSubmit( myThread, wd );

   /* If SchedulePolicy have returned a 'next' value, we have to context switch to
      that WorkDescriptor */
   if ( next ) {
      WD *slice;
      /* We must ensure this 'next' has no sliced components. If it have them we have to
       * queue the remaining parts of 'next' */
      if ( !next->dequeue(&slice) ) {
         mythread->getTeam()->getSchedulePolicy().queue( mythread, *next );

      }
      switchTo ( slice );
   }

}

void Scheduler::submit ( WD ** wds, size_t numElems ) 
{
   if ( sys.getSchedulerConf().getHoldTasksEnabled() && 
         sys.getNetwork()->getNodeNum() == 0) {
      WD *current = myThread->getCurrentWD();
      current->addPresubmittedWDs( numElems, wds );
   } else {
      _submit( wds, numElems );
   }
}

void Scheduler::_submit ( WD ** wds, size_t numElems )
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SCHEDULING, true) );
   if ( numElems == 0 ) return;
   
   BaseThread *mythread = myThread;
   
   // create a vector of threads for each wd
   BaseThread ** threadList = NEW BaseThread*[numElems];
   for( size_t i = 0; i < numElems; ++i )
   {
      WD* wd = wds[i];
      wd->_mcontrol.preInit();
      
      // If the wd is tied to anyone
      BaseThread *wd_tiedto = wd->isTiedTo();
      if ( wd->isTied() && wd_tiedto != mythread ) {
         if ( wd_tiedto->getTeam() == NULL ) {
            //wd_tiedto->addNextWD( &wd );
            // WHAT HERE???
            fatal( "Uncontrolled batch path");
         } else {
            //wd_tiedto->getTeam()->getSchedulePolicy().queue( wd_tiedto, wd );
            threadList[i] = wd_tiedto;
         }
         continue;
      }
      // Otherwise, use mythread
      threadList[i] = mythread;
   }
   
   // Call the scheduling policy
   mythread->getTeam()->getSchedulePolicy().queue( threadList, wds, numElems );
   
   // Release
   delete[] threadList;
}

void Scheduler::updateCreateStats ( WD &wd )
{
   sys.getSchedulerStats()._createdTasks++;
   sys.getSchedulerStats()._totalTasks++;
   wd.setConfigured(); 
}

void Scheduler::updateExitStats ( WD &wd )
{
   sys.throttleTaskOut();
   if ( wd.isConfigured() ) sys.getSchedulerStats()._totalTasks--;
}

struct TestInputs {
   static void call( ProcessingElement *pe, WorkDescriptor *wd ) {
      if ( wd->_mcontrol.isMemoryAllocated() ) {
         pe->testInputs( *wd );
      }
   }
};

template<class behaviour>
inline void Scheduler::idleLoop ()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t total_spins_key  = ID->getEventKey("num-spins"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_yields_key = ID->getEventKey("num-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_blocks_key = ID->getEventKey("num-blocks"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_scheds_key  = ID->getEventKey("num-scheds"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t steal_key  = ID->getEventKey("steal"); )

   NANOS_INSTRUMENT ( static nanos_event_key_t time_yields_key = ID->getEventKey("time-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_blocks_key = ID->getEventKey("time-blocks"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_scheds_key = ID->getEventKey("time-scheds"); )

   NANOS_INSTRUMENT ( nanos_event_key_t Keys[8]; )

   NANOS_INSTRUMENT ( Keys[0] = total_yields_key; )
   NANOS_INSTRUMENT ( Keys[1] = time_yields_key; )
   NANOS_INSTRUMENT ( Keys[2] = total_blocks_key; )
   NANOS_INSTRUMENT ( Keys[3] = time_blocks_key; )
   NANOS_INSTRUMENT ( Keys[4] = total_spins_key; )
   NANOS_INSTRUMENT ( Keys[5] = steal_key; )
   NANOS_INSTRUMENT ( Keys[6] = total_scheds_key; )
   NANOS_INSTRUMENT ( Keys[7] = time_scheds_key; )

   NANOS_INSTRUMENT ( unsigned event_start; )
   NANOS_INSTRUMENT ( unsigned event_num; )

   NANOS_INSTRUMENT( InstrumentState inst(NANOS_IDLE) );

   const int init_spins = sys.getSchedulerConf().getNumSpins();
   const int init_yields = sys.getThreadManagerConf().getNumYields();
   int spins = init_spins;
   int yields = init_yields;
   // Number of times during the spin loop that a steal get was performed
   int num_steals = 0;
   // Number of consecutive calls to getWD that returned NULL
   int num_empty_calls = 0;
   // Modulo to steal tasks
   int steal_mod = sys.getSchedulerConf().getNumStealAfterSpins() + 1;

   NANOS_INSTRUMENT ( unsigned long long total_spins = 0; )  /* Number of spins by idle phase*/
   NANOS_INSTRUMENT ( unsigned long long total_yields = 0; ) /* Number of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long long total_blocks = 0; ) /* Number of blocks by idle phase */
   NANOS_INSTRUMENT ( unsigned long long total_scheds = 0; ) /* Number of scheds by idle phase */

   NANOS_INSTRUMENT ( unsigned long long time_blocks = 0; ) /* Time of blocks by idle phase */
   NANOS_INSTRUMENT ( unsigned long long time_yields = 0; ) /* Time of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long long time_scheds = 0; ) /* Time of yields by idle phase */

   ThreadManager *const thread_manager = sys.getThreadManager();

   WD *current = myThread->getCurrentWD();
   sys.getSchedulerStats()._idleThreads++;
   myThread->setIdle( true );

   for ( ; ; ) {
      BaseThread *thread = getMyThreadSafe();

      if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
         thread->unpause(); //!< If scheduler is enabled thread is not paused, mark it as so
      }
      else {
         thread->pause(); //!< Otherwise thread is paused.
         continue;
      }

      spins--;

      thread_manager->returnMyCpuIfClaimed();

      if ( thread->isSleeping() && !thread_manager->lastActiveThread() && !thread->hasNextWD() ) {
         NANOS_INSTRUMENT (total_spins+= (init_spins - spins); )

         NANOS_INSTRUMENT ( nanos_event_value_t Values[8]; )

         NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_yields; )
         NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) time_yields; )
         NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) total_blocks; )
         NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) time_blocks; )
         NANOS_INSTRUMENT ( Values[4] = (nanos_event_value_t) total_spins; )
         NANOS_INSTRUMENT ( Values[5] = (nanos_event_value_t) 0; /*steal*/ )
         NANOS_INSTRUMENT ( Values[6] = (nanos_event_value_t) total_scheds; )
         NANOS_INSTRUMENT ( Values[7] = (nanos_event_value_t) time_scheds; )

         NANOS_INSTRUMENT ( event_start = 0; event_num = 8; )
         NANOS_INSTRUMENT ( if (total_yields == 0 ) { event_start = 2; event_num = 6; } )
         NANOS_INSTRUMENT ( if (total_yields == 0 && total_blocks == 0) { event_start = 4; event_num = 4; } )
         NANOS_INSTRUMENT ( if (total_scheds == 0 ) { event_num -= 2; } )

         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(event_num, &Keys[event_start], &Values[event_start]); )

         thread->wait();

         NANOS_INSTRUMENT (total_spins = 0; )
         NANOS_INSTRUMENT (total_blocks = 0; )
         NANOS_INSTRUMENT (total_yields = 0; )
         NANOS_INSTRUMENT (total_scheds = 0; )

         NANOS_INSTRUMENT (time_yields = 0; )
         NANOS_INSTRUMENT (time_blocks = 0; )
         NANOS_INSTRUMENT (time_scheds = 0; )

         spins = init_spins;
         yields = init_yields;

      }//thread going to sleep, thread waiking up

      if ( !thread->isRunning() && !thread->hasNextWD() ) {
        // if behaviour is not exiting, it is the implicit one, and can break the loop
        // otherwise we need to switch to implicit wd.
        if ( !behaviour::exiting() ) break;
        else behaviour::switchWD(thread, current, &(thread->getThreadWD()));
      }

      thread->getNextWDQueue().iterate<TestInputs>();
      WD * next = thread->getNextWD();
      
      // Declared here to be used for instrumentation too,
      bool steal = false;

      if ( !next && thread->getTeam() != NULL ) {
         memoryFence();
         if ( sys.getSchedulerStats()._readyTasks > 0 ) {
            NANOS_INSTRUMENT ( total_scheds++; )
            NANOS_INSTRUMENT ( unsigned long long begin_sched = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
            
            // Try to steal only if we spun at least once
            steal = num_empty_calls % steal_mod;
            // Increase the number of steal attempts
            if ( steal ) ++num_steals;
            
            next = behaviour::getWD(thread,current,steal*num_steals);

            NANOS_INSTRUMENT ( unsigned long long end_sched = (unsigned long long) ( OS::getMonotonicTime() * 1.0e9  ); )
            NANOS_INSTRUMENT (time_scheds += ( end_sched - begin_sched ); )
         }
      } 

      // Trigger a wakeup if there's more WDs in the queue
      if ( next && thread->getTeam() != NULL && thread_manager->isGreedy()
            && thread->getTeam()->getSchedulePolicy().testDequeue() ) {
         thread_manager->acquireOne();
      }

      if ( next ) {

         NANOS_INSTRUMENT (total_spins+= (init_spins - spins); )

         NANOS_INSTRUMENT ( nanos_event_value_t Values[8]; )

         NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_yields; )
         NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) time_yields; )
         NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) total_blocks; )
         NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) time_blocks; )
         NANOS_INSTRUMENT ( Values[4] = (nanos_event_value_t) total_spins; )
         NANOS_INSTRUMENT ( Values[5] = (nanos_event_value_t) steal; )
         NANOS_INSTRUMENT ( Values[6] = (nanos_event_value_t) total_scheds; )
         NANOS_INSTRUMENT ( Values[7] = (nanos_event_value_t) time_scheds; )

         NANOS_INSTRUMENT ( event_start = 0; event_num = 8; )
         NANOS_INSTRUMENT ( if (total_yields == 0 ) { event_start = 2; event_num = 6; } )
         NANOS_INSTRUMENT ( if (total_yields == 0 && total_blocks == 0) { event_start = 4; event_num = 4; } )
         NANOS_INSTRUMENT ( if (total_scheds == 0 ) { event_num -= 2; } )

         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(event_num, &Keys[event_start], &Values[event_start]); )

         thread->setIdle( false );
         sys.getSchedulerStats()._idleThreads--;

         behaviour::switchWD(thread, current, next);

         thread = getMyThreadSafe();
         thread->step();

         sys.getSchedulerStats()._idleThreads++;
         thread->setIdle( true );

         NANOS_INSTRUMENT (total_spins = 0; )
         NANOS_INSTRUMENT (total_blocks = 0; )
         NANOS_INSTRUMENT (total_yields = 0; )
         NANOS_INSTRUMENT (total_scheds = 0; )

         NANOS_INSTRUMENT (time_yields = 0; )
         NANOS_INSTRUMENT (time_blocks = 0; )
         NANOS_INSTRUMENT (time_scheds = 0; )

         spins = init_spins;
         yields = init_yields;
         /* gmiranda: If a WD was returned (either by a normal getWD or
          * by a steal operation, reset the num_steals counter */
         num_steals = 0;
         // Also reset the number of empty calls
         num_empty_calls = 0;
         continue;
      }
      
      // Otherwise, getWD returned NULL, increase the counter
      ++num_empty_calls;

      thread->idle();
      //if ( sys.getNetwork()->getNodeNum() > 0 ) {
      //   sys.getNetwork()->poll(0);
      //}

      if ( spins == 0 ) {
         NANOS_INSTRUMENT ( total_spins += init_spins; )

         // Perform yield and/or block
         thread_manager->idle( yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
               , total_yields, total_blocks, time_yields, time_blocks
#endif
               );

         spins = init_spins;
      }
   }
   myThread->setIdle(false);
   sys.getSchedulerStats()._idleThreads--;
   //current->~WorkDescriptor();

   //// This is actually a free(current) but dressed up as C++
   //delete (char*) current;
}

void Scheduler::waitOnCondition (GenericSyncCond *condition)
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SYNCHRONIZATION, true) );

   if (condition->check()) return;

   unsigned int checks = (unsigned int) sys.getSchedulerConf().getNumChecks();

   BaseThread *thread = getMyThreadSafe();
   WD * current = thread->getCurrentWD();
   current->setSyncCond( condition );
   
   bool supportULT = thread->runningOn()->supportsUserLevelThreads();

   ThreadManager *const thread_manager = sys.getThreadManager();

   verbose("Wait on condition");
   while ( !condition->check() /* FIXME:xteruel do we needed? && thread->isRunning() */) {
      if ( checks == 0 ) {
         //verbose("   starting idle loop"); //FIXME:xteruel
         condition->lock();
         if ( !( condition->check() ) ) {

            //! If condition is not acomplished yet, release wd and get more work to do

            //! First checking prefetching queue
            WD * next = thread->getNextWD();
            if ( next != NULL ) {
                verbose("Got wd through getNextWD");
            }

            if ( !thread->isSleeping() ) {
               //! Second calling scheduler policy at block
               if ( !next ) {
                  memoryFence();
                  if ( sys.getSchedulerStats()._readyTasks > 0 ) {
                     if ( sys.getSchedulerConf().getSchedulerEnabled() )
                        next = thread->getTeam()->getSchedulePolicy().atBlock( thread, current );
            if ( next != NULL ) {
                verbose("Got wd through atBlock");
            }
                  }
               }
            }

            // Trigger a wakeup if there's more WDs in the queue
            if ( next && thread->getTeam() != NULL && thread_manager->isGreedy()
                  && thread->getTeam()->getSchedulePolicy().testDequeue() ) {
               thread_manager->acquireOne();
            }

            //! Finally coming back to our Thread's WD (idle task)
            if ( !next && supportULT && sys.getSchedulerConf().getSchedulerEnabled() ) {
               next = &(thread->getThreadWD());
            if ( next != NULL ) {
                verbose("Got wd through getThreadWD");
            }
            }

            //! If found a wd to switch to, execute it
            if ( next ) {
               verbose("   switching to " << next->getId() ); //FIXME:xteruel
               switchTo ( next );
               thread = getMyThreadSafe();
               supportULT = thread->runningOn()->supportsUserLevelThreads();
               thread->step();
            } else {
               condition->unlock();
               thread->atBlock();
            }
         } else condition->unlock();
         checks = (unsigned int) sys.getSchedulerConf().getNumChecks();
      } 
      checks--;
   }

   current->setSyncCond( NULL );
   if ( !current->isReady() ) current->setReady();
}

void Scheduler::wakeUp ( WD *wd )
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SYNCHRONIZATION, true) );
   
   if ( !wd->isReady() ) {
      /* Setting ready wd */
      wd->setReady();
      WD *next = NULL;
      
/*      BaseThread * tiedTo = wd->isTiedTo();
      if ( tiedTo != NULL && sys.getSchedulerConf().getUseBlock() ) {
         // If the thread is blocked, we must not re-submit it's task
         tiedTo->unblock();
         // Note: this will probably break nesting.
         return;
      }*/
      
      if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
         /* atWakeUp must check basic constraints */
         // FIXME: this only works with tied tasks, generalize to untied
         BaseThread *thread = wd->isTied()? wd->isTiedTo(): getMyThreadSafe();
         ThreadTeam *myTeam = thread->getTeam();

         ensure( myTeam, "Trying to wake up a WD from a thread without team." );
         next = myTeam->getSchedulePolicy().atWakeUp( myThread, *wd );
      }

      /* If SchedulePolicy have returned a 'next' value, we have to context switch to
         that WorkDescriptor */
      if ( next ) {
         WD *slice;
         /* We must ensure this 'next' has no sliced components. If it have them we have to
          * queue the remaining parts of 'next' */
         if ( !next->dequeue(&slice) ) {
            myThread->getTeam()->getSchedulePolicy().queue( myThread, *next );
         }
         switchTo ( slice );
      }
   }
}

WD * Scheduler::prefetch( BaseThread *thread, WD &wd )
{
   if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
      //! If the scheduler is running do the prefetch
      WD *prefetchedWD = thread->getTeam()->getSchedulePolicy().atPrefetch( thread, wd );
      if ( prefetchedWD ) {
         prefetchedWD->_mcontrol.preInit();
      }
      return prefetchedWD;
   }
   //! \bug FIXME (#581): Otherwise, do nothing: consequences?
   return NULL;
}

struct WorkerBehaviour
{
   static WD * getWD ( BaseThread *thread, WD *current, int numSteal )
   {
      return thread->getTeam()->getSchedulePolicy().atIdle ( thread, numSteal );
   }

   static void switchWD ( BaseThread *thread, WD *current, WD *next )
   {
      Scheduler::switchTo(next);
   }
   static bool checkThreadRunning( WD *current) { return true; }
   static bool exiting() { return false; }
};

struct AsyncWorkerBehaviour
{
   static WD * getWD ( BaseThread *thread, WD *current, int numSteal )
   {
      if ( !thread->canGetWork() ) return NULL;
      return thread->getTeam()->getSchedulePolicy().atIdle ( thread, numSteal );
   }

   static void switchWD ( BaseThread *thread, WD *current, WD *next )
   {
      if ( next->started() ) {
         // Should not be started in general
         Scheduler::switchTo( next );
      }
      else {
         // Since this is the async behavior, set schedule to false:
         // do not prefetch at this point, as the thread will be always prefetching
         if ( Scheduler::inlineWorkAsync ( next, /* schedule */ false ) ) {
            next->~WorkDescriptor();
            delete[] ( char * ) next;
         }
      }
   }

   static bool exiting() { return false; }
};

void Scheduler::workerLoop ()
{
   idleLoop<WorkerBehaviour>();
}

void Scheduler::asyncWorkerLoop ()
{
   idleLoop<AsyncWorkerBehaviour>();
}

void Scheduler::preOutlineWorkWithThread ( BaseThread * thread, WD *wd )
{
   //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_PRE_OUTLINE_WORK, true); );
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t copy_data_in_key = ID->getEventKey("copy-data-in"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( copy_data_in_key, (nanos_event_value_t) wd->getId() ); )
   //std::cerr << "starting WD " << wd->getId() << " at thd " << thread->getId() << " thd addr " << thread << std::endl; 
   // run it in the current frame
   //WD *oldwd = thread->getCurrentWD();

   //GenericSyncCond *syncCond = oldwd->getSyncCond();
   //if ( syncCond != NULL ) {
   //   syncCond->unlock();
   //}

   //std::cerr << "thd " << myThread->getId() <<  " switching(outlined) to task " << wd << ":" << wd->getId() << std::endl;
   debug( "switching(pre outline) from task " << &(thread->getThreadWD()) << ":" << thread->getThreadWD().getId() << " to " << wd << ":" << wd->getId() );

   //NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(oldwd, NULL, false) );

   // OLD: This ensures that when we return from the inlining is still the same thread
   // OLD: and we don't violate rules about tied WD

   // we tie to when outlining, because we will notify the tied thread when the execution completes
   wd->tieTo( *thread );
   thread->setCurrentWD( *wd );
   if (!wd->started())
      wd->init();

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( copy_data_in_key, 0 ); )
   //NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( NULL, wd, false) );
   //NANOS_INSTRUMENT( inst2.close(); );
}

void Scheduler::preOutlineWork ( WD *wd )
{
   BaseThread *thread = getMyThreadSafe();

   //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_PRE_OUTLINE_WORK, true); );
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t copy_data_in_key = ID->getEventKey("copy-data-in"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( copy_data_in_key, (nanos_event_value_t) wd->getId() ); )
   //std::cerr << "starting WD " << wd->getId() << " at thd " << thread->getId() << " thd addr " << thread << std::endl; 
   // run it in the current frame
   //WD *oldwd = thread->getCurrentWD();

   //GenericSyncCond *syncCond = oldwd->getSyncCond();
   //if ( syncCond != NULL ) {
   //   syncCond->unlock();
   //}

   //std::cerr << "thd " << myThread->getId() <<  " switching(outlined) to task " << wd << ":" << wd->getId() << std::endl;
   debug( "switching(pre outline) from task " << &(thread->getThreadWD()) << ":" << thread->getThreadWD().getId() << " to " << wd << ":" << wd->getId() );

   //NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(oldwd, NULL, false) );

   // OLD: This ensures that when we return from the inlining is still the same thread
   // OLD: and we don't violate rules about tied WD

   // we tie to when outlining, because we will notify the tied thread when the execution completes
   wd->tieTo( *thread );
   thread->setCurrentWD( *wd );
   if (!wd->started())
      wd->init();

   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( copy_data_in_key, 0 ); )
   //NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( NULL, wd, false) );
   //NANOS_INSTRUMENT( inst2.close(); );
}

void Scheduler::prePreOutlineWork ( WD *wd )
{
   BaseThread *thread = getMyThreadSafe();
   wd->_mcontrol.initialize( *(thread->runningOn()) );
}

bool Scheduler::tryPreOutlineWork ( WD *wd )
{
   bool result = false;
   BaseThread *thread = getMyThreadSafe();

   if ( wd->_mcontrol.allocateTaskMemory() ) {
      //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_PRE_OUTLINE_WORK, true); );
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t copy_data_in_key = ID->getEventKey("copy-data-in"); )
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( copy_data_in_key, (nanos_event_value_t) wd->getId() ); )
      debug( "switching(try pre outline) from task " << &(thread->getThreadWD()) << ":" << thread->getThreadWD().getId() << " to " << wd << ":" << wd->getId() );

      result = true;
      wd->tieTo( *thread );
      thread->setCurrentWD( *wd );
      if ( !wd->started() ) {
         wd->init();
      }

      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( copy_data_in_key, 0 ); )
      //NANOS_INSTRUMENT( inst2.close(); );
   }
   return result;
}

void Scheduler::postOutlineWork ( WD *wd, bool schedule, BaseThread *owner )
{
   BaseThread *thread = owner;
   //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_POST_OUTLINE_WORK, true); );

   //std::cerr << "completing WD " << wd->getId() << " at thd " << owner->getId() << " thd addr " << owner << std::endl; 
   //if (schedule && thread->getNextWD() == NULL ) {
   //     thread->setNextWD(thread->getTeam()->getSchedulePolicy().atBeforeExit(thread,*wd));
   //}

   /* If WorkDescriptor has been submitted update statistics */
   updateExitStats (*wd);

   wd->finish();
   wd->done();
   wd->clear();


   //std::cerr << "thd " << myThread->getId() << "exiting task(inlined) " << wd << ":" << wd->getId() <<
   //       " to " << oldwd << ":" << oldwd->getId() << std::endl;
   debug( "exiting task(post outline) " << wd << ":" << std::dec << wd->getId() << " to " << &(thread->getThreadWD()) << ":" << std::dec << thread->getThreadWD().getId() );

   thread->setCurrentWD( thread->getThreadWD() );

   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(wd, NULL, true) );

   //std::cerr << "completed WD " << wd->getId() << " at thd " << owner->getId() << " thd addr " << owner << std::endl; 
   //NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( NULL, oldwd, false) );

   // While we tie the inlined tasks this is not needed
   // as we will always return to the current thread
   #if 0
   if ( oldwd->isTiedTo() != NULL )
      switchToThread(oldwd->isTiedTo());
   #endif

   //ensure(oldwd->isTiedTo() == NULL || thread == oldwd->isTiedTo(),
   //        "Violating tied rules " + toString<BaseThread*>(thread) + "!=" + toString<BaseThread*>(oldwd->isTiedTo()));

   //NANOS_INSTRUMENT( inst2.close(); );
}

void Scheduler::outlineWork( BaseThread *currentThread, WD *wd ) {
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( NULL, wd, false) );
   currentThread->outlineWorkDependent( *wd );
}

void Scheduler::finishWork( WD * wd, bool schedule )
{
   //! \note If WorkDescriptor has been submitted update statistics
   updateExitStats (*wd);

   //! \note getting more work to do (only if not going to sleep)
   if ( !getMyThreadSafe()->isSleeping() && schedule ) {
      BaseThread *thread = getMyThreadSafe();
      ThreadTeam *thread_team = thread->getTeam();
      if ( thread_team ) {
         WD *prefetchedWD = thread_team->getSchedulePolicy().atBeforeExit( thread, *wd, schedule );
         if ( prefetchedWD ) {
            prefetchedWD->_mcontrol.preInit();
            thread->addNextWD( prefetchedWD );
         }
      }
   }

   //! \note Finalizing and cleaning WorkDescriptor
   wd->done();
   wd->clear();
}

bool Scheduler::inlineWork ( WD *wd, bool schedule )
{
   // Getting current thread and WD
   BaseThread *thread = getMyThreadSafe();
   WD *oldwd = thread->getCurrentWD();

   // If old WD have Synchronized Condition, unlock it
   GenericSyncCond *syncCond = oldwd->getSyncCond();
   if ( syncCond != NULL ) syncCond->unlock();

   // Debug information
   debug( "switching(inlined) from task " << oldwd << ":" << oldwd->getId() <<
          " to " << wd << ":" << wd->getId() << " at node " << sys.getNetwork()->getNodeNum() );

   // Initializing wd if necessary. It will be started later in inlineWorkDependent call
   if ( !wd->started() ) { 
      if ( !wd->_mcontrol.isMemoryAllocated() ) {
         wd->_mcontrol.initialize( *(thread->runningOn()) );
         bool result;
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t copy_data_in_key = ID->getEventKey("copy-data-alloc"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( copy_data_in_key, (nanos_event_value_t) wd->getId() ); )
         do {
            result = wd->_mcontrol.allocateTaskMemory();
            if ( !result ) {
               myThread->processTransfers();
            }
         } while( result == false );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( copy_data_in_key, 0 ); )
      }
      wd->init();
   }

   // This ensures that when we return from the inlining is still the same thread
   // and we don't violate rules about tied WD
   if ( oldwd->isTiedTo() != NULL && (wd->isTiedTo() == NULL)) wd->tieTo(*oldwd->isTiedTo());

   // Set current WD to new WD
   thread->setCurrentWD( *wd );

   // Instrumenting context switch: wd enters cpu (last = n/a)
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldwd, wd, false) );

   bool done = thread->inlineWorkDependent(*wd);

   // Reload current thread after running WD due wd may be not tied to thread if
   // both work descriptor were not tied to any thread
   thread = getMyThreadSafe();

   // If WD have already executed (done == true), finishWork
   if ( done ) {
      wd->finish();
      finishWork( wd, schedule );

      // As finishWork potentially may cause a context switch (due to waitCompletion) we need to
      // refresh current thread pointer.
      thread = getMyThreadSafe();

      // Instrumenting context switch: wd leaves cpu and will not come back (last = true)
      // and new_wd enters
      NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(wd, oldwd, true) );
   }

   // Debug information
   debug( "exiting(inlined) from task " << wd << ":" << wd->getId() <<
          " to " << oldwd << ":" << oldwd->getId() << " at node " << sys.getNetwork()->getNodeNum() );

   // Restore current WD to old WD
   thread->setCurrentWD( *oldwd );

   // Tiedness rules
   ensure(oldwd->isTiedTo() == NULL || thread == oldwd->isTiedTo(),
           "Violating tied rules " + toString<BaseThread*>(thread) + "!=" + toString<BaseThread*>(oldwd->isTiedTo()));

  return done;
}

bool Scheduler::inlineWorkAsync ( WD *wd, bool schedule )
{
   // Getting current thread and wd
   BaseThread *thread = getMyThreadSafe();
   WD *oldwd = thread->getCurrentWD();

   GenericSyncCond *syncCond = oldwd->getSyncCond();
   if ( syncCond != NULL ) syncCond->unlock();

   //debug( "switching(inlined) from task " << oldwd << ":" << oldwd->getId() <<
   //       " to " << wd << ":" << wd->getId() );

   // This ensures that when we return from the inlining is still the same thread
   // and we don't violate rules about tied WD
   if ( oldwd->isTiedTo() != NULL && ( wd->isTiedTo() == NULL ) ) wd->tieTo( *oldwd->isTiedTo() );

   //thread->setCurrentWD( *wd );

   /* Instrumenting context switch: wd enters cpu (last = n/a) */
   //NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldwd, wd, false) );

   // Will return false in general
   const bool done = thread->inlineWorkDependent( *wd );

   // reload thread after running WD because wd may be not tied to thread if
   // both work descriptors were not tied to any thread
   thread = getMyThreadSafe();

   if ( schedule ) {
      ThreadTeam *thread_team = thread->getTeam();
      if ( thread_team ) {
         thread->addNextWD( thread_team->getSchedulePolicy().atBeforeExit( thread, *wd, schedule ) );
      }
   }

   if ( done )
      finishWork( oldwd, wd );

   //thread->setCurrentWD( *oldwd );

   ensure( oldwd->isTiedTo() == NULL || thread == oldwd->isTiedTo(),
           "Violating tied rules " + toString<BaseThread*>( thread ) + "!=" + toString<BaseThread*>( oldwd->isTiedTo() ) );

  return done;
}

void Scheduler::switchHelper (WD *oldWD, WD *newWD, void *arg)
{
   myThread->switchHelperDependent(oldWD, newWD, arg);

   GenericSyncCond *syncCond = oldWD->getSyncCond();
   if ( syncCond != NULL ) {
      oldWD->setBlocked();
      syncCond->addWaiter( oldWD );
      syncCond->unlock();
   } else if ( &(myThread->getThreadWD()) != oldWD ) {
      myThread->getTeam()->getSchedulePolicy().queue( myThread, *oldWD );
   }
   myThread->setCurrentWD( *newWD );
}

void Scheduler::switchTo ( WD *to )
{
   if ( myThread->runningOn()->supportsUserLevelThreads() ) {

      if (!to->started()) {
         to->_mcontrol.initialize( *(myThread->runningOn()) );

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t copy_data_in_key = ID->getEventKey("copy-data-alloc"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( copy_data_in_key, (nanos_event_value_t) to->getId() ); )
         bool result;
         do {
            result = to->_mcontrol.allocateTaskMemory();
            if ( !result ) {
               myThread->processTransfers();
            }
         } while( result == false );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( copy_data_in_key, 0 ); )

         to->init();
         to->start(WD::IsAUserLevelThread);
      }

      debug( "switching from task " << myThread->getCurrentWD() << ":" << myThread->getCurrentWD()->getId() <<
            " to " << to << ":" << to->getId() );

      NANOS_INSTRUMENT( WD *oldWD = myThread->getCurrentWD(); )
      NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldWD, to, false ) );

      myThread->switchTo( to, switchHelper );

   } else {
      if (inlineWork(to, /*schedule*/ true)) {
         to->~WorkDescriptor();
         delete[] (char *)to;
      }
   }
}

void Scheduler::yield ()
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SCHEDULING, true) );
   WD *next = myThread->getTeam()->getSchedulePolicy().atYield( myThread, myThread->getCurrentWD() );
   if ( next ) switchTo(next);
}

void Scheduler::switchToThread ( BaseThread *thread )
{
   while ( getMyThreadSafe() != thread )
   {
      NANOS_INSTRUMENT( InstrumentState inst(NANOS_SCHEDULING, true) );

      // Tie wd to target thread
      WD *wd = myThread->getCurrentWD();
      wd->tieTo( *thread );

      // Get a candidate to execute in current thread
      WD *next = myThread->getTeam()->getSchedulePolicy().atYield( myThread, wd );
      if ( next == NULL ) next = &(myThread->getThreadWD());
      if ( next ) switchTo(next);

      // We must notify the thread manager to wake up the thread in case it is sleeping
      sys.getThreadManager()->unblockThread(thread);
   }
}

void Scheduler::exitHelper (WD *oldWD, WD *newWD, void *arg)
{
    myThread->exitHelperDependent(oldWD, newWD, arg);
    myThread->setCurrentWD( *newWD );
    oldWD->~WorkDescriptor();
    delete[] (char *)oldWD;
}

struct ExitBehaviour
{
   static WD * getWD ( BaseThread *thread, WD *current, int numSteal )
   {
      return thread->getTeam()->getSchedulePolicy().atAfterExit( thread, current, numSteal );
   }

   static void switchWD ( BaseThread *thread, WD *current, WD *next )
   {
      if (next->started()){
        Scheduler::exitTo(next);
      }
      else {
        if ( Scheduler::inlineWork ( next /*jb merge */, /*schedule*/ true ) ) {
          next->~WorkDescriptor();
          delete[] (char *)next;
        }
      }
   }
   static bool exiting() { return true; }
};

void Scheduler::exitTo ( WD *to )
 {
//! \bug FIXME: stack reusing was wrongly implementd and it's disabled (see #374)
//    WD *current = myThread->getCurrentWD();

    if (!to->started()) {
       to->_mcontrol.initialize( *(myThread->runningOn()) );
       bool result;
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t copy_data_in_key = ID->getEventKey("copy-data-alloc"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( copy_data_in_key, (nanos_event_value_t) to->getId() ); )
       do {
          result = to->_mcontrol.allocateTaskMemory();
         if ( !result ) {
            myThread->processTransfers();
         }
       } while( result == false );
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( copy_data_in_key, 0 ); )

       to->init();
       //       to->start(true,current);
       to->start(WD::IsAUserLevelThread,NULL);
    }

    debug( "exiting task " << myThread->getCurrentWD() << ":" << myThread->getCurrentWD()->getId() <<
          " to " << to << ":" << to->getId() );

    NANOS_INSTRUMENT( WD *oldWD = myThread->getCurrentWD(); )
    NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldWD, to, true ) );
    myThread->exitTo( to, Scheduler::exitHelper );
}

void Scheduler::exit ( void )
{
   // At this point the WD work is done, so we mark it as such and look for other work to do
   // Deallocation doesn't happen here because:
   // a) We are still running in the WD stack
   // b) Resources can potentially be reused by the next WD
   BaseThread *thread = myThread;

   WD *oldwd = thread->getCurrentWD();

   oldwd->finish();

   finishWork( oldwd, true );

   /* Serve taskset requests */
   sys.getThreadManager()->poll();

   /* update next WorkDescriptor (if any) */
   WD *next = thread->getNextWD();

   if ( !next ) idleLoop<ExitBehaviour>();
   else Scheduler::exitTo(next);

   fatal("A thread should never return from Scheduler::exit");
}

int SchedulerStats::getCreatedTasks() { return _createdTasks.value(); }
int SchedulerStats::getReadyTasks() { return _readyTasks.value(); }
int SchedulerStats::getTotalTasks() { return _totalTasks.value(); }
