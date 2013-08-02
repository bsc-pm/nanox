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
#include "config.hpp"
#include "instrumentationmodule_decl.hpp"
#include "os.hpp"

extern "C" {
   void DLB_UpdateResources_max( int max_resources ) __attribute__(( weak ));
   void DLB_ReturnClaimedCpus( void ) __attribute__(( weak ));
}

using namespace nanos;

void SchedulerConf::config (Config &cfg)
{
   cfg.setOptionsSection ( "Core [Scheduler]", "Policy independent scheduler options"  );

   cfg.registerConfigOption ( "num_spins", NEW Config::UintVar( _numSpins ), "Determines the amount of spinning before sleeping (default = 100)" );
   cfg.registerArgOption ( "num_spins", "spins" );
   cfg.registerEnvOption ( "num_spins", "NX_SPINS" );

   cfg.registerConfigOption ( "num_sleeps", NEW Config::IntegerVar( _numSleeps ), "Determines the amount of sleeping before yielding (default = 20)" );
   cfg.registerArgOption ( "num_sleeps", "sleeps" );
   cfg.registerEnvOption ( "num_sleeps", "NX_SLEEPS" );

   cfg.registerConfigOption ( "sleep_time", NEW Config::IntegerVar( _timeSleep ), "Determines amount of time (in nsec) in each sleeping phase (default = 100)" );
   cfg.registerArgOption ( "sleep_time", "sleep-time" );
   cfg.registerEnvOption ( "sleep_time", "NX_SLEEP_TIME" );
}

void Scheduler::submit ( WD &wd )
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SCHEDULING) );
   BaseThread *mythread = myThread;

   debug ( "submitting task " << wd.getId() );

   wd.submitted();

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

   /* handle tasks which cannot run in current thread */
   if ( !wd.canRunIn(*mythread->runningOn()) ) {
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

void Scheduler::submitAndWait ( WD &wd )
{
   debug ( "submitting and waiting task " << wd.getId() );
   fatal ( "Scheduler::submitAndWait(): This feature is still not supported" );

   // Create a new WorkGroup and add WD
   WG myWG;
   myWG.addWork( wd );

   // Submit WD
   submit( wd );

   // Wait for WD to be finished
   myWG.waitCompletion();
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

template<class behaviour>
inline void Scheduler::idleLoop ()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t total_spins_key  = ID->getEventKey("num-spins"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_yields_key = ID->getEventKey("num-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_sleeps_key = ID->getEventKey("num-sleeps"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_scheds_key  = ID->getEventKey("num-scheds"); )

   NANOS_INSTRUMENT ( static nanos_event_key_t time_yields_key = ID->getEventKey("time-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_sleeps_key = ID->getEventKey("time-sleeps"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_scheds_key = ID->getEventKey("time-scheds"); )

   NANOS_INSTRUMENT ( nanos_event_key_t Keys[7]; )

   NANOS_INSTRUMENT ( Keys[0] = total_yields_key; )
   NANOS_INSTRUMENT ( Keys[1] = time_yields_key; )
   NANOS_INSTRUMENT ( Keys[2] = total_sleeps_key; )
   NANOS_INSTRUMENT ( Keys[3] = time_sleeps_key; )
   NANOS_INSTRUMENT ( Keys[4] = total_spins_key; )
   NANOS_INSTRUMENT ( Keys[5] = total_scheds_key; )
   NANOS_INSTRUMENT ( Keys[6] = time_scheds_key; )

   NANOS_INSTRUMENT ( unsigned event_start; )
   NANOS_INSTRUMENT ( unsigned event_num; )

   NANOS_INSTRUMENT( InstrumentState inst(NANOS_IDLE) );

   const int nspins = sys.getSchedulerConf().getNumSpins();
   const int nsleeps = sys.getSchedulerConf().getNumSleeps();
   const int tsleep = sys.getSchedulerConf().getTimeSleep();
   int spins = nspins;
   int sleeps = nsleeps;

   NANOS_INSTRUMENT ( unsigned long total_spins = 0; )  /* Number of spins by idle phase*/
   NANOS_INSTRUMENT ( unsigned long total_yields = 0; ) /* Number of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long total_sleeps = 0; ) /* Number of sleeps by idle phase */
   NANOS_INSTRUMENT ( unsigned long total_scheds = 0; ) /* Number of scheds by idle phase */

   NANOS_INSTRUMENT ( unsigned long time_sleeps = 0; ) /* Time of sleeps by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_yields = 0; ) /* Time of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_scheds = 0; ) /* Time of yields by idle phase */

   WD *current = myThread->getCurrentWD();
   current->setIdle();
   sys.getSchedulerStats()._idleThreads++;
   for ( ; ; ) {
      BaseThread *thread = getMyThreadSafe();
      spins--;

      if ( !thread->isRunning() && !behaviour::exiting() ) break;

      if ( thread->isTaggedToSleep() && !behaviour::exiting() ) thread->wait();

      WD * next = myThread->getNextWD();
      // This should be ideally performed in getNextWD, but it's const...
      if ( !sys.getSchedulerConf().getSchedulerEnabled() ) {
         // The thread is paused, mark it as so
         myThread->pause();
      }
      else {
         // The thread is not paused, mark it as so
         myThread->unpause();
      }

      if ( !next && thread->getTeam() != NULL ) {
         memoryFence();
         if ( sys.getSchedulerStats()._readyTasks > 0 ) {
            NANOS_INSTRUMENT ( total_scheds++; )
            NANOS_INSTRUMENT ( unsigned long begin_sched = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )

            next = behaviour::getWD(thread,current);

            NANOS_INSTRUMENT (  unsigned long end_sched = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
            NANOS_INSTRUMENT (time_scheds += ( end_sched - begin_sched ); )
         }
      } 

      if ( next ) {
         sys.getSchedulerStats()._idleThreads--;

         NANOS_INSTRUMENT (total_spins+= (nspins - spins); )

         NANOS_INSTRUMENT ( nanos_event_value_t Values[7]; )

         NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_yields; )
         NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) time_yields; )
         NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) total_sleeps; )
         NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) time_sleeps; )
         NANOS_INSTRUMENT ( Values[4] = (nanos_event_value_t) total_spins; )
         NANOS_INSTRUMENT ( Values[5] = (nanos_event_value_t) total_scheds; )
         NANOS_INSTRUMENT ( Values[6] = (nanos_event_value_t) time_scheds; )

         NANOS_INSTRUMENT ( event_start = 0; event_num = 7; )
         NANOS_INSTRUMENT ( if (total_yields == 0 ) { event_start = 2; event_num = 5; } )
         NANOS_INSTRUMENT ( if (total_yields == 0 && total_sleeps == 0) { event_start = 4; event_num = 3; } )
         NANOS_INSTRUMENT ( if (total_scheds == 0 ) { event_num -= 2; } )

         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(event_num, &Keys[event_start], &Values[event_start]); )

         behaviour::switchWD(thread, current, next);

         thread = getMyThreadSafe();
         sys.getSchedulerStats()._idleThreads++;

         NANOS_INSTRUMENT (total_spins = 0; )
         NANOS_INSTRUMENT (total_sleeps = 0; )
         NANOS_INSTRUMENT (total_yields = 0; )
         NANOS_INSTRUMENT (total_scheds = 0; )

         NANOS_INSTRUMENT (time_yields = 0; )
         NANOS_INSTRUMENT (time_sleeps = 0; )
         NANOS_INSTRUMENT (time_scheds = 0; )

         spins = nspins;
         continue;
      }

      if ( spins == 0 ) {
         /* If DLB, return resources if needed */
         if ( sys.dlbEnabled() && DLB_ReturnClaimedCpus && getMyThreadSafe()->getId() == 0 && sys.getPMInterface().isMalleable() )
            DLB_ReturnClaimedCpus();

         NANOS_INSTRUMENT ( total_spins+= nspins; )
         sleeps--;
         if ( sleeps < 0 ) {
            if ( sys.useYield() ) {
               NANOS_INSTRUMENT ( total_yields++; )
               NANOS_INSTRUMENT ( unsigned long begin_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
               thread->yield();
               NANOS_INSTRUMENT ( unsigned long end_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
               NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
            }
            sleeps = nsleeps;
         } else {
            NANOS_INSTRUMENT ( total_sleeps++; )
            struct timespec req ={0,tsleep};
            nanosleep ( &req, NULL );
            NANOS_INSTRUMENT ( time_sleeps += time_sleeps + tsleep; )
         }
         spins = nspins;
      }
      else {
         thread->idle();
      }
   }
   sys.getSchedulerStats()._idleThreads--;
   current->setReady();
   current->~WorkDescriptor();
   delete[] (char *) current;
}

void Scheduler::waitOnCondition (GenericSyncCond *condition)
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t total_spins_key  = ID->getEventKey("num-spins"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_yields_key = ID->getEventKey("num-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_sleeps_key = ID->getEventKey("num-sleeps"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_scheds_key  = ID->getEventKey("num-scheds"); )

   NANOS_INSTRUMENT ( static nanos_event_key_t time_yields_key = ID->getEventKey("time-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_sleeps_key = ID->getEventKey("time-sleeps"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_scheds_key = ID->getEventKey("time-scheds"); )

   NANOS_INSTRUMENT ( nanos_event_key_t Keys[7]; )

   NANOS_INSTRUMENT ( Keys[0] = total_spins_key; )
   NANOS_INSTRUMENT ( Keys[1] = total_yields_key; )
   NANOS_INSTRUMENT ( Keys[2] = total_sleeps_key; )
   NANOS_INSTRUMENT ( Keys[3] = total_scheds_key; )

   NANOS_INSTRUMENT ( Keys[4] = time_yields_key; )
   NANOS_INSTRUMENT ( Keys[5] = time_sleeps_key; )
   NANOS_INSTRUMENT ( Keys[6] = time_scheds_key; )

   NANOS_INSTRUMENT ( unsigned event_start; )
   NANOS_INSTRUMENT ( unsigned event_num; )

   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SYNCHRONIZATION) );

   const int nspins = sys.getSchedulerConf().getNumSpins();
   const int nsleeps = sys.getSchedulerConf().getNumSleeps();
   const int tsleep = sys.getSchedulerConf().getTimeSleep();
   unsigned int spins = nspins; 
   int sleeps = nsleeps;

   NANOS_INSTRUMENT ( unsigned long total_spins = 0; ) /* Number of spins by idle phase*/
   NANOS_INSTRUMENT ( unsigned long total_yields = 0; ) /* Number of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long total_sleeps = 0; ) /* Number of sleeps by idle phase */
   NANOS_INSTRUMENT ( unsigned long total_scheds= 0; ) /* Number of schedulers by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_sleeps = 0; ) /* Time of sleeps by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_yields = 0; ) /* Time of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_scheds = 0; ) /* Time of sched by idle phase */

   WD * current = myThread->getCurrentWD();

   sys.getSchedulerStats()._idleThreads++;
   current->setSyncCond( condition );
   current->setIdle();
   
   BaseThread *thread = getMyThreadSafe();

   while ( !condition->check() && thread->isRunning() ) {
      spins--;
      if ( spins == 0 ) {
         NANOS_INSTRUMENT ( total_spins+= nspins; )
         sleeps--;
         condition->lock();
         if ( !( condition->check() ) ) {
            WD * next = myThread->getNextWD();

            if ( !next ) {
               memoryFence();
               if ( sys.getSchedulerStats()._readyTasks > 0 ) {
                  // If the scheduler is running
                  if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
                     // The thread is not paused, mark it as so
                     thread->unpause();
                     
                     NANOS_INSTRUMENT ( total_scheds++; )
                     NANOS_INSTRUMENT ( unsigned long begin_sched = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )

                     next = thread->getTeam()->getSchedulePolicy().atBlock( thread, current );

                     NANOS_INSTRUMENT ( unsigned long end_sched = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
                     NANOS_INSTRUMENT ( time_scheds += ( end_sched - begin_sched ); )
                  }
                  else {
                     // Pause this thread
                     thread->pause();
                  }
               }
            }

            if ( next ) {
               sys.getSchedulerStats()._idleThreads--;

               NANOS_INSTRUMENT ( nanos_event_value_t Values[7]; )

               NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_spins; )
               NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) total_yields; )
               NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) total_sleeps; )
               NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) total_scheds; )

               NANOS_INSTRUMENT ( Values[4] = (nanos_event_value_t) time_yields; )
               NANOS_INSTRUMENT ( Values[5] = (nanos_event_value_t) time_sleeps; )
               NANOS_INSTRUMENT ( Values[6] = (nanos_event_value_t) time_scheds; )

               NANOS_INSTRUMENT ( event_start = 0; event_num = 7; )
               NANOS_INSTRUMENT ( if (total_yields == 0 ) { event_start = 2; event_num = 5; } )
               NANOS_INSTRUMENT ( if (total_yields == 0 && total_sleeps == 0) { event_start = 4; event_num = 3; } )
               NANOS_INSTRUMENT ( if (total_scheds == 0 ) { event_num -= 2; } )

               NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(event_num, &Keys[event_start], &Values[event_start]); )

               switchTo ( next );
               thread = getMyThreadSafe();

               NANOS_INSTRUMENT ( total_spins = 0; )

               NANOS_INSTRUMENT ( total_yields = 0; )
               NANOS_INSTRUMENT ( total_sleeps = 0; )
               NANOS_INSTRUMENT ( total_scheds = 0; )

               NANOS_INSTRUMENT ( time_sleeps = 0; ) 
               NANOS_INSTRUMENT ( time_yields = 0; )
               NANOS_INSTRUMENT ( time_scheds = 0; )

               sys.getSchedulerStats()._idleThreads++;
            } else {
               /* If DLB, return resources if needed */
               if ( sys.dlbEnabled() && DLB_ReturnClaimedCpus && getMyThreadSafe()->getId() == 0 && sys.getPMInterface().isMalleable() )
                  DLB_ReturnClaimedCpus();

               condition->unlock();
               if ( sleeps < 0 ) {
                  if ( sys.useYield() ) {
                     NANOS_INSTRUMENT ( total_yields++; )
                     NANOS_INSTRUMENT ( unsigned long begin_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); ) 
                     thread->yield();
                     NANOS_INSTRUMENT ( unsigned long end_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
                     NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
                  }
                  sleeps = nsleeps;
               } else {
                  NANOS_INSTRUMENT ( total_sleeps++; )
                  struct timespec req = {0,tsleep};
                  nanosleep ( &req, NULL );
                  NANOS_INSTRUMENT ( time_sleeps += tsleep; )
               }
            }
         } else {
            condition->unlock();
         }
         spins = nspins;
      }

      thread->idle();
   }

   current->setSyncCond( NULL );
   sys.getSchedulerStats()._idleThreads--;
   if ( !current->isReady() ) {
      current->setReady();
   }

   NANOS_INSTRUMENT ( total_spins+= (nspins - spins); )

   NANOS_INSTRUMENT ( nanos_event_value_t Values[7]; )

   NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_spins; )
   NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) total_yields; )
   NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) total_sleeps; )
   NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) total_scheds; )

   NANOS_INSTRUMENT ( Values[4] = (nanos_event_value_t) time_yields; )
   NANOS_INSTRUMENT ( Values[5] = (nanos_event_value_t) time_sleeps; )
   NANOS_INSTRUMENT ( Values[6] = (nanos_event_value_t) time_scheds; )

   NANOS_INSTRUMENT ( event_start = 0; event_num = 7; )
   NANOS_INSTRUMENT ( if (total_yields == 0 ) { event_start = 2; event_num = 5; } )
   NANOS_INSTRUMENT ( if (total_yields == 0 && total_sleeps == 0) { event_start = 4; event_num = 3; } )
   NANOS_INSTRUMENT ( if (total_scheds == 0 ) { event_num -= 2; } )

   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(event_num, &Keys[event_start], &Values[event_start]); )

}

void Scheduler::wakeUp ( WD *wd )
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SYNCHRONIZATION) );
   
   if ( wd->isBlocked() ) {
      /* Setting ready wd */
      wd->setReady();
      WD *next = NULL;
      if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
         // The thread is not paused, mark it as so
         myThread->unpause();
         
         /* atWakeUp must check basic constraints */
         next = getMyThreadSafe()->getTeam()->getSchedulePolicy().atWakeUp( myThread, *wd );
      }
      else {
         // Pause this thread
         myThread->pause();
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
   // If the scheduler is running
   if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
      // The thread is not paused, mark it as so
      thread->unpause();
      
      return thread->getTeam()->getSchedulePolicy().atPrefetch( thread, wd );
   }
   else {
      // Pause this thread
      thread->pause();
   }
   // Otherwise, do nothing
   // FIXME (#581): Consequences?
   return NULL;
}

struct WorkerBehaviour
{
   static WD * getWD ( BaseThread *thread, WD *current )
   {
      if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
         // The thread is not paused, mark it as so
         thread->unpause();
         
         return thread->getTeam()->getSchedulePolicy().atIdle ( thread );
      }
      // Pause this thread
      thread->pause();
      return NULL;
   }

   static void switchWD ( BaseThread *thread, WD *current, WD *next )
   {
      if (next->started()){
        Scheduler::switchTo(next);
      }
      else {
        if ( Scheduler::inlineWork ( next, true ) ) {
          next->~WorkDescriptor();
          delete[] (char *)next;
        }
      }
   }
   static bool exiting() { return false; }
};

struct AsyncWorkerBehaviour
{
   static WD * getWD ( BaseThread *thread, WD *current )
   {
      if ( !thread->canGetWork() ) return NULL;

      if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
         // The thread is not paused, mark it as so
         thread->unpause();

         return thread->getTeam()->getSchedulePolicy().atIdle( thread );
      }
      // Pause this thread
      thread->pause();
      return NULL;
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

void Scheduler::finishWork( WD *oldwd, WD * wd, bool schedule )
{
   /* If WorkDescriptor has been submitted update statistics */
   updateExitStats (*wd);

   BaseThread *thread = getMyThreadSafe();
   // Switch back to oldwd
   thread->setCurrentWD( *oldwd );

   /* Instrumenting context switch: wd leaves cpu and will not come back (last = true) and oldwd enters */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( wd, oldwd, true ) );

   if ( schedule && !getMyThreadSafe()->isTaggedToSleep() ) {
      BaseThread *thread = getMyThreadSafe();
      ThreadTeam *thread_team = thread->getTeam();
      if ( thread_team ) {
         thread->addNextWD( thread_team->getSchedulePolicy().atBeforeExit( thread, *wd, schedule ) );
      }
   }

   wd->done();
   wd->clear();

   /* If DLB, perform the adjustment of resources */
   if ( sys.dlbEnabled() && DLB_UpdateResources_max && getMyThreadSafe()->getId() == 0 ) {
      if ( sys.getPMInterface().isMalleable() )
         DLB_ReturnClaimedCpus();

      int needed_resources = sys.getSchedulerStats()._readyTasks.value() - sys.getNumThreads();
      if ( needed_resources > 0 )
         DLB_UpdateResources_max( needed_resources );
   }

   debug( "exiting task(inlined) " << wd << ":" << wd->getId() <<
          " to " << oldwd << ":" << ( oldwd ? oldwd->getId() : 0 ) );
}

bool Scheduler::inlineWork ( WD *wd, bool schedule )
{
   BaseThread *thread = getMyThreadSafe();

   // run it in the current frame
   WD *oldwd = thread->getCurrentWD();

   GenericSyncCond *syncCond = oldwd->getSyncCond();
   if ( syncCond != NULL ) syncCond->unlock();

   debug( "switching(inlined) from task " << oldwd << ":" << oldwd->getId() <<
          " to " << wd << ":" << wd->getId() );

   // Initializing wd if necessary
   // It will be started later in inlineWorkDependent call
   if ( !wd->started() ) wd->init();

   // This ensures that when we return from the inlining is still the same thread
   // and we don't violate rules about tied WD
   if ( oldwd->isTiedTo() != NULL && (wd->isTiedTo() == NULL)) wd->tieTo(*oldwd->isTiedTo());

   thread->setCurrentWD( *wd );

   /* Instrumenting context switch: wd enters cpu (last = n/a) */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldwd, wd, false) );

   const bool done = thread->inlineWorkDependent(*wd);

   // reload thread after running WD due wd may be not tied to thread if
   // both work descriptor were not tied to any thread
   thread = getMyThreadSafe();

   wd->finish();

   if ( done )
      finishWork( oldwd, wd, schedule );

   thread->setCurrentWD( *oldwd );

   // While we tie the inlined tasks this is not needed
   // as we will always return to the current thread
   #if 0
   if ( oldwd->isTiedTo() != NULL )
      switchToThread(oldwd->isTiedTo());
   #endif

   ensure(oldwd->isTiedTo() == NULL || thread == oldwd->isTiedTo(),
           "Violating tied rules " + toString<BaseThread*>(thread) + "!=" + toString<BaseThread*>(oldwd->isTiedTo()));
   
  return done;
}

bool Scheduler::inlineWorkAsync ( WD *wd, bool schedule )
{
   BaseThread *thread = getMyThreadSafe();

   // run it in the current frame
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
         thread->addNextWD( thread_team->getSchedulePolicy().atBeforeExit( thread, *wd ) );
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
   } else {
      myThread->getTeam()->getSchedulePolicy().queue( myThread, *oldWD );
   }
   myThread->setCurrentWD( *newWD );
}

void Scheduler::switchTo ( WD *to )
{
   if ( myThread->runningOn()->supportsUserLevelThreads() ) {
      if (!to->started()) {
         to->init();
         to->start(WD::IsAUserLevelThread);
      }
      
      debug( "switching from task " << myThread->getCurrentWD() << ":" << myThread->getCurrentWD()->getId() <<
          " to " << to << ":" << to->getId() );
          
      NANOS_INSTRUMENT( WD *oldWD = myThread->getCurrentWD(); )
      NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldWD, to, false ) );

      myThread->switchTo( to, switchHelper );

   } else {
      if (inlineWork(to)) {
         to->~WorkDescriptor();
         delete[] (char *)to;
      }
   }
}

void Scheduler::yield ()
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SCHEDULING) );
   // If the scheduler is running
   if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
      // The thread is not paused, mark it as so
      myThread->unpause();
      
      WD *next = myThread->getTeam()->getSchedulePolicy().atYield( myThread, myThread->getCurrentWD() );
      if ( next ) {
         switchTo(next);
      }
   }
   else {
      // Pause this thread
      myThread->pause();
   }
}

void Scheduler::switchToThread ( BaseThread *thread )
{
   while ( getMyThreadSafe() != thread )
        yield();
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
   static WD * getWD ( BaseThread *thread, WD *current )
   {
      if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
         // The thread is not paused, mark it as so
         thread->unpause();
         
         return thread->getTeam()->getSchedulePolicy().atAfterExit( thread, current );
      }
      
      // Pause this thread
      thread->pause();
      return NULL;
   }

   static void switchWD ( BaseThread *thread, WD *current, WD *next )
   {
      Scheduler::exitTo(next);
   }
   static bool exiting() { return true; }
};

void Scheduler::exitTo ( WD *to )
 {
//   FIXME: stack reusing was wrongly implementd and it's disabled (see #374)
//    WD *current = myThread->getCurrentWD();

    if (!to->started()) {
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

   /* get next WorkDescriptor (if any) */
   WD *next =  thread->getNextWD();

   oldwd->finish();

   finishWork( next, oldwd, ( next == NULL ) );

   /* update next WorkDescriptor (if any) */
   next = ( next == NULL ) ? thread->getNextWD() : next;

   if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
      // The thread is not paused, mark it as so
      thread->unpause();
   } else {
      // Pause this thread (only if we have no next wd to execute )
      if ( !next ) thread->pause();
   }

   if ( !next ) idleLoop<ExitBehaviour>();
   else Scheduler::exitTo(next);

   fatal("A thread should never return from Scheduler::exit");
}

int SchedulerStats::getCreatedTasks() { return _createdTasks.value(); }
int SchedulerStats::getReadyTasks() { return _readyTasks.value(); }
int SchedulerStats::getTotalTasks() { return _totalTasks.value(); }
