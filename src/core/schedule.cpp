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
#include "smpthread.hpp"
#include "system.hpp"
#include "config.hpp"
#include "synchronizedcondition.hpp"
#include "instrumentationmodule_decl.hpp"
#include "os.hpp"

#ifdef CLUSTER_DEV
#include "clusterthread_decl.hpp"
#include "clusternode_decl.hpp"
#include "gpudd.hpp"
#include "smpdd.hpp"
#endif
#include "wddeque.hpp"
#include "dlb.hpp"
#include "smpthread.hpp"
#include "nanos-int.h"

#include <iostream>

using namespace nanos;

void SchedulerConf::config (Config &cfg)
{
   cfg.setOptionsSection ( "Core [Scheduler]", "Policy independent scheduler options"  );

   cfg.registerConfigOption( "yield-opt", NEW Config::FlagOption( _useYield, true ),
                             "Thread yield on idle and condition waits (default is disabled)" );
   cfg.registerArgOption( "yield-opt", "enable-yield" );

   cfg.registerConfigOption( "block-opt", NEW Config::FlagOption( _useBlock, true ),
                             "Thread block on idle and condition waits (default is disabled)" );
   cfg.registerArgOption( "block-opt", "enable-block" );

   cfg.registerConfigOption ( "num-spins", NEW Config::UintVar( _numSpins ), "Set number of spins before yield (default = 1)" );
   cfg.registerArgOption ( "num-spins", "spins" );

   cfg.registerConfigOption ( "num-checks", NEW Config::UintVar( _numChecks ), "Set number of checks before schedule on wait conditions (default = 1)" );
   cfg.registerArgOption ( "num-checks", "checks" );

   cfg.registerConfigOption ( "num-yields", NEW Config::UintVar( _numYields ), "Set number of yields before block (default = 1)" );
   cfg.registerArgOption ( "num-yields", "yields" );

}

void Scheduler::submit ( WD &wd, bool force_queue )
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SCHEDULING) );
   BaseThread *mythread = myThread;

   debug ( "submitting task " << wd.getId() );

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

   /* handle tasks which cannot run in current thread */
   if ( force_queue || !wd.canRunIn(*mythread->runningOn()) ) {
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
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SCHEDULING) );
   
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

   NANOS_INSTRUMENT ( static nanos_event_key_t time_yields_key = ID->getEventKey("time-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_blocks_key = ID->getEventKey("time-blocks"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_scheds_key = ID->getEventKey("time-scheds"); )

   NANOS_INSTRUMENT ( nanos_event_key_t Keys[7]; )

   NANOS_INSTRUMENT ( Keys[0] = total_yields_key; )
   NANOS_INSTRUMENT ( Keys[1] = time_yields_key; )
   NANOS_INSTRUMENT ( Keys[2] = total_blocks_key; )
   NANOS_INSTRUMENT ( Keys[3] = time_blocks_key; )
   NANOS_INSTRUMENT ( Keys[4] = total_spins_key; )
   NANOS_INSTRUMENT ( Keys[5] = total_scheds_key; )
   NANOS_INSTRUMENT ( Keys[6] = time_scheds_key; )

   NANOS_INSTRUMENT ( unsigned event_start; )
   NANOS_INSTRUMENT ( unsigned event_num; )

   NANOS_INSTRUMENT( InstrumentState inst(NANOS_IDLE) );

   const int init_spins = sys.getSchedulerConf().getNumSpins();
   const int init_yields = sys.getSchedulerConf().getNumYields();
   const bool use_yield = sys.getSchedulerConf().getUseYield();
   const bool use_block = sys.getSchedulerConf().getUseBlock();
   int spins = init_spins;
   int yields = init_yields;

   NANOS_INSTRUMENT ( unsigned long total_spins = 0; )  /* Number of spins by idle phase*/
   NANOS_INSTRUMENT ( unsigned long total_yields = 0; ) /* Number of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long total_blocks = 0; ) /* Number of blocks by idle phase */
   NANOS_INSTRUMENT ( unsigned long total_scheds = 0; ) /* Number of scheds by idle phase */

   NANOS_INSTRUMENT ( unsigned long time_blocks = 0; ) /* Time of blocks by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_yields = 0; ) /* Time of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_scheds = 0; ) /* Time of yields by idle phase */

   WD *current = myThread->getCurrentWD();
   sys.getSchedulerStats()._idleThreads++;
   myThread->setIdle( true );

   for ( ; ; ) {
      BaseThread *thread = getMyThreadSafe();
      spins--;

      if ( !thread->isRunning() && !behaviour::exiting() ) break;

      //! \note thread can only wait if not in exit behaviour, meaning that it has no user's work
      // descriptor in its stack frame
      if ( thread->isSleeping() && !behaviour::exiting() ) thread->wait();

      myThread->getNextWDQueue().iterate<TestInputs>();
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

         NANOS_INSTRUMENT (total_spins+= (init_spins - spins); )

         NANOS_INSTRUMENT ( nanos_event_value_t Values[7]; )

         NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_yields; )
         NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) time_yields; )
         NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) total_blocks; )
         NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) time_blocks; )
         NANOS_INSTRUMENT ( Values[4] = (nanos_event_value_t) total_spins; )
         NANOS_INSTRUMENT ( Values[5] = (nanos_event_value_t) total_scheds; )
         NANOS_INSTRUMENT ( Values[6] = (nanos_event_value_t) time_scheds; )

         NANOS_INSTRUMENT ( event_start = 0; event_num = 7; )
         NANOS_INSTRUMENT ( if (total_yields == 0 ) { event_start = 2; event_num = 5; } )
         NANOS_INSTRUMENT ( if (total_yields == 0 && total_blocks == 0) { event_start = 4; event_num = 3; } )
         NANOS_INSTRUMENT ( if (total_scheds == 0 ) { event_num -= 2; } )

         NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(event_num, &Keys[event_start], &Values[event_start]); )

         behaviour::switchWD(thread, current, next);

         thread = getMyThreadSafe();
         sys.getSchedulerStats()._idleThreads++;

         NANOS_INSTRUMENT (total_spins = 0; )
         NANOS_INSTRUMENT (total_blocks = 0; )
         NANOS_INSTRUMENT (total_yields = 0; )
         NANOS_INSTRUMENT (total_scheds = 0; )

         NANOS_INSTRUMENT (time_yields = 0; )
         NANOS_INSTRUMENT (time_blocks = 0; )
         NANOS_INSTRUMENT (time_scheds = 0; )

         spins = init_spins;
         continue;
      }

      thread->idle();
      if ( sys.getNetwork()->getNodeNum() > 0 ) { sys.getNetwork()->poll(0); }

      if ( spins == 0 ) {
         NANOS_INSTRUMENT ( total_spins += init_spins; )
         dlb_returnCpusIfNeeded();
#if 0
         if ( sys.dlbEnabled() && DLB_ReturnClaimedCpus && getMyThreadSafe()->getId() == 0 && sys.getPMInterface().isMalleable() )
            DLB_ReturnClaimedCpus();
#endif
         if ( yields == 0 || !use_yield ) {
            if ( use_block ) {
               WD * currentWD = thread->getCurrentWD();
               // If it's not tied to the current thread, tie it until the thread is resumed
               bool tiedTemporally = false;
               if ( currentWD->isTiedTo() == NULL )
               {
                  currentWD->tieTo( *thread );
                  tiedTemporally = true;
               }
               
               NANOS_INSTRUMENT ( total_blocks++; )
               NANOS_INSTRUMENT ( unsigned long begin_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
               thread->block(); //FIXME:xteruel
               NANOS_INSTRUMENT ( unsigned long end_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
               NANOS_INSTRUMENT ( time_blocks += ( end_block - begin_block ); )
                     
               // Having reached this point, if we temporally tied the wd to the thread, undo it
               if ( tiedTemporally )
                  currentWD->untie();
            }
            yields = init_yields;
         } else if ( use_yield ) {
            NANOS_INSTRUMENT ( total_yields++; )
            NANOS_INSTRUMENT ( unsigned long begin_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
            thread->yield();
            NANOS_INSTRUMENT ( unsigned long end_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
            NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
            if ( use_block ) yields--;
         }
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
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t total_spins_key  = ID->getEventKey("num-spins"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_yields_key = ID->getEventKey("num-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_blocks_key = ID->getEventKey("num-blocks"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_scheds_key  = ID->getEventKey("num-scheds"); )

   NANOS_INSTRUMENT ( static nanos_event_key_t time_yields_key = ID->getEventKey("time-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_blocks_key = ID->getEventKey("time-blocks"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_scheds_key = ID->getEventKey("time-scheds"); )

   NANOS_INSTRUMENT ( nanos_event_key_t Keys[7]; )

   NANOS_INSTRUMENT ( Keys[0] = total_spins_key; )
   NANOS_INSTRUMENT ( Keys[1] = total_yields_key; )
   NANOS_INSTRUMENT ( Keys[2] = total_blocks_key; )
   NANOS_INSTRUMENT ( Keys[3] = total_scheds_key; )

   NANOS_INSTRUMENT ( Keys[4] = time_yields_key; )
   NANOS_INSTRUMENT ( Keys[5] = time_blocks_key; )
   NANOS_INSTRUMENT ( Keys[6] = time_scheds_key; )

   NANOS_INSTRUMENT ( unsigned event_start; )
   NANOS_INSTRUMENT ( unsigned event_num; )

   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SYNCHRONIZATION) );

   NANOS_INSTRUMENT ( unsigned long total_spins = 0; ) /* Number of spins by idle phase*/
   NANOS_INSTRUMENT ( unsigned long total_yields = 0; ) /* Number of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long total_blocks = 0; ) /* Number of blocks by idle phase */
   NANOS_INSTRUMENT ( unsigned long total_scheds= 0; ) /* Number of schedulers by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_blocks = 0; ) /* Time of blocks by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_yields = 0; ) /* Time of yields by idle phase */
   NANOS_INSTRUMENT ( unsigned long time_scheds = 0; ) /* Time of sched by idle phase */

   if (condition->check()) {
       return;
   }

   const int init_spins = sys.getSchedulerConf().getNumSpins();
   const int init_checks = sys.getSchedulerConf().getNumChecks();
   const int init_yields = sys.getSchedulerConf().getNumYields();

   const bool use_yield = sys.getSchedulerConf().getUseYield();
   const bool use_block = sys.getSchedulerConf().getUseBlock();

   unsigned int checks = init_checks; 
   unsigned int spins = init_spins;
   unsigned int yields = init_yields;

   WD * current = myThread->getCurrentWD();

   current->setSyncCond( condition );
   sys.getSchedulerStats()._idleThreads++;
   myThread->setIdle( true );
   
   BaseThread *thread = getMyThreadSafe();

   while ( !condition->check()
         && thread->isRunning() ) {
      checks--;
      if ( checks == 0 ) {
         checks = init_checks;
         condition->lock();
         if ( !( condition->check() ) ) {
            //! Init of schedule phase
            spins--;

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

               NANOS_INSTRUMENT ( total_spins+= spins; )
               NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_spins; )
               NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) total_yields; )
               NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) total_blocks; )
               NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) total_scheds; )

               NANOS_INSTRUMENT ( Values[4] = (nanos_event_value_t) time_yields; )
               NANOS_INSTRUMENT ( Values[5] = (nanos_event_value_t) time_blocks; )
               NANOS_INSTRUMENT ( Values[6] = (nanos_event_value_t) time_scheds; )

               NANOS_INSTRUMENT ( event_start = 0; event_num = 7; )
               NANOS_INSTRUMENT ( if (total_yields == 0 ) { event_start = 2; event_num = 5; } )
               NANOS_INSTRUMENT ( if (total_yields == 0 && total_blocks == 0) { event_start = 4; event_num = 3; } )
               NANOS_INSTRUMENT ( if (total_scheds == 0 ) { event_num -= 2; } )

               NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(event_num, &Keys[event_start], &Values[event_start]); )

               switchTo ( next );
               thread = getMyThreadSafe();

               NANOS_INSTRUMENT ( total_spins = 0; )

               NANOS_INSTRUMENT ( total_yields = 0; )
               NANOS_INSTRUMENT ( total_blocks = 0; )
               NANOS_INSTRUMENT ( total_scheds = 0; )

               NANOS_INSTRUMENT ( time_blocks = 0; ) 
               NANOS_INSTRUMENT ( time_yields = 0; )
               NANOS_INSTRUMENT ( time_scheds = 0; )

               sys.getSchedulerStats()._idleThreads++;

               spins = init_spins;
            }

            condition->unlock();

            if ( spins == 0 ) {
               NANOS_INSTRUMENT ( total_spins+= init_spins; )
               /* If DLB, return resources if needed */
               dlb_returnCpusIfNeeded();
#if 0
               if ( sys.dlbEnabled() && DLB_ReturnClaimedCpus && getMyThreadSafe()->getId() == 0 && sys.getPMInterface().isMalleable() )
                  DLB_ReturnClaimedCpus();
#endif
               if ( yields == 0 || !use_yield ) {
                  if ( use_block ) {
                     condition->lock();
                     if ( !condition->check() ) {
                        WD * currentWD = thread->getCurrentWD();
                        // If it's not tied to the current thread, tie it until the thread is resumed
                        bool tiedTemporally = false;
                        if ( currentWD->isTiedTo() == NULL )
                        {
                           currentWD->tieTo( *thread );
                           tiedTemporally = true;
                        }
                        
                        // Unblock other threads so that they can work
                        for ( int t = 0; t < sys.getNumWorkers(); ++t )
                        {
                           BaseThread * worker = sys.getWorker( t );
                           if ( worker == thread ) continue;
                           // wake up, Neo
                           worker->unblock();
                        }
                        
                        currentWD->setBlocked();   // Very important
                        condition->addWaiter( currentWD );
                        condition->unlock(); // FIXME: may cause race condition
                        
                        NANOS_INSTRUMENT ( total_blocks++; )
                        NANOS_INSTRUMENT ( unsigned long begin_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
                        thread->block(); //FIXME:xteruel
                        NANOS_INSTRUMENT ( unsigned long end_block = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
                        NANOS_INSTRUMENT ( time_blocks += ( end_block - begin_block ); )
                        
                        // Having reached this point, if we temporally tied the wd to the thread, undo it
                        if ( tiedTemporally )
                           currentWD->untie();
                     }
                     else {
                        condition->unlock();                        
                     }
                  }
                  yields = init_yields;
               } else if ( use_yield ) {
                  NANOS_INSTRUMENT ( total_yields++; )
                  NANOS_INSTRUMENT ( unsigned long begin_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
                  thread->yield();
                  NANOS_INSTRUMENT ( unsigned long end_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  ); )
                  NANOS_INSTRUMENT ( time_yields += ( end_yield - begin_yield ); )
                  if ( use_block ) yields--;
               }
               spins = init_spins;
            }
         } else {
            condition->unlock();
         }
      }
      thread->idle();
   }

   current->setSyncCond( NULL );
   sys.getSchedulerStats()._idleThreads--;
   if ( !current->isReady() ) {
      current->setReady();
   }

   NANOS_INSTRUMENT ( total_spins+= (init_spins - spins); )

   NANOS_INSTRUMENT ( nanos_event_value_t Values[7]; )

   NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_spins; )
   NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) total_yields; )
   NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) total_blocks; )
   NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) total_scheds; )

   NANOS_INSTRUMENT ( Values[4] = (nanos_event_value_t) time_yields; )
   NANOS_INSTRUMENT ( Values[5] = (nanos_event_value_t) time_blocks; )
   NANOS_INSTRUMENT ( Values[6] = (nanos_event_value_t) time_scheds; )

   NANOS_INSTRUMENT ( event_start = 0; event_num = 7; )
   NANOS_INSTRUMENT ( if (total_yields == 0 ) { event_start = 2; event_num = 5; } )
   NANOS_INSTRUMENT ( if (total_yields == 0 && total_blocks == 0) { event_start = 4; event_num = 3; } )
   NANOS_INSTRUMENT ( if (total_scheds == 0 ) { event_num -= 2; } )

   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(event_num, &Keys[event_start], &Values[event_start]); )

}

void Scheduler::wakeUp ( WD *wd )
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SYNCHRONIZATION) );
   
   if ( !wd->isReady() ) {
      /* Setting ready wd */
      wd->setReady();
      WD *next = NULL;
      
      BaseThread * tiedTo = wd->isTiedTo();
      if ( tiedTo != NULL && sys.getSchedulerConf().getUseBlock() ) {
         // If the thread is blocked, we must not re-submit it's task
         tiedTo->unblock();
         // Note: this will probably break nesting.
         return;
      }
      
      if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
         // The thread is not paused, mark it as so
         myThread->unpause();
         
         /* atWakeUp must check basic constraints */
         ThreadTeam *myTeam = getMyThreadSafe()->getTeam();
         if ( myTeam ) next = myTeam->getSchedulePolicy().atWakeUp( myThread, *wd );
         else fatal("Trying to wake up a WD from a thread without team.");
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
   if ( sys.getSchedulerConf().getSchedulerEnabled() ) {
      //! If the scheduler is running
      //! The thread is not paused, mark it as so...
      thread->unpause();
      
      //! ... and do the prefetch
      WD *prefetchedWD = thread->getTeam()->getSchedulePolicy().atPrefetch( thread, wd );
      if ( prefetchedWD ) {
         prefetchedWD->_mcontrol.preInit();
      }
      return prefetchedWD;
   } else {
      //! Otherwise, pause this thread
      thread->pause();
   }
   //! \bug FIXME (#581): Otherwise, do nothing: consequences?
   return NULL;
}

#ifdef CLUSTER_DEV
WD * Scheduler::getClusterWD( BaseThread *thread, int inGPU )
{
   WD * wd = NULL;
   if ( thread->getTeam() != NULL ) {
      wd = thread->getNextWD();
      if ( wd ) {
#ifdef GPU_DEV
         if ( ( inGPU == 1 && !wd->canRunIn( ext::GPU ) ) || ( inGPU == 0 && !wd->canRunIn( ext::SMP ) ) )
#else
         if ( inGPU == 0 && !wd->canRunIn( ext::SMP ) )
#endif
         { // found a non compatible wd in "nextWD", ignore it
            wd = thread->getTeam()->getSchedulePolicy().atIdle ( thread );
            //if(wd!=NULL)std::cerr << "GN got a wd with depth " <<wd->getDepth() << std::endl;
         } else {
           // thread->resetNextWD();
            std::cerr << "FIXME" << std::endl;
         }
      } else {
         wd = thread->getTeam()->getSchedulePolicy().atIdle ( thread );
         //if(wd!=NULL)std::cerr << "got a wd with depth " <<wd->getDepth() << std::endl;
      }
   }
   return wd;
}
#endif

#ifdef CLUSTER_DEV
void Scheduler::workerClusterLoop ()
{
   BaseThread *parent = myThread;
   myThread = myThread->getNextThread();

   for ( ; ; ) {
      if ( !parent->isRunning() ) break; //{ std::cerr << "FINISHING CLUSTER THD!" << std::endl; break; }

      if ( parent != myThread ) // if parent == myThread, then there are no "soft" threads and just do nothing but polling.
      {
         ext::ClusterThread * volatile myClusterThread = dynamic_cast< ext::ClusterThread * >( myThread );
         if ( myClusterThread->tryLock() ) {
            ext::ClusterNode *thisNode = dynamic_cast< ext::ClusterNode * >( myThread->runningOn() );
            thisNode->disableDevice( 1 ); 
            myClusterThread->clearCompletedWDsSMP2();

            if ( myClusterThread->hasWaitingDataWDs() ) {
               WD * wd_waiting = myClusterThread->getWaitingDataWD();
               if ( wd_waiting->isInputDataReady() ) {
                  myClusterThread->addRunningWDSMP( wd_waiting );
                  outlineWork( myThread, wd_waiting );
               } else {
                  myClusterThread->addWaitingDataWD( wd_waiting );


// Try to get a WD normally, this is needed because otherwise we will keep only checking the WaitingData WDs
               if ( myClusterThread->hasAPendingWDToInit() ) {
                  WD * wd = myClusterThread->getPendingInitWD();
                  if ( Scheduler::tryPreOutlineWork(wd) ) {
                     myThread->preOutlineWorkDependent( *wd );
                     //std::cerr << "GOT A PENDIGN WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                     if ( wd->isInputDataReady() ) {
                        myClusterThread->addRunningWDSMP( wd );
                     //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK); );
                        outlineWork( myThread, wd );
                     //NANOS_INSTRUMENT( inst2.close(); );
                     } else {
                        myClusterThread->addWaitingDataWD( wd );
                     }
                  } else {
                     //std::cerr << "REPEND WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                     myClusterThread->setPendingInitWD( wd );
                  }
               } else {
                  if ( myClusterThread->acceptsWDsSMP() )
                  {
                     WD * wd = getClusterWD( myThread, 0 );
                     if ( wd )
                     {
                        Scheduler::prePreOutlineWork(wd); 
                        if ( Scheduler::tryPreOutlineWork(wd) ) {
                           myThread->preOutlineWorkDependent( *wd );
                           if ( wd->isInputDataReady() ) {
                              //std::cerr << "SUCCED WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                              myClusterThread->addRunningWDSMP( wd );
                              //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK); );
                              outlineWork( myThread, wd );
                              //NANOS_INSTRUMENT( inst2.close(); );
                           } else {
                              myClusterThread->addWaitingDataWD( wd );
                           }
                        } else {
                           //std::cerr << "ADDED A PENDIGN WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                           myClusterThread->setPendingInitWD( wd );
                        }
                     }
                  }// else { std::cerr << "Max presend reached "<<myClusterThread->getId()  << std::endl; }
               }




               }
            } else {

               if ( myClusterThread->hasAPendingWDToInit() ) {
                  WD * wd = myClusterThread->getPendingInitWD();
                  if ( Scheduler::tryPreOutlineWork(wd) ) {
                     myThread->preOutlineWorkDependent( *wd );
                     //std::cerr << "GOT A PENDIGN WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                     if ( wd->isInputDataReady() ) {
                        myClusterThread->addRunningWDSMP( wd );
                     //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK); );
                        outlineWork( myThread, wd );
                     //NANOS_INSTRUMENT( inst2.close(); );
                     } else {
                        myClusterThread->addWaitingDataWD( wd );
                     }
                  } else {
                     //std::cerr << "REPEND WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                     myClusterThread->setPendingInitWD( wd );
                  }
               } else {
                  if ( myClusterThread->acceptsWDsSMP() )
                  {
                     WD * wd = getClusterWD( myThread, 0 );
                     if ( wd )
                     {
                        Scheduler::prePreOutlineWork(wd); 
                        if ( Scheduler::tryPreOutlineWork(wd) ) {
                           myThread->preOutlineWorkDependent( *wd );
                           if ( wd->isInputDataReady() ) {
                              //std::cerr << "SUCCED WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                              myClusterThread->addRunningWDSMP( wd );
                              //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK); );
                              outlineWork( myThread, wd );
                              //NANOS_INSTRUMENT( inst2.close(); );
                           } else {
                              myClusterThread->addWaitingDataWD( wd );
                           }
                        } else {
                           //std::cerr << "ADDED A PENDIGN WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                           myClusterThread->setPendingInitWD( wd );
                        }
                     }
                  }// else { std::cerr << "Max presend reached "<<myClusterThread->getId()  << std::endl; }
               }
            }
            thisNode->enableDevice( 1 ); 
#ifdef GPU_DEV
            thisNode->disableDevice( 0 ); 
            myClusterThread->clearCompletedWDsGPU2();

            if ( myClusterThread->hasWaitingDataWDs() ) {
               WD * wd_waiting = myClusterThread->getWaitingDataWD();
               if ( wd_waiting->isInputDataReady() ) {
                  myClusterThread->addRunningWDGPU( wd_waiting );
                  outlineWork( myThread, wd_waiting );
               } else {
                  myClusterThread->addWaitingDataWD( wd_waiting );


// Try to get a WD normally, this is needed because otherwise we will keep only checking the WaitingData WDs
               if ( myClusterThread->hasAPendingWDToInit() ) {
                  WD * wd = myClusterThread->getPendingInitWD();
                  if ( Scheduler::tryPreOutlineWork(wd) ) {
                     myThread->preOutlineWorkDependent( *wd );
                     //std::cerr << "GOT A PENDIGN WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                     if ( wd->isInputDataReady() ) {
                        myClusterThread->addRunningWDGPU( wd );
                     //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK); );
                        outlineWork( myThread, wd );
                     //NANOS_INSTRUMENT( inst2.close(); );
                     } else {
                        myClusterThread->addWaitingDataWD( wd );
                     }
                  } else {
                     //std::cerr << "REPEND WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                     myClusterThread->setPendingInitWD( wd );
                  }
               } else {
                  if ( myClusterThread->acceptsWDsGPU() )
                  {
                     WD * wd = getClusterWD( myThread, 0 );
                     if ( wd )
                     {
                        Scheduler::prePreOutlineWork(wd); 
                        if ( Scheduler::tryPreOutlineWork(wd) ) {
                           myThread->preOutlineWorkDependent( *wd );
                           if ( wd->isInputDataReady() ) {
                              //std::cerr << "SUCCED WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                              myClusterThread->addRunningWDGPU( wd );
                              //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK); );
                              outlineWork( myThread, wd );
                              //NANOS_INSTRUMENT( inst2.close(); );
                           } else {
                              myClusterThread->addWaitingDataWD( wd );
                           }
                        } else {
                           //std::cerr << "ADDED A PENDIGN WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                           myClusterThread->setPendingInitWD( wd );
                        }
                     }
                  }// else { std::cerr << "Max presend reached "<<myClusterThread->getId()  << std::endl; }
               }




               }
            } else {

               if ( myClusterThread->hasAPendingWDToInit() ) {
                  WD * wd = myClusterThread->getPendingInitWD();
                  if ( Scheduler::tryPreOutlineWork(wd) ) {
                     myThread->preOutlineWorkDependent( *wd );
                     //std::cerr << "GOT A PENDIGN WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                     if ( wd->isInputDataReady() ) {
                        myClusterThread->addRunningWDGPU( wd );
                     //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK); );
                        outlineWork( myThread, wd );
                     //NANOS_INSTRUMENT( inst2.close(); );
                     } else {
                        myClusterThread->addWaitingDataWD( wd );
                     }
                  } else {
                     //std::cerr << "REPEND WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                     myClusterThread->setPendingInitWD( wd );
                  }
               } else {
                  if ( myClusterThread->acceptsWDsGPU() )
                  {
                     WD * wd = getClusterWD( myThread, 0 );
                     if ( wd )
                     {
                        Scheduler::prePreOutlineWork(wd); 
                        if ( Scheduler::tryPreOutlineWork(wd) ) {
                           myThread->preOutlineWorkDependent( *wd );
                           if ( wd->isInputDataReady() ) {
                              //std::cerr << "SUCCED WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                              myClusterThread->addRunningWDGPU( wd );
                              //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK); );
                              outlineWork( myThread, wd );
                              //NANOS_INSTRUMENT( inst2.close(); );
                           } else {
                              myClusterThread->addWaitingDataWD( wd );
                           }
                        } else {
                           //std::cerr << "ADDED A PENDIGN WD for thd " << myThread->getId() <<" wd is " << wd->getId() << std::endl;
                           myClusterThread->setPendingInitWD( wd );
                        }
                     }
                  }// else { std::cerr << "Max presend reached "<<myClusterThread->getId()  << std::endl; }
               }
            }
            thisNode->enableDevice( 0 ); 
#endif
            myClusterThread->unlock();
         }
      }
      sys.getNetwork()->poll(parent->getId());
      myThread = myThread->getNextThread();
   }

   ext::SMPMultiThread *parentM = dynamic_cast< ext::SMPMultiThread * >( parent );
   for ( unsigned int i = 0; i < parentM->getNumThreads(); i += 1 ) {
      myThread = parentM->getThreadVector()[ i ];
      myThread->leaveTeam();
      myThread->joined();
   }
   myThread = parent;
}
#endif


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
        if ( Scheduler::inlineWork ( next /*jb merge */, true ) ) {
          next->~WorkDescriptor();
          delete[] (char *)next;
        }
      }
   }
   static bool checkThreadRunning( WD *current) { return true; }
   static bool exiting() { return false; }
};

void Scheduler::workerLoop ()
{
   idleLoop<WorkerBehaviour>();
}

void Scheduler::preOutlineWorkWithThread ( BaseThread * thread, WD *wd )
{
   //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_PRE_OUTLINE_WORK); );
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

   //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_PRE_OUTLINE_WORK); );
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
      //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_PRE_OUTLINE_WORK); );
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
   //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_POST_OUTLINE_WORK); );

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
   debug( "exiting task(post outline) " << wd << ":" << wd->getId() << " to " << &(thread->getThreadWD()) << ":" << thread->getThreadWD().getId() );

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
   /* If WorkDescriptor has been submitted update statistics */
   updateExitStats (*wd);

   //! \note getting more work to do (only if not going to sleep)
   if ( schedule && !getMyThreadSafe()->isSleeping() ) {
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

   wd->done();
   wd->clear();

   /* If DLB, perform the adjustment of resources */
   if ( sys.getPMInterface().isMalleable() )
	dlb_updateAvailableCpus();

/*   if ( sys.dlbEnabled() && DLB_UpdateResources_max && getMyThreadSafe()->getId() == 0 ) {
      if ( sys.getPMInterface().isMalleable() )
         DLB_ReturnClaimedCpus();

      int needed_resources = sys.getSchedulerStats()._readyTasks.value() - sys.getNumThreads();
      if ( needed_resources > 0 )
         DLB_UpdateResources_max( needed_resources );
   }*/

}

bool Scheduler::inlineWork ( WD *wd, bool schedule )
{
   BaseThread *thread = getMyThreadSafe();

   // run it in the current frame
   WD *oldwd = thread->getCurrentWD();

   GenericSyncCond *syncCond = oldwd->getSyncCond();
   if ( syncCond != NULL ) syncCond->unlock();

   debug( "switching(inlined) from task " << oldwd << ":" << oldwd->getId() <<
          " to " << wd << ":" << wd->getId() << " at node " << sys.getNetwork()->getNodeNum() );

   // Initializing wd if necessary
   // It will be started later in inlineWorkDependent call
   
   if ( !wd->started() ) { 
      if ( !wd->_mcontrol.isMemoryAllocated() ) {
         wd->_mcontrol.initialize( *(thread->runningOn()) );
         bool result;
         do {
            result = wd->_mcontrol.allocateTaskMemory();
         } while( result == false );
      }
      wd->init();
   }

   // This ensures that when we return from the inlining is still the same thread
   // and we don't violate rules about tied WD
   if ( oldwd->isTiedTo() != NULL && (wd->isTiedTo() == NULL)) wd->tieTo(*oldwd->isTiedTo());

   thread->setCurrentWD( *wd );

   /* Instrumenting context switch: wd enters cpu (last = n/a) */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( oldwd, wd, false) );

   bool done = thread->inlineWorkDependent(*wd);

   // reload thread after running WD due wd may be not tied to thread if
   // both work descriptor were not tied to any thread
   thread = getMyThreadSafe();

   if ( done ) {
      wd->finish();

      finishWork( wd, schedule );
      /* Instrumenting context switch: wd leaves cpu and will not come back (last = true) and new_wd enters */
      NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(wd, oldwd, true) );
   }

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
         to->_mcontrol.initialize( *(myThread->runningOn()) );
         bool result;
         do {
            result = to->_mcontrol.allocateTaskMemory();
         } while( result == false );

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
//! \bug FIXME: stack reusing was wrongly implementd and it's disabled (see #374)
//    WD *current = myThread->getCurrentWD();

    if (!to->started()) {
       to->_mcontrol.initialize( *(myThread->runningOn()) );
       bool result;
       do {
          result = to->_mcontrol.allocateTaskMemory();
       } while( result == false );

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

   finishWork( oldwd, ( next == NULL ) );

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
