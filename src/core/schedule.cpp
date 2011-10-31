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

#ifdef CLUSTER_DEV
#include "clusterthread_decl.hpp"
#include "clusternode_decl.hpp"
#include "gpudd.hpp"
#include "smpdd.hpp"
#endif

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
   BaseThread *mythread = getMyThreadSafe();

   sys.getSchedulerStats()._createdTasks++;
   sys.getSchedulerStats()._totalTasks++;

   debug ( "submitting task " << wd.getId() );

   wd.submitted();

   /* handle tied tasks */
   if ( wd.isTied() && wd.isTiedTo() != mythread ) {
      wd.isTiedTo()->getTeam()->getSchedulePolicy().queue( wd.isTiedTo(), wd );
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
   myWG.waitCompletionAndSignalers();
}

void Scheduler::updateExitStats ( WD &wd )
{
   if ( wd.isSubmitted() ) 
     sys.getSchedulerStats()._totalTasks--;
}

template<class behaviour>
inline void Scheduler::idleLoop ()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_spins_key = ID->getEventKey("num-spins"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_yields_key = ID->getEventKey("num-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_sleeps_key = ID->getEventKey("time-sleeps"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_yields_key = ID->getEventKey("time-yields"); )
   NANOS_INSTRUMENT ( nanos_event_key_t Keys[4]; )
   NANOS_INSTRUMENT ( Keys[0] = total_spins_key; )
   NANOS_INSTRUMENT ( Keys[1] = total_yields_key; )
   NANOS_INSTRUMENT ( Keys[2] = time_sleeps_key; )
   NANOS_INSTRUMENT ( Keys[3] = time_yields_key; )

   NANOS_INSTRUMENT( InstrumentState inst(NANOS_IDLE) );

   const int nspins = sys.getSchedulerConf().getNumSpins();
   const int nsleeps = sys.getSchedulerConf().getNumSleeps();
   const int tsleep = sys.getSchedulerConf().getTimeSleep();
   int spins = nspins;
   int sleeps = nsleeps;
   unsigned long total_spins = 0;  /* Number of spins by idle phase*/
   unsigned long total_yields = 0; /* Number of yields by idle phase */
   unsigned long time_sleeps = 0;  /* Time of sleeps by idle phase */
   unsigned long time_yields = 0;  /* Time of yields by idle phase */

   WD *current = myThread->getCurrentWD();
   //WD *prefetchedWD = NULL; 
   //current->setIdle();
   sys.getSchedulerStats()._idleThreads++;
   for ( ; ; ) {
      BaseThread *thread = getMyThreadSafe();
      spins--;

      if ( !thread->isRunning() ) break;

      if ( thread->getTeam() != NULL ) {
         WD * next = myThread->getNextWD();

         if ( next ) {
         //  if ( !next->isSubmitted() && !next->started() ) 
         //    sys.getSchedulerStats()._readyTasks++;
         //} else if ( prefetchedWD != NULL ) {
         //   next = prefetchedWD;
         //      std::cerr << "executing prefetched wd " << next->getId() << std::endl;
         //   prefetchedWD = NULL;
            myThread->resetNextWD();
         } else {
           memoryFence();
           if ( sys.getSchedulerStats()._readyTasks > 0 ) {
              NANOS_INSTRUMENT( InstrumentState inst1(NANOS_SCHEDULING) )
              next = behaviour::getWD(thread,current);
              NANOS_INSTRUMENT( inst1.close() );
           }
         } 

         if ( next ) {
            sys.getSchedulerStats()._idleThreads--;

            total_spins+= (nspins - spins);
            NANOS_INSTRUMENT ( nanos_event_value_t Values[4]; )
            NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_spins; )
            NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) total_yields; )
            NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) time_sleeps; )
            NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) time_yields; )
            NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEventNkvs(4, Keys, Values); )

            NANOS_INSTRUMENT( InstrumentState inst2(NANOS_RUNTIME) )
            behaviour::switchWD(thread,current, next);
//std::cerr <<"n:" << sys.getNetwork()->getNodeNum() << " finished switchWD" << std::endl;
            thread = getMyThreadSafe();
            NANOS_INSTRUMENT( inst2.close() );
            //prefetchedWD = next->getPrefetchedWD();
            //if (prefetchedWD != NULL)
            //   std::cerr << "setting prefetched wd " << prefetchedWD->getId() << std::endl;
            sys.getSchedulerStats()._idleThreads++;
            total_spins = 0;
            total_yields = 0;
            time_sleeps = 0;
            time_yields = 0;
            spins = nspins;
            continue;
         }
      }


      if ( spins == 0 ) {
        total_spins+= nspins;
        sleeps--;
        if ( sleeps < 0 ) {
           if ( sys.useYield() ) {
              total_yields++;
              unsigned long begin_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  );
              thread->yield();
              unsigned long end_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  );
              time_yields += ( end_yield - begin_yield );
           }
           sleeps = nsleeps;
        } else {
           struct timespec req ={0,tsleep};
           nanosleep ( &req, NULL );
           time_sleeps += time_sleeps + tsleep;
        }
        spins = nspins;
      }
      else {
         thread->idle();
      }
   }
   sys.getSchedulerStats()._idleThreads--;
   current->setReady();
   //current->~WorkDescriptor();
   //delete[] (char *) current;
}

void Scheduler::waitOnCondition (GenericSyncCond *condition)
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_spins_key = ID->getEventKey("num-spins"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t total_yields_key = ID->getEventKey("num-yields"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_sleeps_key = ID->getEventKey("time-sleeps"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t time_yields_key = ID->getEventKey("time-yields"); )
   NANOS_INSTRUMENT ( nanos_event_key_t Keys[4]; )
   NANOS_INSTRUMENT ( Keys[0] = total_spins_key; )
   NANOS_INSTRUMENT ( Keys[1] = total_yields_key; )
   NANOS_INSTRUMENT ( Keys[2] = time_sleeps_key; )
   NANOS_INSTRUMENT ( Keys[3] = time_yields_key; )

   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SYNCHRONIZATION) );

   const int nspins = sys.getSchedulerConf().getNumSpins();
   const int nsleeps = sys.getSchedulerConf().getNumSleeps();
   const int tsleep = sys.getSchedulerConf().getTimeSleep();
   unsigned int spins = nspins; 
   int sleeps = nsleeps;
   unsigned long total_spins = 0;  /* Number of spins by idle phase*/
   unsigned long total_yields = 0; /* Number of yields by idle phase */
   unsigned long time_sleeps = 0;  /* Time of sleeps by idle phase */
   unsigned long time_yields = 0;  /* Time of yields by idle phase */

   WD * current = myThread->getCurrentWD();

   sys.getSchedulerStats()._idleThreads++;
   current->setSyncCond( condition );
   //current->setIdle();
   
   BaseThread *thread = getMyThreadSafe();

   while ( !condition->check() && thread->isRunning() ) {
      spins--;
      if ( spins == 0 ) {
         total_spins+= nspins;
         sleeps--;
         condition->lock();
         if ( !( condition->check() ) ) {
            condition->addWaiter( current );

            WD * next = myThread->getNextWD();

            if ( next) {
               myThread->resetNextWD();
            } else {
               memoryFence();
               if ( sys.getSchedulerStats()._readyTasks > 0 ) {
                  NANOS_INSTRUMENT( InstrumentState inst1(NANOS_SCHEDULING) )
                  next = thread->getTeam()->getSchedulePolicy().atBlock( thread, current );
                  NANOS_INSTRUMENT( inst1.close() );
                }
            }

            if ( next ) {
               sys.getSchedulerStats()._idleThreads--;

               NANOS_INSTRUMENT ( nanos_event_value_t Values[4]; )
               NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_spins; )
               NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) total_yields; )
               NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) time_sleeps; )
               NANOS_INSTRUMENT ( Values[3] = (nanos_event_value_t) time_yields; )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEventNkvs(3, Keys, Values); )

               NANOS_INSTRUMENT( InstrumentState inst2(NANOS_RUNTIME); );
               switchTo ( next );
               thread = getMyThreadSafe();
               NANOS_INSTRUMENT( inst2.close() );

               total_spins = 0;
               total_yields = 0;
               time_sleeps = 0;
               time_yields = 0;

               sys.getSchedulerStats()._idleThreads++;
            } else {
               condition->unlock();
               if ( sleeps < 0 ) {
                  if ( sys.useYield() ) {
                     total_yields++;
                     unsigned long begin_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  );
                     thread->yield();
                     unsigned long end_yield = (unsigned long) ( OS::getMonotonicTime() * 1.0e9  );
                     time_yields += ( end_yield - begin_yield );
                  }
                  sleeps = nsleeps;
               } else {
                  struct timespec req = {0,tsleep};
                  nanosleep ( &req, NULL );
                  time_sleeps += time_sleeps + tsleep;
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

   total_spins+= (nspins - spins);
   NANOS_INSTRUMENT ( nanos_event_value_t Values[3]; )
   NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) total_spins; )
   NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) total_yields; )
   NANOS_INSTRUMENT ( Values[2] = (nanos_event_value_t) time_yields; )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEventNkvs(3, Keys, Values); )
}

void Scheduler::wakeUp ( WD *wd )
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SYNCHRONIZATION) );

   if ( wd->isBlocked() ) {
      /* Setting ready wd */
      wd->setReady();
      if ( checkBasicConstraints ( *wd, *myThread ) ) {
         WD *next = getMyThreadSafe()->getTeam()->getSchedulePolicy().atWakeUp( myThread, *wd );
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
      } else {
         myThread->getTeam()->getSchedulePolicy().queue( myThread, *wd );
      }
   }
}

WD * Scheduler::prefetch( BaseThread *thread, WD &wd )
{
   return thread->getTeam()->getSchedulePolicy().atPrefetch( thread, wd );
}

WD * Scheduler::getClusterWD( BaseThread *thread, int inGPU )
{
	WD * wd = NULL;
	if ( thread->getTeam() != NULL )
	{
		wd = thread->getNextWD();
		if ( wd )
		{
         if ( ( inGPU == 1 && !wd->canRunIn( ext::GPU ) ) || ( inGPU == 0 && !wd->canRunIn( ext::SMP ) ) )
         { // found a non compatible wd in "nextWD", ignore it
            wd = thread->getTeam()->getSchedulePolicy().atIdle ( thread );
         }
         else {
            thread->resetNextWD();
         }
		}
		else
		{
			wd = thread->getTeam()->getSchedulePolicy().atIdle ( thread );
		}
	}
	return wd;
}


void Scheduler::workerClusterLoop ()
{
	BaseThread *parent = myThread;
	myThread = myThread->getNextThread();

	sys.preMainBarrier();

	for ( ; ; ) {
		if ( !parent->isRunning() ) break; //{ std::cerr << "FINISHING CLUSTER THD!" << std::endl; break; }

		if ( parent != myThread ) // if parent == myThread, then there are no "soft" threads and just do nothing but polling.
		{
			ext::ClusterThread *myClusterThread = dynamic_cast< ext::ClusterThread * >( myThread );
			ext::ClusterNode *thisNode = dynamic_cast< ext::ClusterNode * >( myThread->runningOn() );
			thisNode->disableDevice( 1 ); 
			if ( ( (int) myClusterThread->numRunningWDsSMP() ) < ext::ClusterInfo::getSmpPresend() )
			{
				myClusterThread->clearCompletedWDsSMP2();
				//WD * wd = myClusterThread->getBlockingWDSMP();
				//WD * wd = myClusterThread->fetchBlockingWDSMP();
				//if ( wd )
				//{ 
				//	if ( !wd->canBeBlocked() ) 
				//	{
				//		myClusterThread->addRunningWDSMP( wd );
				//		//myClusterThread->setBlockingWDSMP( NULL );
				//		Scheduler::preOutlineWork(wd);
				//		myThread->outlineWorkDependent(*wd);
				//	} else myClusterThread->addBlockingWDSMP( wd );
				////	else
				////		std::cerr << "Data can block me AGAIN SMP (node " << thisNode->getClusterNodeNum() << ") task is " << wd->getId() <<std::endl;
				//}
				//else
				{
					WD * wd = getClusterWD( myThread, 0 );
					if ( wd )
					{
						if ( /*!wd->canBeBlocked()*/ true ) 
						{
							myClusterThread->addRunningWDSMP( wd );
							Scheduler::preOutlineWork(wd);
							myThread->outlineWorkDependent(*wd);
						}
						else
						{
							//std::cerr << "Data can block me SMP (node " << thisNode->getClusterNodeNum() << ") task is " << wd->getId() <<std::endl;
							//myClusterThread->setBlockingWDSMP( wd );
							myClusterThread->addBlockingWDSMP( wd );
						}
					}
				}
			}
			//if ( myClusterThread->areThereCompletedWDsSMP() )
			//{
			//   WD *completedWD = myClusterThread->fetchCompletedWDSMP();
			//   Scheduler::postOutlineWork( completedWD, true );
			//   delete completedWD;
			//}
			thisNode->enableDevice( 1 ); 
#ifdef GPU_DEV
			thisNode->disableDevice( 0 ); 
         myClusterThread->clearCompletedWDsGPU2();
			if ( ( (int) myClusterThread->numRunningWDsGPU() ) < ext::ClusterInfo::getGpuPresend() )
				//if ( !myThread->isWorking(1) )
			{
				//WD * runningWD = myClusterThread->getRunningWD( 1 );
				//if ( runningWD != NULL )
				//{
				//   myClusterThread->clearRunningWD( 1 );
				//   Scheduler::postOutlineWork( runningWD, false, myThread );
				//   delete runningWD;
				//} 
				WD * wd = myClusterThread->fetchBlockingWDGPU();
				if ( wd )
				{ 
					if ( !wd->canBeBlocked() ) 
					{
						myClusterThread->addRunningWDGPU( wd );
							//message("adding a GPU task for node " << thisNode->getClusterNodeNum() << " task is " << wd->getId() << " RELEASE TASK " );
						Scheduler::preOutlineWork(wd);
						myThread->outlineWorkDependent(*wd);
					} else {
                  myClusterThread->addBlockingWDGPU(wd );
						//message("Data can block me AGAIN GPU (node " << thisNode->getClusterNodeNum() << ") task is " << wd->getId() );
               }
				}
				else
				{
					WD* newwd = getClusterWD( myThread, 1 );
					if ( newwd )
					{
						//if ( true /*!wd->canBeBlocked() */) 
						if ( !newwd->canBeBlocked() ) 
						{
							//message("adding a GPU task for node " << thisNode->getClusterNodeNum() << " task is " << newwd->getId());
							myClusterThread->addRunningWDGPU( newwd );
							Scheduler::preOutlineWork(newwd);
							myThread->outlineWorkDependent(*newwd);
						}
						else
						{
							//message( "Data can block me GPU (node " << thisNode->getClusterNodeNum() << ") task is " << newwd->getId() );
							myClusterThread->addBlockingWDGPU(newwd );
						}
					}
				}
			}
			thisNode->enableDevice( 0 ); 
#endif
		}
		sys.getNetwork()->poll(parent->getId());
		myThread = myThread->getNextThread();
	}
}


struct WorkerBehaviour
{
   static WD * getWD ( BaseThread *thread, WD *current )
   {
      return thread->getTeam()->getSchedulePolicy().atIdle ( thread );
   }

   static void switchWD ( BaseThread *thread, WD *current, WD *next )
   {
      if (next->started()){
        Scheduler::switchTo(next);
      }
      else {
        Scheduler::inlineWork ( next /*, true */ );
        next->~WorkDescriptor();
        delete[] (char *)next;
      }
   }
   static bool checkThreadRunning( WD *current) { return true; }
};

void Scheduler::workerLoop ()
{
   idleLoop<WorkerBehaviour>();
}

void Scheduler::preOutlineWork ( WD *wd )
{
   BaseThread *thread = getMyThreadSafe();

               NANOS_INSTRUMENT( InstrumentState inst2(NANOS_PRE_OUTLINE_WORK); );
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

   //NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( NULL, wd, false) );
               NANOS_INSTRUMENT( inst2.close(); );
}

void Scheduler::postOutlineWork ( WD *wd, bool schedule, BaseThread *owner )
{
   BaseThread *thread = owner;
               NANOS_INSTRUMENT( InstrumentState inst2(NANOS_POST_OUTLINE_WORK); );

   //std::cerr << "completing WD " << wd->getId() << " at thd " << owner->getId() << " thd addr " << owner << std::endl; 
   if (schedule && thread->getNextWD() == NULL ) {
        thread->setNextWD(thread->getTeam()->getSchedulePolicy().atBeforeExit(thread,*wd));
   }

   /* If WorkDescriptor has been submitted update statistics */
   updateExitStats (*wd);

   wd->done();
   wd->clear();

   //NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(wd, NULL, false) );


   //std::cerr << "thd " << myThread->getId() << "exiting task(inlined) " << wd << ":" << wd->getId() <<
   //       " to " << oldwd << ":" << oldwd->getId() << std::endl;
   debug( "exiting task(post outline) " << wd << ":" << wd->getId() << " to " << &(thread->getThreadWD()) << ":" << thread->getThreadWD().getId() );


   thread->setCurrentWD( thread->getThreadWD() );

   //NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch( NULL, oldwd, false) );

   // While we tie the inlined tasks this is not needed
   // as we will always return to the current thread
   #if 0
   if ( oldwd->isTiedTo() != NULL )
      switchToThread(oldwd->isTiedTo());
   #endif

   //ensure(oldwd->isTiedTo() == NULL || thread == oldwd->isTiedTo(),
   //        "Violating tied rules " + toString<BaseThread*>(thread) + "!=" + toString<BaseThread*>(oldwd->isTiedTo()));

               NANOS_INSTRUMENT( inst2.close(); );
}

void Scheduler::inlineWork ( WD *wd, bool schedule )
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
   if ( !wd->started() ) wd->init();

   // This ensures that when we return from the inlining is still the same thread
   // and we don't violate rules about tied WD
   if ( oldwd->isTiedTo() != NULL && (wd->isTiedTo() == NULL)) wd->tieTo(*oldwd->isTiedTo());

   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(oldwd, NULL, false) );
   thread->setCurrentWD( *wd );
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(NULL, wd, false) );

   thread->inlineWorkDependent(*wd);

   // reload thread after running WD due wd may be not tied to thread if
   // both work descriptor were not tied to any thread
   thread = getMyThreadSafe();

   if (schedule) {
        if ( thread->reserveNextWD() ) {
           thread->setReservedNextWD(thread->getTeam()->getSchedulePolicy().atBeforeExit(thread,*wd));
        }
   }

   /* If WorkDescriptor has been submitted update statistics */
   updateExitStats (*wd);

   wd->done();
   wd->clear();
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(wd, NULL, true) );

   debug( "exiting task(inlined) " << wd << ":" << wd->getId() <<
          " to " << oldwd << ":" << oldwd->getId() );

   thread->setCurrentWD( *oldwd );
   /* Instrumenting context switch: wd leaves cpu and will not come back (last = true) and oldwd enters */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(NULL, oldwd, false) );

   // While we tie the inlined tasks this is not needed
   // as we will always return to the current thread
   #if 0
   if ( oldwd->isTiedTo() != NULL )
      switchToThread(oldwd->isTiedTo());
   #endif

   ensure(oldwd->isTiedTo() == NULL || thread == oldwd->isTiedTo(),
           "Violating tied rules " + toString<BaseThread*>(thread) + "!=" + toString<BaseThread*>(oldwd->isTiedTo()));

}

void Scheduler::switchHelper (WD *oldWD, WD *newWD, void *arg)
{
   /* Instrumenting context switch: oldwd leaves cpu but will come back (last = false) and newWD enters */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdSwitch(oldWD, newWD, false) );

   myThread->switchHelperDependent(oldWD, newWD, arg);

   GenericSyncCond *syncCond = oldWD->getSyncCond();
   if ( syncCond != NULL ) {
      oldWD->setBlocked();
      syncCond->unlock();
   } else {
      myThread->getTeam()->getSchedulePolicy().queue( myThread, *oldWD );
   }

   myThread->setCurrentWD( *newWD );
}

void Scheduler::switchTo ( WD *to )
{
   if ( false /*myThread->runningOn()->supportsUserLevelThreads() */) {
      if (!to->started()) {
         to->init();
         to->start(WD::IsAUserLevelThread);
      }
      
      debug( "switching from task " << myThread->getCurrentWD() << ":" << myThread->getCurrentWD()->getId() << " to " << to << ":" << to->getId() );
          
      myThread->switchTo( to, switchHelper );
   } else {
      inlineWork(to);
      to->~WorkDescriptor();
      delete[] (char *)to;
   }
}

void Scheduler::yield ()
{
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_SCHEDULING) );
   WD *next = myThread->getTeam()->getSchedulePolicy().atYield( myThread, myThread->getCurrentWD() );

   if ( next ) {
      switchTo(next);
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
    NANOS_INSTRUMENT ( sys.getInstrumentation()->wdSwitch(oldWD,NULL,true) );
    myThread->setCurrentWD( *newWD );
    oldWD->~WorkDescriptor();
    delete[] (char *)oldWD;
    NANOS_INSTRUMENT ( sys.getInstrumentation()->wdSwitch(NULL,newWD,false) );
}

struct ExitBehaviour
{
   static WD * getWD ( BaseThread *thread, WD *current )
   {
      return thread->getTeam()->getSchedulePolicy().atAfterExit( thread, current );
   }

   static void switchWD ( BaseThread *thread, WD *current, WD *next )
   {
      Scheduler::exitTo(next);
   }
   static bool checkThreadRunning( WD *current ) { return true; }
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

    //std::cerr << "thd " << myThread->getId() << "exiting task " << myThread->getCurrentWD() << ":" << myThread->getCurrentWD()->getId() <<
    //      " to " << to << ":" << to->getId() << std::endl;;
    debug( "exiting task " << myThread->getCurrentWD() << ":" << myThread->getCurrentWD()->getId() <<
          " to " << to << ":" << to->getId() );

    myThread->exitTo ( to, Scheduler::exitHelper );
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

  /* if getNextWD() has returned a WD, we need to resetNextWD(). If no WD has
   * been returned call scheduler policy */
   if (next) thread->resetNextWD();
   else next = thread->getTeam()->getSchedulePolicy().atBeforeExit(thread,*oldwd);

   updateExitStats (*oldwd);
   oldwd->done();
   oldwd->clear();

   if (!next) {
     idleLoop<ExitBehaviour>();
   } else {
     Scheduler::exitTo(next);
   } 

   fatal("A thread should never return from Scheduler::exit");
}

