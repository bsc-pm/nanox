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
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"

namespace nanos {
   namespace ext {

      class BreadthFirst : public SchedulePolicy
      {
        private:
           struct TeamData : public ScheduleTeamData
           {
              WDPool *_readyQueue;

              TeamData () : ScheduleTeamData(), _readyQueue( NULL )
              {
                if ( _usePriority || _useSmartPriority ) _readyQueue = NEW WDPriorityQueue<>( true /* enableDeviceCounter */, true /* optimise option */ );
                else _readyQueue = NEW WDDeque( true /* enableDeviceCounter */ );
              }
              ~TeamData () { delete _readyQueue; }
           };

         public:
           static bool       _useStack;
           static bool       _usePriority;
           static bool       _useSmartPriority;

           BreadthFirst() : SchedulePolicy("Breadth First")
           {
              /* If priorities are disabled by the user and detected
                 by the compiler, disable them. If enabled by the
                 user (default) and not detected by the compiler,
                 disable too
               */
               _usePriority = _usePriority && sys.getPrioritiesNeeded();
           }
           virtual ~BreadthFirst () {}

         private:

           virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
           virtual size_t getThreadDataSize () const { return 0; }

           virtual ScheduleTeamData * createTeamData ()
           {
              return NEW TeamData();
           }

           virtual ScheduleThreadData * createThreadData ()
           {
              return 0;
           }

           virtual void queue ( BaseThread *thread, WD &wd )
           {
              BaseThread *targetThread = wd.isTiedTo();
              if ( targetThread ) targetThread->addNextWD(&wd);
              else {
                 TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
                 if ( _useStack ) return tdata._readyQueue->push_front( &wd );
                 else tdata._readyQueue->push_back( &wd );
              }
           }

            virtual void queue ( BaseThread ** threads, WD ** wds, size_t numElems )
            {
               fatal_cond( numElems == 0, "Cannot queue 0 elements.");

               // First step: check if all threads have the same team
               ThreadTeam* team = threads[0]->getTeam();

               for ( size_t i = 1; i < numElems; ++i )
               {
                  if ( threads[i]->getTeam() == team )
                     continue;

                  fatal( "Batch submission does not support different teams" );
               }

               // If they have the same team, we can insert in batch
               TeamData &tdata = (TeamData &) *team->getScheduleData();
               if ( _useStack ) tdata._readyQueue->push_front( wds, numElems );
               else tdata._readyQueue->push_back( wds, numElems );

               // Unblock all participant threads
               for ( size_t i = 1; i < numElems; ++i ) {
                  sys.getThreadManager()->unblockThread(threads[i]);
               }
            }

            /*! This scheduling policy supports all WDs, no restrictions. */
            bool isValidForBatch ( const WD * wd ) const
            {
               return true;
            }


            /*!
             * \brief This method performs the main task of the smart priority
             * scheduler, which is to propagate the priority of a WD to its
             * immediate predecessors. It is meant to be invoked from
             * DependenciesDomain::submitWithDependenciesInternal.
             * \param [in/out] predecessor The preceding DependableObject.
             * \param [in] successor DependableObject whose WD priority has to be
             * propagated.
             */
//            void successorFound( DependableObject *predecessor, DependableObject *successor )
            void atSuccessor   ( DependableObject &successor, DependableObject &predecessor )
            {
               //debug( "Scheduler::successorFound" );

               if ( ! _useSmartPriority ) return;


 //              if ( predecessor == NULL || successor == NULL ) return;

               WD *pred = ( WD* ) predecessor.getRelatedObject();
               if ( pred == NULL ) return;

               WD *succ = ( WD* ) successor.getRelatedObject();
               if ( succ == NULL ) {
                  fatal( "SmartPriority::successorFound  successor->getRelatedObject() is NULL" );
               }

               debug ( "Propagating priority from "
                  << (void*)succ << ":" << succ->getId() << " to "
                  << (void*)pred << ":"<< pred->getId()
                  << ", old priority: " << pred->getPriority()
                  << ", new priority: " << std::max( pred->getPriority(),
                  succ->getPriority() )
               );

               // Propagate priority
               if ( pred->getPriority() < succ->getPriority() ) {
                  pred->setPriority( succ->getPriority() );

                  // Reorder
                  TeamData &tdata = (TeamData &) *myThread->getTeam()->getScheduleData();
                  WDPriorityQueue<> *q = (WDPriorityQueue<> *) tdata._readyQueue;
                  q->reorderWD( pred );
               }
            }

           virtual WD *atSubmit ( BaseThread *thread, WD &newWD )
           {
              queue( thread, newWD );
              return 0;
           }

           WD * atIdle ( BaseThread *thread, int numSteal )
           {
              WD * next = thread->getNextWD();
              if (!next) {
                 TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
                 next = tdata._readyQueue->pop_front( thread );
              }
              return next;
           }

           WD * atPrefetch ( BaseThread *thread, WD &current )
           {
              WD * found = current.getImmediateSuccessor(*thread);
              if ( found && (_usePriority || _useSmartPriority) ) {
                 WDPriorityQueue<> &tdata = (WDPriorityQueue<> &) *((TeamData *) thread->getTeam()->getScheduleData())->_readyQueue;
                 if (found->getPriority() < tdata.maxPriority() ) {
                    queue(thread, *found);
                    found = NULL;
                 }
              }
              return found != NULL ? found : atIdle(thread,false);
           }

           WD * atBeforeExit ( BaseThread *thread, WD &current, bool schedule )
           {
              WD * found = schedule ? current.getImmediateSuccessor(*thread) : NULL;
              if ( found && (_usePriority || _useSmartPriority) ) {
                 WDPriorityQueue<> &tdata = (WDPriorityQueue<> &) *((TeamData *) thread->getTeam()->getScheduleData())->_readyQueue;
                 if (found->getPriority() < tdata.maxPriority() ) {
                    queue(thread, *found);
                    found = NULL;
                 }
              }
              return found;
           }

            bool reorderWD ( BaseThread *t, WD *wd )
            {
              //! \bug FIXME flags of priority must be in queue
               if ( _usePriority || _useSmartPriority ) {
                  WDPriorityQueue<> *q = (WDPriorityQueue<> *) wd->getMyQueue();
                  return q? q->reorderWD( wd ) : true;
               } else {
                  return true;
               }
            }

            bool testDequeue()
            {
               TeamData &tdata = (TeamData &) *myThread->getTeam()->getScheduleData();
               return tdata._readyQueue->testDequeue();
            }

            bool usingPriorities() const
            {
               return _usePriority || _useSmartPriority;
            }
      };

      bool BreadthFirst::_useStack = false;
      bool BreadthFirst::_usePriority = true;
      bool BreadthFirst::_useSmartPriority = false;

      class BFSchedPlugin : public Plugin
      {

         public:
            BFSchedPlugin() : Plugin( "BF scheduling Plugin",1 ) {}

            virtual void config ( Config &cfg )
            {
               cfg.setOptionsSection( "BF module", "Breadth-first scheduling module" );
               cfg.registerConfigOption ( "bf-use-stack", NEW Config::FlagOption( BreadthFirst::_useStack ), "Stack usage for the breadth-first policy");
               cfg.registerArgOption( "bf-use-stack", "bf-use-stack" );

               cfg.registerAlias ( "bf-use-stack", "bf-stack", "Stack usage for the breadth-first policy" );
               cfg.registerArgOption ( "bf-stack", "bf-stack" );

               cfg.registerConfigOption ( "schedule-priority", NEW Config::FlagOption( BreadthFirst::_usePriority ), "Priority queue used as ready task queue");
               cfg.registerArgOption( "schedule-priority", "schedule-priority" );

               cfg.registerConfigOption ( "schedule-smart-priority", NEW Config::FlagOption( BreadthFirst::_useSmartPriority ), "Smart priority queue propagates high priorities to predecessors");
               cfg.registerArgOption( "schedule-smart-priority", "schedule-smart-priority" );

            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW BreadthFirst());
            }
      };

   }
}

DECLARE_PLUGIN("sched-bf",nanos::ext::BFSchedPlugin);
