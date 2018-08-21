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

      class MultiPriorityQueue : public SchedulePolicy
      {
        private:
           struct TeamData : public ScheduleTeamData
           {
              WDPool **_readyQueue;

              TeamData () : ScheduleTeamData(), _readyQueue()
              {
                _readyQueue = (WDPool **) malloc(sizeof(WDPool *)* _numQueues);
                for (int i=0; i<_numQueues; i++)
                   _readyQueue[i] = NEW WDDeque();
               //fprintf(stderr,"Nanos++: Max priority = %d, number of queues = %d\n",_maxPriority, _numQueues);
              }
              ~TeamData () {
                 for (int i=0; i<_numQueues; i++)
                    delete _readyQueue[i];
              }
           };

         public:
           static bool       _useStack;
           static bool       _usePriority;
           static bool       _useSmartPriority;
           static int        _numQueues;
           static int        _maxPriority;

           MultiPriorityQueue() : SchedulePolicy("Multi Priority Queue") {
              _usePriority = _usePriority && sys.getPrioritiesNeeded();
           }
           virtual ~MultiPriorityQueue () {}

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
              TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

              int priority = (int) (wd.getPriority()*(_numQueues-1))/ ((_maxPriority == 0)?1:_maxPriority);
              if( priority >= _numQueues) {
                 fprintf(stderr,"Nanox++: Priority value has been saturated\n");
                 priority = _numQueues-1;
              }

              if ( _useStack ) return tdata._readyQueue[priority]->push_front( &wd );
              else tdata._readyQueue[priority]->push_back( &wd );
           }

            virtual void queue ( BaseThread ** threads, WD ** wds, size_t numElems )
            {
               fatal_cond( true , "Cannot queue 0 (or N) elements ."); //FIXME
               
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
               
               LockBlock lock( tdata._readyQueue[0]->getLock() );
               
               if ( _useStack ) return tdata._readyQueue[0]->push_front( wds, numElems );
               else tdata._readyQueue[0]->push_back( wds, numElems );
            }
            
            /*! This scheduling policy supports all WDs, no restrictions. */
            bool isValidForBatch ( const WD * wd ) const
            {
               return false;
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
               // if ( ! _useSmartPriority ) return;
               return; //FIXME


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
                  WDPriorityQueue<> *q = (WDPriorityQueue<> *) tdata._readyQueue[0];
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
              TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
              
              WD *rv = NULL;
              for (int i=_numQueues-1; i >=0; i-- ) {
                 rv = tdata._readyQueue[i]->pop_front( thread );
                 if ( rv != NULL) break;
              }
              return rv;
           }

           WD * atPrefetch ( BaseThread *thread, WD &current )
           {
              if ( !_usePriority && !_useSmartPriority ) {
                 WD * found = current.getImmediateSuccessor(*thread);
                 return found != NULL ? found : atIdle(thread,false);
              } else {
                 return atIdle(thread,false);
              }
           }
        
           WD * atBeforeExit ( BaseThread *thread, WD &current, bool schedule )
           {
              if ( !_usePriority && !_useSmartPriority && schedule ) {
                 return current.getImmediateSuccessor(*thread);
              } else {
                 return 0;
              }
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
            
            bool usingPriorities() const
            {
               return _usePriority || _useSmartPriority;
            }
      };

      bool MultiPriorityQueue::_useStack = false;
      bool MultiPriorityQueue::_usePriority = true;
      bool MultiPriorityQueue::_useSmartPriority = false;
      int  MultiPriorityQueue::_maxPriority = 0;
      int  MultiPriorityQueue::_numQueues = 8;

      class BFSchedPlugin : public Plugin
      {

         public:
            BFSchedPlugin() : Plugin( "BF scheduling Plugin",1 ) {}

            virtual void config ( Config &cfg )
            {
               cfg.setOptionsSection( "MPQ module", "Multi-priority queue scheduling module" );

               cfg.registerConfigOption ( "schedule-priority", NEW Config::FlagOption( MultiPriorityQueue::_usePriority ), "Priority queue used as ready task queue");
               cfg.registerArgOption( "schedule-priority", "schedule-priority" );

               cfg.registerConfigOption ( "schedule-smart-priority", NEW Config::FlagOption( MultiPriorityQueue::_useSmartPriority ), "Smart priority queue propagates high priorities to predecessors");
               cfg.registerArgOption( "schedule-smart-priority", "schedule-smart-priority" );

               cfg.registerConfigOption( "schedule-max-priority", NEW Config::PositiveVar( MultiPriorityQueue::_maxPriority ),
                                         "Defines max priority which can be used in the program" );
               cfg.registerArgOption( "schedule-max-priority", "schedule-max-priority" );

               cfg.registerConfigOption( "schedule-num-queues", NEW Config::PositiveVar( MultiPriorityQueue::_numQueues ),
                                         "Defines number of queues that will be used by the scheduler" );
               cfg.registerArgOption( "schedule-num-queues", "schedule-num-queues" );

            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW MultiPriorityQueue());
            }
      };

   }
}

DECLARE_PLUGIN("sched-bf",nanos::ext::BFSchedPlugin);
