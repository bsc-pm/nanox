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
                if ( _usePriority || _useSmartPriority ) _readyQueue = NEW WDPriorityQueue<>( true /* optimise option */ );
                else _readyQueue = NEW WDDeque();
              }
              ~TeamData () { delete _readyQueue; }
           };

         public:
           static bool       _useStack;
           static bool       _usePriority;
           static bool       _useSmartPriority;

           BreadthFirst() : SchedulePolicy("Breadth First") {}
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
              TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
              if ( _useStack ) return tdata._readyQueue->push_front( &wd );
              else tdata._readyQueue->push_back( &wd );
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
            void successorFound( DependableObject *predecessor, DependableObject *successor )
            {
               debug( "Scheduler::successorFound" );

               if ( ! _useSmartPriority ) return;


               if ( predecessor == NULL || successor == NULL ) return;
               
               WD *pred = ( WD* ) predecessor->getRelatedObject();
               if ( pred == NULL ) return;

               WD *succ = ( WD* ) successor->getRelatedObject();
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

           WD * atIdle ( BaseThread *thread )
           {
              TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
              
              return tdata._readyQueue->pop_front( thread );
           }

           WD * atPrefetch ( BaseThread *thread, WD &current )
           {
              if ( !_usePriority && !_useSmartPriority ) {
                 WD * found = current.getImmediateSuccessor(*thread);
                 return found != NULL ? found : atIdle(thread);
              } else {
                 return atIdle(thread);
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
      };

      bool BreadthFirst::_useStack = false;
      bool BreadthFirst::_usePriority = false;
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
