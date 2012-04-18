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

      /*!
       * \brief Smart Priority scheduler.
       * This priority-based scheduler propagates the priority of a task to its
       * preceeding tasks (those the task depends on).
       */
      class SmartPriority : public SchedulePolicy
      {
         private:
            Lock _lock;
         
         private:
            struct TeamData : public ScheduleTeamData
            {
               WDPriorityQueue           _readyQueue;
               
               TeamData () : ScheduleTeamData (), _readyQueue( SmartPriority::_optimise, SmartPriority::_fifo ) {}
               ~TeamData () {}
            };
         public:
            /*! \brief When this is enabled, elements with the same priority
             * as the one in the back will be inserted at the back.
             * \note When this is enabled, it will override the LIFO behaviour
             * in the above case.
             */
            static bool _optimise;
            //! \brief Insert WDs with the same after the current ones?
            static bool _fifo;
   
         public:
            SmartPriority () : SchedulePolicy ( "SmartPriority" ) {}
            virtual ~SmartPriority () {}
            
            /*!
             * \brief This method performs the main task of the smart priority
             * scheduler, which is to propagate the priority of a WD to its
             * immediate predecessors. It is meant to be invoked from
             * DependenciesDomain::submitWithDependenciesInternal.
             * \param [in/out] predecessor The preceeding DependableObject.
             * \param [in] successor DependableObject whose WD priority has to be
             * propagated.
             */
            void successorFound( DependableObject *predecessor, DependableObject *successor )
            {
               debug( "SmartPriority::successorFound" );
               if ( predecessor == NULL ) {
                  debug( "SmartPriority::successorFound predecessor is NULL" );
                  return;
               }
               if ( successor == NULL ) {
                  debug( "SmartPriority::successorFound successor is NULL" );
                  return;
               }
               
               WD *pred = ( WD* ) predecessor->getRelatedObject();
               if ( pred == NULL ) {
                  debug( "SmartPriority::successorFound predecessor->getRelatedObject() is NULL" )
                  return;
               }
               
               WD *succ = ( WD* ) successor->getRelatedObject();
               if ( succ == NULL ) {
                  fatal( "SmartPriority::successorFound  successor->getRelatedObject() is NULL" );
               }
               
               //debug( "Predecessor[" << pred->getId() << "]" << pred << ", Successor[" << succ->getId() << "]" );
               
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
                  TeamData &tdata = ( TeamData & ) *nanos::myThread->getTeam()->getScheduleData();
                  tdata._readyQueue.reorderWD( pred );
               }
            }
   
         private:
            
            virtual size_t getTeamDataSize () const { return sizeof ( TeamData ) ; }
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
               TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();
               tdata._readyQueue.push( &wd );
            }
            
            virtual WD *atSubmit ( BaseThread *thread, WD &newWD )
            {
               queue( thread, newWD );
               return 0;
            }
            
            WD * atIdle ( BaseThread *thread )
            {
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
              
               return tdata._readyQueue.pop( thread );
            }
      };
      
      bool SmartPriority::_optimise = true;
      bool SmartPriority::_fifo = true;

      class SmartPrioritySchedPlugin : public Plugin
      {

      public:
         SmartPrioritySchedPlugin() : Plugin( "SmartPriority scheduling Plugin",1 ) {}

         virtual void config ( Config &cfg )
         {
            cfg.setOptionsSection( "SmartPriority module", "SmartPriority scheduling module" );
            cfg.registerConfigOption ( "smartpriority-optimise", NEW Config::FlagOption( SmartPriority::_optimise ), "Insert WDs right in the back of the queue if they have the same or lower priority than the element in the back.");
            cfg.registerArgOption( "smartpriority-optimise", "smartpriority-optimise" );
            
            cfg.registerConfigOption ( "smartpriority-fifo", NEW Config::FlagOption( SmartPriority::_fifo ), "When enabled (default behaviour), WDs with the same priority are inserted after the current ones.");
            cfg.registerArgOption( "smartpriority-fifo", "smartpriority-fifo" );
         }

         virtual void init() {
            sys.setDefaultSchedulePolicy(NEW SmartPriority());
         }
      };

   }
}

DECLARE_PLUGIN("sched-smartpriority",nanos::ext::SmartPrioritySchedPlugin);
