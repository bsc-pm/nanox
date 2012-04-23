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

      class Priority : public SchedulePolicy
      {
         private:
            struct TeamData : public ScheduleTeamData
            {
               WDPriorityQueue           _readyQueue;
               
               TeamData () : ScheduleTeamData(), _readyQueue( Priority::_optimise, Priority::_fifo ) {}
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
            Priority() : SchedulePolicy("Priority First") {}
            virtual ~Priority () {}
   
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
      
      bool Priority::_optimise = true;
      bool Priority::_fifo = true;

      class PrioritySchedPlugin : public Plugin
      {

      public:
         PrioritySchedPlugin() : Plugin( "Priority scheduling Plugin",1 ) {}

         virtual void config ( Config &cfg )
         {
            cfg.setOptionsSection( "Priority module", "Priority scheduling module" );
            //cfg.registerConfigOption ( "priority-optimise", NEW Config::FlagOption( Priority::_optimise ), "Insert WDs right in the back of the queue if they have the same or lower priority than the element in the back.");
            //cfg.registerArgOption( "priority-optimise", "priority-optimise" );
            
            cfg.registerConfigOption ( "priority-fifo", NEW Config::FlagOption( Priority::_fifo ), "When enabled (default behaviour), WDs with the same priority are inserted after the current ones.");
            cfg.registerArgOption( "priority-fifo", "priority-fifo" );
         }

         virtual void init() {
            sys.setDefaultSchedulePolicy(NEW Priority());
         }
      };

   }
}

DECLARE_PLUGIN("sched-priority",nanos::ext::PrioritySchedPlugin);
