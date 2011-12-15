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

              TeamData () : ScheduleTeamData(), _readyQueue() {}
              ~TeamData () {}
           };

         public:
           Priority() : SchedulePolicy("Priority First") {}
           virtual ~Priority () {}

         private:
            
           virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
           virtual size_t getThreadDataSize () const { return 0; }

           virtual ScheduleTeamData * createTeamData ( ScheduleTeamData *preAlloc )
           {
              TeamData *data;

              if ( preAlloc ) data = new (preAlloc) TeamData();
              else data = NEW TeamData();

              return data;
           }

           virtual ScheduleThreadData * createThreadData ( ScheduleThreadData *preAlloc )
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

      class PrioritySchedPlugin : public Plugin
      {

         public:
            PrioritySchedPlugin() : Plugin( "Priority scheduling Plugin",1 ) {}

            virtual void config ( Config &cfg )
            {
               cfg.setOptionsSection( "Priority module", "Priority scheduling module" );
            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW Priority());
            }
      };

   }
}

DECLARE_PLUGIN("sched-priority",nanos::ext::PrioritySchedPlugin);
