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

#ifndef _NANOS_SCHEDULE
#define _NANOS_SCHEDULE

#include <stddef.h>
#include <string>

#include "workdescriptor.hpp"
#include "wddeque.hpp"
#include "basethread.hpp"
#include "atomic.hpp"
#include "functors.hpp"
#include <algorithm>

namespace nanos
{

// singleton class to encapsulate scheduling data and methods

   class GenericSyncCond;
   typedef void SchedulerHelper ( WD *oldWD, WD *newWD, void *arg);

   class Scheduler
   {
      private:
         static void queue ( WD &wd );
         static void switchHelper (WD *oldWD, WD *newWD, void *arg);
         static void exitHelper (WD *oldWD, WD *newWD, void *arg);
         
         template<class behaviour>
         static void idleLoop (void);

      public:
         static void inlineWork ( WD *work );

         static void submit ( WD &wd );
         static void switchTo ( WD *to );
         static void exitTo ( WD *next );
         static void switchToThread ( BaseThread * thread );

         static void workerLoop ( void );
         static void yield ( void );

         static void exit ( void );

         static void waitOnCondition ( GenericSyncCond *condition );
         static void wakeUp ( WD *wd );
   };

   class System;
   class SchedulerStats
   {
      friend class Scheduler;
      friend class System;
      
      private:
        Atomic<int>          _createdTasks;
        Atomic<int>          _readyTasks;
        Atomic<int>          _idleThreads;
        Atomic<int>          _totalTasks;

      public:
         SchedulerStats () : _createdTasks(0), _idleThreads(0), _totalTasks(1) {}
   };

   class ScheduleTeamData {
      public:
         ScheduleTeamData() {}
         virtual ~ScheduleTeamData() {}
   };

   class ScheduleThreadData {
      public:
         ScheduleThreadData() {}
         virtual ~ScheduleThreadData() {}
   };

   class SchedulePolicy
   {
      private:
         std::string    _name;
         
      public:

         SchedulePolicy ( std::string &name ) : _name(name) {}
         SchedulePolicy ( const char *name ) : _name(name) {}
         
         virtual ~SchedulePolicy () {};

         const std::string & getName () const { return _name; }

         virtual size_t getTeamDataSize() const = 0;
         virtual size_t getThreadDataSize() const = 0;
         virtual ScheduleTeamData * createTeamData ( ScheduleTeamData *preAlloc ) = 0;
         virtual ScheduleThreadData * createThreadData ( ScheduleThreadData *preAlloc ) = 0;
         
         virtual WD * atSubmit   ( BaseThread *thread, WD &wd ) = 0;
         virtual WD * atIdle     ( BaseThread *thread ) = 0;
         virtual WD * atExit     ( BaseThread *thread, WD *current ) { return atIdle( thread ); }
         virtual WD * atBlock    ( BaseThread *thread, WD *current ) { return atIdle( thread ); }
         virtual WD * atYield    ( BaseThread *thread, WD *current) { return atIdle(thread); };
         virtual WD * atWakeUp   ( BaseThread *thread, WD &wd ) { return 0; }

         virtual void queue ( BaseThread *thread, WD &wd )  = 0;
   };
   
};

#endif

