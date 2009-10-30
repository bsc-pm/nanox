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

#ifndef _BASE_THREAD_ELEMENT
#define _BASE_THREAD_ELEMENT

#include "workdescriptor.hpp"
#include "atomic.hpp"

namespace nanos
{

// forward declarations

   class ProcessingElement;

   class SchedulingGroup;

   class SchedulingData;

   class ThreadTeam;

// Threads are binded to a PE for its life-time

   class BaseThread
   {

      private:
         static Atomic<int> idSeed;
         Lock   mlock;

         // Thread info
         int id;
         int cpu_id;

         ProcessingElement *pe;
         WD & threadWD;

         // Thread status
         bool started;
         volatile bool mustStop;
         WD *    currentWD;

         // Team info
         bool  has_team;
         ThreadTeam *team;
         int local_single;

         // scheduling info
         SchedulingGroup *schedGroup;
         SchedulingData  *schedData;

         //disable copy and assigment
         BaseThread( const BaseThread & );
         const BaseThread operator= ( const BaseThread & );

         virtual void run_dependent () = 0;

      public:

         // constructor
         BaseThread ( WD &wd, ProcessingElement *creator=0 ) :
               id( idSeed++ ), cpu_id( id ), pe( creator ), threadWD( wd ), started( false ), mustStop( false ), has_team( false ), team( NULL ), local_single( 0 ) {}

         // destructor
         virtual ~BaseThread() {}

         // atomic access
         void lock () { mlock++; }

         void unlock () { mlock--; }

         virtual void start () = 0;
         void run();
         void stop() { mustStop = true; }

         virtual void join() = 0;
         virtual void bind() {};

         // WD micro-scheduling
         virtual void inlineWork ( WD *work ) = 0;
         virtual void switchTo( WD *work ) = 0;
         virtual void exitTo( WD *work ) = 0;

         // set/get methods
         void setCurrentWD ( WD &current ) { currentWD = &current; }

         WD * getCurrentWD () const { return currentWD; }

         WD & getThreadWD () const { return threadWD; }

         // team related methods
         void reserve() { has_team = 1; }

         void enterTeam( ThreadTeam *newTeam ) { has_team=1; team = newTeam; }

         bool hasTeam() const { return has_team; }

         void leaveTeam() { has_team = 0; team = 0; }

         ThreadTeam * getTeam() const { return team; }

         SchedulingGroup * getSchedulingGroup () const { return schedGroup; }

         SchedulingData * getSchedulingData () const { return schedData; }

         void setScheduling ( SchedulingGroup *sg, SchedulingData *sd )  { schedGroup = sg; schedData = sd; }

         bool isStarted () const { return started; }

         bool isRunning () const { return started && !mustStop; }

         ProcessingElement * runningOn() const { return pe; }

         void associate();

         int getId() { return id; }

         int getCpuId() { return cpu_id; }

         bool singleGuard();
   };

   extern __thread BaseThread *myThread;

}

#endif
