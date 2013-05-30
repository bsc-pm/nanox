
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

#include "schedule_decl.hpp"
#include "workdescriptor_decl.hpp"
#include "atomic.hpp"
#include "functors.hpp"
#include <algorithm>
#include "synchronizedcondition_fwd.hpp"
#include "system_fwd.hpp"
#include "basethread_decl.hpp"

using namespace nanos;

inline bool Scheduler::checkBasicConstraints ( WD &wd, BaseThread &thread )
{
   return wd.canRunIn(*thread.runningOn()) && ( !wd.isTied() || wd.isTiedTo() == &thread ) && wd.tryAcquireCommutativeAccesses();
}

inline unsigned int SchedulerConf::getNumSpins () const
{
   return _numSpins;
}

inline void SchedulerConf::setNumSpins ( const unsigned int num )
{
   _numSpins = num;
}
   
inline int SchedulerConf::getNumSleeps () const
{
   return _numSleeps;
}

inline void SchedulerConf::setNumSleeps ( const unsigned int num )
{
   _numSleeps = num;
}

inline int SchedulerConf::getTimeSleep () const
{
   return _timeSleep;
}

inline void SchedulerConf::setSchedulerEnabled ( bool value )
{
   _schedulerEnabled = value;
}

inline bool SchedulerConf::getSchedulerEnabled () const
{
   return _schedulerEnabled;
}
inline const std::string & SchedulePolicy::getName () const
{
   return _name;
}

inline WD * SchedulePolicy::atBeforeExit  ( BaseThread *thread, WD &current, bool schedule )
{
   return 0;
}

inline WD * SchedulePolicy::atAfterExit   ( BaseThread *thread, WD *current )
{
   return atIdle( thread );
}

inline WD * SchedulePolicy::atBlock       ( BaseThread *thread, WD *current )
{
   return atIdle( thread );
}

inline WD * SchedulePolicy::atYield       ( BaseThread *thread, WD *current)
{
   return atIdle( thread );
}

inline WD * SchedulePolicy::atWakeUp      ( BaseThread *thread, WD &wd )
{
   // Ticket #716: execute earlier tasks that have been waiting for children
   // If the WD was waiting for something
   if ( wd.started() ) {
      BaseThread * prefetchThread = NULL;
      // Check constraints since they won't be checked in Schedule::wakeUp
      if ( Scheduler::checkBasicConstraints ( wd, *thread ) ) {
         prefetchThread = thread;
      }
      else
         prefetchThread = wd.isTiedTo();
      
      // Returning the wd here makes the application to hang
      // Use prefetching instead.
      if ( prefetchThread != NULL ) {
         prefetchThread->addNextWD( &wd );
         
         return NULL;
      }
   }
   
   // otherwise, as usual
   queue( thread, wd );
   
   return NULL;
}

inline WD * SchedulePolicy::atPrefetch    ( BaseThread *thread, WD &current )
{
   return atIdle( thread );
}

inline void SchedulePolicySuccessorFunctor::operator() ( DependableObject *predecessor, DependableObject *successor )
{
   _obj.successorFound( predecessor, successor );
}

#endif

