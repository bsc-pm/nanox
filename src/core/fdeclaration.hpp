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

#ifndef _NANOS_FORWARD_DECLARATION_H
#define _NANOS_FORWARD_DECLARATION_H

namespace nanos
{
   class Accelerator;

   class Barrier;

   class TeamData;
   class BaseThread;

   class CopyData;

   class DependableObject;
   class DOSubmit;
   class DOWait;

   class DependenciesDomain;
   class Dependency;

   class Instrumentor;

   class ProcessingElement;

   class Scheduler;
   class SchedulerStats;
   class ScheduleTeamData;
   class ScheduleThreadData;
   class SchedulePolicy;

   class Slicer;
   class SlicerData;
   class SlicedWD;
   class SlicerDataRepeatN;
   class SlicerDataFor;

   class ConditionChecker;
   class EqualConditionChecker;
   class LessOrEqualConditionChecker;
   class GenericSyncCond;
   class SynchronizedCondition;
   class SingleSyncCond;
   class MultipleSyncCond;

   class System;

   class ThreadTeam;

   class ThrottlePolicy;

   class TrackableObject;

   class SchedulePredicate;
   class WDDeque;

   class Device;
   class DeviceData;
   class WorkDescriptor;

   class WorkGroup;
};

#endif

