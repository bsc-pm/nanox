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

using namespace nanos;

/*! class TaskStealData
  * \brief Specialization of SchedulingData class for CILK-like scheduler
  */


class TaskStealData : public SchedulingData
{

      friend class TaskStealPolicy;

   protected:
      //  int schId;
      /*! queue of ready tasks to be executed */
      WDDeque readyQueue;

   public:
      // constructor
      TaskStealData ( int id = 0 ) : SchedulingData ( id ) {}

      //TODO: copy & assigment costructor

      // destructor
      ~TaskStealData() {}
};

/*! class TaskStealPolicy
  * \brief Implements a CILK-like scheduler
  */

class TaskStealPolicy : public SchedulingGroup
{

   public:
      // constructor
      TaskStealPolicy() : SchedulingGroup ( "task-steal-sch" ) {}

      TaskStealPolicy ( int groupsize ) : SchedulingGroup ( "task-steal-sch", groupsize ) {}

      // TODO: copy and assigment operations
      // destructor
      virtual ~TaskStealPolicy() {}

      virtual WD *atCreation ( BaseThread *thread, WD &newWD );
      virtual WD *atIdle ( BaseThread *thread );
      virtual void queue ( BaseThread *thread, WD &wd );
      virtual SchedulingData * createMemberData ( BaseThread &thread );
};

/*! \fn queue( BaseThread *thread, WD &wd )
  \brief Enqueue a work descriptor in the readyQueue of the passed thread
  \param thread pointer to the thread to which readyQueue the task must be appended
  \param wd a reference to the work descriptor to be enqueued
  \sa TaskStealData, WD and BaseThread
*/
void TaskStealPolicy::queue ( BaseThread *thread, WD &wd )
{
   TaskStealData *data = ( TaskStealData * ) thread->getSchedulingData();
   data->readyQueue.push_front ( &wd );
}

/*! \fn atCreation( BaseThread *thread, WD &newWD )
  \brief Function called when a new task must be created: the new created task is directly executed (Depth-First policy)
  \param thread pointer to the thread to which belongs the new task
  \param wd a reference to the work descriptor of the new task
  \sa WD and BaseThread
*/
WD * TaskStealPolicy::atCreation ( BaseThread *thread, WD &newWD )
{
   //NEW: now it does not enqueue the created task, but it moves down to the generated son: DEPTH-FIRST
   return &newWD;
}

/*! \fn atIdle( BaseThread *thread)
  \brief Function called by the scheduler when a thread becomes idle to schedule it: implements the CILK-scheduler algorithm
  \param thread pointer to the thread to be scheduled
  \sa BaseThread
*/
WD * TaskStealPolicy::atIdle ( BaseThread *thread )
{
   WorkDescriptor * wd;

   TaskStealData *data = ( TaskStealData * ) thread->getSchedulingData();
   /*!
      First try to schedule the thread with a task from its queue
    */

   if ( ( wd = data->readyQueue.pop_front ( thread ) ) != NULL ) {
      return wd;
   } else {
      /*!
         If the local queue is empty, try to steal the parent (possibly enqueued in the queue of another thread)
       */
      if ( ( wd = ( thread->getCurrentWD() )->getParent() ) != NULL ) {
         //removing it from the queue. Try to remove from one queue: if someone move it, I stop looking for it to avoid ping-pongs.
         if ( ( wd->isEnqueued() ) == true && ( ! ( wd )->isTied() || ( wd )->isTiedTo() == thread ) ) { //not in queue = in execution, in queue = not in execution
            if ( wd->getMyQueue()->removeWD ( wd ) == true ) { //found it!
               return wd;
            }
         }
      }

      /*!
         If also the parent is NULL or if someone moved it to another queue while I was trying to steal it, try to steal tasks from other queues
         \warning other queues are checked cyclically: should be random
       */
      int newposition = ( ( data->getSchId() ) + 1 ) % getSize();

      while ( ( newposition != data->getSchId() ) && ( ( wd = ( ( ( TaskStealData * ) ( getMemberData ( newposition ) ) )->readyQueue.pop_back ( thread ) ) ) == NULL ) ) {
         newposition = ( newposition + 1 ) % getSize();
      }

      return wd;
   }
}

/*! \fn createMemberData ( BaseThread &thread )
  \brief creates a new instance of TaskStealData
  \param thread unused argument (for now all threads performs the same scheduling algorithm)
  \sa BaseThread and TaskStealData
*/
SchedulingData * TaskStealPolicy::createMemberData ( BaseThread &thread )
{
   return new TaskStealData();
}

/*! \fn createTaskStealPolicy ( )
  \brief creates a new instance of TaskStealPolicy
*/

SchedulingGroup * createTaskStealPolicy ()
{
   return new TaskStealPolicy();
}

SchedulingGroup * createTaskStealPolicy ( int groupsize )
{
   return new TaskStealPolicy ( groupsize );
}

class CilkSchedPlugin : public Plugin
{

   public:
      CilkSchedPlugin() : Plugin( "Cilk scheduling Plugin",1 ) {}

      virtual void init() {
         sys.setDefaultSGFactory( createTaskStealPolicy );
      }
};

CilkSchedPlugin NanosXPlugin;
