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

namespace nanos {
   namespace ext {

      class DistributedBFPolicy : public SchedulePolicy
      {
         public:
            using SchedulePolicy::queue;
            static bool       _usePriority;
            static bool       _useSmartPriority;
         private:
            /** \brief DistributedBF Scheduler data associated to each thread
              *
              */
            struct ThreadData : public ScheduleThreadData
            {
               /*! queue of ready tasks to be executed */
               WDPool *_readyQueue;

               ThreadData () : ScheduleThreadData(), _readyQueue( NULL )
               {
                 if ( _usePriority || _useSmartPriority ) _readyQueue = NEW WDPriorityQueue<>( true /* enableDeviceCounter */, true /* optimise option */ );
                 else _readyQueue = NEW WDDeque( true /* enableDeviceCounter */ );
               }
               virtual ~ThreadData () { delete _readyQueue; }
            };

            /* disable copy and assigment */
            explicit DistributedBFPolicy ( const DistributedBFPolicy & );
            const DistributedBFPolicy & operator= ( const DistributedBFPolicy & );

         public:
            // constructor
            DistributedBFPolicy() : SchedulePolicy ( "Cilk" )
            {
              _usePriority = _usePriority && sys.getPrioritiesNeeded();
            }

            // destructor
            virtual ~DistributedBFPolicy() {}

            virtual size_t getTeamDataSize () const { return 0; }
            virtual size_t getThreadDataSize () const { return sizeof(ThreadData); }

            virtual ScheduleTeamData * createTeamData ()
            {
               return 0;
            }

            virtual ScheduleThreadData * createThreadData ()
            {
               return NEW ThreadData();
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
            void atSuccessor   ( DependableObject &successor, DependableObject &predecessor )
            {
               //debug( "Scheduler::successorFound" );

               if ( ! _useSmartPriority ) return;


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
                  ThreadData &tdata = (ThreadData &) *myThread->getTeamData()->getScheduleData();
                  WDPriorityQueue<> *q = (WDPriorityQueue<> *) tdata._readyQueue;
                  q->reorderWD( pred );
               }
            }

            /*!
            *  \brief Enqueue a work descriptor in the readyQueue of the passed thread
            *  \param thread pointer to the thread to which readyQueue the task must be appended
            *  \param wd a reference to the work descriptor to be enqueued
            *  \sa ThreadData, WD and BaseThread
            */
            virtual void queue ( BaseThread *thread, WD &wd )
            {
               BaseThread *targetThread = wd.isTiedTo();
               if ( targetThread ) targetThread->addNextWD(&wd);
               else {
                  ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
                  data._readyQueue->push_front( &wd );
                  sys.getThreadManager()->unblockThread(thread);
               }
            }

            /*!
            *  \brief Function called when a new task must be created: the new created task
            *          is directly queued (Breadth-First policy)
            *  \param thread pointer to the thread to which belongs the new task
            *  \param wd a reference to the work descriptor of the new task
            *  \sa WD and BaseThread
            */
            virtual WD * atSubmit ( BaseThread *thread, WD &newWD )
            {
               queue(thread,newWD);

               return 0;
            }


            virtual WD *atIdle ( BaseThread *thread, int numSteal );

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

            bool testDequeue()
            {
               ThreadData &data = ( ThreadData & ) *myThread->getTeamData()->getScheduleData();
               return data._readyQueue->testDequeue();
            }
      };



      /*! 
       *  \brief Function called by the scheduler when a thread becomes idle to schedule it: implements the CILK-scheduler algorithm
       *  \param thread pointer to the thread to be scheduled
       *  \sa BaseThread
       */
      WD * DistributedBFPolicy::atIdle ( BaseThread *thread, int numSteal )
      {
         WorkDescriptor * wd = thread->getNextWD();

         if ( wd ) return wd;

         WorkDescriptor * next = NULL; 

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();

         //! First try to schedule the thread with a task from its queue
         if ( ( wd = data._readyQueue->pop_front ( thread ) ) != NULL ) {
            return wd;
         } else {
            //! If the local queue is empty, try to steal the parent (possibly enqueued in the queue of another thread)
            if ( ( wd = thread->getCurrentWD()->getParent() ) != NULL ) {
               WDPool *pq; //!< Parent queue
               //! Removing it from the queue, if someone move it stop looking for it to avoid ping-pongs
               if ( (pq = wd->getMyQueue()) != NULL ) {
                  //! Not in queue = in execution, in queue = not in execution
                  if ( pq->removeWD( thread, wd, &next ) ) return next; //!< Found it!
               }
            }

            //! If also the parent is NULL or if someone moved it to another queue while was trying to steal it, 
            //! try to steal tasks from other queues
            //! \warning other queues are checked cyclically: should be random
            int size = thread->getTeam()->getFinalSize();
            int thid = rand() % size;
            int count = 0;
            wd = NULL;

            do {
               thid = ( thid + 1 ) % size;

               BaseThread &victim = thread->getTeam()->getThread(thid);

               if ( victim.getTeam() != NULL ) {
                 ThreadData &tdata = ( ThreadData & ) *victim.getTeamData()->getScheduleData();
                 wd = tdata._readyQueue->pop_back ( thread );
               }

               count++;

            } while ( wd == NULL && count < size );

            return wd;
         }
      }

      bool DistributedBFPolicy::_usePriority = true;
      bool DistributedBFPolicy::_useSmartPriority = false;

      class DistributedBFSchedPlugin : public Plugin
      {
         public:
            DistributedBFSchedPlugin() : Plugin( "Distributed Breadth-First scheduling Plugin",1 ) {}

            virtual void config( Config& cfg )
            {
               
               cfg.setOptionsSection( "DBF module", "Distributed Breadth-first scheduling module" );

               cfg.registerConfigOption ( "schedule-priority", NEW Config::FlagOption( DistributedBFPolicy::_usePriority ), "Priority queue used as ready task queue");
               cfg.registerArgOption( "schedule-priority", "schedule-priority" );

               cfg.registerConfigOption ( "schedule-smart-priority", NEW Config::FlagOption( DistributedBFPolicy::_useSmartPriority ), "Smart priority queue propagates high priorities to predecessors");
               cfg.registerArgOption( "schedule-smart-priority", "schedule-smart-priority" );

               
            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW DistributedBFPolicy());
            }
      };

   }
}

DECLARE_PLUGIN("sched-dbf",nanos::ext::DistributedBFSchedPlugin);
