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

#ifndef _NANOS_THREAD_TEAM_DECL_H
#define _NANOS_THREAD_TEAM_DECL_H

#include <vector>
#include <list>
#include "basethread_decl.hpp"
#include "schedule_decl.hpp"
#include "barrier_decl.hpp"
#include "task_reduction_decl.hpp"


namespace nanos {

   class ThreadTeamData
   {
      private:
         /*! \brief ThreadTeamData copy constructor (private)
          */
         ThreadTeamData ( ThreadTeamData &ttd );
         /*! \brief ThreadTeamData copy assignment operator (private)
          */
         ThreadTeamData& operator=  ( ThreadTeamData &ttd );
      public:
         /*! \brief ThreadTeamData default constructor
          */
         ThreadTeamData() {}
         /*! \brief ThreadTeamData destructor
          */
         virtual ~ThreadTeamData() {}

         virtual void init( ThreadTeam *parent ) {}
   };

   class ThreadTeam
   {
      private:
         typedef std::list<nanos_reduction_t*>     ReductionList;  /**< List of Reduction op's (Bursts) */
         typedef std::map<unsigned, BaseThread *>  ThreadTeamList; /**< List of team members */
         typedef std::map<unsigned, bool>          ThreadTeamIdList; /**< List of team members */
         typedef std::list<TaskReduction *>        task_reduction_list_t;  //< List of task reductions type
         typedef std::set<BaseThread *>            ThreadSet;

         ThreadTeamList               _threads;          /**< Threads that make up the team */
         ThreadTeamIdList             _idList;           /**< List of id usage (reusing old id's) */
         ThreadSet                    _expectedThreads;  /**< Threads expected to form the team */
         Atomic<size_t>               _starSize;
         int                          _idleThreads;
         int                          _numTasks;
         Barrier &                    _barrier;
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
         int                 _singleGuardCount;
#else
         volatile int                 _singleGuardCount;
#endif
         SchedulePolicy &             _schedulePolicy;
         ScheduleTeamData *           _scheduleData;
         ThreadTeamData &             _threadTeamData;
         ThreadTeam *                 _parent;           /**< Parent ThreadTeam */
         int                          _level;            /**< Nesting level of the team */
         int                          _creatorId;        /**< Team Id of the thread that created the team */
         nanos_ws_desc_t             *_wsDescriptor;     /**< Worksharing queue (pointer managed due specific atomic op's over these pointers) */
         ReductionList                _redList;          /**< Reduction List */
         Lock                         _lock;
      private:

         /*! \brief ThreadTeam default constructor (disabled)
          */
         ThreadTeam();

         /*! \brief ThreadTeam copy constructor (disabled)
          */
         ThreadTeam( const ThreadTeam &sys );

         /*! \brief ThreadTeam copy assignment operator (disabled)
          */
         const ThreadTeam & operator= ( const ThreadTeam &sys );

      public:
         /*! \brief ThreadTeam constructor - 1
          */
         ThreadTeam ( int maxThreads, SchedulePolicy &policy, ScheduleTeamData *data, Barrier &barrier, ThreadTeamData & ttd, ThreadTeam * parent );

         /*! \brief ThreadTeam destructor
          */
         ~ThreadTeam ();

         // atomic access
         void lock();
         void unlock();

         unsigned size() const;

         /*! \brief Initializes team structures dependent on the number of threads.
          *
          *  This method initializes the team structures that depend on the number of threads.
          *  It *must* be called after all threads have entered the team
          *  It *must* be called by a single thread
          */
         void init ();

         /*! This method should be called when there's a change in the team size to readjust all structures
          *  \warn Not implemented yet!
          */
         void resized ();

         const BaseThread & getThread ( int i ) const;
         BaseThread & getThread ( int i );

         const BaseThread & operator[]  ( int i ) const;
         BaseThread & operator[]  ( int i );

         /*! \brief adds a thread to the team pool, returns the thread id in the team
          *  \param star: If true thread will adopt an starring role within the team
          *  \param creator: If true the thread ID is set as the creatorID for this team
          */
         unsigned addThread ( BaseThread *thread, bool star = false, bool creator = false );
         /*! \brief removes a thread from the team pool
          *  
          *  \returns Team final size
          */
         size_t  removeThread ( unsigned id );

         /*! \brief removes and returns the last thread from the team pool
          */
         BaseThread * popThread();

         void barrier();

         bool singleGuard( int local );
         bool enterSingleBarrierGuard( int local );
         void releaseSingleBarrierGuard( void );
         void waitSingleBarrierGuard( int local );

         ScheduleTeamData * getScheduleData() const;
         SchedulePolicy & getSchedulePolicy() const;

        /*! \brief Returns the ThreadTeamData
         */
         ThreadTeamData & getThreadTeamData() const;

        /*! \brief Returns the parent of this team, if any
         */
         ThreadTeam * getParent() const;

        /*! \brief returns the depth level of the Team
         */
         int getLevel() const;

        /*! \brief returns the team's creator Id, -1 if not set
         */
         int getCreatorId() const;

        /*! \brief returns WorkSharing 
         */
         nanos_ws_desc_t  *getWorkSharingDescriptor( void );

        /*! \brief returns WorkSharing Address
         */
         nanos_ws_desc_t **getWorkSharingDescriptorAddr( void );

        /*! \brief Return number of starring thread
         */
         unsigned getNumStarringThreads( void ) const;

        /*! \brief Return number of starring thread and fill a vector of pointers to threads
         */
         unsigned getStarringThreads( BaseThread *list_of_threads[] ) const;

        /*! \brief Return number of supporting thread
         */
         unsigned getNumSupportingThreads( void ) const;

        /*! \brief Return number of supporting thread and fill a vector of pointers to threads
         */
         unsigned getSupportingThreads( BaseThread *list_of_threads[] ) const;

        /*! \brief Register a new reduction to execute at next barrier 
         */
         void createReduction( nanos_reduction_t *red );

        /*! \brief Returns private data for a given source data
         */
         void *getReductionPrivateData ( void* s );

        /*! \brief Returns private data for a given source data
         */
         nanos_reduction_t *getReduction ( void* s );

        /*! \brief Clean readuction list
         */
         void cleanUpReductionList ( void );

        /*! \brief Compute reduction
         */
         void computeVectorReductions ( void );

        /*! \brief Get final size
         */
         size_t getFinalSize ( void ) const;

        /*! \brief Check whether team has the expected members
         */
         bool isStable ( void );

        /*! \brief Add an expected member that will enter by itself in the team
         */
         void addExpectedThread( BaseThread *thread );

        /*! \brief Remove an expected team member. It should leave by itself
         */
         void removeExpectedThread( BaseThread *thread );
   };

} // namespace nanos

#endif
