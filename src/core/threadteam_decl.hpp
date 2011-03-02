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

#ifndef _NANOS_THREAD_TEAM_DECL_H
#define _NANOS_THREAD_TEAM_DECL_H

#include <vector>
#include "basethread_decl.hpp"
#include "schedule_decl.hpp"
#include "barrier_decl.hpp"


namespace nanos
{

   class ThreadTeam
   {
      private:
         std::vector<BaseThread *>    _threads;
         int                          _idleThreads;
         int                          _numTasks;
         Barrier &                    _barrier;
         int                          _singleGuardCount;
         SchedulePolicy &             _schedulePolicy;
         ScheduleTeamData *           _scheduleData;
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
         ThreadTeam ( int maxThreads, SchedulePolicy &policy, ScheduleTeamData *data, Barrier &barrier );

         /*! \brief ThreadTeam destructor
          */
         ~ThreadTeam ();

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

         BaseThread & getThread ( int i ) const;
         BaseThread & getThread ( int i );

         const BaseThread & operator[]  ( int i ) const;
         BaseThread & operator[]  ( int i );

         /*! \brief adds a thread to the team pool, returns the thread id in the team
          *
          */
         unsigned addThread ( BaseThread *thread );

         void barrier();

         bool singleGuard( int local );

         ScheduleTeamData * getScheduleData() const;
         SchedulePolicy & getSchedulePolicy() const;
   };

}

#endif
