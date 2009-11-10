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

#ifndef _NANOS_THREAD_TEAM
#define _NANOS_THREAD_TEAM

#include <vector>
#include "basethread.hpp"
#include "schedule.hpp"
#include "barrier.hpp"


namespace nanos
{

   class TeamData
   {
   };

   class ThreadTeam
   {

      private:
         std::vector<BaseThread *> threads;
         int  idleThreads;
         int  numTasks;
         Barrier * barrAlgorithm;
         int  single;

         // disable copy constructor & assignment operation
         ThreadTeam( const ThreadTeam &sys );
         const ThreadTeam & operator= ( const ThreadTeam &sys );

      public:

         ThreadTeam ( int maxThreads, SG &policy ) : idleThreads( 0 ), numTasks( 0 ), single( 0 )
           { threads.reserve( maxThreads ); }

         ~ThreadTeam () { /*TODO*/ }

         unsigned size() const { return threads.size(); }

         const BaseThread & operator[]  ( int i ) const { return *threads[i]; }

         BaseThread & operator[]  ( int i ) { return *threads[i]; }

         void addThread ( BaseThread *thread ) {
            threads.push_back( thread );
         }

         void setBarrAlgorithm( Barrier * barrAlg ) { barrAlgorithm = barrAlg; }

         void barrier() { barrAlgorithm->barrier(); }

         bool singleGuard( int local );
   };

}

#endif
