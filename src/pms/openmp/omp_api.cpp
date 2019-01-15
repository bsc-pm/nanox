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

#include "basethread.hpp"
#include "threadteam.hpp"
#include "system.hpp"
#include "omp_wd_data.hpp"
#include "omp_threadteam_data.hpp"
#include "nanos_omp.h"

using namespace nanos;
using namespace nanos::OpenMP;

extern "C"
{
   NANOS_API_DEF(int, omp_get_num_threads, ( void ))
   {
      return myThread->getTeam()->getFinalSize();
   }

   int nanos_omp_get_num_threads ( void ) __attribute__ ((alias ("omp_get_num_threads")));
   int nanos_omp_get_num_threads_ ( void ) __attribute__ ((alias ("omp_get_num_threads")));

   NANOS_API_DEF(int, omp_get_max_threads, ( void ))
   {
      return sys.getPMInterface().getMaxThreads();
   }

   int nanos_omp_get_max_threads ( void ) __attribute__ ((alias ("omp_get_max_threads")));
   int nanos_omp_get_max_threads_ ( void ) __attribute__ ((alias ("omp_get_max_threads")));

   void omp_set_num_threads( int nthreads )
   {
      sys.getPMInterface().setNumThreads( nthreads );
   }

   void omp_set_num_threads_(int *nthreads);
   void omp_set_num_threads_(int *nthreads)
   {
      omp_set_num_threads(*nthreads);
   }

   void nanos_omp_set_num_threads ( int nthreads ) __attribute__ ((alias ("omp_set_num_threads")));
   void nanos_omp_set_num_threads_ ( int nthreads ) __attribute__ ((alias ("omp_set_num_threads")));

   NANOS_API_DEF(int, omp_get_thread_num, ( void ))
   {
      //! \todo check if master always gets a 0 -> ensure condition ?
      if (myThread && myThread->getTeamData()) {
         return myThread->getTeamData()->getId();
      } else {
         return -1;
      }
   }

   int nanos_omp_get_thread_num ( void ) __attribute__ ((alias ("omp_get_thread_num")));
   int nanos_omp_get_thread_num_ ( void ) __attribute__ ((alias ("omp_get_thread_num")));

   NANOS_API_DEF(int, omp_get_num_procs, ( void ))
   {
      return sys.getSMPPlugin()->getCpuCount();
   }

   NANOS_API_DEF(int, omp_in_parallel, ( void ))
   {
      return myThread->getTeam()->getFinalSize() > 1;
   }

   void omp_set_dynamic( int dynamic_threads )
   {
      OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
      data->icvs()->setDynamic((bool) dynamic_threads);
   }

   void omp_set_dynamic_( int* dynamic_threads );
   void omp_set_dynamic_( int* dynamic_threads )
   {
      omp_set_dynamic(*dynamic_threads);
   }

   NANOS_API_DEF(int, omp_get_dynamic, ( void ))
   {
      OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
      return (int) data->icvs()->getDynamic();
   }

   void omp_set_nested ( int nested )
   {
      OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
      data->icvs()->setNested((bool) nested);
   }

   void omp_set_nested_ ( int* nested );
   void omp_set_nested_ ( int* nested )
   {
      omp_set_nested(*nested);
   }

   NANOS_API_DEF(int, omp_get_nested, ( void ))
   {
      OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
      return (int) data->icvs()->getNested();
   }

   void omp_set_schedule ( omp_sched_t kind, int modifier )
   {
      OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
      data->icvs()->setSchedule( LoopSchedule(kind,modifier) );
   }

   void omp_set_schedule_ ( omp_sched_t *kind, int *modifier );
   void omp_set_schedule_ ( omp_sched_t *kind, int *modifier )
   {
      omp_set_schedule(*kind, *modifier);
   }

   NANOS_API_DEF(void, omp_get_schedule, ( omp_sched_t *kind, int *modifier ))
   {
      OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
      const LoopSchedule &schedule = data->icvs()->getSchedule();

      *kind = schedule._kind;
      *modifier = schedule._modifier;
   }

   NANOS_API_DEF(int, omp_get_thread_limit, ( void ))
   {
      return globalState->getThreadLimit();
   }

   void omp_set_max_active_levels( int max_active_levels )
   {
      if (!omp_in_parallel() )
         globalState->setMaxActiveLevels(max_active_levels);
   }

   void omp_set_max_active_levels_( int *max_active_levels );
   void omp_set_max_active_levels_( int *max_active_levels )
   {
      omp_set_max_active_levels(*max_active_levels);
   }

   NANOS_API_DEF(int, omp_get_max_active_levels, ( void ))
   {
      return globalState->getMaxActiveLevels();
   }

   NANOS_API_DEF(int, omp_get_level, ( void ))
   {
      return getMyThreadSafe()->getTeam()->getLevel();
   }

   int omp_get_ancestor_thread_num ( int level )
   {
      ThreadTeam* ancestor = getMyThreadSafe()->getTeam();
      int currentLevel = ancestor->getLevel();

      if ( level >= 0 && level <= currentLevel ) {
         while ( level != currentLevel ) {
            ancestor = ancestor->getParent();
            currentLevel = ancestor->getLevel();
         }
         int id = ancestor->getCreatorId();
         ensure ( id != -1, "Error in OpenMP Team initialization, team creator id was not set" );
         return id;
      }
      return -1;
   }

   int omp_get_ancestor_thread_num_(int* level);
   int omp_get_ancestor_thread_num_(int* level)
   {
      return omp_get_ancestor_thread_num(*level);
   }

   int omp_get_team_size( int level )
   {
      ThreadTeam* ancestor = getMyThreadSafe()->getTeam();
      int currentLevel = ancestor->getLevel();

      if ( level >= 0 && level <= currentLevel ) {
         while ( level != currentLevel ) {
            ancestor = ancestor->getParent();
            currentLevel = ancestor->getLevel();
         }
         return ancestor->size();
      }
      return -1;
   }

   int omp_get_team_size_ ( int *level );
   int omp_get_team_size_ ( int *level )
   {
      return omp_get_team_size(*level);
   }

   NANOS_API_DEF(int, omp_get_active_level, ( void ))
   {
      return ((OmpThreadTeamData &)getMyThreadSafe()->getTeam()->getThreadTeamData()).getActiveLevel();
   }

   NANOS_API_DEF(int, omp_in_final, ( void ))
   {
      OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
      return (int)data->isFinal();
   }

   NANOS_API_DEF(int, nanos_omp_get_num_threads_next_parallel, ( int threads_requested ))
   {
      OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
      if ( threads_requested <= 0 ) {
         int avail_cpus = sys.getThreadManager()->borrowResources();
         if ( avail_cpus <= 0 ) {
            // If ThreadManager is disabled (default) and the user did not specify nthreads:
            threads_requested = data->icvs()->getNumThreads();
         } else {
            // ThreadManager is enabled:
            threads_requested = avail_cpus;
         }
      }

      int num_threads = 0;
      int threads_busy = 1; // FIXME: Should we keep track of it?
      int active_parallel_regions = getMyThreadSafe()->getTeam()->getLevel();
      int threads_available = globalState->getThreadLimit() - threads_busy + 1;

      if ( active_parallel_regions >= 1 && !data->icvs()->getNested() ) {
         num_threads = 1;
      }
      else if ( active_parallel_regions == globalState->getMaxActiveLevels() ) {
         num_threads = 1;
      }
      else if ( threads_requested > threads_available ) {
         num_threads = threads_available;
      }
      else {
         num_threads = threads_requested;
      }

      return num_threads;
   }
}
