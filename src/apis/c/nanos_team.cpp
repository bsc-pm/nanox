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

#include "nanos.h"
#include "system.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "instrumentationmodule_decl.hpp"

using namespace nanos;

nanos_err_t nanos_create_team( nanos_team_t *team, nanos_sched_t sp, unsigned int *nthreads,
                               nanos_constraint_t * constraints, bool reuse, nanos_thread_t *info )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","create_team",NANOS_RUNTIME) );

   try {
      if ( *team ) warning( "pre-allocated team not supported yet" );
      if ( sp ) warning ( "selecting scheduling policy not supported yet");

     /* fourth parameter is equal to false because related threads are not entering new team.
      * They must enter explicetely by calling thread->enterTeam()
      * fifth parameter is equal to true because threads will be all stars
      */
      ThreadTeam *new_team = sys.createTeam( *nthreads,constraints,reuse,false,true );

      *team = new_team;

      *nthreads = new_team->size();

      for ( unsigned i = 0; i < new_team->size(); i++ )
         info[i] = ( nanos_thread_t ) &( *new_team )[i];
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_create_team_mapped ( nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads,
                                       unsigned int *mapping )
{
   return NANOS_UNIMPLEMENTED;
}

nanos_err_t nanos_leave_team ()
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","leave_team",NANOS_RUNTIME) );

   try {
      sys.releaseWorker(myThread);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

nanos_err_t nanos_end_team ( nanos_team_t team )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","end_team",NANOS_RUNTIME) );

   try {
      sys.endTeam((ThreadTeam *)team);
   } catch ( ... ) {
         return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

/*!
   Implements the team barrier by invoking the barrier function of the team.
   The actual barrier algorithm is loaded at the run-time startup.
*/
nanos_err_t nanos_team_barrier ( )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","team_barrier",NANOS_SYNCHRONIZATION) );

   try {
      myThread->getTeam()->barrier();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_team_get_num_starring_threads ( int *n )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_num_starring_threads",NANOS_RUNTIME) );

   try {
      *n = myThread->getTeam()->getNumStarringThreads();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_team_get_starring_threads ( int *n, nanos_thread_t *list_of_threads )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_starring_threads",NANOS_RUNTIME) );

   try {
      *n = myThread->getTeam()->getStarringThreads( (BaseThread **) list_of_threads );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_team_get_num_supporting_threads ( int *n )
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_num_supporting_threads",NANOS_RUNTIME) );

   try {
      *n = myThread->getTeam()->getNumSupportingThreads();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_team_get_supporting_threads ( int *n, nanos_thread_t *list_of_threads)
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_supporting_threads",NANOS_RUNTIME) );

   try {
      *n = myThread->getTeam()->getSupportingThreads( (BaseThread **) list_of_threads );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}
