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
#include "instrumentationmodule_decl.hpp"

using namespace nanos;

NANOS_API_DEF(const char *, nanos_get_default_architecture, ())
{
   return (sys.getDefaultArch()).c_str();
}

NANOS_API_DEF(const char *, nanos_get_pm, ())
{
   return (sys.getPMInterface()).getDescription().c_str();
}

NANOS_API_DEF(nanos_err_t, nanos_get_num_running_tasks, ( int *num ))
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","get_num_running_tasks",RUNTIME) );

   try {
      *num = sys.getRunningTasks();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(const char *, nanos_get_default_scheduler, ())
{
   return (sys.getDefaultSchedule()).c_str();
}

NANOS_API_DEF(nanos_err_t, nanos_stop_scheduler, ())
{
   try {
      sys.stopScheduler();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_start_scheduler, ())
{
   try {
      sys.startScheduler();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_scheduler_enabled, ( bool *res ))
{
   try {
      *res = sys.isSchedulerStopped();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_wait_until_threads_paused, ())
{
   try {
      sys.waitUntilThreadsPaused();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_wait_until_threads_unpaused, ())
{
   try {
      sys.waitUntilThreadsUnpaused();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_delay_start, ())
{
   try {
      sys.setDelayedStart(true);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_start, ())
{
   try {
      sys.start();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_finish, ()) 
{
   try {
      sys.finish();
   } catch ( ... ) { 
      return NANOS_UNKNOWN_ERR;
   }   

   return NANOS_OK;
}

