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

/*! \file nanos_sched.cpp
 *  \brief 
 */
#include "nanos.h"
#include "system.hpp"

/*! \defgroup capi_sched Scheduler services.
 *  \ingroup capi
 */
/*! \addtogroup capi_sched
 *  \{
 */

using namespace nanos;

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

NANOS_API_DEF(nanos_err_t, nanos_scheduler_get_stealing, ( bool *res ))
{
   try {
      *res = sys.getDefaultSchedulePolicy()->getStealing();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_scheduler_set_stealing, ( bool value ))
{
   try {
      sys.getDefaultSchedulePolicy()->setStealing( value );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}


/*!
 * \}
 */ 
