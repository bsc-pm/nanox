/*************************************************************************************/
/*      Copyright 2010 Barcelona Supercomputing Center                               */
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

#include "chpl_nanos.h"
#include <assert.h>

#include "system.hpp"
#include "basethread.hpp"
#include "schedule.hpp"
#include "smpdd.hpp"

// // TODO: include chpl headers?
// typedef char * chpl_string;
// typedef bool chpl_bool;
// typedef void (*chpl_fn_p) (void *);
// typedef void * chpl_task_list_p;
// typedef int chpl_fn_int_t;
// typedef int chpl_taskID_t;

extern chpl_fn_p chpl_ftable[];


using namespace nanos;
using namespace nanos::ext;

namespace nanos
{
   namespace Chapel
   {
      static void init()
      {
	 sys.setDelayedStart(true);
      }
   }

   System::Init externInit = Chapel::init;
}

//
// interface function with begin-statement
//
void CHPL_BEGIN(chpl_fn_p fp,
                void* a,
                chpl_bool ignore_serial,  // always add task to pool
                chpl_bool serial_state,
                chpl_task_list_p ltask) {

   assert(!ltask);

   WD * wd = new WD( new SMPDD( fp ), 0, a );
   sys.submit(*wd);
}

// Tasks

void CHPL_TASKING_INIT() 
{
   sys.setInitialMode( System::POOL );
   sys.setUntieMaster(true);
   sys.setNumPEs(maxThreads);
   sys.start();

/*  tp = chpl_alloc(sizeof(thread_private_data_t), CHPL_RT_MD_THREAD_PRIVATE_DATA, 0, 0);
  threadlayer_set_thread_private_data(tp);
  tp->serial_state = false;*/
}


void CHPL_TASKING_EXIT()
{
   sys.finalize();
}

void CHPL_ADD_TO_TASK_LIST(chpl_fn_int_t fid, void* arg,
                           chpl_task_list_p *task_list,
                           int32_t task_list_locale,
                           chpl_bool call_chpl_begin,
                           int lineno, chpl_string filename) {
    chpl_fn_p fp = chpl_ftable[fid];
    CHPL_BEGIN(fp, arg, false, false, NULL);
}

void CHPL_PROCESS_TASK_LIST(chpl_task_list_p task_list)
{
}

void CHPL_EXECUTE_TASKS_IN_LIST(chpl_task_list_p task_list)
{
}

void CHPL_FREE_TASK_LIST(chpl_task_list_p task_list)
{
}

//TODO
void CHPL_TASK_SLEEP(int secs)
{
  sleep(secs);
}

//TODO
chpl_bool CHPL_GET_SERIAL(void)
{
  return 0;
}

//TODO
void CHPL_SET_SERIAL(chpl_bool state)
{
}


// Task stats routines

uint32_t CHPL_NUMQUEUEDTASKS(void)
{
   return sys.getReadyNum();
}

uint32_t CHPL_NUMRUNNINGTASKS(void)
{
  return sys.getRunningTasks();
}

int32_t  CHPL_NUMBLOCKEDTASKS(void)
{
  return sys.getTaskNum() - sys.getReadyNum() -  sys.getRunningTasks();
}

//TODO
chpl_taskID_t CHPL_TASK_ID(void)
{
  return myThread->getCurrentWD()->getId();
}

// Threads stat routines

int32_t  CHPL_THREADS_GETMAXTHREADS(void)
{
   // TODO: Alex
   return 0;
}

int32_t  CHPL_THREADS_MAXTHREADSLIMIT(void)
{
   // TODO: Alex
   return 0;
}

uint32_t CHPL_NUMTHREADS(void)
{
   return sys.getNumWorkers();
}

uint32_t CHPL_NUMIDLETHREADS(void)
{
    return sys.getIdleNum();
}

