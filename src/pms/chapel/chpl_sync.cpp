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

#include "system.hpp"
#include "chpl_nanos.h"

using namespace nanos;

// // TODO: include chpl headers?
// typedef char * chpl_string;
// typedef bool chpl_bool;
// typedef void (*chpl_fn_p) (void *);
// typedef void * chpl_task_list_p;
// typedef int chpl_fn_int_t;
// typedef int chpl_taskID_t;
// 
// extern chpl_fn_p chpl_ftable[];

// Mutex interface

void CHPL_MUTEX_INIT(chpl_mutex_t* mutex)
{
}

chpl_mutex_t* CHPL_MUTEX_NEW(void)
{
   return (chpl_mutex_t*) new Lock();
}

void CHPL_MUTEX_LOCK(chpl_mutex_t* mutex)
{
   Lock *l = ( Lock * ) mutex;
   l++;
}

void CHPL_MUTEX_UNLOCK(chpl_mutex_t* mutex)
{
   Lock *l = ( Lock * ) mutex;
   l--;
}

// Sync variables interface

void CHPL_SYNC_INIT_AUX(chpl_sync_aux_t *s)
{
   s->is_full = false;

   s->empty = (void * )new SingleSyncCond<EqualConditionChecker<bool> >( EqualConditionChecker<bool>( &s->is_full, false ) );
   s->full = (void *) new SingleSyncCond<EqualConditionChecker<bool> >( EqualConditionChecker<bool>( &s->is_full, true ) );
}

void CHPL_SYNC_DESTROY_AUX(chpl_sync_aux_t *s)
{
  delete (GenericSyncCond *)s->empty;
  delete (GenericSyncCond *)s->full;
}

void CHPL_SYNC_WAIT_FULL_AND_LOCK(chpl_sync_aux_t *s,
                                  int32_t lineno, chpl_string filename)
{
  GenericSyncCond *sync = (GenericSyncCond *) s->full;
  sync->wait();
}

void CHPL_SYNC_WAIT_EMPTY_AND_LOCK(chpl_sync_aux_t *s,
                                   int32_t lineno, chpl_string filename)
{
  GenericSyncCond *sync = (GenericSyncCond *) s->full;
  sync->signal_one();
}

void CHPL_SYNC_MARK_AND_SIGNAL_FULL(chpl_sync_aux_t *s)
{
  s->is_full = true;
  GenericSyncCond *sync = (GenericSyncCond *) s->full;
  sync->signal_one();
}

void CHPL_SYNC_MARK_AND_SIGNAL_EMPTY(chpl_sync_aux_t *s)
{
  s->is_full = false;
  GenericSyncCond *sync = (GenericSyncCond *) s->full;
  sync->signal_one();
}

chpl_bool CHPL_SYNC_IS_FULL(void *val_ptr,
                            chpl_sync_aux_t *s,
                            chpl_bool simple_sync_var)
{
  return s->is_full;
}

void CHPL_SYNC_LOCK(chpl_sync_aux_t *s)
{
}

void CHPL_SYNC_UNLOCK(chpl_sync_aux_t *s)
{
}

// Single variables interface

void CHPL_SINGLE_INIT_AUX(chpl_single_aux_t *s)
{
   s->is_full = false;

   s->full = (void *) new SingleSyncCond<EqualConditionChecker<bool> >( EqualConditionChecker<bool>( &s->is_full, true ) );
}

void CHPL_SINGLE_DESTROY_AUX(chpl_single_aux_t *s)
{
   delete (GenericSyncCond *)s->full;
}

void CHPL_SINGLE_WAIT_FULL(chpl_single_aux_t *s,
                           int32_t lineno, chpl_string filename)
{
  GenericSyncCond *sync = (GenericSyncCond *) s->full;
  sync->wait();
}

void CHPL_SINGLE_MARK_AND_SIGNAL_FULL(chpl_single_aux_t *s)
{
  s->is_full = true;
  GenericSyncCond *sync = (GenericSyncCond *) s->full;
  sync->signal_one();
}

chpl_bool CHPL_SINGLE_IS_FULL(void *val_ptr,
                              chpl_single_aux_t *s,
                              chpl_bool simple_single_var)
{
  return s->is_full;
}


