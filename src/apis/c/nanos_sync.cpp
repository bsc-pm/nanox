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

#include "workgroup.hpp"
#include "nanos.h"
#include "schedule.hpp"
#include "system.hpp"
#include "synchronizedcondition.hpp"

using namespace nanos;

nanos_err_t nanos_wg_wait_completion ( nanos_wg_t uwg )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( WG_WAIT_COMPLETION, RUNTIME );
      WG *wg = ( WG * )uwg;
      wg->waitCompletion();
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_create_int_sync_cond ( nanos_sync_cond_t *sync_cond, volatile int *p, int condition )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( SYNC_COND, RUNTIME );
      *sync_cond = ( nanos_sync_cond_t * ) new SingleSyncCond<EqualConditionChecker<int> >( EqualConditionChecker<int>( p, condition ) );
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_create_bool_sync_cond ( nanos_sync_cond_t *sync_cond, volatile bool *p, bool condition )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( SYNC_COND, RUNTIME );
      *sync_cond = ( nanos_sync_cond_t * ) new SingleSyncCond<EqualConditionChecker<bool> >( EqualConditionChecker<bool>( p, condition ) );
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_sync_cond_wait ( nanos_sync_cond_t *sync_cond )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( SYNC_COND, RUNTIME );
      GenericSyncCond * syncCond = (GenericSyncCond *) *sync_cond;
      syncCond->wait();
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_sync_cond_signal ( nanos_sync_cond_t *sync_cond )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( SYNC_COND, RUNTIME );
      GenericSyncCond * syncCond = (GenericSyncCond *) *sync_cond;
      syncCond->signal();
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_destroy_sync_cond ( nanos_sync_cond_t *sync_cond )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( SYNC_COND, RUNTIME );
      GenericSyncCond * syncCond = (GenericSyncCond *) *sync_cond;
      delete syncCond;
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_wait_on ( size_t num_deps, nanos_dependence_t *deps )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( WAIT_ON, RUNTIME );
      if ( deps != NULL ) {
         sys.waitOn( num_deps, deps );
         return NANOS_OK;
      }

   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_init_lock ( nanos_lock_t *lock )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( LOCK, RUNTIME );
      *lock = ( nanos_lock_t ) new Lock();
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_set_lock ( nanos_lock_t lock )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( LOCK, RUNTIME );
      Lock *l = ( Lock * ) lock;
      l++;
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_unset_lock ( nanos_lock_t lock )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( LOCK, RUNTIME );
      Lock *l = ( Lock * ) lock;
      l--;
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_try_lock ( nanos_lock_t lock, bool *result )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( LOCK, RUNTIME);
      Lock *l = ( Lock * ) lock;

      *result = l->tryAcquire();
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}

nanos_err_t nanos_destroy_lock ( nanos_lock_t lock )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( LOCK, RUNTIME );
      delete ( Lock * )lock;
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}


nanos_err_t nanos_single_guard ( bool *b )
{
   try {
      sys.getInstrumentor()->enterRuntimeAPI( SINGLE_GUARD, RUNTIME );
      *b = myThread->singleGuard();
   } catch ( ... ) {
      sys.getInstrumentor()->leaveRuntimeAPI();
      return NANOS_UNKNOWN_ERR;
   }

   sys.getInstrumentor()->leaveRuntimeAPI();
   return NANOS_OK;
}
