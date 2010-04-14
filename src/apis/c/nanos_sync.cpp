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
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","wg_wait_completion"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      WG *wg = ( WG * )uwg;
      wg->waitCompletion();
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_create_int_sync_cond ( nanos_sync_cond_t *sync_cond, volatile int *p, int condition )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","*_create_sync_cond"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      *sync_cond = ( nanos_sync_cond_t * ) new SingleSyncCond<EqualConditionChecker<int> >( EqualConditionChecker<int>( p, condition ) );
   } catch ( ... ) {
      NANOS_INSTRUMENTOR( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_create_bool_sync_cond ( nanos_sync_cond_t *sync_cond, volatile bool *p, bool condition )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","*_create_sync_cond"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      *sync_cond = ( nanos_sync_cond_t * ) new SingleSyncCond<EqualConditionChecker<bool> >( EqualConditionChecker<bool>( p, condition ) );
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_sync_cond_wait ( nanos_sync_cond_t *sync_cond )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","sync_cond_wait"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      GenericSyncCond * syncCond = (GenericSyncCond *) *sync_cond;
      syncCond->wait();
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_sync_cond_signal ( nanos_sync_cond_t *sync_cond )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","sync_cond_signal"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      GenericSyncCond * syncCond = (GenericSyncCond *) *sync_cond;
      syncCond->signal();
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_destroy_sync_cond ( nanos_sync_cond_t *sync_cond )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","destroy_sync_cond"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      GenericSyncCond * syncCond = (GenericSyncCond *) *sync_cond;
      delete syncCond;
   } catch ( ... ) {
      NANOS_INSTRUMENTOR( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_wait_on ( size_t num_deps, nanos_dependence_t *deps )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","wait_on"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      if ( deps != NULL ) {
         sys.waitOn( num_deps, deps );
         return NANOS_OK;
      }

   } catch ( ... ) {
      NANOS_INSTRUMENTOR( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_init_lock ( nanos_lock_t *lock )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","*_lock"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      *lock = ( nanos_lock_t ) new Lock();
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_set_lock ( nanos_lock_t lock )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","*_lock"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      Lock *l = ( Lock * ) lock;
      l++;
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_unset_lock ( nanos_lock_t lock )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","*_lock"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      Lock *l = ( Lock * ) lock;
      l--;
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_try_lock ( nanos_lock_t lock, bool *result )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","*_lock"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      Lock *l = ( Lock * ) lock;

      *result = l->tryAcquire();
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}

nanos_err_t nanos_destroy_lock ( nanos_lock_t lock )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","*_lock"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      delete ( Lock * )lock;
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}


nanos_err_t nanos_single_guard ( bool *b )
{
   NANOS_INSTRUMENTOR( static Instrumentor *inst = sys.getInstrumentor() );
   try {
      NANOS_INSTRUMENTOR(static nanos_event_value_t val = inst->getInstrumentorDictionary()->getEventValue("api","single_guard"));
      NANOS_INSTRUMENTOR( inst->enterRuntimeAPI(val,RUNTIME) );
      *b = myThread->singleGuard();
   } catch ( ... ) {
      NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
      return NANOS_UNKNOWN_ERR;
   }

   NANOS_INSTRUMENTOR ( inst->leaveRuntimeAPI() );
   return NANOS_OK;
}
