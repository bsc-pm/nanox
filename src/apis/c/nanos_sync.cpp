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

/*! \file nanos_sync.cpp
 *  \brief
 */

#include "nanos.h"
#include "schedule.hpp"
#include "system.hpp"
#include "atomic.hpp"
#include "synchronizedcondition.hpp"
#include "instrumentationmodule_decl.hpp"
#include "instrumentation.hpp"

/*! \defgroup capi_sync Synchronization services.
 *  \ingroup capi
 */
/*! \addtogroup capi_sync
 *  \{
 */

using namespace nanos;

NANOS_API_DEF(nanos_err_t, nanos_wg_wait_completion_mandatory, ( nanos_wg_t uwg, bool avoid_flush ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","wg_wait_completion",NANOS_SYNCHRONIZATION) );

   NANOS_INSTRUMENT( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT( static nanos_event_key_t taskwait  = ID->getEventKey("taskwait"); )
   NANOS_INSTRUMENT( unsigned wd_id = ((WD *)uwg)->getId( ); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &taskwait, (nanos_event_value_t *) &wd_id); )

   try {
      WD *wg = ( WD * )uwg;
      wg->waitCompletion( avoid_flush );
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_wg_wait_completion, ( nanos_wg_t uwg, bool avoid_flush ))
{
   WD *wg = ( WD * )uwg;
   if ( wg->isFinal() ) return NANOS_OK;
   return nanos_wg_wait_completion_mandatory( uwg, avoid_flush );
}

NANOS_API_DEF(nanos_err_t, nanos_create_int_sync_cond, ( nanos_sync_cond_t *sync_cond, volatile int *p, int condition ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","*_create_sync_cond",NANOS_RUNTIME ) );

   try {
      *sync_cond = ( nanos_sync_cond_t ) NEW MultipleSyncCond<EqualConditionChecker<int> >( EqualConditionChecker<int>(
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
                  // We cannot change the external interface of this API now
                  (int*)p,
#else
                  p,
#endif
                  condition ) );
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_create_bool_sync_cond, ( nanos_sync_cond_t *sync_cond, volatile bool *p, bool condition ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","*_create_sync_cond",NANOS_RUNTIME) );

   try {
      *sync_cond = ( nanos_sync_cond_t ) NEW MultipleSyncCond<EqualConditionChecker<bool> >( EqualConditionChecker<bool>(
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
                  // We cannot change the external interface of this API now
                  (bool*)p,
#else
                  p,
#endif
                  condition ) );
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_sync_cond_wait, ( nanos_sync_cond_t sync_cond ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","sync_cond_wait",NANOS_SYNCHRONIZATION) );

   try {
      GenericSyncCond * syncCond = (GenericSyncCond *) sync_cond;
      syncCond->wait();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_sync_cond_signal, ( nanos_sync_cond_t sync_cond ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","sync_cond_signal",NANOS_SYNCHRONIZATION) );

   try {
      GenericSyncCond * syncCond = (GenericSyncCond *) sync_cond;
      syncCond->signal();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_destroy_sync_cond, ( nanos_sync_cond_t sync_cond ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","destroy_sync_cond",NANOS_RUNTIME) );

   try {
      GenericSyncCond * syncCond = (GenericSyncCond *) sync_cond;
      delete syncCond;
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_wait_on, ( size_t num_data_accesses, nanos_data_access_t *data_accesses ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","wait_on",NANOS_SYNCHRONIZATION ); )

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t wd_num_deps = ID->getEventKey("wd-num-deps"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t wd_deps_ptr = ID->getEventKey("wd-deps-ptr"); )

   NANOS_INSTRUMENT ( nanos_event_key_t Keys[2]; )
   NANOS_INSTRUMENT ( nanos_event_value_t Values[2]; )

   NANOS_INSTRUMENT ( Keys[0] = wd_num_deps; )
   NANOS_INSTRUMENT ( Values[0] = (nanos_event_value_t) num_data_accesses; )

   NANOS_INSTRUMENT ( Keys[1] = wd_deps_ptr; );
   NANOS_INSTRUMENT ( Values[1] = (nanos_event_value_t) data_accesses; )

   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(2, Keys, Values); )

   try {
      if ( data_accesses != NULL ) {
         sys.waitOn( num_data_accesses, data_accesses );
         return NANOS_OK;
      }

   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_init_lock, ( nanos_lock_t **lock ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","init_lock",NANOS_RUNTIME) );

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t Keys = ID->getEventKey("lock-addr"); )
   NANOS_INSTRUMENT ( nanos_event_value_t Values = (nanos_event_value_t) *lock; )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &Keys, &Values); )

   try {
      *lock = NEW Lock();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_init_lock_at, ( nanos_lock_t *lock ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","init_lock",NANOS_RUNTIME) );

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t Keys = ID->getEventKey("lock-addr"); )
   NANOS_INSTRUMENT ( nanos_event_value_t Values = (nanos_event_value_t) lock; )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &Keys, &Values); )

   try {
      new ( lock ) Lock();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_set_lock, ( nanos_lock_t *lock ))
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","set_lock",NANOS_SYNCHRONIZATION) );

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t Keys = ID->getEventKey("lock-addr"); )
   NANOS_INSTRUMENT ( nanos_event_value_t Values = (nanos_event_value_t) lock; )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &Keys, &Values); )

   try {
      Lock &l = *( Lock * ) lock;
      l++;
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_unset_lock, ( nanos_lock_t *lock ))
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","unset_lock",NANOS_SYNCHRONIZATION) );

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t Keys = ID->getEventKey("lock-addr"); )
   NANOS_INSTRUMENT ( nanos_event_value_t Values = (nanos_event_value_t) 0; )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &Keys, &Values); )

   try {
      Lock &l = *( Lock * ) lock;
      l--;
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_try_lock, ( nanos_lock_t *lock, bool *result ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","try_lock",NANOS_SYNCHRONIZATION) );

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t Keys = ID->getEventKey("lock-addr"); )
   NANOS_INSTRUMENT ( nanos_event_value_t Values = (nanos_event_value_t) lock; )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &Keys, &Values); )

   try {
      Lock &l = *( Lock * ) lock;

      *result = l.tryAcquire();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_destroy_lock, ( nanos_lock_t *lock ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","destroy_lock",NANOS_RUNTIME) );

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )

   NANOS_INSTRUMENT ( static nanos_event_key_t Keys = ID->getEventKey("lock-addr"); )
   NANOS_INSTRUMENT ( nanos_event_value_t Values = (nanos_event_value_t) lock; )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, &Keys, &Values); )

   try {
      delete ( Lock * )lock;
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}


NANOS_API_DEF(nanos_err_t, nanos_single_guard, ( bool *b ))
{
   NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","single_guard",NANOS_SYNCHRONIZATION) );

   try {
      *b = myThread->singleGuard();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_enter_sync_init, ( bool *b ))
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","single_guard",NANOS_SYNCHRONIZATION) );

   try {
      *b = myThread->enterSingleBarrierGuard();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_wait_sync_init, ( void ))
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","single_guard",NANOS_SYNCHRONIZATION) );

   try {
      myThread->waitSingleBarrierGuard();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_release_sync_init, ( void ))
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","single_guard",NANOS_SYNCHRONIZATION) );

   try {
      myThread->releaseSingleBarrierGuard();
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_memory_fence, (void))
{
    nanos::memoryFence();
    return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_get_lock_address, ( void *addr, nanos_lock_t **lock ))
{
   //NANOS_INSTRUMENT( InstrumentStateAndBurst inst("api","nanos_get_lock",NANOS_SYNCHRONIZATION) );

   try {
      *lock = sys.getLockAddress(addr);
   } catch ( nanos_err_t e) {
      return e;
   }

   return NANOS_OK;
}
/*!
 * \}
 */
