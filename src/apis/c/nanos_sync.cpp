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
#include "instrumentormodule_decl.hpp"

using namespace nanos;

nanos_err_t nanos_wg_wait_completion ( nanos_wg_t uwg )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","wg_wait_completion",RUNTIME) );

   try {
      WG *wg = ( WG * )uwg;
      wg->waitCompletion();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_create_int_sync_cond ( nanos_sync_cond_t *sync_cond, volatile int *p, int condition )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","*_create_sync_cond",RUNTIME) );

   try {
      *sync_cond = ( nanos_sync_cond_t * ) new SingleSyncCond<EqualConditionChecker<int> >( EqualConditionChecker<int>( p, condition ) );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_create_bool_sync_cond ( nanos_sync_cond_t *sync_cond, volatile bool *p, bool condition )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","*_create_sync_cond",RUNTIME) );

   try {
      *sync_cond = ( nanos_sync_cond_t * ) new SingleSyncCond<EqualConditionChecker<bool> >( EqualConditionChecker<bool>( p, condition ) );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_sync_cond_wait ( nanos_sync_cond_t *sync_cond )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","sync_cond_wait",RUNTIME) );

   try {
      GenericSyncCond * syncCond = (GenericSyncCond *) *sync_cond;
      syncCond->wait();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_sync_cond_signal ( nanos_sync_cond_t *sync_cond )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","sync_cond_signal",RUNTIME) );

   try {
      GenericSyncCond * syncCond = (GenericSyncCond *) *sync_cond;
      syncCond->signal();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_destroy_sync_cond ( nanos_sync_cond_t *sync_cond )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","destroy_sync_cond",RUNTIME) );

   try {
      GenericSyncCond * syncCond = (GenericSyncCond *) *sync_cond;
      delete syncCond;
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_wait_on ( size_t num_deps, nanos_dependence_t *deps )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","wait_on",RUNTIME) );

   NANOS_INSTRUMENTOR ( static nanos_event_key_t wait_key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("wait-on") );
   NANOS_INSTRUMENTOR ( static nanos_event_key_t dep_key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("nanos-dep") );
   NANOS_INSTRUMENTOR ( unsigned int nkvs = num_deps+1 );
   NANOS_INSTRUMENTOR ( nanos_event_key_t *Keys = new nanos_event_key_t(nkvs) );
   NANOS_INSTRUMENTOR ( nanos_event_value_t *Values = new nanos_event_value_t(nkvs) );
   NANOS_INSTRUMENTOR ( Keys[0] = (unsigned int) wait_key );
   NANOS_INSTRUMENTOR ( Values[0] = (nanos_event_value_t) myThread->getCurrentWD()->getId() );
   NANOS_INSTRUMENTOR ( for (unsigned int i = 1; i< nkvs; i++) {
       Keys[i] = (unsigned int) dep_key;
       Values[i] = (nanos_event_value_t) &deps[i-1];
   } );
   NANOS_INSTRUMENTOR( sys.getInstrumentor()->throwPointEventNkvs(nkvs, Keys, Values) );

   try {
      if ( deps != NULL ) {
         sys.waitOn( num_deps, deps );
         return NANOS_OK;
      }

   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_init_lock ( nanos_lock_t *lock )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","*_lock",RUNTIME) );

   try {
      *lock = ( nanos_lock_t ) new Lock();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_set_lock ( nanos_lock_t lock )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","*_lock",RUNTIME) );

   try {
      Lock *l = ( Lock * ) lock;
      l++;
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_unset_lock ( nanos_lock_t lock )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","*_lock",RUNTIME) );

   try {
      Lock *l = ( Lock * ) lock;
      l--;
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_try_lock ( nanos_lock_t lock, bool *result )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","*_lock",RUNTIME) );

   try {
      Lock *l = ( Lock * ) lock;

      *result = l->tryAcquire();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_destroy_lock ( nanos_lock_t lock )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","*_lock",RUNTIME) );

   try {
      delete ( Lock * )lock;
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}


nanos_err_t nanos_single_guard ( bool *b )
{
   NANOS_INSTRUMENTOR( InstrumentorStateAndBurst inst("api","single_guard",RUNTIME) );

   try {
      *b = myThread->singleGuard();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

