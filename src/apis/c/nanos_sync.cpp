#include "workgroup.hpp"
#include "nanos.h"
#include "schedule.hpp"

using namespace nanos;

nanos_err_t nanos_wg_wait_completation ( nanos_wg_t uwg )
{
  try {
    WG *wg = (WG *)uwg;
    wg->waitCompletation();
  } catch (...) {
    return NANOS_UNKNOWN_ERR;
  }
  return NANOS_OK;
}

nanos_err_t nanos_wait_on_int ( volatile int *p, int condition )
{
   try {
      Scheduler::blockOnCondition<int>(p,condition);
   } catch (...) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

nanos_err_t nanos_wait_on_bool ( volatile _Bool *p, _Bool condition )
{
   try {
      Scheduler::blockOnCondition<_Bool>(p,condition);
   } catch (...) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

nanos_err_t nanos_init_lock ( nanos_lock_t *lock )
{
   try {
      *lock = (nanos_lock_t ) new Lock();
   } catch (...) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

nanos_err_t nanos_set_lock ( nanos_lock_t lock )
{
    try {
       Lock *l = (Lock *) lock;
       l++;
    } catch (...) {
       return NANOS_UNKNOWN_ERR;
    }
    return NANOS_OK;
}

nanos_err_t nanos_unset_lock ( nanos_lock_t lock )
{
    try {
       Lock *l = (Lock *) lock;
       l--;
    } catch (...) {
       return NANOS_UNKNOWN_ERR;
    }
    return NANOS_OK;
}

nanos_err_t nanos_try_lock ( nanos_lock_t lock, bool *result )
{
    try {
       Lock *l = (Lock *) lock;
       
       *result = l->tryAcquire();
    } catch (...) {
       return NANOS_UNKNOWN_ERR;
    }
    return NANOS_OK;
}

nanos_err_t nanos_destroy_lock ( nanos_lock_t lock )
{
   try {
      delete (Lock *)lock;
   } catch (...) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}


nanos_err_t nanos_single_guard ( bool *b )
{
   try {
        *b = myThread->singleGuard();
   } catch (...) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}
