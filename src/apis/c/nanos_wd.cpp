#include "nanos.h"
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "workdescriptor.hpp"
#include "smpwd.hpp"

using namespace nanos;

// TODO: move smp to some dependent part
typedef struct {
   void (*outline) (void *);
} nanos_smp_args_t;

void * nanos_smp_factory(void *args)
{
    nanos_smp_args_t *smp = (nanos_smp_args_t *) args;

    return (void *)smp->outline;
}

// FIX-ME: currently, it works only for SMP
nanos_err_t nanos_create_wd ( nanos_device_t *devices, nanos_wd_t **uwd, size_t data_size,
                              void ** data, nanos_wd_props_t *props )
{   
   try {
      // TODO: check throttling
      // if ( sys.throttleTask() ) ...
      
      int size_to_allocate = (*uwd == NULL) ? sizeof(SMPWD) : 0 +
                             (*data == NULL) ? data_size : 0
                             ;
      char *chunk=0;
      
      if (size_to_allocate)
        chunk = new char[size_to_allocate];

      if ( *uwd == NULL ) {
          *uwd = (nanos_wd_t*) chunk;
          chunk += sizeof(SMPWD);
      }
      if ( *data == NULL) {
         *data = chunk;
      }

      WD * wd = new (*uwd) SMPWD((void (*) (WD*)) devices[0].factory(&devices[0].factory_args), (WorkData *)*data);
      
      // set properties
      if ( props != NULL ) {
          if (props->tied) wd->tied();
          if (props->tie_to) wd->tieTo(*(BaseThread *)props->tie_to);
      }

   } catch (...) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

nanos_err_t nanos_submit ( nanos_wd_t *uwd, nanos_dependence_t *deps, nanos_team_t *team )
{
   if (deps != NULL) {
      warning("Dependence support not implemented yet");
   }
   if (team != NULL) {
      warning("Submitting to another team not implemented yet");
   }
   try {
     WD * wd = (WD *) uwd;
     sys.submit(*wd);
   } catch (...) {
     return NANOS_UNKNOWN_ERR;
   }
   
   return NANOS_OK;
}

nanos_err_t nanos_create_wd_and_run ( nanos_device_t *devices, void * data, nanos_dependence_t *deps,
                                      nanos_wd_props_t *props )
{
   return NANOS_UNIMPLEMENTED;
}

