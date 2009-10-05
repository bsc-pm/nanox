#include "nanos.h"
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "workdescriptor.hpp"
#include "smpdd.hpp"

using namespace nanos;

// TODO: move to dependent part
void * nanos_smp_factory(void *args)
{
    nanos_smp_args_t *smp = (nanos_smp_args_t *) args;

    return (void *)new SMPDD(smp->outline);
}

nanos_wd_t nanos_current_wd()
{
   return myThread->getCurrentWD();
}

// FIX-ME: currently, it works only for SMP
nanos_err_t nanos_create_wd (  nanos_wd_t *uwd, size_t num_devices, nanos_device_t *devices, size_t data_size,
                              void ** data, nanos_wg_t uwg, nanos_wd_props_t *props )
{   
   try {
      if ( ( (!props) ||  (props && !props->mandatory_creation) ) && !sys.throttleTask() ) {
         *uwd = 0; 
         return NANOS_OK;
      }

      if ( num_devices > 1 ) warning("Multiple devices not yet supported. Using first one");

#if 0
      // there is problem at destruction with this right now
      
      int size_to_allocate = (*uwd == NULL) ? sizeof(SMPWD) : 0 +
                             (*data == NULL) ? data_size : 0
                             ;
      char *chunk=0;
      verbose("allocating " << data_size);
      if (size_to_allocate)
        chunk = new char[size_to_allocate];

      if ( *uwd == NULL ) {
          *uwd = (nanos_wd_t) chunk;
          chunk += sizeof(SMPWD);
      }
      
      if ( *data == NULL) {
         *data = chunk;
      }

      WD * wd = new (*uwd) SMPWD((void (*) (void *)) devices[0].factory(&devices[0].factory_args), *data);
#else
      WD *wd;

      if (*data == NULL)
        *data = new char[data_size];
      if (*uwd ==  NULL)
        *uwd = wd =  new WD((DD*) devices[0].factory(devices[0].arg), *data);
      else
        wd = (WD *)*uwd;
#endif

      // add to workgroup
      if ( uwg ) {
        WG * wg = (WG *)uwg;
        wg->addWork(*wd);
      }

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

nanos_err_t nanos_submit ( nanos_wd_t uwd, nanos_dependence_t *deps, nanos_team_t team )
{
   try {
     if (deps != NULL) {
        warning("Dependence support not implemented yet");
     }
     if (team != NULL) {
        warning("Submitting to another team not implemented yet");
     }
     ensure(uwd,"NULL WD received");
     WD * wd = (WD *) uwd;
     sys.submit(*wd);
   } catch (...) {
     return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

// data must be not null
nanos_err_t nanos_create_wd_and_run ( size_t num_devices, nanos_device_t *devices, void * data,
                                      nanos_dependence_t *deps, nanos_wd_props_t *props )
{
   try {
      if ( num_devices > 1 ) warning("Multiple devices not yet supported. Using first one");
      if ( deps != NULL) warning("Dependence support not implemented yet");

      WD wd((DD*) devices[0].factory(devices[0].arg), data);
      sys.inlineWork(wd);
      
   } catch (...) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

