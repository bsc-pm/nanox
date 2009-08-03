#include "nanos.h"
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "workdescriptor.hpp"

using namespace nanos;

nanos_err_t nanos_create_wd ( nanos_device_t *devices, nanos_wd_t **wd, size_t data_size,
                              void ** data, nanos_wd_props_t *props )
{
   return NANOS_UNIMPLEMENTED;
}

nanos_err_t nanos_submit ( nanos_wd_t *wd, nanos_dependence_t *deps, nanos_team_t *team )
{
   if (deps != NULL) {
      warning("Dependence support not implemented yet");
   }
   if (team != NULL) {
      warning("Submitting to another team not implemented yet");
   }

   WD * iwd = (WD *) wd;

   try {
     sys.submit(*iwd);
   } catch (...) {
     return NANOS_ERR;
   }
   
   return NANOS_OK;
}

nanos_err_t nanos_create_wd_and_run ( nanos_device_t *devices, void * data, nanos_dependence_t *deps,
                                      nanos_wd_props_t *props )
{
   return NANOS_UNIMPLEMENTED;
}

