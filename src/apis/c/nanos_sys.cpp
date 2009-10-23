#include "nanos.h"
#include "system.hpp"

using namespace nanos;

nanos_err_t nanos_get_num_running_tasks ( int *num )
{
   try {
      *num = sys.getRunningTasks();
   } catch (...) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}