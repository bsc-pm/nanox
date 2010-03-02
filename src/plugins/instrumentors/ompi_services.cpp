#include "system.hpp"

namespace nanos {

extern "C" {

   void OMPItrace_neventandcounters (unsigned int count, unsigned int *types, unsigned int *values);

   unsigned int nanos_ompitrace_get_max_threads ( void )
   {
      return sys.getNumPEs();
   }

   unsigned int nanos_ompitrace_get_thread_num ( void )
   { 
      return myThread->getId(); 
   }

}

} // namespace nanos

