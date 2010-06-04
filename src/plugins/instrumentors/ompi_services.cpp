#include "system.hpp"

namespace nanos {

extern "C" {

   void OMPItrace_neventandcounters (unsigned int count, unsigned int *types, unsigned int *values);

   unsigned int nanos_ompitrace_get_max_threads ( void )
   {
      return sys.getNumPEs()+2;
   }

   unsigned int nanos_ompitrace_get_thread_num ( void )
   { 
      if ( myThread == NULL ) return 0;
      else return myThread->getId(); 
   }

}

} // namespace nanos

