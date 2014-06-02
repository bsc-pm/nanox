#include "printbt_decl.hpp"

#include <stdlib.h>
#include <execinfo.h>
#include <iostream>

void nanos::printBt( std::ostream &o ) {
   void* tracePtrs[100];
   int count = backtrace( tracePtrs, 100 );
   char** funcNames = backtrace_symbols( tracePtrs, count );
   o << "+--------------------------------------" << std::endl;

   // Print the stack trace
   for( int ii = 0; ii < count; ii++ )
      o << "| " << funcNames[ii] << std::endl;

   // Free the string pointers
   free( funcNames );
   o << "+--------------------------------------" << std::endl;
}

