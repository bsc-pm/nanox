#include "printbt_decl.hpp"

#include <stdlib.h>
#include <execinfo.h>
#include <iostream>

void nanos::printBt() {
   void* tracePtrs[100];
   int count = backtrace( tracePtrs, 100 );
   char** funcNames = backtrace_symbols( tracePtrs, count );
   std::cerr << "+--------------------------------------" << std::endl;

   // Print the stack trace
   for( int ii = 0; ii < count; ii++ )
      std::cerr << "| " << funcNames[ii] << std::endl;

   // Free the string pointers
   free( funcNames );
   std::cerr << "+--------------------------------------" << std::endl;
}

