#include "printbt_decl.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <execinfo.h>

void nanos::printBt() {
   void* tracePtrs[100];
   int count = backtrace( tracePtrs, 100 );
   char** funcNames = backtrace_symbols( tracePtrs, count );

   // Print the stack trace
   for( int ii = 0; ii < count; ii++ )
      fprintf(stderr, "%s\n", funcNames[ii] );

   // Free the string pointers
   free( funcNames );
}

