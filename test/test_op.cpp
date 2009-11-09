#include "config.hpp"
#include "nanos.h"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include <string.h>

using namespace std;
using namespace nanos;
using namespace nanos::ext;

void operExecutor ( void * arg )
{
   int * arg
   int cop = ( int ) ( ( *( int * ) arg ) );

   while ( true ) {
      switch ( cop ) {
      case 1: { printf( "first operation" ); break; }
      case 2: { printf( "second operation" ); break; }
      default: { printf( "default operation" ); break; }
      }
   }
}


int main (int argc, char **argv)
{
      cout << "start" << endl;
      //
      int cop = 1;
      for ( int i = 1; i < sys.getNumPEs(); i++ ) {
         WD * wd = new WD( new SMPDD ( operExecutor ), &cop );
         sys.submit( *wd );
      }

      operExecutor( NULL );

      cout << "end" << endl;
}

