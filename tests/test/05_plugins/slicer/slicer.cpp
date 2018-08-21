/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

/*
<testinfo>
test_generator=gens/core-generator
</testinfo>
*/

#include "config.hpp"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include "slicer.hpp"
#include "plugin.hpp"
#include <string.h>

using namespace std;

using namespace nanos;
using namespace nanos::ext;

int a = 1234;
std::string b( "default" );
bool c = false;

typedef struct {
   nanos_repeat_n_info_t repeat_n_info;
   int a;
   char b[20];
   std::string c;
} hello_world_args;

void hello_world ( void *args );
void hello_world ( void *args )
{
   hello_world_args *hargs = ( hello_world_args * ) args;
   cout << "hello_world "
        << hargs->a << " "
        << hargs->b << " "
        << hargs->c
        << endl;
}

int main ( int argc, char **argv )
{
   cout << "PEs = " << sys.getSMPPlugin()->getNumPEs() << endl;
   cout << "Mode = " << sys.getExecutionMode() << endl;
   cout << "Verbose = " << sys.getVerbose() << endl;
   cout << "Args" << endl;
   for ( int i = 0; i < argc; i++ )
      cout << argv[i] << endl;
   cout << "start" << endl;

   hello_world_args *data;
   const char *str;

   // Work arguments
   str = "std::string(1)";
   data = new hello_world_args();
   data->a = 1;
   strncpy(data->b, "char *string(1)", strlen("char *string(1)"));
   data->c = str;

   // Work descriptor creation
   WD * wd1 = new WD( new SMPDD( hello_world ), sizeof(hello_world_args), __alignof__(hello_world_args), data );

   // Work arguments
   str = "std::string(2)";
   data = new hello_world_args();
   data->repeat_n_info.n = 10;
   data->a = 2;
   strncpy(data->b, "char *string(2)", strlen("char *string(2)"));
   data->c = str;

   // loading RepeatN Slicer Plugin
   sys.loadPlugin( "slicer-repeat_n" );
   Slicer *slicer = sys.getSlicer ( "repeat_n" );
 
   // Work descriptor creation
   WD * wd2 = new WorkDescriptor( new SMPDD( hello_world ), sizeof(hello_world_args), __alignof__(hello_world_args),data,0,NULL,NULL );
   wd2->setSlicer(slicer);

   // Work Group affiliation and work submision
   WD *wg = getMyThreadSafe()->getCurrentWD();
   wg->addWork( *wd1 );
   wg->addWork( *wd2 );

   if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
      char *idata = NEW char[sys.getPMInterface().getInternalDataSize()];
      sys.getPMInterface().initInternalData( idata );
      wd1->setInternalData( idata );
   }
   sys.setupWD(*wd1, (nanos::WD *) wg);
   sys.submit( *wd1 );

   if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
      char *idata = NEW char[sys.getPMInterface().getInternalDataSize()];
      sys.getPMInterface().initInternalData( idata );
      wd2->setInternalData( idata );
   }
   sys.setupWD(*wd1, (nanos::WD *) wg);
   sys.setupWD(*wd2, (nanos::WD *) wg);
   sys.submit( *wd2 );

   wg->waitCompletion();

   cout << "end" << endl;
}

