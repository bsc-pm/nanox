/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "config.hpp"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"
#include "copydata.hpp"
#include <string.h>

using namespace std;

using namespace nanos;
using namespace nanos::ext;

int a = 1234;
std::string b( "default" );
bool c = false;

typedef struct {
   int a;
   std::string b;
} hello_world_args;

void hello_world ( void *args )
{
   WD *wd = myThread->getCurrentWD();
   hello_world_args *hargs = ( hello_world_args * ) args;
   CopyData* cd = wd->getCopies();

   if ( cd[0].getAddress() !=  (void *)&(hargs->a) )
      std::cout << "Error: CopyData address '" << cd[0].getAddress() << "' does not match argument with address '"
                << &(hargs->a) << "'." << std::endl;
   else std::cout << "Checking for CopyData address correctness... PASS" << std::endl;
   if ( (void *)cd[1].getAddress() != (void *) &(hargs->b) )
      std::cout << "Error: CopyData address '" << cd[1].getAddress() << "' does not match argument with address '"
                << &(hargs->b) << "'." << std::endl;
   else std::cout << "Checking for CopyData address correctness... PASS" << std::endl;

   if ( cd[0].getSize() != sizeof(hargs->a) )
      std::cout << "Error: CopyData size '" << cd[0].getSize() << "' does not match argument with size '"
                << sizeof((hargs->b)) << "'." << std::endl;
   else std::cout << "Checking for CopyData size correctness... PASS" << std::endl;
   if ( cd[1].getSize() != sizeof(hargs->b) )
      std::cout << "Error: CopyData size '" << cd[1].getSize() << "' does not match argument with size '"
                << sizeof((hargs->b)) << "'." << std::endl;
   else std::cout << "Checking for CopyData size correctness... PASS" << std::endl;

   if ( !cd[0].isInput() )
      std::cout << "Error: CopyData was supposed to be input." << std::endl;
   else std::cout << "Checking for CopyData direction correctness... PASS" << std::endl;
   if ( !cd[1].isOutput() )
      std::cout << "Error: CopyData was supposed to be output." << std::endl;
   else std::cout << "Checking for CopyData direction correctness... PASS" << std::endl;
   
}

int main ( int argc, char **argv )
{
   const char *a = "alex";

   hello_world_args *data = new hello_world_args();

   data->a = 1;

   data->b = a;

   CopyData cd[2] = { CopyData((void *)&data->a, true, false, sizeof(data->a) ), CopyData( (void *)&data->b, false, true, sizeof(data->b) ) };

   WD * wd = new WD( new SMPDD( hello_world ), sizeof( hello_world_args ), data, 2, cd );

   WG *wg = myThread->getCurrentWD();

   wg->addWork( *wd );

   sys.submit( *wd );

   usleep( 500 );

   wg->waitCompletation();

}
