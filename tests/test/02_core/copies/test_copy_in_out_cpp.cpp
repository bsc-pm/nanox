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
#include "copydata.hpp"
#include <string.h>
#include <unistd.h>

using namespace std;

using namespace nanos;
using namespace nanos::ext;

typedef struct {
   int a;
   std::string b;
} hello_world_args;

void hello_world ( void *args );

void hello_world ( void *args )
{
   WD *wd = getMyThreadSafe()->getCurrentWD();
   hello_world_args *hargs = ( hello_world_args * ) args;
   CopyData* cd = wd->getCopies();

   if ( (void *)cd[0].getAddress() !=  (void *)&(hargs->a) ) {
      std::cout << "Error: CopyData address '" << cd[0].getAddress() << "' does not match argument with address '"
                << &(hargs->a) << "'." << std::endl;
      abort();
   } else {
      std::cout << "Checking for CopyData address correctness... PASS" << std::endl;
   }

   if ( (void *)( (char *)hargs + (unsigned long)cd[1].getAddress() ) != (void *) &(hargs->b) ) {
      std::cout << "Error: CopyData address '" << cd[1].getAddress() << "' does not match argument with address '"
                << &(hargs->b) << "'." << std::endl;
      abort();
   } else {
      std::cout << "Checking for CopyData address correctness... PASS" << std::endl;
   }

   if ( cd[0].getSize() != sizeof(hargs->a) ) {
      std::cout << "Error: CopyData size '" << cd[0].getSize() << "' does not match argument with size '"
                << sizeof((hargs->b)) << "'." << std::endl;
      abort();
   } else {
      std::cout << "Checking for CopyData size correctness... PASS" << std::endl;
   }

   if ( cd[1].getSize() != sizeof(hargs->b) ) {
      std::cout << "Error: CopyData size '" << cd[1].getSize() << "' does not match argument with size '"
                << sizeof((hargs->b)) << "'." << std::endl;
      abort();
   } else {
      std::cout << "Checking for CopyData size correctness... PASS" << std::endl;
   }

   if ( !cd[0].isInput() ) {
      std::cout << "Error: CopyData was supposed to be input." << std::endl;
      abort();
   } else {
      std::cout << "Checking for CopyData direction correctness... PASS" << std::endl;
   }

   if ( !cd[1].isOutput() ) {
      std::cout << "Error: CopyData was supposed to be output." << std::endl;
      abort();
   } else {
      std::cout << "Checking for CopyData direction correctness... PASS" << std::endl;
   }
   
   if ( !cd[0].isShared() ) {
      std::cout << "Error: CopyData was supposed to be NANOS_SHARED." <<  std::endl;
      abort();
   } else {
      std::cout << "Checking for CopyData sharing... PASS" << std::endl;
   }

   if ( !cd[1].isPrivate() ) {
      std::cout << "Error: CopyData was supposed to be NANOS_PRIVATE." <<  std::endl;
      abort();
   } else {
      std::cout << "Checking for CopyData sharing... PASS" << std::endl;
   }
}

int main ( int argc, char **argv )
{
   const char *a = "alex";

   hello_world_args *data = new hello_world_args();

   data->a = 1;

   data->b = a;


   nanos_region_dimension_internal_t dims[2];

   dims[0] = (nanos_region_dimension_internal_t) {sizeof(data->a), 0, sizeof(data->a)};
   dims[1] = (nanos_region_dimension_internal_t) {sizeof(data->b), 0, sizeof(data->b)};

   CopyData cd[2] = { CopyData( (uint64_t)&data->a, NANOS_SHARED, true, false, 1, &dims[0], 0 ),
                      CopyData( (uint64_t)&data->b, NANOS_PRIVATE, true, true, 1, &dims[1], 0 ) };

   WD * wd = new WD( new SMPDD( hello_world ), sizeof( hello_world_args ), __alignof__( hello_world_args ), data, 2, cd );

   WD *wg = getMyThreadSafe()->getCurrentWD();

   wg->addWork( *wd );

   if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
      char *idata = NEW char[sys.getPMInterface().getInternalDataSize()];
      sys.getPMInterface().initInternalData( idata );
      wd->setInternalData( idata );
   }

   sys.setupWD(*wd, (nanos::WD *) wg);
   sys.submit( *wd );

   usleep( 500 );

   wg->waitCompletion();

}
