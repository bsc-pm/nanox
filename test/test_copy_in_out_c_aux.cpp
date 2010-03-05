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

extern "C" {

void * aux_get_copies_addr( unsigned int i )
{
   WD *wd = myThread->getCurrentWD();
   CopyData* cd = wd->getCopies();
   return cd[i].getAddress();
}

nanos_sharing_t aux_get_sharing( unsigned int i )
{
   WD *wd = myThread->getCurrentWD();
   CopyData* cd = wd->getCopies();
   return cd[i].getSharing();
}


void check_hardcoded_copy_data ()
{
   WD *wd = myThread->getCurrentWD();
   CopyData* cd = wd->getCopies();

   if ( cd[0].getAddress() !=  (void *)1280 ) 
      std::cout << "Error: CopyData address '" << (unsigned long)cd[0].getAddress() << "' does not match argument with address '"
                << 1280 << "'." << std::endl;
   else std::cout << "Checking for CopyData address correctness... PASS" << std::endl;
   if ( cd[1].getAddress() !=  (void *)1024 ) 
      std::cout << "Error: CopyData address '" << (unsigned long)cd[1].getAddress() << "' does not match argument with address '"
                << 1024 << "'." << std::endl;
   else std::cout << "Checking for CopyData address correctness... PASS" << std::endl;

   if ( cd[0].getSize() != 255 )
      std::cout << "Error: CopyData size '" << cd[0].getSize() << "' does not match argument with size '"
                << 255 << "'." << std::endl;
   else std::cout << "Checking for CopyData size correctness... PASS" << std::endl;
   if ( cd[1].getSize() != 127 )
      std::cout << "Error: CopyData size '" << cd[1].getSize() << "' does not match argument with size '"
                << 127 << "'." << std::endl;
   else std::cout << "Checking for CopyData size correctness... PASS" << std::endl;

   if ( !cd[0].isInput() )
      std::cout << "Error: CopyData was supposed to be input." << std::endl;
   else std::cout << "Checking for CopyData direction correctness... PASS" << std::endl;
   if ( !cd[1].isOutput() )
      std::cout << "Error: CopyData was supposed to be output." << std::endl;
   else std::cout << "Checking for CopyData direction correctness... PASS" << std::endl;

   if ( !cd[0].isShared() )
      std::cout << "Error: CopyData was supposed to be NX_SHARED." <<  std::endl;
   else std::cout << "Checking for CopyData sharing... PASS" << std::endl;
   if ( !cd[1].isPrivate() )
      std::cout << "Error: CopyData was supposed to be NX_PRIVATE." <<  std::endl;
   else std::cout << "Checking for CopyData sharing... PASS" << std::endl;
}

}

