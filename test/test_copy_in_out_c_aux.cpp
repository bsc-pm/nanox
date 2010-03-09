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

uint64_t aux_get_copies_addr( unsigned int i )
{
   WD *wd = myThread->getCurrentWD();
   CopyData* cd = wd->getCopies();
   return (uint64_t)cd[i].getAddress();
}

nanos_sharing_t aux_get_sharing( unsigned int i )
{
   WD *wd = myThread->getCurrentWD();
   CopyData* cd = wd->getCopies();
   return cd[i].getSharing();
}

}

