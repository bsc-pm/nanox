/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#include <iostream>
#include <unistd.h>
#include "os.hpp"
#include "system.hpp"
#include "debug.hpp"
#include "cpuset.hpp"

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255
#endif

using namespace nanos;

int main( int argc, char *argv[] )
{
   int pid = getpid();

   const CpuSet& cpu_set = sys.getCpuProcessMask();

   char hostname[HOST_NAME_MAX];
   gethostname( hostname, HOST_NAME_MAX );

   std::cout << "Nanos++: " << hostname << "::" << pid
      << " [ " << cpu_set.toString() << " ]" << std::endl;

   return 0;
}
