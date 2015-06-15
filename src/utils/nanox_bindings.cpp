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

#include <sched.h>
#include <iostream>
#include <unistd.h>
#include "os.hpp"
#include "system.hpp"
#include "debug.hpp"

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255
#endif

int main( int argc, char *argv[] )
{
   int pid = getpid();
   char hostname[HOST_NAME_MAX];
   std::ostringstream output;
   cpu_set_t cpu_set = sys.getCpuProcessMask();

   gethostname( hostname, HOST_NAME_MAX );

   output << "Nanos++: " << hostname << "::" << pid << " [ ";
   for ( int i=0; i<OS::getMaxProcessors(); i++ )
      if ( CPU_ISSET(i, &cpu_set) )
         output << i << " ";

   output << "]" << std::endl;
   std::cout << output.str();

   return 0;
}
