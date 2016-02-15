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
#include <getopt.h>
#include "os.hpp"
#include "system.hpp"
#include "debug.hpp"
#include "cpuset.hpp"

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255
#endif

using namespace nanos;

static void print_version()
{
   std::cout << PACKAGE << " " << VERSION << " (" << NANOX_BUILD_VERSION << ")" <<  std::endl;
   std::cout << "Configured with: " << NANOX_CONFIGURE_ARGS << std::endl;
}

static void print_help( const char* program )
{
   std::cout << "usage: " << program << " [-a|--all]" << std::endl;
   std::cout << std::endl;
   std::cout << "Report Nanos++ bindings according to the process affinity mask and the runtime workers" << std::endl;
   std::cout << std::endl;
   std::cout << "Options:" << std::endl;
   std::cout << "  -a, --all:    report all available bindings, ignoring number of threads" << std::endl;
   std::cout << "  -h, --help:   print this help" << std::endl;
   std::cout << std::endl;
   std::cout << "Examples:" << std::endl;
   std::cout << "  > nanox-bindings" << std::endl;
   std::cout << "  Nanos++: hostname::pid [ 0-15 ]" << std::endl;
   std::cout << "  > NX_ARGS=\"--smp-workers=4 --binding-stride=2\" nanox-bindings" << std::endl;
   std::cout << "  Nanos++: hostname::pid [ 0,2,4,6 ]" << std::endl;
}

int main( int argc, char *argv[] )
{
   bool do_all = false;
   bool do_help = false;
   bool do_version = false;

   int opt;
   struct option long_options[] = {
      {"all",     no_argument, 0, 'a'},
      {"help",    no_argument, 0, 'h'},
      {"version", no_argument, 0, 'v'},
      {0,         0,           0, 0 }
   };

   while ( (opt = getopt_long(argc, argv, "ahv", long_options, NULL)) != -1 ) {
      switch (opt) {
         case 'a':
            do_all = true;
            break;
         case 'h':
            do_help = true;
            break;
         case 'v':
            do_version = true;
            break;
         default:
            print_help( argv[0] );
            exit( EXIT_SUCCESS );
      }

   }

   if (do_version) {
      print_version();
      exit( EXIT_SUCCESS );
   }

   if (do_help) {
      print_help( argv[0] );
      exit( EXIT_SUCCESS );
   }

   pid_t pid = getpid();

   const CpuSet& cpu_set = do_all ? sys.getCpuProcessMask() : sys.getCpuActiveMask();

   char hostname[HOST_NAME_MAX];
   gethostname( hostname, HOST_NAME_MAX );

   std::cout << "Nanos++: " << hostname << "::" << pid
      << " [ " << cpu_set.toString() << " ]" << std::endl;

   exit( EXIT_SUCCESS );
}
