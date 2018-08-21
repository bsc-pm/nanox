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

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>

int main()
{
   int i;
   int MAX_CPUS = 2;
   std::ostringstream exec_versions;
   exec_versions << "exec_versions=\"";
   std::ostringstream versions_env;
   char *max_cpus = getenv("NX_TEST_MAX_CPUS");
   if ( max_cpus != NULL) MAX_CPUS = atoi(max_cpus);

//   printf("exec_versions=\"normal cilk\"\n");

   for ( i=1; i<=MAX_CPUS; i++ ) {
      exec_versions << "normal" << i << " ";
      exec_versions << "cilk" << i << " ";
      versions_env << "test_ENV_normal" << i << "=\"NX_ARGS='--smp-cpus=" << i <<"'\"" << std::endl;
      versions_env << "test_ENV_cilk" << i << "=\"NX_ARGS='--schedule=cilk --smp-cores=" << i << "'\"" << std::endl;
//      printf("test_ENV_normal=\"NX_ARGS=''\"\n");
//      printf("test_ENV_cilk=\"NX_ARGS='--schedule cilk'\"\n");
   }
   exec_versions << "\"" << std::endl;

   std::cout << exec_versions.str();
   std::cout << versions_env.str();
}

