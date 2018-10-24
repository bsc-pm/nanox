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
#include "system.hpp"
#include "os.hpp"

/*
<testinfo>
   test_generator="gens/core-generator -a --no-warmup-threads|--warmup-threads"
   test_generator_ENV=( "NX_TEST_MAX_CPUS=1"
                        "NX_TEST_SCHEDULE=bf"
                        "NX_TEST_ARCH=smp")
   test_exec_command="timeout 1m"
</testinfo>
*/

#define ITERS 1000

using namespace nanos;

int main ( int argc, char *argv[])
{
   int i, error = 0;
   unsigned nths = 0;
   int max_threads = std::min( OS::getMaxProcessors(), 4 );

   for ( i=0; i<ITERS; i++ ) {

      nths = (nths % max_threads) + 1;

      sys.updateActiveWorkers( nths );

      fprintf(stdout,"[%d/%d] Team final size is %d and %d is expected\n", i, ITERS, (int) myThread->getTeam()->getFinalSize(), nths );

      if ( myThread->getTeam()->getFinalSize() != nths ) error++;
   }

   fprintf(stdout,"Result is %s\n", error? "UNSUCCESSFUL":"successful");

   return error;
}

