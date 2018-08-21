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

#include "cpuset.hpp"
#include "system.hpp"
#include <iostream>
#include <assert.h>

/*
<testinfo>
   test_generator="gens/core-generator -a --no-warmup-threads|--warmup-threads"
   test_generator_ENV=( "NX_TEST_MAX_CPUS=1"
                        "NX_TEST_SCHEDULE=bf"
                        "NX_TEST_ARCH=smp")
   test_exec_command="timeout 1m"
</testinfo>
*/

using namespace nanos;
using namespace nanos::ext;

enum { MIN_CPUS = 2 };

int main ( int argc, char *argv[])
{
   /* Skip test if binding is disabled */
   if ( !sys.getSMPPlugin()->getBinding() ) {
      return EXIT_SUCCESS;
   }

   /* Skip test if process mask does not contains at least MIN_CPUS */
   const CpuSet& process_mask = sys.getCpuProcessMask();
   if ( process_mask.size() < MIN_CPUS ) {
      std::cout << "Skipping " << argv[0] << " test, not enough CPUs" << std::endl;
      return EXIT_SUCCESS;
   }

   /* Test active mask provided by Nanos++ and by the system */
   const CpuSet initial_active_mask( sys.getCpuActiveMask() );
   CpuSet initial_sched_mask;
   sched_getaffinity( 0, sizeof(cpu_set_t), initial_sched_mask.get_cpu_set_pointer() );
   assert( initial_sched_mask.isSubsetOf( initial_active_mask) );

   /* Set active mask to first MIN_CPUS of the process mask */
   CpuSet mask;
   for ( CpuSet::const_iterator it=process_mask.begin();
         it!=process_mask.end() && mask.size() < MIN_CPUS;
         ++it ) {
      mask.set(*it);
   }
   sys.setCpuActiveMask( mask );
   assert( myThread->getTeam()->getFinalSize() == MIN_CPUS );
   CpuSet sched_mask;
   sched_getaffinity( 0, sizeof(cpu_set_t), sched_mask.get_cpu_set_pointer() );
   assert( sched_mask.isSubsetOf( mask) );

   return EXIT_SUCCESS;
}

