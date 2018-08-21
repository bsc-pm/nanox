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
test_generator="gens/core-generator -d plain,regions,perfect-regions"
test_generator_ENV=( "NX_TEST_SCHEDULE=bf" )
</testinfo>
*/

#include "config.hpp"
#include "nanos.h"
#include "atomic.hpp"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"

using namespace std;

using namespace nanos;
using namespace nanos::ext;

const unsigned arraySize = 128;

bool check = true;

int *array;

void main_loop( void *args );

void main_loop( void *args )
{
   unsigned i;
   
   Scheduler::yield();
   
   for ( i = 0; i < arraySize; i++ )
   {
      // give a chance to execute other tasks
      Scheduler::yield();
      
      // If it's not zero, error if we're using regions
      if ( array[i] != 0 )
      {
         check = false;
      }
      
      // give a chance to execute other tasks
      Scheduler::yield();
   }
   
   // So... was everything ok? If we're using regions, that's good
   // Otherwise, it's an error)
   if( sys.getDefaultDependenciesManager() == "plain" )
      check = !check;

}
typedef struct {
   unsigned index;
} task_data_t;

void try_to_fail( void *args );

void try_to_fail( void *args )
{
   task_data_t* data = (task_data_t*) args;
   array[data->index] = 1;
}

int main ( int argc, char **argv )
{
   unsigned i;
   
   if ( posix_memalign( (void**) &array, sizeof( int [arraySize] ), sizeof(int [arraySize] ) ) != 0)
      return -1;

   memset( array, 0, sizeof( int ) * arraySize );
   
   // Stop scheduler
   sys.stopScheduler();
   sys.waitUntilThreadsPaused();
   WD *wg = getMyThreadSafe()->getCurrentWD();
   
   
   nanos_region_dimension_t dimLoop[1] = {{ sizeof( int )*arraySize, 0, sizeof( int )*arraySize }};
   nanos_data_access_t depsLoop[] = {{(void *)&array, {1,1,0,0,0}, 1, dimLoop, 0} };
   
   WD* wd = new WD( new SMPDD( main_loop ) );
   wg->addWork( *wd );
   sys.submitWithDependencies( *wd, 1, (nanos::DataAccess*)&depsLoop );
   
   for ( i = 0; i < (ptrdiff_t)arraySize; i++ )
   {
      nanos_region_dimension_t dimFail[1] = {{ sizeof( int ), (size_t) i, sizeof( int ) }};
      nanos_data_access_t depsFail[] = {{(void *)&array, {0,1,0,0,0}, 1, dimFail, (ptrdiff_t)i} };
      task_data_t task_data;
      task_data.index = i;
      
      wd = new WD( new SMPDD( try_to_fail ), sizeof( task_data ), __alignof__(task_data ), ( void * ) &task_data );
      wg->addWork( *wd );
      sys.submitWithDependencies( *wd, 1, (nanos::DataAccess*)&depsFail );
   }
   
   
   // Re-enable the scheduler
   sys.startScheduler();
   sys.waitUntilThreadsUnpaused();

   
   // barrier (kind of)
   wg->waitCompletion();
   
   if( !check ){
      printf( "Error!\n" );
      return 1;
   }
   
   return 0;
   
}
